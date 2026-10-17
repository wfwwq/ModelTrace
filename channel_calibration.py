"""多渠道整合与指纹库校准。

同一个模型往往能从多个渠道拿到：官方 API、订阅客户端、各家中转站……
单一渠道采集的指纹会混入该渠道特有的偏差（系统提示词、包装层、采样参数、甚至被掉包）。
本模块负责：

1. 渠道一致性校验：用“其他渠道 / 已有可信渠道”建立的指纹去识别每个渠道的回答，
   找出与同名模型明显不符的可疑渠道（例如中转站把 Opus 换成了 Sonnet）。
2. 整合：只把通过校验的渠道并入参考数据，每条回答带 channel_id。
3. 校准：bank_builder 在建库时按渠道等权求模型中心、额外投影掉“渠道干扰方向”，
   并用跨渠道留出交叉验证拟合温度，使概率反映“识别一个新渠道”的真实难度。

命令行：
    python channel_calibration.py report  --bank claude [--model claude-opus-4-7]
    python channel_calibration.py import  --bank claude --model claude-opus-4-7 \
        --channel 官方=official.jsonl --channel 中转A=relay_a.jsonl [--keep-suspect] [--dry-run]
    python channel_calibration.py rebuild [--bank claude]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import secrets
from collections import Counter
from pathlib import Path

import numpy as np

from bank_builder import build_bank, channel_of, fit_robust_artifacts, read_rows
from challenge_suite import fingerprint_suite
from fingerprint import count_numbers, js_similarity, parse_numbers, robust_score_numbers, softmax


PROJECT = Path(__file__).resolve().parent
DATA_DIR = PROJECT / "data"
CUSTOM_BANKS_FILE = DATA_DIR / "custom_banks.json"
UNIFIED_BANK_FILE = DATA_DIR / "unified_bank.json"
BUILTIN_BANKS = {"gpt": "GPT", "claude": "Claude"}

# 判定阈值
CONSISTENT_TOP1 = 0.6   # 单条回答命中同名模型的比例达到该值视为一致
SUSPECT_TOP1 = 0.3      # 低于该值且整体归因不是同名模型 -> 可疑
DEFAULT_BETA_1 = 7.0
DEFAULT_BETA_3 = 12.0


# ---------------------------------------------------------------- 基础工具

def slugify_channel(name: str) -> str:
    slug = re.sub(r"[^0-9a-zA-Z\u4e00-\u9fff]+", "-", name.strip()).strip("-")
    return slug or f"channel-{secrets.token_hex(3)}"


def prepare_row(row: dict) -> dict:
    """为新采集的行补上 numbers / counts，与 read_rows 的输出格式对齐。"""
    if "numbers" in row and "counts" in row:
        return row
    numbers = parse_numbers(row["text"])
    return {**row, "numbers": numbers, "counts": count_numbers(numbers)}


def bank_paths(bank_id: str) -> tuple[Path, Path]:
    return DATA_DIR / f"{bank_id}_reference.jsonl", DATA_DIR / f"{bank_id}_bank.json"


def bank_labels() -> dict[str, str]:
    labels = dict(BUILTIN_BANKS)
    if CUSTOM_BANKS_FILE.exists():
        for item in json.loads(CUSTOM_BANKS_FILE.read_text(encoding="utf-8")):
            labels[item["id"]] = item["label"]
    return labels


def load_family_rows() -> list[dict]:
    """读取所有指纹库的有效参考数据，附带家族信息（与 app.global_reference_rows 一致）。"""
    rows = []
    for bank_id, label in bank_labels().items():
        data_file, _ = bank_paths(bank_id)
        if not data_file.exists():
            continue
        rows.extend(
            {**row, "family_id": bank_id, "family_name": label}
            for row in read_rows(data_file)
        )
    return rows


def _beta(calibration: dict | None, key: str, default: float) -> float:
    try:
        return float(calibration[key]["beta"])
    except (TypeError, KeyError, ValueError):
        return default


# ---------------------------------------------------------------- 一致性校验

def _score_against(rows_to_test: list[dict], train: list[dict], model_ids: list[str]) -> list[list[float]]:
    bank = {"robust": fit_robust_artifacts(train, model_ids)}
    return [robust_score_numbers(row["numbers"], bank)["fused"] for row in rows_to_test]


def _summarize(
    scores: list[list[float]],
    model_ids: list[str],
    target: str | None,
    beta_1: float,
    beta_3: float,
) -> dict:
    top_models = [model_ids[int(np.argmax(item))] for item in scores]
    pooled = np.mean(np.asarray(scores), axis=0)
    # 汇总概率按“3 次查询”的温度换算，回答越多越自信，但封顶在 3 次的校准上。
    pooled_probabilities = softmax([beta_3 * value for value in pooled.tolist()])
    order = sorted(range(len(model_ids)), key=lambda index: pooled_probabilities[index], reverse=True)
    single_probabilities = [softmax([beta_1 * value for value in item]) for item in scores]
    summary = {
        "responses": len(scores),
        "pooled_prediction": model_ids[order[0]],
        "pooled_probability": pooled_probabilities[order[0]],
        "top_alternatives": [
            {"model": model_ids[index], "probability": pooled_probabilities[index]}
            for index in order[:3]
        ],
        "per_response_votes": dict(Counter(top_models).most_common(3)),
    }
    if target is not None and target in model_ids:
        target_index = model_ids.index(target)
        summary.update(
            {
                "target_top1_rate": sum(model == target for model in top_models) / len(top_models),
                "target_mean_probability": float(np.mean([item[target_index] for item in single_probabilities])),
                "target_pooled_probability": pooled_probabilities[target_index],
            }
        )
    return summary


def _verdict(summary: dict, target: str) -> tuple[str, str]:
    top1 = summary.get("target_top1_rate")
    if top1 is None:
        return "unknown", "库中没有可对照的同名指纹"
    pooled_ok = summary["pooled_prediction"] == target
    if pooled_ok and top1 >= CONSISTENT_TOP1:
        return "consistent", "与同名模型指纹一致"
    if not pooled_ok and top1 < SUSPECT_TOP1:
        return "suspect", f"更像 {summary['pooled_prediction']}（{summary['pooled_probability']:.1%}）"
    return "weak", "一致性偏弱，可能存在包装层/参数差异，建议多采集或人工确认"


def channel_report(
    rows: list[dict],
    model_id: str,
    candidate_channels: list[str] | None = None,
    calibration: dict | None = None,
) -> dict:
    """校验 model_id 的各个渠道。

    rows：全库有效行（包含所有模型，用作对照候选）。
    candidate_channels：待校验渠道；其余同名渠道视为可信基准。为空时校验全部渠道（留一渠道法）。

    - 有可信基准：用“可信渠道 + 其他模型”建库，识别每个候选渠道。
    - 无可信基准但有 ≥2 个渠道：留一渠道法，用其余渠道的同名指纹做对照。
    - 只有一个渠道且无基准：去掉同名模型，看库内其他模型是否能高置信认领它（掉包预警）。
    """
    rows = [prepare_row(row) for row in rows]
    all_models = list(dict.fromkeys(row["source"] for row in rows))
    model_rows = [row for row in rows if row["source"] == model_id]
    channels = sorted({channel_of(row) for row in model_rows})
    if candidate_channels is None:
        candidate_channels = channels
        trusted = []
    else:
        trusted = [channel for channel in channels if channel not in candidate_channels]
    beta_1 = _beta(calibration, "1", DEFAULT_BETA_1)
    beta_3 = _beta(calibration, "3", DEFAULT_BETA_3)

    other_rows = [row for row in rows if row["source"] != model_id]
    entries = []
    for channel in candidate_channels:
        held = [row for row in model_rows if channel_of(row) == channel]
        if not held:
            continue
        if trusted:
            reference = [row for row in model_rows if channel_of(row) in trusted]
            mode = "against_trusted"
        else:
            reference = [row for row in model_rows if channel_of(row) != channel]
            mode = "leave_one_channel_out"
        if reference:
            train = other_rows + reference
            model_ids = all_models
            target = model_id
        else:
            # 没有同名对照：只能看其他模型是否认领它
            mode = "anchor_only"
            train = other_rows
            model_ids = [model for model in all_models if model != model_id]
            target = None
        if len(model_ids) < 2:
            entries.append({"channel": channel, "responses": len(held), "mode": mode, "verdict": "unknown",
                            "reason": "库内候选模型不足，无法校验"})
            continue
        summary = _summarize(_score_against(held, train, model_ids), model_ids, target, beta_1, beta_3)
        if target is None:
            if summary["pooled_probability"] >= 0.9 and len(held) >= 3:
                verdict, reason = "warning", (
                    f"库内 {summary['pooled_prediction']} 以 {summary['pooled_probability']:.1%} 认领该渠道，"
                    "请确认它不是被掉包的已有模型"
                )
            else:
                verdict, reason = "unknown", "无同名对照；库内没有其他模型能高置信认领，视为新指纹"
        else:
            verdict, reason = _verdict(summary, model_id)
        entries.append({"channel": channel, "mode": mode, **summary, "verdict": verdict, "reason": reason})

    pooled = {
        channel: [sum(row["counts"][index] for row in model_rows if channel_of(row) == channel)
                  for index in range(len(model_rows[0]["counts"]))]
        for channel in channels
    } if model_rows else {}
    pairwise = [
        {"left": left, "right": right, "distribution_similarity": js_similarity(pooled[left], pooled[right])}
        for position, left in enumerate(channels)
        for right in channels[position + 1:]
    ]
    return {
        "model": model_id,
        "channels": channels,
        "trusted_channels": trusted,
        "candidate_channels": list(candidate_channels),
        "entries": entries,
        "pairwise": pairwise,
        "suspect_channels": [entry["channel"] for entry in entries if entry["verdict"] == "suspect"],
    }


def bank_channel_overview(rows: list[dict], calibration: dict | None = None) -> list[dict]:
    """对库中所有多渠道模型做留一渠道校验。"""
    by_model: dict[str, set[str]] = {}
    for row in rows:
        by_model.setdefault(row["source"], set()).add(channel_of(row))
    return [
        channel_report(rows, model_id, None, calibration)
        for model_id, channels in by_model.items()
        if len(channels) > 1
    ]


# ---------------------------------------------------------------- 整合 + 校准

def row_key(row: dict) -> tuple[str, str, str]:
    """同一模型、同一渠道、同一道题的重复采集视为同一条数据。

    没有 challenge_id 时退回 row_id：那种行的题目身份未知，按行身份处理才不会互相顶掉。
    """
    challenge = str(row.get("challenge_id") or row.get("row_id") or "")
    return (str(row.get("source")), channel_of(row), challenge)


def dedupe_rows(existing: list[dict], incoming: list[dict]) -> tuple[list[dict], int]:
    """按（模型, 渠道, 题目）去重：同键的新行替换旧行，其余追加在后面。"""
    fresh = {row_key(row): row for row in incoming}
    kept = [row for row in existing if row_key(row) not in fresh]
    return kept + list(fresh.values()), len(existing) - len(kept)


def append_rows(rows: list[dict], data_file: Path, replace: bool = False) -> int:
    """写入参考数据行；replace=True 时按（模型, 渠道, 题目）幂等替换。

    replace 只在重采/导入路径开启；同一次采集内部不去重。
    返回被替换掉的旧行数。
    """
    data_file.parent.mkdir(parents=True, exist_ok=True)
    clean_rows = [
        {key: value for key, value in row.items() if key not in {"numbers", "counts"}}
        for row in rows
    ]
    if not replace:
        with data_file.open("a", encoding="utf-8") as handle:
            for row in clean_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        return 0

    existing = [json.loads(line) for line in data_file.read_text(encoding="utf-8").splitlines() if line] \
        if data_file.exists() else []
    merged, replaced = dedupe_rows(existing, clean_rows)
    temp_file = data_file.with_name(data_file.name + ".tmp")
    with temp_file.open("w", encoding="utf-8", newline="\n") as handle:
        for row in merged:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temp_file, data_file)
    return replaced


def rebuild_bank_file(data_file: Path, bank_file: Path) -> dict:
    previous = json.loads(bank_file.read_text(encoding="utf-8")) if bank_file.exists() else None
    bank = build_bank(read_rows(data_file), previous["calibration"] if previous else None)
    bank_file.write_text(json.dumps(bank, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return bank


def rebuild_unified() -> dict:
    bank = build_bank(load_family_rows())
    UNIFIED_BANK_FILE.write_text(json.dumps(bank, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return bank


def integrate_channels(
    bank_id: str,
    model_id: str,
    new_rows: list[dict],
    keep_suspect: bool = False,
    dry_run: bool = False,
    data_file: Path | None = None,
    bank_file: Path | None = None,
) -> dict:
    """把多个渠道的新回答整合进指定指纹库并重新校准。

    校验时以“全部指纹库”作为对照候选（可以发现被换成其他家族模型的渠道），
    通过校验的渠道写入 bank 的 reference.jsonl，并重建该库与统一库。
    """
    default_data, default_bank = bank_paths(bank_id)
    data_file = data_file or default_data
    bank_file = bank_file or default_bank
    new_rows = [prepare_row({**row, "source": model_id, "model_id": model_id, "exact_version": model_id})
                for row in new_rows if row.get("strict_valid", True)]
    if not new_rows:
        raise ValueError("没有可用回答，指纹库未修改")
    candidates = sorted({channel_of(row) for row in new_rows})

    calibration = None
    if UNIFIED_BANK_FILE.exists():
        calibration = json.loads(UNIFIED_BANK_FILE.read_text(encoding="utf-8")).get("calibration")
    label = bank_labels().get(bank_id, bank_id)
    reference_rows = load_family_rows()
    tagged_new = [{**row, "family_id": bank_id, "family_name": label} for row in new_rows]
    report = channel_report(reference_rows + tagged_new, model_id, candidates, calibration)

    rejected = set() if keep_suspect else set(report["suspect_channels"])
    accepted_rows = [row for row in new_rows if channel_of(row) not in rejected]
    result = {
        "model": model_id,
        "bank_id": bank_id,
        "report": report,
        "accepted_channels": sorted({channel_of(row) for row in accepted_rows}),
        "rejected_channels": sorted(rejected),
        "accepted_responses": len(accepted_rows),
        "dry_run": dry_run,
    }
    if dry_run or not accepted_rows:
        return result
    result["replaced_responses"] = append_rows(accepted_rows, data_file, replace=True)
    bank = rebuild_bank_file(data_file, bank_file)
    unified = rebuild_unified()
    result["bank_calibration"] = bank["calibration"]
    result["unified_calibration"] = unified["calibration"]
    result["channel_calibration"] = unified.get("channel_calibration")
    result.update(fingerprint_scope(accepted_rows, unified))
    return result


def fingerprint_scope(rows: list[dict], bank: dict) -> dict:
    """报告有多少行真正参与了指纹估计。

    建库只采用“每个模型都有数据”的完整环境，其余环境的行虽然写进了参考数据，
    却不会进入质心与校准。不报出来就会变成静默丢弃。
    """
    used = set(bank.get("robust", {}).get("complete_environments") or [])
    if not used:
        return {"excluded_conditions": [], "excluded_responses": 0}
    excluded = sorted({str(row.get("condition_id")) for row in rows} - used)
    return {
        "excluded_conditions": excluded,
        "excluded_responses": sum(1 for row in rows if str(row.get("condition_id")) in excluded),
    }


# ---------------------------------------------------------------- 外部文件导入

def _suite_lookup() -> dict[str, dict]:
    return {task["challenge_id"]: task for task in fingerprint_suite()}


def load_channel_file(path: Path, channel_name: str, model_id: str, bank_id: str) -> list[dict]:
    """读取一个渠道导出的回答文件。

    支持：
    - ModelTrace 参考数据格式（*.jsonl，含 text / challenge_id / condition_id）；
    - 精简 jsonl：每行 {"challenge_id": "query-07", "text": "..."}，环境由标准挑战集补全；
    - 纯文本：用单独一行 ===OUTPUT=== 分隔多份回答（视为 manual 环境）。
    """
    raw = path.read_text(encoding="utf-8")
    return rows_from_text(raw, channel_name, model_id, bank_id, path.suffix.lower() in {".jsonl", ".json"})


def rows_from_text(raw: str, channel_name: str, model_id: str, bank_id: str, is_jsonl: bool = False) -> list[dict]:
    """把一个渠道的回答文本（jsonl 或 ===OUTPUT=== 分隔）转成参考数据行。"""
    suite = _suite_lookup()
    channel_id = slugify_channel(channel_name)
    items: list[dict] = []
    if is_jsonl:
        for line in raw.splitlines():
            if line.strip():
                items.append(json.loads(line))
    else:
        for block in re.split(r"(?m)^\s*===OUTPUT===\s*$", raw):
            if block.strip():
                items.append({"text": block.strip(), "challenge_id": f"manual-{secrets.token_hex(6)}",
                              "condition_id": "manual"})
    rows = []
    for item in items:
        text = str(item.get("text", ""))
        task = suite.get(item.get("challenge_id", ""))
        condition = item.get("condition_id") or (task["condition"] if task else "manual")
        expected = int(item.get("requested_count") or (task["expected_count"] if task else 0))
        numbers = parse_numbers(text)
        threshold = max(80, math.ceil(expected * 0.55)) if expected else 80
        rows.append(
            {
                **{key: value for key, value in item.items() if key not in {"numbers", "counts"}},
                "row_id": item.get("row_id") or f"{condition}-{secrets.token_hex(8)}",
                "bank_id": bank_id,
                "source": model_id,
                "model_id": model_id,
                "exact_version": model_id,
                "condition_id": condition,
                "nuisance_condition_id": item.get("nuisance_condition_id") or condition,
                "challenge_id": item.get("challenge_id") or f"manual-{secrets.token_hex(6)}",
                "requested_count": expected,
                "parsed_count": len(numbers),
                "strict_threshold": threshold,
                "strict_valid": len(numbers) >= threshold,
                "provider": channel_id,
                "channel_id": channel_id,
                "channel_name": channel_name,
                "text": text,
            }
        )
    return rows


# ---------------------------------------------------------------- 命令行

def _print_report(report: dict) -> None:
    print(f"\n模型 {report['model']} · 渠道 {', '.join(report['channels'])}")
    if report["trusted_channels"]:
        print(f"  可信基准：{', '.join(report['trusted_channels'])}")
    for entry in report["entries"]:
        rate = entry.get("target_top1_rate")
        rate_text = f"单条命中 {rate:.0%} · " if rate is not None else ""
        print(
            f"  [{entry['verdict']:>10}] {entry['channel']:<20} {entry['responses']:>3} 份 · {rate_text}"
            f"整体归因 {entry.get('pooled_prediction', '-')} {entry.get('pooled_probability', 0):.1%} · {entry['reason']}"
        )
    for pair in report["pairwise"]:
        print(f"  分布相似度 {pair['left']} ↔ {pair['right']}: {pair['distribution_similarity']:.1%}")


def main() -> None:
    parser = argparse.ArgumentParser(description="多渠道整合与指纹库校准")
    sub = parser.add_subparsers(dest="command", required=True)

    report_cmd = sub.add_parser("report", help="校验已有数据中的多渠道模型")
    report_cmd.add_argument("--bank", help="只看某个指纹库的模型")
    report_cmd.add_argument("--model", help="只看某个模型")

    import_cmd = sub.add_parser("import", help="导入多个渠道的回答并校准")
    import_cmd.add_argument("--bank", required=True)
    import_cmd.add_argument("--model", required=True, help="指纹名称（同一模型的统一标签）")
    import_cmd.add_argument("--channel", action="append", required=True, metavar="名称=文件",
                            help="可重复，例如 --channel 官方=official.jsonl")
    import_cmd.add_argument("--keep-suspect", action="store_true", help="可疑渠道也并入")
    import_cmd.add_argument("--dry-run", action="store_true", help="只校验不写入")

    rebuild_cmd = sub.add_parser("rebuild", help="按渠道感知算法重建并重新校准")
    rebuild_cmd.add_argument("--bank", help="只重建某个库（统一库总会重建）")

    args = parser.parse_args()
    if args.command == "report":
        rows = load_family_rows()
        if args.bank:
            models = {row["source"] for row in rows if row["family_id"] == args.bank}
        else:
            models = {row["source"] for row in rows}
        if args.model:
            models &= {args.model}
        calibration = json.loads(UNIFIED_BANK_FILE.read_text(encoding="utf-8")).get("calibration") \
            if UNIFIED_BANK_FILE.exists() else None
        reports = [report for report in bank_channel_overview(rows, calibration) if report["model"] in models]
        if not reports:
            print("没有找到拥有两个以上渠道的模型。")
        for report in reports:
            _print_report(report)
    elif args.command == "import":
        new_rows = []
        for spec in args.channel:
            if "=" not in spec:
                parser.error(f"--channel 需要 名称=文件 格式：{spec}")
            name, file_name = spec.split("=", 1)
            new_rows.extend(load_channel_file(Path(file_name), name, args.model, args.bank))
        result = integrate_channels(args.bank, args.model, new_rows, args.keep_suspect, args.dry_run)
        _print_report(result["report"])
        print(f"\n并入渠道：{', '.join(result['accepted_channels']) or '无'}；"
              f"拒绝：{', '.join(result['rejected_channels']) or '无'}；回答 {result['accepted_responses']} 份"
              + (f"（替换同渠道旧回答 {result['replaced_responses']} 份）" if result.get("replaced_responses") else "")
              + ("（dry-run，未写入）" if result["dry_run"] else ""))
        if result.get("excluded_responses"):
            print(f"注意：其中 {result['excluded_responses']} 份所在环境（{'、'.join(result['excluded_conditions'])}）"
                  "并非所有模型都覆盖，已写入参考数据但不参与指纹估计。")
        if "unified_calibration" in result:
            print(json.dumps({"unified_calibration": result["unified_calibration"],
                              "channel_calibration": result["channel_calibration"]}, ensure_ascii=False, indent=2))
    elif args.command == "rebuild":
        bank_ids = [args.bank] if args.bank else list(bank_labels())
        for bank_id in bank_ids:
            data_file, bank_file = bank_paths(bank_id)
            if data_file.exists() and read_rows(data_file):
                bank = rebuild_bank_file(data_file, bank_file)
                print(f"{bank_id}: {len(bank['models'])} 个模型 · {bank['channel_calibration']}")
        unified = rebuild_unified()
        print(json.dumps({"unified_calibration": unified["calibration"],
                          "channel_calibration": unified["channel_calibration"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
