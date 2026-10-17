"""多渠道整合校准的回归测试：python tests/test_channel_calibration.py

1. 单渠道（旧数据）建库结果与仓库内 unified_bank.json 完全一致 —— 新算法向后兼容。
2. 同一模型的真实新渠道被判为 consistent，被掉包成其他模型的渠道被判为 suspect。
3. 多渠道建库启用渠道干扰投影与跨渠道留出校准。
4. 重采幂等：同一（模型, 渠道, 题目）重复导入是替换而不是累积。
5. 采集调度按主机分组：同一网关的渠道排队，不同网关并行。
全部在内存 / 临时目录中完成，不修改 data/。
"""

from __future__ import annotations

import json
import random
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bank_builder import build_bank, read_rows  # noqa: E402
from channel_calibration import (  # noqa: E402
    append_rows,
    channel_report,
    fingerprint_scope,
    load_family_rows,
    prepare_row,
)
from enrollment import group_channels_by_host  # noqa: E402

TARGET = "claude-opus-4-7"


def synthetic_channel(rows: list[dict], channel: str, seed: int, resample: bool) -> list[dict]:
    rng = random.Random(seed)
    output = []
    for index, row in enumerate(rows):
        numbers = list(row["numbers"])
        if resample:
            numbers = [rng.choice(row["numbers"]) for _ in numbers]
        output.append(prepare_row({
            **{key: value for key, value in row.items() if key not in {"numbers", "counts"}},
            "row_id": f"{channel}-{index}",
            "source": TARGET, "model_id": TARGET,
            "channel_id": channel, "provider": channel,
            "text": json.dumps(numbers),
        }))
    return output


def test_backward_compatible() -> None:
    rows = load_family_rows()
    rows = [row for row in rows if row["family_id"] in {"gpt", "claude"}]
    rebuilt = build_bank(rows)
    stored = json.loads((ROOT / "data" / "unified_bank.json").read_text(encoding="utf-8"))
    for part in ("hellinger", "ordered_blocks"):
        for key in ("centroids", "nuisance_basis"):
            difference = np.abs(np.asarray(rebuilt["robust"][part][key]) - np.asarray(stored["robust"][part][key])).max()
            assert difference < 1e-9, (part, key, difference)
    for key in ("1", "2", "3"):
        assert abs(rebuilt["calibration"][key]["beta"] - stored["calibration"][key]["beta"]) < 1e-9
    assert rebuilt["channel_calibration"]["enabled"] is False
    print("ok  单渠道结果与原版一致")


def test_detects_swapped_channel() -> None:
    rows = load_family_rows()
    claude = read_rows(ROOT / "data" / "claude_reference.jsonl")
    good = synthetic_channel([row for row in claude if row["source"] == TARGET], "relay-good", 1, True)
    swapped = synthetic_channel([row for row in claude if row["source"] == "claude-sonnet-4-6"], "relay-swapped", 2, False)
    report = channel_report(rows + good + swapped, TARGET, ["relay-good", "relay-swapped"])
    verdicts = {entry["channel"]: entry["verdict"] for entry in report["entries"]}
    assert verdicts["relay-good"] == "consistent", verdicts
    assert verdicts["relay-swapped"] == "suspect", verdicts
    print("ok  掉包渠道被识别：", verdicts)
    return good


def test_channel_aware_bank(good: list[dict]) -> None:
    rows = load_family_rows() + good
    bank = build_bank(rows)
    assert bank["channel_calibration"]["enabled"] is True
    assert bank["channel_calibration"]["channel_nuisance_rank"] >= 1
    assert TARGET in bank["multi_channel_models"]
    assert bank["calibration"]["3"]["cv_accuracy"] >= 0.95
    print("ok  渠道感知建库：", bank["channel_calibration"], "3 次查询 CV 准确率", bank["calibration"]["3"]["cv_accuracy"])


def channel_rows(channel: str, challenges: list[str], text: str) -> list[dict]:
    return [
        prepare_row(
            {
                "row_id": f"{channel}-{challenge}",
                "source": TARGET,
                "model_id": TARGET,
                "challenge_id": challenge,
                "condition_id": "environment-01",
                "channel_id": channel,
                "provider": channel,
                "strict_valid": True,
                "text": text,
            }
        )
        for challenge in challenges
    ]


def test_reimport_is_idempotent() -> None:
    """重采一个渠道不该让它的行数翻倍——否则该渠道会被重复计权。"""
    challenges = ["query-01", "query-02", "query-03"]
    with tempfile.TemporaryDirectory() as workspace:
        data_file = Path(workspace) / "reference.jsonl"

        assert append_rows(channel_rows("relay-a", challenges, "[1, 2, 3]"), data_file, replace=True) == 0
        assert len(data_file.read_text(encoding="utf-8").splitlines()) == 3

        replaced = append_rows(channel_rows("relay-a", challenges, "[4, 5, 6]"), data_file, replace=True)
        stored = [json.loads(line) for line in data_file.read_text(encoding="utf-8").splitlines()]
        assert replaced == 3, replaced
        assert len(stored) == 3, len(stored)
        assert {row["text"] for row in stored} == {"[4, 5, 6]"}, stored

        assert append_rows(channel_rows("relay-b", challenges, "[7, 8, 9]"), data_file, replace=True) == 0
        stored = [json.loads(line) for line in data_file.read_text(encoding="utf-8").splitlines()]
        assert len(stored) == 6, len(stored)
        assert sorted(row["channel_id"] for row in stored) == ["relay-a"] * 3 + ["relay-b"] * 3

        # 不带 replace 时保持原行为：同一次采集内部不去重
        assert append_rows(channel_rows("relay-b", challenges, "[7, 8, 9]"), data_file) == 0
        assert len(data_file.read_text(encoding="utf-8").splitlines()) == 9
    print("ok  重采幂等：同渠道替换、跨渠道新增")


def test_host_grouping() -> None:
    gateway = "http://114.66.28.184:16521/v1"
    channels = [
        {"name": "商汤", "base_url": gateway},
        {"name": "官方", "base_url": "https://api.anthropic.com"},
        {"name": "另一家中转", "base_url": gateway + "/"},
        {"name": "OpenAI 直连", "base_url": "https://api.openai.com/v1"},
    ]
    groups = group_channels_by_host(channels)
    assert sorted(groups) == ["114.66.28.184:16521", "api.anthropic.com", "api.openai.com"], groups
    assert sorted(item["name"] for item in groups["114.66.28.184:16521"]) == ["另一家中转", "商汤"], groups
    print("ok  采集按主机分组：", {host: len(items) for host, items in groups.items()})


def test_reports_rows_that_skip_the_fingerprint() -> None:
    """粘贴路径落在“manual”环境：写进了参考数据却不参与指纹，必须如实报告。"""
    bank = build_bank(load_family_rows())
    complete = bank["robust"]["complete_environments"]
    assert "manual" not in complete, complete

    scope = fingerprint_scope([{"condition_id": "manual"}], bank)
    assert scope == {"excluded_conditions": ["manual"], "excluded_responses": 1}, scope

    assert fingerprint_scope([{"condition_id": complete[0]}], bank) == {
        "excluded_conditions": [],
        "excluded_responses": 0,
    }
    print("ok  未参与指纹的回答会被如实报告")


if __name__ == "__main__":
    test_backward_compatible()
    test_channel_aware_bank(test_detects_swapped_channel())
    test_reimport_is_idempotent()
    test_host_grouping()
    test_reports_rows_that_skip_the_fingerprint()
