from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parent
sys.path.insert(0, str(PROJECT))

from fingerprint import (  # noqa: E402
    ALPHA,
    DIMENSION,
    ORDERED_BLOCK_WEIGHT,
    count_numbers,
    hellinger_feature,
    ordered_block_feature,
    parse_numbers,
    robust_score_numbers,
)


DEFAULT_RESPONSES = PROJECT / "data" / "gpt_reference.jsonl"
DEFAULT_OUTPUT = PROJECT / "data" / "gpt_bank.json"


def read_rows(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    output = []
    for row in rows:
        if not row.get("strict_valid"):
            continue
        numbers = parse_numbers(row["text"])
        output.append({**row, "numbers": numbers, "counts": count_numbers(numbers)})
    return output


CHANNEL_NUISANCE_RANK = 2


def channel_of(row: dict) -> str:
    """同一模型的不同采集渠道（中转站、官方 API、订阅客户端等）。

    旧数据没有 channel_id 字段，回退到 provider，保证单渠道库的结果与原版完全一致。
    """
    return str(row.get("channel_id") or row.get("provider") or "default")


def model_channels(rows: list[dict], model_ids: list[str]) -> dict[str, list[str]]:
    return {
        model_id: sorted({channel_of(row) for row in rows if row["source"] == model_id})
        for model_id in model_ids
    }


def balanced_centroid(values: np.ndarray, labels: np.ndarray, channels: np.ndarray, model_id: str) -> np.ndarray:
    """按渠道等权求模型中心：每个渠道先取均值再平均，避免某个渠道样本多就主导指纹。"""
    selected = labels == model_id
    channel_ids = sorted(set(channels[selected].tolist()))
    if len(channel_ids) <= 1:
        return values[selected].mean(axis=0)
    return np.stack(
        [values[selected & (channels == channel_id)].mean(axis=0) for channel_id in channel_ids]
    ).mean(axis=0)


def channel_nuisance_basis(
    values: np.ndarray,
    labels: np.ndarray,
    channels: np.ndarray,
    environments: np.ndarray,
    model_ids: list[str],
) -> np.ndarray:
    """估计“渠道干扰方向”：同一模型、同一环境下，不同渠道之间的平均偏移。

    按 (模型, 环境) 分组比较，避免渠道覆盖的环境不同而把环境差异误当成渠道差异。
    只有拥有两个以上渠道的模型参与；不存在多渠道模型时返回空基，行为与原算法一致。
    输入 values 已经去除了环境方向，因此返回的基与环境基正交。
    """
    offsets = []
    for model_id in model_ids:
        selected = labels == model_id
        if len(set(channels[selected].tolist())) < 2:
            continue
        for environment in sorted(set(environments[selected].tolist())):
            group = selected & (environments == environment)
            channel_ids = sorted(set(channels[group].tolist()))
            if len(channel_ids) < 2:
                continue
            means = np.stack(
                [values[group & (channels == channel_id)].mean(axis=0) for channel_id in channel_ids]
            )
            offsets.extend(means - means.mean(axis=0, keepdims=True))
    if not offsets:
        return np.empty((0, values.shape[1]), dtype=np.float64)
    matrix = np.stack(offsets)
    _, singular, right = np.linalg.svd(matrix, full_matrices=False)
    if not singular.size or not singular[0]:
        return np.empty((0, values.shape[1]), dtype=np.float64)
    rank = min(CHANNEL_NUISANCE_RANK, int(np.sum(singular > singular[0] * 1e-8)))
    return right[:rank]


def fit_robust_artifacts(rows: list[dict], model_ids: list[str]) -> dict:
    environments = sorted({row["condition_id"] for row in rows})
    complete = [
        environment
        for environment in environments
        if {row["source"] for row in rows if row["condition_id"] == environment}
        == set(model_ids)
    ]
    robust_ready = bool(complete)
    if complete:
        robust_rows = [row for row in rows if row["condition_id"] in complete]
    else:
        robust_rows = rows

    features = np.stack([hellinger_feature(row["counts"]) for row in robust_rows])
    feature_mean = features.mean(axis=0)
    feature_scale = features.std(axis=0)
    feature_scale[feature_scale < 1e-12] = 1.0
    standardized = (features - feature_mean) / feature_scale

    nuisance_environments = sorted(
        {
            row.get("nuisance_condition_id", row["condition_id"])
            for row in robust_rows
        }
    )
    environment_means = []
    for environment in nuisance_environments:
        indices = np.asarray(
            [
                row.get("nuisance_condition_id", row["condition_id"]) == environment
                for row in robust_rows
            ],
            dtype=bool,
        )
        environment_means.append(standardized[indices].mean(axis=0))
    offsets = np.stack(environment_means)
    offsets -= offsets.mean(axis=0, keepdims=True)
    basis = np.empty((0, DIMENSION), dtype=np.float64)
    if len(offsets) > 1:
        _, singular, right = np.linalg.svd(offsets, full_matrices=False)
        rank = min(2, int(np.sum(singular > singular[0] * 1e-8))) if singular[0] else 0
        basis = right[:rank]
        standardized -= (standardized @ basis.T) @ basis

    labels = np.asarray([row["source"] for row in robust_rows])
    channels = np.asarray([channel_of(row) for row in robust_rows])
    row_environments = np.asarray([row["condition_id"] for row in robust_rows])
    channel_basis = channel_nuisance_basis(standardized, labels, channels, row_environments, model_ids)
    if len(channel_basis):
        standardized -= (standardized @ channel_basis.T) @ channel_basis
        basis = np.vstack([basis, channel_basis])
    centroids = np.stack(
        [balanced_centroid(standardized, labels, channels, model_id) for model_id in model_ids]
    )
    centroid_norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    centroids /= np.maximum(centroid_norms, 1e-12)

    ordered_features = np.stack([ordered_block_feature(row["numbers"]) for row in robust_rows])
    ordered_mean = ordered_features.mean(axis=0)
    ordered_scale = ordered_features.std(axis=0)
    ordered_scale[ordered_scale < 1e-12] = 1.0
    ordered_standardized = (ordered_features - ordered_mean) / ordered_scale
    ordered_environment_means = []
    for environment in nuisance_environments:
        indices = np.asarray(
            [
                row.get("nuisance_condition_id", row["condition_id"]) == environment
                for row in robust_rows
            ],
            dtype=bool,
        )
        ordered_environment_means.append(ordered_standardized[indices].mean(axis=0))
    ordered_offsets = np.stack(ordered_environment_means)
    ordered_offsets -= ordered_offsets.mean(axis=0, keepdims=True)
    ordered_basis = np.empty((0, ordered_standardized.shape[1]), dtype=np.float64)
    if len(ordered_offsets) > 1:
        _, singular, right = np.linalg.svd(ordered_offsets, full_matrices=False)
        rank = min(2, int(np.sum(singular > singular[0] * 1e-8))) if singular[0] else 0
        ordered_basis = right[:rank]
        ordered_standardized -= (ordered_standardized @ ordered_basis.T) @ ordered_basis
    ordered_channel_basis = channel_nuisance_basis(
        ordered_standardized, labels, channels, row_environments, model_ids
    )
    if len(ordered_channel_basis):
        ordered_standardized -= (ordered_standardized @ ordered_channel_basis.T) @ ordered_channel_basis
        ordered_basis = np.vstack([ordered_basis, ordered_channel_basis])
    ordered_centroids = np.stack(
        [balanced_centroid(ordered_standardized, labels, channels, model_id) for model_id in model_ids]
    )
    ordered_centroids /= np.maximum(np.linalg.norm(ordered_centroids, axis=1, keepdims=True), 1e-12)
    ordered_environment_centroids = []
    unprojected_ordered = (ordered_features - ordered_mean) / ordered_scale
    template_environments = complete or [None]
    for environment in template_environments:
        environment_centroids = []
        for model_id in model_ids:
            indices = np.asarray(
                [
                    (environment is None or row["condition_id"] == environment)
                    and row["source"] == model_id
                    for row in robust_rows
                ],
                dtype=bool,
            )
            centroid = balanced_centroid(
                unprojected_ordered[indices], labels[indices], channels[indices], model_id
            )
            centroid /= max(float(np.linalg.norm(centroid)), 1e-12)
            environment_centroids.append(centroid)
        ordered_environment_centroids.append(np.stack(environment_centroids))
    ordered_weight = ORDERED_BLOCK_WEIGHT

    return {
        "model_order": model_ids,
        "robust_ready": robust_ready,
        "training_rows": len(robust_rows),
        "complete_environments": complete,
        "channel_nuisance_rank": int(len(channel_basis)),
        "hellinger": {
            "feature_mean": feature_mean.tolist(),
            "feature_scale": feature_scale.tolist(),
            "nuisance_rank": len(basis),
            "nuisance_environments": nuisance_environments,
            "nuisance_basis": basis.tolist(),
            "centroids": centroids.tolist(),
        },
        "ordered_blocks": {
            "weight": ordered_weight,
            "feature": "four position blocks x 16 value bins plus final-digit distribution",
            "feature_mean": ordered_mean.tolist(),
            "feature_scale": ordered_scale.tolist(),
            "nuisance_rank": len(ordered_basis),
            "nuisance_basis": ordered_basis.tolist(),
            "centroids": ordered_centroids.tolist(),
            "environment_centroids": [centroids.tolist() for centroids in ordered_environment_centroids],
        },
    }


def calibration_records(
    rows: list[dict],
    model_ids: list[str],
    query_count: int,
    cross_channel: bool = True,
) -> list[tuple[list[float], int]]:
    """分组交叉验证，产出 (各模型得分, 真实模型序号) 记录用于拟合温度。

    - 按挑战分组留出：留出的挑战不参与训练（原算法）。
    - 渠道感知：同一模型的每个渠道单独组成一组留出回答，因此多渠道模型会贡献多条记录。
    - cross_channel：若该模型还有其他渠道，额外把被测渠道的全部回答从训练集中移除，
      让温度反映“用其他渠道的指纹识别一个新渠道”的真实难度，防止渠道特有偏差泄漏。
    """
    records = []
    channels_by_model = model_channels(rows, model_ids)
    conditions = sorted({row["condition_id"] for row in rows})
    for condition in conditions:
        challenge_ids = sorted({row["challenge_id"] for row in rows if row["condition_id"] == condition})
        challenge_ids = [
            challenge_id
            for challenge_id in challenge_ids
            if all(
                any(
                    row["source"] == model_id and row["challenge_id"] == challenge_id
                    for row in rows
                )
                for model_id in model_ids
            )
        ]
        if len(challenge_ids) < query_count:
            continue
        for held_ids in itertools.combinations(challenge_ids, query_count):
            held_set = set(held_ids)
            train = [row for row in rows if row["challenge_id"] not in held_set]
            shared_bank = None
            # 跨渠道留出按渠道缓存：同一渠道（如同一家中转站）承载的多个模型共用一次拟合。
            channel_banks: dict[str, dict] = {}
            for truth_index, model_id in enumerate(model_ids):
                channel_ids = channels_by_model[model_id]
                for channel_id in channel_ids:
                    held = [
                        row
                        for row in rows
                        if row["source"] == model_id
                        and channel_of(row) == channel_id
                        and row["challenge_id"] in held_set
                    ]
                    if len(held) != query_count:
                        continue
                    if cross_channel and len(channel_ids) > 1:
                        if channel_id not in channel_banks:
                            channel_train = [
                                row
                                for row in train
                                if not (
                                    channel_of(row) == channel_id
                                    and len(channels_by_model[row["source"]]) > 1
                                )
                            ]
                            channel_banks[channel_id] = {
                                "robust": fit_robust_artifacts(channel_train, model_ids)
                            }
                        score_bank = channel_banks[channel_id]
                    else:
                        if shared_bank is None:
                            shared_bank = {"robust": fit_robust_artifacts(train, model_ids)}
                        score_bank = shared_bank
                    response_scores = [
                        robust_score_numbers(row["numbers"], score_bank)["fused"] for row in held
                    ]
                    combined = [
                        sum(scores[index] for scores in response_scores) / query_count
                        for index in range(len(model_ids))
                    ]
                    records.append((combined, truth_index))
    return records


def fit_beta(records: list[tuple[list[float], int]]) -> tuple[float, float]:
    best_beta = 1.0
    best_loss = float("inf")
    for step in range(801):
        beta = math.exp(math.log(0.05) + step * (math.log(12.0) - math.log(0.05)) / 800)
        loss = 0.0
        for scores, truth in records:
            scaled = [beta * value for value in scores]
            maximum = max(scaled)
            log_sum = maximum + math.log(sum(math.exp(value - maximum) for value in scaled))
            loss += log_sum - scaled[truth]
        loss /= len(records)
        if loss < best_loss:
            best_beta = beta
            best_loss = loss
    return best_beta, best_loss


def build_bank(
    rows: list[dict],
    fallback_calibration: dict | None = None,
    cross_channel: bool = True,
) -> dict:
    model_ids = list(dict.fromkeys(row["source"] for row in rows))
    model_entries = []
    for model_id in model_ids:
        selected = [row for row in rows if row["source"] == model_id]
        pooled = [sum(row["counts"][index] for row in selected) for index in range(DIMENSION)]
        family_id = selected[0].get("family_id")
        family_name = selected[0].get("family_name")
        model_entries.append(
            {
                "id": model_id,
                "display_name": model_id,
                "family": family_id,
                "family_name": family_name,
                "response_count": len(selected),
                "valid_number_count": sum(len(row["numbers"]) for row in selected),
                "conditions": dict(Counter(row["condition_id"] for row in selected)),
                "channels": dict(Counter(channel_of(row) for row in selected)),
                "counts": pooled,
            }
        )

    robust = fit_robust_artifacts(rows, model_ids)
    calibration = {}
    for query_count in (1, 2, 3):
        records = calibration_records(rows, model_ids, query_count, cross_channel)
        key = str(query_count)
        if records:
            beta, nll = fit_beta(records)
            correct = sum(max(range(len(scores)), key=lambda index: scores[index]) == truth for scores, truth in records)
            calibration[key] = {
                "beta": beta,
                "cv_accuracy": correct / len(records),
                "cv_correct": correct,
                "cv_samples": len(records),
                "cv_nll": nll,
                "fallback": False,
            }
        else:
            previous = fallback_calibration.get(key) if fallback_calibration else None
            calibration[key] = {
                **(
                    previous
                    or {
                        "beta": 1.0,
                        "cv_accuracy": None,
                        "cv_correct": 0,
                        "cv_samples": 0,
                        "cv_nll": None,
                    }
                ),
                "fallback": True,
            }

    providers = sorted({row.get("provider") for row in rows if row.get("provider")})
    multi_channel_models = {
        model_id: channel_ids
        for model_id, channel_ids in model_channels(rows, model_ids).items()
        if len(channel_ids) > 1
    }
    source_scope = "Reference outputs enrolled for the listed model labels."
    method_name = "Ordered-block + nuisance-Hellinger"
    return {
        "schema": "robust-number-fingerprint-bank",
        "built_at": datetime.now(timezone.utc).isoformat(),
        "method": {
            "name": method_name,
            "range": [1, 355],
            "alpha": ALPHA,
            "ordered_block_weight": robust["ordered_blocks"]["weight"],
            "parser": "longest digit run; alphabetic separators split runs; retain values 1..355",
            "response_score": "shared nuisance-Hellinger and ordered-block environment/nuisance fusion",
            "aggregation": "mean of response-wise fused class z-scores",
            "probability": "uniform-prior softmax with fusion-specific grouped cross-validation temperature scaling",
        },
        "recommended_queries": 3,
        "minimum_valid_numbers": 80,
        "source_scope": source_scope,
        "providers": providers,
        "multi_channel_models": multi_channel_models,
        "channel_calibration": {
            "enabled": bool(multi_channel_models),
            "cross_channel_holdout": bool(cross_channel and multi_channel_models),
            "channel_nuisance_rank": robust["channel_nuisance_rank"],
            "centroid_weighting": "channel-balanced" if multi_channel_models else "row-mean",
        },
        "models": model_entries,
        "robust": robust,
        "calibration": calibration,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--responses", type=Path, default=DEFAULT_RESPONSES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    fallback = None
    if args.output.exists():
        fallback = json.loads(args.output.read_text(encoding="utf-8"))["calibration"]
    bank = build_bank(read_rows(args.responses), fallback)
    args.output.write_text(json.dumps(bank, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = {
        "output": str(args.output),
        "method": bank["method"]["name"],
        "robust_training_rows": bank["robust"]["training_rows"],
        "complete_environments": len(bank["robust"]["complete_environments"]),
        "nuisance_rank": bank["robust"]["hellinger"]["nuisance_rank"],
        "models": {
            model["id"]: {
                "responses": model["response_count"],
                "valid_numbers": model["valid_number_count"],
            }
            for model in bank["models"]
        },
        "calibration": bank["calibration"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
