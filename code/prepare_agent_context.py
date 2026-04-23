#!/usr/bin/env python3
"""Prepare AI-agent-ready context files from grid safety analysis outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = (BASE_DIR / "../output").resolve()


def to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def to_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def percentile(sorted_values: list[float], p: float) -> float:
    if not sorted_values:
        return 0.0
    if p <= 0:
        return float(sorted_values[0])
    if p >= 100:
        return float(sorted_values[-1])
    position = (len(sorted_values) - 1) * (p / 100.0)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    lower_val = sorted_values[lower]
    upper_val = sorted_values[upper]
    return float(lower_val + (upper_val - lower_val) * (position - lower))


def resolve_score_csv(path_arg: str | None) -> Path:
    if path_arg:
        path = Path(path_arg).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Score file not found: {path}")
        return path

    candidates = [
        OUTPUT_DIR / "grid_scores.csv",
        OUTPUT_DIR / "final_grid_safety_2026.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "No score CSV found. Expected one of: "
        f"{candidates[0]} or {candidates[1]}"
    )


def load_grid_scores(score_csv: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with score_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = set(reader.fieldnames or [])
        if "grid_id" not in fieldnames or "safety_score" not in fieldnames:
            raise ValueError(
                "Score CSV must include at least 'grid_id' and 'safety_score' columns."
            )

        for raw in reader:
            grid_id = to_int(raw.get("grid_id"), default=0)
            if grid_id <= 0:
                continue
            row = {
                "grid_id": grid_id,
                "pred_property_annual": to_float(raw.get("pred_property_annual"), 0.0),
                "pred_violence_annual": to_float(raw.get("pred_violence_annual"), 0.0),
                "risk": to_float(raw.get("risk"), 0.0),
                "risk_norm": to_float(raw.get("risk_norm"), 0.0),
                "safety_score": to_float(raw.get("safety_score"), 0.0),
                "safety_level": (raw.get("safety_level") or "Unknown").strip() or "Unknown",
                "risk_rank": to_int(raw.get("risk_rank"), 0),
                "hotspot_2026_top10pct": to_int(raw.get("hotspot_2026_top10pct"), 0),
            }
            rows.append(row)

    if not rows:
        raise ValueError(f"No valid rows loaded from score CSV: {score_csv}")

    if not any(row["risk_rank"] > 0 for row in rows):
        rank_sorted = sorted(rows, key=lambda x: x["risk"], reverse=True)
        for idx, row in enumerate(rank_sorted, start=1):
            row["risk_rank"] = idx

    return rows


def load_monthly_trends(
    monthly_csv: Path | None,
    target_year: int,
    focus_grid_ids: set[int],
) -> tuple[list[dict[str, Any]], dict[int, list[dict[str, Any]]], int]:
    if monthly_csv is None or not monthly_csv.exists():
        return [], {}, 0

    city_monthly_map: dict[str, dict[str, float]] = defaultdict(
        lambda: {"pred_property_total": 0.0, "pred_violence_total": 0.0}
    )
    focus_trends: dict[int, list[dict[str, Any]]] = defaultdict(list)
    monthly_rows = 0
    prefix = f"{target_year:04d}-"

    with monthly_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            monthly_rows += 1
            grid_id = to_int(raw.get("grid_id"), 0)
            month_start = (raw.get("month_start") or "").strip()
            pred_property = to_float(raw.get("pred_property"), 0.0)
            pred_violence = to_float(raw.get("pred_violence"), 0.0)

            if month_start.startswith(prefix):
                month_bucket = city_monthly_map[month_start]
                month_bucket["pred_property_total"] += pred_property
                month_bucket["pred_violence_total"] += pred_violence

            if grid_id in focus_grid_ids:
                focus_trends[grid_id].append(
                    {
                        "month_start": month_start,
                        "pred_property": round(pred_property, 6),
                        "pred_violence": round(pred_violence, 6),
                        "pred_total": round(pred_property + pred_violence, 6),
                    }
                )

    city_monthly = []
    for month_start in sorted(city_monthly_map.keys()):
        item = city_monthly_map[month_start]
        city_monthly.append(
            {
                "month_start": month_start,
                "pred_property_total": round(item["pred_property_total"], 4),
                "pred_violence_total": round(item["pred_violence_total"], 4),
                "pred_total": round(
                    item["pred_property_total"] + item["pred_violence_total"], 4
                ),
            }
        )

    for grid_id in list(focus_trends.keys()):
        focus_trends[grid_id] = sorted(
            focus_trends[grid_id], key=lambda x: x["month_start"]
        )

    return city_monthly, dict(focus_trends), monthly_rows


def build_summary(score_rows: list[dict[str, Any]]) -> dict[str, Any]:
    safety_scores = sorted(row["safety_score"] for row in score_rows)
    risk_values = [row["risk"] for row in score_rows]
    hotspot_count = sum(1 for row in score_rows if row["hotspot_2026_top10pct"] == 1)
    level_counts = Counter(row["safety_level"] for row in score_rows)

    return {
        "grid_count": len(score_rows),
        "hotspot_top10_count": hotspot_count,
        "avg_safety_score": round(sum(safety_scores) / len(safety_scores), 4),
        "min_safety_score": round(min(safety_scores), 4),
        "max_safety_score": round(max(safety_scores), 4),
        "p10_safety_score": round(percentile(safety_scores, 10), 4),
        "p50_safety_score": round(percentile(safety_scores, 50), 4),
        "p90_safety_score": round(percentile(safety_scores, 90), 4),
        "avg_risk": round(sum(risk_values) / len(risk_values), 6),
        "safety_level_counts": dict(level_counts),
    }


def build_prompt_snippet(
    target_year: int,
    summary: dict[str, Any],
    top_risky: list[dict[str, Any]],
    top_safe: list[dict[str, Any]],
) -> str:
    lines = [
        f"Grid safety context for {target_year}:",
        f"- Total grids: {summary['grid_count']}",
        f"- Hotspot grids (top 10% risk): {summary['hotspot_top10_count']}",
        (
            "- Safety score stats: "
            f"mean={summary['avg_safety_score']}, "
            f"p10={summary['p10_safety_score']}, "
            f"p50={summary['p50_safety_score']}, "
            f"p90={summary['p90_safety_score']}"
        ),
        "- Top risky grids (risk_rank asc): "
        + ", ".join(
            f"{r['grid_id']}[score={r['safety_score']}, rank={r['risk_rank']}]"
            for r in top_risky
        ),
        "- Top safe grids (safety_score desc): "
        + ", ".join(
            f"{r['grid_id']}[score={r['safety_score']}, rank={r['risk_rank']}]"
            for r in top_safe
        ),
    ]
    return "\n".join(lines)


def prepare_context(
    score_csv: Path,
    monthly_csv: Path | None,
    target_year: int,
    top_k: int,
    focus_k: int,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    score_rows = load_grid_scores(score_csv)
    summary = build_summary(score_rows)

    top_risky = sorted(score_rows, key=lambda x: (x["risk_rank"], -x["risk"]))[:top_k]
    top_safe = sorted(score_rows, key=lambda x: (-x["safety_score"], x["risk_rank"]))[:top_k]
    focus_ids = {row["grid_id"] for row in top_risky[:focus_k]}

    city_monthly, focus_trends, monthly_row_count = load_monthly_trends(
        monthly_csv=monthly_csv,
        target_year=target_year,
        focus_grid_ids=focus_ids,
    )

    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    source_meta: dict[str, Any] = {
        "score_csv": str(score_csv),
        "score_row_count": len(score_rows),
    }
    if monthly_csv is not None and monthly_csv.exists():
        source_meta["monthly_csv"] = str(monthly_csv)
        source_meta["monthly_row_count"] = monthly_row_count
    else:
        source_meta["monthly_csv"] = None
        source_meta["monthly_row_count"] = 0

    basic_context = {
        "schema_version": "1.0",
        "generated_at_utc": generated_at,
        "target_year": target_year,
        "source_files": source_meta,
        "data_dictionary": {
            "safety_score": "0-100, higher means safer",
            "risk": "annual weighted risk: 0.4*property + 0.6*violence",
            "risk_rank": "1 means highest risk grid",
            "hotspot_2026_top10pct": "1 if grid is in top 10% risk for 2026",
        },
        "summary": summary,
        "top_risky_grids": top_risky,
        "top_safe_grids": top_safe,
        "city_monthly_trend": city_monthly,
        "focus_grid_trends": {
            str(grid_id): values for grid_id, values in sorted(focus_trends.items())
        },
    }

    grid_index = {
        "schema_version": "1.0",
        "generated_at_utc": generated_at,
        "target_year": target_year,
        "grid_profiles": {
            str(row["grid_id"]): {
                "grid_id": row["grid_id"],
                "safety_score": row["safety_score"],
                "safety_level": row["safety_level"],
                "risk": row["risk"],
                "risk_rank": row["risk_rank"],
                "hotspot_2026_top10pct": row["hotspot_2026_top10pct"],
                "pred_property_annual": row["pred_property_annual"],
                "pred_violence_annual": row["pred_violence_annual"],
            }
            for row in score_rows
        },
    }

    prompt_snippet = build_prompt_snippet(
        target_year=target_year,
        summary=summary,
        top_risky=top_risky[: min(10, len(top_risky))],
        top_safe=top_safe[: min(10, len(top_safe))],
    )
    return basic_context, grid_index, prompt_snippet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare agent-ready context JSON from grid safety outputs."
    )
    parser.add_argument(
        "--score-csv",
        type=str,
        default=None,
        help="Path to grid score CSV (default: auto-detect from output directory).",
    )
    parser.add_argument(
        "--monthly-csv",
        type=str,
        default=str(OUTPUT_DIR / "future_monthly_predictions_2025_to_2026.csv"),
        help="Path to monthly prediction CSV. Use empty string to skip.",
    )
    parser.add_argument(
        "--target-year",
        type=int,
        default=2026,
        help="Year used for monthly city trend aggregation.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Number of top risky/safe grids to include in basic context.",
    )
    parser.add_argument(
        "--focus-k",
        type=int,
        default=5,
        help="Number of risky grids to include monthly trend details.",
    )
    parser.add_argument(
        "--basic-json",
        type=str,
        default=str(OUTPUT_DIR / "agent_grid_context_basic.json"),
        help="Output path for basic context JSON.",
    )
    parser.add_argument(
        "--index-json",
        type=str,
        default=str(OUTPUT_DIR / "agent_grid_profile_index_2026.json"),
        help="Output path for full grid profile index JSON.",
    )
    parser.add_argument(
        "--prompt-txt",
        type=str,
        default=str(OUTPUT_DIR / "agent_grid_prompt_snippet.txt"),
        help="Output path for text prompt snippet.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    score_csv = resolve_score_csv(args.score_csv)

    monthly_csv: Path | None
    if args.monthly_csv and args.monthly_csv.strip():
        monthly_csv = Path(args.monthly_csv).expanduser().resolve()
    else:
        monthly_csv = None

    basic_context, grid_index, prompt_snippet = prepare_context(
        score_csv=score_csv,
        monthly_csv=monthly_csv,
        target_year=args.target_year,
        top_k=max(1, args.top_k),
        focus_k=max(1, args.focus_k),
    )

    basic_json_path = Path(args.basic_json).expanduser().resolve()
    index_json_path = Path(args.index_json).expanduser().resolve()
    prompt_txt_path = Path(args.prompt_txt).expanduser().resolve()

    basic_json_path.parent.mkdir(parents=True, exist_ok=True)
    index_json_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_txt_path.parent.mkdir(parents=True, exist_ok=True)

    with basic_json_path.open("w", encoding="utf-8") as f:
        json.dump(basic_context, f, ensure_ascii=False, indent=2)
    with index_json_path.open("w", encoding="utf-8") as f:
        json.dump(grid_index, f, ensure_ascii=False, indent=2)
    prompt_txt_path.write_text(prompt_snippet, encoding="utf-8")

    print(f"Wrote: {basic_json_path}")
    print(f"Wrote: {index_json_path}")
    print(f"Wrote: {prompt_txt_path}")
    print(f"Grid count: {basic_context['summary']['grid_count']}")
    print(f"Hotspot count: {basic_context['summary']['hotspot_top10_count']}")


if __name__ == "__main__":
    main()
