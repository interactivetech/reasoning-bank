#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_preds(path: Path) -> dict:
    return json.loads(path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge split SWE-Bench preds.json files and report empty-patch tasks."
    )
    parser.add_argument(
        "--parts",
        nargs="+",
        required=True,
        help="Input preds.json files to merge.",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output merged predictions JSON path.",
    )
    parser.add_argument(
        "--missing-out",
        help="Optional path to write empty-patch task ids, one per line.",
    )
    parser.add_argument(
        "--target-count",
        type=int,
        default=30,
        help="Expected total number of unique tasks in the merged result.",
    )
    args = parser.parse_args()

    merged: dict = {}
    for part in args.parts:
        path = Path(part)
        data = load_preds(path)
        overlap = set(merged) & set(data)
        if overlap:
            raise SystemExit(f"Duplicate task ids found in {path}: {sorted(overlap)}")
        merged.update(data)

    out_path = Path(args.out)
    out_path.write_text(json.dumps(merged, indent=2))

    empty_patch_ids = sorted(
        task_id for task_id, pred in merged.items() if not pred.get("model_patch")
    )
    nonempty_patch_ids = sorted(task_id for task_id in merged if task_id not in empty_patch_ids)

    print(f"merged_predictions: {out_path}")
    print(f"total_unique_tasks: {len(merged)}")
    print(f"nonempty_patch_tasks: {len(nonempty_patch_ids)}")
    print(f"empty_patch_tasks: {len(empty_patch_ids)}")

    if args.target_count:
        print(f"target_task_count: {args.target_count}")
        print(f"missing_valid_predictions_for_target: {max(args.target_count - len(nonempty_patch_ids), 0)}")

    if empty_patch_ids:
        print("\nempty_patch_ids:")
        for task_id in empty_patch_ids:
            print(task_id)

    if args.missing_out:
        missing_path = Path(args.missing_out)
        missing_path.write_text("".join(f"{task_id}\n" for task_id in empty_patch_ids))
        print(f"\nmissing_task_file: {missing_path}")


if __name__ == "__main__":
    main()
