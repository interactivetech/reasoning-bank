#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_events(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def fmt_pct(numerator: float, denominator: float) -> str:
    if denominator == 0:
        return "0.0%"
    return f"{100.0 * numerator / denominator:.1f}%"


def event_labels(event: dict) -> list[str]:
    if "quality_labels" in event:
        return event.get("quality_labels", [])
    return event.get("pollution_flags", [])


def retrieval_has_labeled_memory(event: dict) -> bool:
    if "selected_quality_labels" in event:
        return any(labels for labels in event.get("selected_quality_labels", []))
    return any(score > 0 for score in event.get("selected_pollution_scores", []))


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze ReasoningBank memory pollution telemetry.")
    parser.add_argument("telemetry_path", type=Path, help="Path to *_telemetry.jsonl")
    args = parser.parse_args()

    events = load_events(args.telemetry_path)
    writes = [e for e in events if e.get("event") == "write"]
    retrievals = [e for e in events if e.get("event") == "retrieval"]

    labeled_writes = [e for e in writes if event_labels(e)]
    labeled_retrievals = [e for e in retrievals if retrieval_has_labeled_memory(e)]

    print(f"telemetry_file: {args.telemetry_path}")
    print(f"total_events: {len(events)}")
    print(f"writes: {len(writes)}")
    print(f"retrievals: {len(retrievals)}")
    print(f"writes_with_quality_labels: {len(labeled_writes)} ({fmt_pct(len(labeled_writes), len(writes))})")
    print(
        f"retrievals_with_labeled_memory: {len(labeled_retrievals)} "
        f"({fmt_pct(len(labeled_retrievals), len(retrievals))})"
    )

    label_counts: dict[str, int] = {}
    for event in writes:
        for label in event_labels(event):
            label_counts[label] = label_counts.get(label, 0) + 1

    print("\nwrite_label_counts:")
    if not label_counts:
        print("  none")
    else:
        for label, count in sorted(label_counts.items(), key=lambda kv: (-kv[1], kv[0])):
            print(f"  {label}: {count}")

    feature_counts: dict[str, int] = {}
    for event in writes:
        features = event.get("quality_features", {})
        for key, value in features.items():
            if isinstance(value, bool) and value:
                feature_counts[key] = feature_counts.get(key, 0) + 1

    print("\nwrite_feature_counts:")
    if not feature_counts:
        print("  none")
    else:
        for key, count in sorted(feature_counts.items(), key=lambda kv: (-kv[1], kv[0])):
            print(f"  {key}: {count}")

    print("\nwrite_timeline:")
    for idx, event in enumerate(writes, 1):
        features = event.get("quality_features", {})
        print(
            f"  {idx:03d} task={event['task_id']} status={event.get('status')} "
            f"bank_before={event.get('memory_bank_size_before')} "
            f"items={features.get('item_count', event.get('item_count'))} "
            f"structured={features.get('looks_structured', 'na')} "
            f"labels={','.join(event_labels(event)) or 'none'}"
        )

    print("\nretrieval_timeline:")
    for idx, event in enumerate(retrievals, 1):
        selected = event.get("selected_task_ids", [])
        if "selected_quality_labels" in event:
            labels = event.get("selected_quality_labels", [])
            zipped = ", ".join(
                f"{task}:{'|'.join(task_labels) if task_labels else 'clean'}"
                for task, task_labels in zip(selected, labels)
            )
        else:
            scores = event.get("selected_pollution_scores", [])
            zipped = ", ".join(f"{task}:{score:.2f}" for task, score in zip(selected, scores))
        print(
            f"  {idx:03d} task={event['task_id']} bank_size={event.get('memory_bank_size')} "
            f"selected={event.get('selected_count')} labeled={retrieval_has_labeled_memory(event)} "
            f"hits=[{zipped}]"
        )


if __name__ == "__main__":
    main()
