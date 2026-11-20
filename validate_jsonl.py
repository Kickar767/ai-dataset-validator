import json
import re
from pathlib import Path


REQUIRED_KEYS = {"prompt", "response", "language"}


def validate_jsonl_dataset(file_path: str) -> dict:
    """Validate JSONL records used in LLM fine-tuning and RLHF evaluation."""
    report = {
        "total_rows": 0,
        "valid_rows": 0,
        "errors": [],
    }

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    with path.open("r", encoding="utf-8") as f:
        for line_num, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue

            report["total_rows"] += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                report["errors"].append(f"Line {line_num}: Invalid JSON ({exc})")
                continue

            missing = REQUIRED_KEYS - set(record.keys())
            if missing:
                report["errors"].append(f"Line {line_num}: Missing keys {sorted(missing)}")
                continue

            if not str(record["response"]).strip():
                report["errors"].append(f"Line {line_num}: Empty model response")
                continue

            if re.search(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", str(record["response"])):
                report["errors"].append(f"Line {line_num}: Invalid control characters detected")
                continue

            report["valid_rows"] += 1

    return report


if __name__ == "__main__":
    print("AI dataset validator ready: validate_jsonl_dataset('dataset.jsonl')")
