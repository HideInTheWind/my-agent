import json
from datetime import datetime, timezone
from pathlib import Path

AUDIT_LOG = Path(__file__).parent / "audit.log"


def log_tool_call(
    tool_name: str,
    args: dict,
    result,
    latency_ms: float,
) -> None:
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tool_name": tool_name,
        "args": args,
        "result": result,
        "latency_ms": latency_ms,
    }
    with open(AUDIT_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
