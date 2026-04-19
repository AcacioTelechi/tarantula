from __future__ import annotations

import json
import logging
from pathlib import Path


def configure(verbosity: int, log_path: Path | None) -> None:
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG

    handlers: list[logging.Handler] = [logging.StreamHandler()]
    for h in handlers:
        h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))

    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setFormatter(_JSONLFormatter())
        fh.setLevel(logging.DEBUG)
        handlers.append(fh)

    root = logging.getLogger()
    root.setLevel(level)
    for h in handlers:
        root.addHandler(h)


class _JSONLFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        return json.dumps({
            "ts": self.formatTime(record),
            "level": record.levelname,
            "name": record.name,
            "msg": record.getMessage(),
        })
