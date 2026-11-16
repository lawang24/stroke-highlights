"""Portable shot list format shared by the CLI and browser."""

import json


def write_browser_review(path, source, metadata, events, classifier, settings=None):
    """Existing browser Load shot list format; use the full source duration."""
    data = {"version": 1, "classifier": classifier, "settings": settings or {},
            "source": {"name": source.name, "size": source.stat().st_size,
                       "lastModified": source.stat().st_mtime_ns // 1_000_000,
                       "duration": metadata["duration"]}, "events": events}
    path.write_text(json.dumps(data, indent=2) + "\n")
