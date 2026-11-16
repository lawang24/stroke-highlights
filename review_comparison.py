"""Review an archived classifier experiment; this viewer never runs inference."""

import argparse
import csv
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
import re
from urllib.parse import urlsplit

ASSETS = Path(__file__).resolve().parent / "web/comparison"


def ordered_matches(rows, events, tolerance):
    """One-to-one timestamp matches: most matches, then smallest total offset.

    On a time axis an optimal assignment can be ordered, so dynamic programming
    keeps the review server independent of NumPy/SciPy/model environments.
    """
    left = sorted(range(len(rows)), key=lambda i: rows[i]["time"])
    right = sorted(range(len(events)), key=lambda i: events[i]["time"])
    choices = [bytearray(len(right) + 1) for _ in range(len(left) + 1)]
    previous = [(0, 0.0)] * (len(right) + 1)
    for i, li in enumerate(left, 1):
        current = [(0, 0.0)]
        for j, ri in enumerate(right, 1):
            best, choice = previous[j], 1
            if current[j-1] > best:
                best, choice = current[j-1], 2
            delta = abs(rows[li]["time"] - events[ri]["time"])
            if delta <= tolerance:
                match = (previous[j-1][0] + 1, previous[j-1][1] - delta)
                if match > best:
                    best, choice = match, 3
            current.append(best)
            choices[i][j] = choice
        previous = current
    i, j, matches = len(left), len(right), []
    while i and j:
        choice = choices[i][j]
        if choice == 3:
            matches.append((left[i-1], right[j-1]))
            i, j = i-1, j-1
        elif choice == 1:
            i -= 1
        else:
            j -= 1
    return matches


def load_comparison(directory, video=None):
    directory = Path(directory).resolve()
    reviews = {name: json.loads((directory / name / "stroke-review.json").read_text())
               for name in ("heuristic", "adaspot")}
    source = reviews["heuristic"]["source"]
    other = reviews["adaspot"]["source"]
    if any(source[key] != other[key] for key in ("name", "size", "duration")):
        raise ValueError("The two classifier results refer to different videos")
    if video is None:
        # Only look for this exact filename beside the comparison, not across disk.
        video = directory.parent / Path(source["name"]).name
    video = Path(video).resolve()
    if not video.is_file() or video.name != source["name"] or video.stat().st_size != source["size"]:
        raise ValueError(f"Use --video to select the original {source['name']} ({source['size']} bytes)")
    rows = []
    with (directory / "comparison.csv").open(newline="") as handle:
        for index, row in enumerate(csv.DictReader(handle)):
            result = {"id": f"proposal-{index}", "status": row["status"]}
            times = []
            for name in reviews:
                time = row[f"{name}_time"]
                result[name] = None
                if time != "":
                    time = float(time)
                    times.append(time)
                    result[name] = next(e for e in reviews[name]["events"] if abs(float(e["time"]) - time) < .0001)
            if not times:
                raise ValueError("Comparison contains a row without a timestamp")
            result["time"] = sum(times) / len(times)
            rows.append(result)
    payload = {"source": source, "rows": rows,
               "settings": {name: review.get("settings", {}) for name, review in reviews.items()}}
    payload["reviewId"] = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:20]
    # Keep the base draft identity and proposal IDs stable when a new candidate
    # is added, so an in-progress human review survives reloading the page.
    payload["models"] = [{"id": "heuristic", "label": "Motion heuristic"},
                         {"id": "adaspot", "label": "AdaSpot"}]
    candidate_path = directory / "tennis-rnn/stroke-review.json"
    if candidate_path.exists():
        candidate = json.loads(candidate_path.read_text())
        if any(source[key] != candidate["source"][key] for key in ("name", "size", "duration")):
            raise ValueError("Tennis RNN results refer to a different video")
        tolerance = payload["settings"]["adaspot"].get("compare_tolerance", .5)
        matches = ordered_matches(rows, candidate["events"], tolerance)
        for i, j in matches:
            rows[i]["tennis-rnn"] = candidate["events"][j]
            labels = {rows[i][k]["stroke"] for k in ("heuristic", "adaspot", "tennis-rnn") if rows[i].get(k)}
            rows[i]["status"] = "different_label" if len(labels) > 1 else "agree"
        used = {j for _, j in matches}
        for j, event in enumerate(candidate["events"]):
            if j not in used:
                rows.append({"id": f"tennis-rnn-{event['time']:.3f}", "time": event["time"],
                             "status": "tennis_rnn_only", "heuristic": None, "adaspot": None,
                             "tennis-rnn": event})
        payload["settings"]["tennis-rnn"] = candidate.get("settings", {})
        payload["models"].append({"id": "tennis-rnn", "label": "Tennis RNN"})
        rows.sort(key=lambda r: r["time"])
    return video, payload


def byte_range(header, size):
    if header is None:
        return 0, size - 1, False
    match = re.fullmatch(r"bytes=(\d*)-(\d*)", header)
    if not match or not any(match.groups()):
        raise ValueError("Invalid range")
    first, last = match.groups()
    if first:
        start, end = int(first), min(int(last) if last else size - 1, size - 1)
    else:
        length = int(last)
        if length <= 0:
            raise ValueError("Invalid suffix range")
        start, end = max(0, size - length), size - 1
    if not 0 <= start <= end < size:
        raise ValueError("Range outside video")
    return start, end, True


def handler_for(video, payload, loader=None):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def end_headers(self):
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Cross-Origin-Resource-Policy", "same-origin")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Security-Policy", "default-src 'self'; media-src 'self'; style-src 'self'; script-src 'self'; connect-src 'self'; frame-ancestors 'none'; object-src 'none'; base-uri 'none'")
            super().end_headers()

        def do_HEAD(self):
            self.respond(head=True)

        def do_GET(self):
            self.respond(head=False)

        def respond(self, head=False):
            path = urlsplit(self.path).path
            if path == "/video":
                size = video.stat().st_size
                try:
                    start, end, partial = byte_range(self.headers.get("Range"), size)
                except ValueError:
                    self.send_response(416)
                    self.send_header("Content-Range", f"bytes */{size}")
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                self.send_response(206 if partial else 200)
                self.send_header("Content-Type", mimetypes.guess_type(video.name)[0] or "video/mp4")
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("Content-Length", str(end - start + 1))
                if partial:
                    self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
                self.end_headers()
                if not head:
                    try:
                        with video.open("rb") as handle:
                            handle.seek(start)
                            remaining = end - start + 1
                            while remaining:
                                block = handle.read(min(256 * 1024, remaining))
                                if not block:
                                    break
                                self.wfile.write(block)
                                remaining -= len(block)
                    except (BrokenPipeError, ConnectionResetError):
                        pass  # Seeking cancels an earlier range request.
                return
            assets = {"/": ("index.html", "text/html"), "/review.js": ("review.js", "text/javascript"),
                      "/review.css": ("review.css", "text/css")}
            if path == "/api/comparison":
                try:
                    current = loader()[1] if loader else payload
                except (ValueError, OSError, KeyError, StopIteration):
                    self.send_error(409, "Comparison is being updated or has an incompatible source")
                    return
                body, mime = json.dumps(current).encode(), "application/json"
            elif path in assets:
                name, mime = assets[path]
                body = (ASSETS / name).read_bytes()
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", f"{mime}; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            if not head:
                self.wfile.write(body)
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("comparison_dir", type=Path)
    parser.add_argument("--video", type=Path)
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()
    try:
        video, payload = load_comparison(args.comparison_dir, args.video)
        server = ThreadingHTTPServer(("127.0.0.1", args.port), handler_for(
            video, payload, lambda: load_comparison(args.comparison_dir, args.video)))
    except (ValueError, OSError, KeyError, StopIteration) as exc:
        parser.exit(1, f"Cannot open comparison: {exc}\n")
    print(f"Review comparison: http://127.0.0.1:{server.server_port}\nVideo and annotations stay on this computer. Ctrl+C stops the server.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
