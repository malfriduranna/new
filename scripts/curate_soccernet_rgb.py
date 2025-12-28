import argparse
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


DEFAULT_CLASSES = ["Pass", "Shot", "Ball Touch", "Foul"]
DEFAULT_COMPETITION = "England Premier League"
DEFAULT_PATTERN = "{game_id}/{half}_720p.{ext}"


@dataclass
class Event:
    game_id: str
    split: str
    label: str
    timestamp: float
    half: int


def _parse_time_seconds(time_str: str) -> float:
    # Example format: "1 - 00:12:34"
    half_part, clock = [p.strip() for p in time_str.split("-")]
    half = int(half_part)
    h, m, s = clock.split(":")
    return half, int(h) * 3600 + int(m) * 60 + int(s)


def _event_timestamp(event: Dict) -> float:
    if "position" in event and event["position"] is not None:
        pos = float(event["position"])
        return pos / 1000.0 if pos > 1e3 else pos
    if "gameTime" in event and event["gameTime"]:
        _, ts = _parse_time_seconds(event["gameTime"])
        return float(ts)
    raise ValueError(f"Event is missing position/gameTime fields: {event}")


def _normalize_games(payload) -> List[Dict]:
    if isinstance(payload, dict):
        if "games" in payload:
            games = payload["games"]
        elif "annotations" in payload:
            games = [payload]
        else:
            raise ValueError("Labels JSON must contain a 'games' list or 'annotations' field.")
    elif isinstance(payload, list):
        games = payload
    else:
        raise ValueError(f"Unexpected labels structure: {type(payload)}")
    return games


def _game_identifier(game: Dict, labels_path: Path) -> str:
    default_id = labels_path.parent.name
    raw = game.get("game") or game.get("id") or game.get("match") or game.get("UrlLocal") or default_id
    raw = str(raw).strip()
    return raw.rstrip("/")


def _load_events(labels_path: Path, keep_classes: Sequence[str], competition: str) -> List[Event]:
    data = json.loads(labels_path.read_text())
    games = _normalize_games(data)
    events: List[Event] = []
    for game in games:
        if competition and game.get("competition") and game["competition"] != competition:
            continue
        game_id = _game_identifier(game, labels_path)
        split = game.get("subset") or "train"
        for ann in game.get("annotations", []):
            label = ann.get("label")
            if label not in keep_classes:
                continue
            half = int(ann.get("half") or ann.get("period") or 1)
            ts = _event_timestamp(ann)
            # If the video contains both halves, shift second-half timestamps by 45 minutes.
            if half == 2 and ts < 60 * 60:
                ts = ts + 45 * 60
            events.append(Event(game_id=game_id, split=split, label=label, timestamp=ts, half=half))
    return events


def _maybe_glob(game_id: str, half: int, videos_root: Path) -> Path:
    candidates: Iterable[Path] = videos_root.glob(f"**/{game_id}*{half}*.mp4")
    for candidate in candidates:
        return candidate
    mkv_candidates: Iterable[Path] = videos_root.glob(f"**/{game_id}*{half}*.mkv")
    for candidate in mkv_candidates:
        return candidate
    raise FileNotFoundError(f"Could not locate a video for {game_id} half {half} under {videos_root}")


def _resolve_video(game_id: str, half: int, videos_root: Path, pattern: str | None) -> Path:
    if pattern:
        for ext in ("mp4", "mkv"):
            candidate = videos_root / pattern.format(game_id=game_id, half=half, ext=ext)
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"Pattern path not found for {game_id} half {half}")
    try:
        return _maybe_glob(game_id, half, videos_root)
    except FileNotFoundError:
        fallback = videos_root / f"{game_id}.mp4"
        if fallback.exists():
            return fallback
        raise


def _cut_clip(video_path: Path, start: float, duration: float, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{start:.3f}",
        "-i",
        str(video_path),
        "-t",
        f"{duration:.3f}",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-pix_fmt",
        "yuv420p",
        "-an",
        str(output_path),
    ]
    subprocess.run(cmd, check=True)


def _iter_label_files(root: Path) -> Iterable[Path]:
    stack = [root]
    visited: set[Path] = set()
    while stack:
        current = stack.pop()
        if current in visited:
            continue
        visited.add(current)
        try:
            if current.is_file():
                if current.name == "Labels-v2.json":
                    yield current
                continue
            if current.is_dir():
                for child in current.iterdir():
                    stack.append(child)
        except OSError as exc:
            print(f"Warning: skipping {current} ({exc})")
            continue


def _gather_label_files(path: Path) -> List[Path]:
    if path.is_file():
        return [path]
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist.")
    matches = sorted(_iter_label_files(path))
    if matches:
        return matches
    raise FileNotFoundError(f"No Labels-v2.json files found under {path}")


def curate(
    labels_path: Path,
    videos_root: Path,
    output_root: Path,
    keep_classes: Sequence[str],
    *,
    competition: str = DEFAULT_COMPETITION,
    clip_seconds: float = 2.0,
    pattern: str | None = None,
) -> Dict[str, int]:
    events = _load_events(labels_path, keep_classes, competition)
    class_to_idx = {name: idx for idx, name in enumerate(keep_classes)}
    counts = {name: 0 for name in keep_classes}
    for event in events:
        start = max(event.timestamp - clip_seconds / 2, 0.0)
        try:
            video_path = _resolve_video(event.game_id, event.half, videos_root, pattern)
        except FileNotFoundError as exc:
            print(f"Warning: skipping {event.game_id} half {event.half}: {exc}")
            continue
        clip_name = f"{event.game_id}_h{event.half}_{int(event.timestamp * 1000)}.mp4"
        clip_dir = output_root / event.split / event.label
        clip_path = clip_dir / clip_name
        _cut_clip(video_path, start=start, duration=clip_seconds, output_path=clip_path)
        counts[event.label] += 1
    summary = {label: counts[label] for label in keep_classes}
    print(f"Finished processing {labels_path}. Clips per class: {summary}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Curate SoccerNet RGB clips for VideoMAE.")
    parser.add_argument(
        "--labels-json",
        type=Path,
        default=Path("SoccerNet/england_epl/2014-2015"),
        help="Path to a Labels-v2.json file or directory to scan (default: SoccerNet/england_epl/2014-2015).",
    )
    parser.add_argument(
        "--videos-root",
        type=Path,
        default=Path("SoccerNet"),
        help="Root directory containing 720p videos (default: SoccerNet).",
    )
    parser.add_argument("--output-root", type=Path, default=Path("data/soccernet_rgb"), help="Directory to write curated clips.")
    parser.add_argument(
        "--classes",
        nargs="+",
        default=DEFAULT_CLASSES,
        help="Action classes to retain (default: Pass, Shot, Ball Touch, Foul).",
    )
    parser.add_argument(
        "--competition",
        type=str,
        default=DEFAULT_COMPETITION,
        help="Filter matches by competition (default: England Premier League).",
    )
    parser.add_argument(
        "--video-pattern",
        type=str,
        default=DEFAULT_PATTERN,
        help="Format string to locate videos (default: '{game_id}/{half}_720p.{ext}').",
    )
    parser.add_argument("--clip-seconds", type=float, default=2.0, help="Clip duration in seconds (default: 2.0).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    label_files = _gather_label_files(args.labels_json)
    totals = {label: 0 for label in args.classes}
    for label_path in label_files:
        try:
            summary = curate(
                labels_path=label_path,
                videos_root=args.videos_root,
                output_root=args.output_root,
                keep_classes=args.classes,
                competition=args.competition,
                clip_seconds=args.clip_seconds,
                pattern=args.video_pattern,
            )
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            print(f"Warning: skipping {label_path} due to error: {exc}")
            continue
        else:
            for label, count in summary.items():
                totals[label] += count
    print(f"All files processed. Total clips per class: {totals}")


if __name__ == "__main__":
    main()
