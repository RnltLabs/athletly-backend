"""Activity storage: persist and retrieve parsed training activities."""

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from src.tools.fit_parser import is_activity_file, parse_fit_file

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent.parent / "data"
ACTIVITIES_DIR = DATA_DIR / "activities"
GFIT_DIR = DATA_DIR / "gfit"
MANIFEST_PATH = DATA_DIR / "import_manifest.json"


def store_activity(activity_data: dict, storage_dir: str | Path | None = None) -> Path:
    """Store parsed activity as JSON. Returns the file path.

    Filename is based on start_time and sport for easy identification.
    """
    dest = Path(storage_dir) if storage_dir else ACTIVITIES_DIR
    dest.mkdir(parents=True, exist_ok=True)

    start_time = activity_data.get("start_time", "")
    sport = activity_data.get("sport", "unknown")

    # Create a sortable filename from start time
    if start_time:
        try:
            dt = datetime.fromisoformat(start_time)
            ts = dt.strftime("%Y-%m-%d_%H%M%S")
        except ValueError:
            ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    else:
        ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")

    filename = f"{ts}_{sport}.json"
    path = dest / filename
    path.write_text(json.dumps(activity_data, indent=2))
    return path


def list_activities(
    storage_dir: str | Path | None = None,
    sport: str | None = None,
    after: str | None = None,
    before: str | None = None,
) -> list[dict]:
    """List stored activities with optional filters.

    Args:
        storage_dir: Directory to read from (default: data/activities/)
        sport: Filter by sport type (e.g., "running")
        after: Only include activities after this ISO date string
        before: Only include activities before this ISO date string

    Returns:
        List of activity dicts, sorted by start_time ascending.
    """
    src = Path(storage_dir) if storage_dir else ACTIVITIES_DIR
    if not src.exists():
        return []

    activities = []
    for path in sorted(src.glob("*.json")):
        data = json.loads(path.read_text())

        # Filter by sport
        if sport and data.get("sport") != sport:
            continue

        # Filter by date range
        start = data.get("start_time", "")
        if after and start and start < after:
            continue
        if before and start and start >= before:
            continue

        activities.append(data)

    return activities


def get_weekly_summary(activities: list[dict]) -> dict:
    """Summarize a collection of activities.

    Returns:
        dict with total_sessions, total_duration_minutes, total_distance_km,
        avg_hr, sessions_by_sport, total_trimp (if available).
    """
    if not activities:
        return {
            "total_sessions": 0,
            "total_duration_minutes": 0,
            "total_distance_km": 0.0,
            "avg_hr": None,
            "sessions_by_sport": {},
        }

    total_duration_sec = 0
    total_distance_m = 0
    hr_sum = 0
    hr_count = 0
    sessions_by_sport: dict[str, int] = {}

    for act in activities:
        # Duration
        dur = act.get("duration_seconds")
        if dur:
            total_duration_sec += dur

        # Distance
        dist = act.get("distance_meters")
        if dist:
            total_distance_m += dist

        # Heart rate
        hr_data = act.get("heart_rate")
        if hr_data and hr_data.get("avg"):
            hr_sum += hr_data["avg"]
            hr_count += 1

        # Sport counts
        sport = act.get("sport", "unknown")
        sessions_by_sport[sport] = sessions_by_sport.get(sport, 0) + 1

    return {
        "total_sessions": len(activities),
        "total_duration_minutes": round(total_duration_sec / 60, 1),
        "total_distance_km": round(total_distance_m / 1000, 2),
        "avg_hr": round(hr_sum / hr_count) if hr_count > 0 else None,
        "sessions_by_sport": sessions_by_sport,
    }


# ---------------------------------------------------------------------------
# Import manifest management
# ---------------------------------------------------------------------------


def load_manifest(manifest_path: str | Path | None = None) -> dict:
    """Load the import manifest from disk.

    Returns an empty dict if the file doesn't exist yet.
    """
    path = Path(manifest_path) if manifest_path else MANIFEST_PATH
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def save_manifest(manifest: dict, manifest_path: str | Path | None = None) -> None:
    """Write the import manifest to disk."""
    path = Path(manifest_path) if manifest_path else MANIFEST_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2))


def file_hash(file_path: str | Path) -> str:
    """Return SHA-256 hex digest of a file's contents."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Two-pass auto-import pipeline
# ---------------------------------------------------------------------------


def _existing_source_files(storage_dir: Path) -> set[str]:
    """Return the set of source_file values already stored as activity JSON."""
    source_files: set[str] = set()
    if not storage_dir.exists():
        return source_files
    for path in storage_dir.glob("*.json"):
        try:
            data = json.loads(path.read_text())
            sf = data.get("source_file")
            if sf:
                source_files.add(sf)
        except Exception:
            continue
    return source_files


def import_new_activities(
    gfit_dir: str | Path | None = None,
    storage_dir: str | Path | None = None,
    manifest_path: str | Path | None = None,
) -> list[dict]:
    """Scan for new FIT files, classify, parse activities, compute metrics, and store.

    Two-pass pipeline:
      Pass 0: Load manifest, find new files
      Pass 1: Classify each new file (activity vs non-activity)
      Pass 2: Parse activity files, compute metrics, store JSON

    Returns a list of newly imported activity dicts (for the startup report).
    """
    gfit = Path(gfit_dir) if gfit_dir else GFIT_DIR
    storage = Path(storage_dir) if storage_dir else ACTIVITIES_DIR
    m_path = Path(manifest_path) if manifest_path else MANIFEST_PATH

    # Pass 0 -- manifest load and new file detection
    manifest = load_manifest(m_path)

    if not gfit.exists():
        return []

    all_fit_files = sorted(gfit.glob("*.fit"))
    new_files = [f for f in all_fit_files if f.name not in manifest]

    if not new_files:
        return []

    # For dedup safety net: check which source_files already exist in storage
    existing_sources = _existing_source_files(storage)

    now = datetime.now(timezone.utc).isoformat()

    # Pass 1 -- classification (~1ms/file)
    activity_files: list[Path] = []
    for fit_file in new_files:
        try:
            if is_activity_file(str(fit_file)):
                activity_files.append(fit_file)
            else:
                manifest[fit_file.name] = {
                    "status": "non_activity",
                    "imported_at": now,
                    "activity_path": None,
                }
        except Exception as e:
            logger.warning("Error classifying %s: %s", fit_file.name, e)
            manifest[fit_file.name] = {
                "status": "error",
                "error": str(e),
                "file_hash": None,
                "imported_at": now,
                "activity_path": None,
            }

    # Save manifest after Pass 1 (non-activity files won't be re-scanned)
    save_manifest(manifest, m_path)

    # Pass 2 -- parse + store (~50ms/file)
    imported: list[dict] = []
    for fit_file in activity_files:
        try:
            # Dedup safety net: skip if already stored (e.g. manifest was deleted)
            if fit_file.name in existing_sources:
                manifest[fit_file.name] = {
                    "status": "activity",
                    "file_hash": file_hash(fit_file),
                    "imported_at": now,
                    "activity_path": None,  # already exists, don't know path
                    "note": "dedup_skipped",
                }
                continue

            activity = parse_fit_file(str(fit_file))

            # TRIMP/zone computation requires user_id + DB-defined formulas.
            # Callers with a user_id should call compute_metric() after import.
            activity["trimp"] = None
            activity["hr_zone"] = None

            # Zone distribution: use device data if present, otherwise leave as None
            # (fallback from record-level HR can be added later when needed)
            if activity.get("zone_distribution") is not None:
                pass  # already has device zone distribution
            else:
                activity["zone_distribution"] = None
                activity["zone_distribution_source"] = None

            # Store the activity
            stored_path = store_activity(activity, storage_dir=storage)

            # Record in manifest
            rel_path = str(stored_path.relative_to(DATA_DIR)) if stored_path.is_relative_to(DATA_DIR) else str(stored_path)
            manifest[fit_file.name] = {
                "status": "activity",
                "file_hash": file_hash(fit_file),
                "imported_at": now,
                "activity_path": rel_path,
            }

            imported.append(activity)

        except Exception as e:
            logger.warning("Error importing %s: %s", fit_file.name, e)
            manifest[fit_file.name] = {
                "status": "error",
                "error": str(e),
                "file_hash": None,
                "imported_at": now,
                "activity_path": None,
            }

    # Save manifest after Pass 2
    save_manifest(manifest, m_path)

    return imported
