"""User model with belief-driven memory following Mem0/Zep/Graphiti patterns.

Central representation of the athlete. The profile is a structured PROJECTION
of the user model. Beliefs have confidence, temporal metadata, stability,
durability, and embedding vectors for similarity search.
"""

import json
import uuid
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from src.agent.llm import get_client

DATA_DIR = Path(__file__).parent.parent.parent / "data"
MODEL_DIR = DATA_DIR / "user_model"
MODEL_PATH = MODEL_DIR / "model.json"
ARCHIVE_PATH = MODEL_DIR / "beliefs_archive.json"

EMBEDDING_MODEL = "models/text-embedding-004"

# Valid category values for beliefs
BELIEF_CATEGORIES = {
    "preference",
    "constraint",
    "history",
    "motivation",
    "physical",
    "fitness",
    "scheduling",
    "personality",
    "meta",
}


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _today_iso() -> str:
    return datetime.now().date().isoformat()


class UserModel:
    """Belief-driven user model with embedding-based similarity search."""

    def __init__(self, data_dir: Path | None = None):
        self._data_dir = Path(data_dir) if data_dir else MODEL_DIR
        self._model_path = self._data_dir / "model.json"
        self._archive_path = self._data_dir / "beliefs_archive.json"

        self.structured_core: dict = {
            "name": None,
            "sports": [],
            "goal": {
                "event": None,
                "target_date": None,
                "target_time": None,
                "goal_type": None,
            },
            "fitness": {
                "estimated_vo2max": None,
                "threshold_pace_min_km": None,
                "weekly_volume_km": None,
                "trend": "unknown",
            },
            "constraints": {
                "training_days_per_week": None,
                "max_session_minutes": None,
                "available_sports": [],
            },
        }
        self.beliefs: list[dict] = []
        self.meta: dict = {
            "created_at": _now_iso(),
            "updated_at": _now_iso(),
            "sessions_completed": 0,
            "last_interaction": None,
        }

    # ── Belief CRUD ──────────────────────────────────────────────

    def add_belief(
        self,
        text: str,
        category: str,
        confidence: float = 0.7,
        source: str = "conversation",
        source_ref: str | None = None,
        durability: str = "global",
        stability: str = "stable",
        valid_from: str | None = None,
        valid_until: str | None = None,
        embedding: list[float] | None = None,
    ) -> dict:
        """Create and store a new belief. Returns the belief dict."""
        now = _now_iso()
        belief = {
            "id": f"belief_{uuid.uuid4().hex[:8]}",
            "text": text,
            "category": category if category in BELIEF_CATEGORIES else "preference",
            "confidence": max(0.0, min(1.0, confidence)),
            "stability": stability,
            "durability": durability,
            "source": source,
            "source_ref": source_ref,
            "first_observed": now,
            "last_confirmed": now,
            "valid_from": valid_from or _today_iso(),
            "valid_until": valid_until,
            "learned_at": now,
            "archived_at": None,
            "active": True,
            "superseded_by": None,
            "embedding": embedding,
            # Outcome tracking fields (P6: active memory)
            "utility": 0.0,
            "outcome_count": 0,
            "last_outcome": None,
            "outcome_history": [],
        }
        self.beliefs.append(belief)
        self.meta["updated_at"] = now
        return belief

    def update_belief(
        self,
        belief_id: str,
        new_text: str | None = None,
        new_confidence: float | None = None,
    ) -> dict | None:
        """Update an existing belief. Bumps last_confirmed. Returns updated belief or None."""
        for belief in self.beliefs:
            if belief["id"] == belief_id and belief["active"]:
                now = _now_iso()
                if new_text is not None:
                    belief["text"] = new_text
                    belief["embedding"] = None  # needs re-embedding
                if new_confidence is not None:
                    belief["confidence"] = max(0.0, min(1.0, new_confidence))
                belief["last_confirmed"] = now
                self.meta["updated_at"] = now
                return belief
        return None

    def invalidate_belief(
        self,
        belief_id: str,
        superseded_by: str | None = None,
    ) -> dict | None:
        """Mark belief as inactive (never delete). Sets archived_at. Returns the belief or None."""
        for belief in self.beliefs:
            if belief["id"] == belief_id and belief["active"]:
                now = _now_iso()
                belief["active"] = False
                belief["archived_at"] = now
                belief["valid_until"] = _today_iso()
                if superseded_by:
                    belief["superseded_by"] = superseded_by
                self.meta["updated_at"] = now
                return belief
        return None

    def get_active_beliefs(
        self,
        category: str | None = None,
        min_confidence: float = 0.0,
    ) -> list[dict]:
        """Retrieve active beliefs, optionally filtered by category and confidence."""
        results = []
        for b in self.beliefs:
            if not b["active"]:
                continue
            if category and b["category"] != category:
                continue
            if b["confidence"] < min_confidence:
                continue
            results.append(b)
        return results

    # ── Outcome Recording (P6: active memory) ───────────────────

    def record_outcome(
        self,
        belief_id: str,
        outcome: str,
        detail: str = "",
    ) -> dict | None:
        """Record an outcome for a belief, updating confidence and utility.

        Args:
            belief_id: ID of the belief to update.
            outcome: "confirmed" or "contradicted".
            detail: Optional description of the evidence.

        Returns:
            The updated belief, or None if not found.
        """
        for belief in self.beliefs:
            if belief["id"] == belief_id and belief["active"]:
                now = _now_iso()

                # Ensure outcome fields exist (backwards compatibility)
                belief.setdefault("utility", 0.0)
                belief.setdefault("outcome_count", 0)
                belief.setdefault("last_outcome", None)
                belief.setdefault("outcome_history", [])

                # Update confidence based on outcome
                if outcome == "confirmed":
                    belief["confidence"] = min(1.0, belief["confidence"] + 0.05)
                    belief["utility"] = min(1.0, belief["utility"] + 0.1)
                elif outcome == "contradicted":
                    belief["confidence"] = max(0.0, belief["confidence"] - 0.1)
                    belief["utility"] = max(0.0, belief["utility"] - 0.05)

                belief["outcome_count"] += 1
                belief["last_outcome"] = outcome
                belief["last_confirmed"] = now
                belief["outcome_history"].append({
                    "date": _today_iso(),
                    "type": outcome,
                    "detail": detail,
                })

                self.meta["updated_at"] = now
                return belief
        return None

    def get_high_utility_beliefs(
        self,
        min_utility: float = 0.3,
        min_confidence: float = 0.6,
    ) -> list[dict]:
        """Retrieve active beliefs with proven utility from outcome tracking."""
        results = []
        for b in self.beliefs:
            if not b["active"]:
                continue
            utility = b.get("utility", 0.0)
            if utility >= min_utility and b["confidence"] >= min_confidence:
                results.append(b)
        results.sort(key=lambda b: b.get("utility", 0.0), reverse=True)
        return results

    # ── User Model Summary (for prompt injection) ────────────────

    def get_model_summary(self) -> str:
        """Return a concise text summary of the user model for LLM prompt injection.

        This implements the PrefEval reminder injection pattern:
        active beliefs are formatted as COACH'S NOTES for every LLM call.
        """
        lines = []

        # Structured core summary
        core = self.structured_core
        if core.get("name"):
            lines.append(f"Athlete: {core['name']}")
        if core.get("sports"):
            lines.append(f"Sports: {', '.join(core['sports'])}")

        goal = core.get("goal", {})
        if goal.get("event"):
            parts = [f"Goal: {goal['event']}"]
            if goal.get("target_date"):
                parts.append(f"by {goal['target_date']}")
            if goal.get("target_time"):
                parts.append(f"in {goal['target_time']}")
            lines.append(" ".join(parts))

        from src.utils.pace_format import decimal_min_to_mmss
        fitness = core.get("fitness", {})
        fit_parts = []
        if fitness.get("estimated_vo2max"):
            fit_parts.append(f"VO2max ~{fitness['estimated_vo2max']}")
        threshold_pretty = decimal_min_to_mmss(
            fitness.get("threshold_pace_min_km")
        )
        if threshold_pretty is not None:
            # Pre-formatted: never leak the decimal into the prompt.
            fit_parts.append(f"threshold {threshold_pretty} /km")
        if fitness.get("weekly_volume_km"):
            fit_parts.append(f"{fitness['weekly_volume_km']} km/week")
        if fit_parts:
            lines.append(f"Fitness: {', '.join(fit_parts)}")

        constraints = core.get("constraints", {})
        if constraints.get("training_days_per_week"):
            lines.append(
                f"Constraints: {constraints['training_days_per_week']} days/week, "
                f"max {constraints.get('max_session_minutes', '?')} min/session"
            )

        # Active beliefs grouped by category
        active = self.get_active_beliefs(min_confidence=0.6)
        if active:
            lines.append("\nCOACH'S NOTES ON THIS ATHLETE:")
            by_cat: dict[str, list[str]] = {}
            for b in active:
                cat = b["category"]
                if cat not in by_cat:
                    by_cat[cat] = []
                by_cat[cat].append(f"- {b['text']} (confidence: {b['confidence']:.1f})")

            for cat in sorted(by_cat):
                lines.append(f"  [{cat.upper()}]")
                for line in by_cat[cat]:
                    lines.append(f"  {line}")

        return "\n".join(lines)

    # ── Profile Projection (backward compatibility) ──────────────

    def project_profile(self) -> dict:
        """Generate a profile.json-compatible dict from structured_core.

        This ensures backward compatibility with existing generate_plan(),
        assess_training(), and other Step 1-5 functions.
        """
        core = self.structured_core
        now = _now_iso()
        return {
            "name": core.get("name") or "Athlete",
            "sports": core.get("sports") or [],
            "goal": {
                "event": core.get("goal", {}).get("event"),
                "target_date": core.get("goal", {}).get("target_date"),
                "target_time": core.get("goal", {}).get("target_time"),
            },
            "fitness": {
                "estimated_vo2max": core.get("fitness", {}).get("estimated_vo2max"),
                "threshold_pace_min_km": core.get("fitness", {}).get("threshold_pace_min_km"),
                "weekly_volume_km": core.get("fitness", {}).get("weekly_volume_km"),
                "trend": core.get("fitness", {}).get("trend", "unknown"),
            },
            "constraints": {
                "training_days_per_week": core.get("constraints", {}).get("training_days_per_week") or 5,
                "max_session_minutes": core.get("constraints", {}).get("max_session_minutes") or 90,
                "available_sports": core.get("constraints", {}).get("available_sports") or core.get("sports") or [],
            },
            "created_at": self.meta.get("created_at", now),
            "updated_at": self.meta.get("updated_at", now),
        }

    # ── Structured Core Updates ──────────────────────────────────

    def update_structured_core(self, field_path: str, value) -> None:
        """Update a nested field in structured_core using dot-notation.

        Example: update_structured_core("goal.target_date", "2026-10-15")
        """
        parts = field_path.split(".")
        target = self.structured_core
        for part in parts[:-1]:
            if part not in target or not isinstance(target[part], dict):
                target[part] = {}
            target = target[part]
        target[parts[-1]] = value
        self.meta["updated_at"] = _now_iso()

    # ── Embedding & Similarity Search ────────────────────────────

    def embed_belief(self, belief: dict) -> list[float] | None:
        """Generate embedding vector via Gemini Embedding API and store in belief.

        Returns the embedding vector or None if the API call fails / is disabled.
        Embeddings are optional - belief storage and dedup work without them.
        Set ATHLETLY_EMBEDDINGS_DISABLED=1 to skip entirely (e.g. when the
        Gemini key is unavailable).
        """
        import os

        if os.environ.get("ATHLETLY_EMBEDDINGS_DISABLED") == "1":
            return None
        if not os.environ.get("GEMINI_API_KEY"):
            return None
        try:
            client = get_client()
            response = client.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=belief["text"],
            )
            embedding = response.embeddings[0].values
            belief["embedding"] = list(embedding)
            return belief["embedding"]
        except Exception:
            return None

    def find_similar_beliefs(
        self,
        candidate_text: str,
        top_k: int = 3,
    ) -> list[tuple[dict, float]]:
        """Embed candidate text and find top-k similar active beliefs via cosine similarity.

        Returns list of (belief, similarity_score) tuples sorted by similarity descending.
        Falls back to returning all active beliefs if < 10 beliefs exist (Mem0 fallback pattern).
        """
        import os

        active = self.get_active_beliefs()

        # Fallback: if few beliefs, return all (no need for embedding search)
        if len(active) < 10:
            return [(b, 1.0) for b in active]

        # Embeddings can be disabled via env or missing Gemini key. In that
        # case fall back to returning all active beliefs - the caller gets
        # the same set, just without similarity ranking.
        if (
            os.environ.get("ATHLETLY_EMBEDDINGS_DISABLED") == "1"
            or not os.environ.get("GEMINI_API_KEY")
        ):
            return [(b, 1.0) for b in active]

        # Embed candidate
        try:
            client = get_client()
            response = client.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=candidate_text,
            )
            candidate_embedding = np.array(response.embeddings[0].values)
        except Exception:
            # If embedding fails, fall back to returning all
            return [(b, 1.0) for b in active]

        # Compute cosine similarity against beliefs with embeddings
        scored = []
        for b in active:
            if b.get("embedding"):
                b_emb = np.array(b["embedding"])
                # Cosine similarity
                dot = np.dot(candidate_embedding, b_emb)
                norm = np.linalg.norm(candidate_embedding) * np.linalg.norm(b_emb)
                sim = float(dot / norm) if norm > 0 else 0.0
                scored.append((b, sim))
            else:
                scored.append((b, 0.0))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    # ── Forget Phase ─────────────────────────────────────────────

    def prune_stale_beliefs(
        self,
        max_age_days: int = 30,
        min_confidence: float = 0.5,
    ) -> list[dict]:
        """Archive beliefs that are stale and low-confidence (the Forget phase).

        Archives beliefs that meet ALL of:
        - confidence < min_confidence
        - last_confirmed > max_age_days ago

        Also archives session-only beliefs that weren't promoted to global.

        Returns list of archived beliefs.
        """
        now = datetime.now()
        cutoff = now - timedelta(days=max_age_days)
        archived = []

        for belief in self.beliefs:
            if not belief["active"]:
                continue

            should_archive = False

            # Session beliefs that survived past their session
            if belief["durability"] == "session":
                should_archive = True

            # Low-confidence + stale
            if belief["confidence"] < min_confidence:
                try:
                    last_confirmed = datetime.fromisoformat(belief["last_confirmed"])
                    if last_confirmed < cutoff:
                        should_archive = True
                except (ValueError, TypeError):
                    pass

            if should_archive:
                self.invalidate_belief(belief["id"])
                archived.append(belief)

        # Persist archived beliefs
        if archived:
            self._append_to_archive(archived)

        return archived

    def _append_to_archive(self, beliefs: list[dict]) -> None:
        """Append superseded beliefs to beliefs_archive.json."""
        existing = []
        if self._archive_path.exists():
            try:
                existing = json.loads(self._archive_path.read_text())
            except (json.JSONDecodeError, ValueError):
                existing = []

        # Strip embeddings from archived beliefs to save space
        for b in beliefs:
            archived_copy = {k: v for k, v in b.items() if k != "embedding"}
            existing.append(archived_copy)

        self._archive_path.parent.mkdir(parents=True, exist_ok=True)
        self._archive_path.write_text(json.dumps(existing, indent=2))

    # ── Persistence ──────────────────────────────────────────────

    def save(self) -> Path:
        """Save user model to disk. Returns the path."""
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self.meta["updated_at"] = _now_iso()

        data = {
            "structured_core": self.structured_core,
            "beliefs": self.beliefs,
            "meta": self.meta,
        }
        self._model_path.write_text(json.dumps(data, indent=2))
        return self._model_path

    def load(self) -> "UserModel":
        """Load user model from disk. Returns self for chaining.

        Backfills missing outcome fields on old beliefs for backwards compatibility.
        """
        if not self._model_path.exists():
            raise FileNotFoundError(f"No user model found at {self._model_path}")

        data = json.loads(self._model_path.read_text())
        self.structured_core = data.get("structured_core", self.structured_core)
        self.beliefs = data.get("beliefs", [])
        self.meta = data.get("meta", self.meta)

        # Backfill outcome fields for beliefs created before P6
        for belief in self.beliefs:
            belief.setdefault("utility", 0.0)
            belief.setdefault("outcome_count", 0)
            belief.setdefault("last_outcome", None)
            belief.setdefault("outcome_history", [])

        return self

    @classmethod
    def load_or_create(cls, data_dir: Path | None = None) -> "UserModel":
        """Load existing model or create a new one."""
        model = cls(data_dir=data_dir)
        try:
            model.load()
        except FileNotFoundError:
            pass
        return model
