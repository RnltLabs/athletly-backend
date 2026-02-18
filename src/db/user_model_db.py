"""Supabase-backed UserModel that replaces JSON file persistence with PostgreSQL + pgvector.

Drop-in replacement for :class:`src.memory.user_model.UserModel`.  All mutations
are persisted immediately to the ``profiles`` and ``beliefs`` tables in Supabase,
so there is no risk of data loss from a crashed process.

Embedding-based similarity search delegates to the ``match_beliefs()`` PostgreSQL
function which uses pgvector cosine distance -- far more scalable than the
numpy-based search in the file-backed implementation.

Usage::

    from src.db.user_model_db import UserModelDB

    model = UserModelDB(user_id="uuid-here")
    model.load()
    model.add_belief("Runs easy too fast", "pattern", confidence=0.7)
    # Belief is already persisted -- no save() call required.
    # Call save() only when you update structured_core or meta.
"""

import logging
import uuid
from datetime import datetime, timedelta

from src.db.client import get_supabase

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "models/text-embedding-004"

# Valid category values for beliefs (must match the CHECK constraint in the DB).
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
    """ISO-8601 timestamp at second precision."""
    return datetime.now().isoformat(timespec="seconds")


def _today_iso() -> str:
    """ISO-8601 date string for today."""
    return datetime.now().date().isoformat()


class UserModelDB:
    """Supabase-backed user model.  Same public interface as ``UserModel``
    but persists to PostgreSQL (profiles + beliefs tables with pgvector).

    Beliefs are written/updated immediately on mutation.  The ``save()``
    method only needs to be called to persist changes to
    ``structured_core`` or ``meta``.
    """

    # ------------------------------------------------------------------ init

    def __init__(self, user_id: str):
        self.user_id = user_id
        self._db = get_supabase()

        # Same in-memory structure as UserModel for full compatibility.
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

    # ── Loading / Persistence ────────────────────────────────────

    def load(self) -> "UserModelDB":
        """Load profile + active beliefs from Supabase.  Returns *self* for chaining.

        If no profile row exists yet the in-memory defaults are kept (equivalent
        to the file-based ``load_or_create`` behaviour).
        """
        self._load_profile()
        self._load_beliefs()
        return self

    def _load_profile(self) -> None:
        """Fetch the profile row and populate structured_core + meta."""
        try:
            result = (
                self._db.table("profiles")
                .select("*")
                .eq("user_id", self.user_id)
                .maybe_single()
                .execute()
            )
        except Exception:
            logger.exception("Failed to load profile for user %s", self.user_id)
            return

        # maybe_single() returns an APIResponse whose .data is None when no row
        # matches, or the response object itself can be None in some client
        # versions.
        if result is None or not result.data:
            return

        row = result.data
        self._from_profile_row(row)

    def _load_beliefs(self) -> None:
        """Fetch all active beliefs and populate self.beliefs."""
        try:
            result = (
                self._db.table("beliefs")
                .select("*")
                .eq("user_id", self.user_id)
                .eq("active", True)
                .execute()
            )
        except Exception:
            logger.exception("Failed to load beliefs for user %s", self.user_id)
            return

        self.beliefs = [self._from_belief_row(r) for r in (result.data or [])]

    @classmethod
    def load_or_create(cls, user_id: str) -> "UserModelDB":
        """Load an existing model from Supabase, or return a fresh instance."""
        model = cls(user_id=user_id)
        model.load()
        return model

    def save(self) -> None:
        """Upsert the profile row in Supabase (structured_core + meta).

        Beliefs are persisted individually on mutation, so this only handles
        the profile / meta portion.
        """
        profile_data = self._to_profile_row()
        try:
            self._db.table("profiles").upsert(
                profile_data, on_conflict="user_id"
            ).execute()
        except Exception:
            logger.exception("Failed to save profile for user %s", self.user_id)

    # ── Row <-> dict mapping ─────────────────────────────────────

    def _from_profile_row(self, row: dict) -> None:
        """Populate structured_core and meta from a profiles table row."""
        self.structured_core["name"] = row.get("name")
        self.structured_core["sports"] = row.get("sports") or []

        self.structured_core["goal"] = {
            "event": row.get("goal_event"),
            "target_date": row.get("goal_target_date"),
            "target_time": row.get("goal_target_time"),
            "goal_type": row.get("goal_type"),
        }
        self.structured_core["fitness"] = {
            "estimated_vo2max": row.get("estimated_vo2max"),
            "threshold_pace_min_km": row.get("threshold_pace_min_km"),
            "weekly_volume_km": row.get("weekly_volume_km"),
            "trend": row.get("fitness_trend") or "unknown",
        }
        self.structured_core["constraints"] = {
            "training_days_per_week": row.get("training_days_per_week"),
            "max_session_minutes": row.get("max_session_minutes"),
            "available_sports": row.get("available_sports") or [],
        }

        # Meta from the meta JSONB column + timestamps
        db_meta = row.get("meta") or {}
        self.meta = {
            "created_at": row.get("created_at") or _now_iso(),
            "updated_at": row.get("updated_at") or _now_iso(),
            "sessions_completed": db_meta.get("sessions_completed", 0),
            "last_interaction": db_meta.get("last_interaction"),
        }

    def _to_profile_row(self) -> dict:
        """Build a profiles-table row dict from in-memory state."""
        core = self.structured_core
        goal = core.get("goal", {})
        fitness = core.get("fitness", {})
        constraints = core.get("constraints", {})

        now = _now_iso()
        self.meta["updated_at"] = now

        return {
            "user_id": self.user_id,
            "name": core.get("name"),
            "sports": core.get("sports") or [],
            "goal_event": goal.get("event"),
            "goal_target_date": goal.get("target_date"),
            "goal_target_time": goal.get("target_time"),
            "goal_type": goal.get("goal_type"),
            "estimated_vo2max": fitness.get("estimated_vo2max"),
            "threshold_pace_min_km": fitness.get("threshold_pace_min_km"),
            "weekly_volume_km": fitness.get("weekly_volume_km"),
            "fitness_trend": fitness.get("trend"),
            "training_days_per_week": constraints.get("training_days_per_week"),
            "max_session_minutes": constraints.get("max_session_minutes"),
            "available_sports": constraints.get("available_sports") or [],
            "onboarding_complete": True,
            "meta": {
                "sessions_completed": self.meta.get("sessions_completed", 0),
                "last_interaction": self.meta.get("last_interaction"),
            },
            "updated_at": now,
        }

    @staticmethod
    def _from_belief_row(row: dict) -> dict:
        """Convert a beliefs-table row into the in-memory belief dict format
        used by the original ``UserModel``.
        """
        return {
            "id": row.get("id"),
            "text": row.get("text"),
            "category": row.get("category"),
            "confidence": row.get("confidence", 0.7),
            "stability": row.get("stability", "stable"),
            "durability": row.get("durability", "global"),
            "source": row.get("source", "conversation"),
            "source_ref": row.get("source_ref"),
            "first_observed": row.get("first_observed"),
            "last_confirmed": row.get("last_confirmed"),
            "valid_from": row.get("valid_from"),
            "valid_until": row.get("valid_until"),
            "learned_at": row.get("first_observed"),  # alias kept for compat
            "archived_at": row.get("archived_at"),
            "active": row.get("active", True),
            "superseded_by": row.get("superseded_by"),
            # Embeddings are NOT loaded into memory (handled by pgvector).
            "embedding": None,
            # Outcome tracking
            "utility": row.get("utility", 0.0),
            "outcome_count": row.get("outcome_count", 0),
            "last_outcome": row.get("last_outcome"),
            "outcome_history": row.get("outcome_history") or [],
        }

    @staticmethod
    def _to_belief_insert(
        user_id: str,
        text: str,
        category: str,
        confidence: float,
        stability: str,
        durability: str,
        source: str,
        source_ref: str | None,
        valid_from: str | None,
        valid_until: str | None,
        embedding: list[float] | None,
    ) -> dict:
        """Build a beliefs-table INSERT dict."""
        now = _now_iso()
        today = _today_iso()
        return {
            "user_id": user_id,
            "text": text,
            "category": category if category in BELIEF_CATEGORIES else "preference",
            "confidence": max(0.0, min(1.0, confidence)),
            "stability": stability,
            "durability": durability,
            "source": source,
            "source_ref": source_ref,
            "valid_from": valid_from or today,
            "valid_until": valid_until,
            "embedding": embedding,
            "first_observed": now,
            "last_confirmed": now,
            "active": True,
            "utility": 0.0,
            "outcome_count": 0,
            "last_outcome": None,
            "outcome_history": [],
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
        """Create a new belief and persist it to Supabase immediately.

        If *embedding* is ``None`` an embedding is generated via Gemini
        automatically (failure is non-fatal -- the belief is stored without one).

        Returns the belief dict (same shape as ``UserModel.add_belief``).
        """
        # Generate embedding if not provided
        if embedding is None:
            embedding = self._generate_embedding(text)

        row = self._to_belief_insert(
            user_id=self.user_id,
            text=text,
            category=category,
            confidence=confidence,
            stability=stability,
            durability=durability,
            source=source,
            source_ref=source_ref,
            valid_from=valid_from,
            valid_until=valid_until,
            embedding=embedding,
        )

        try:
            result = self._db.table("beliefs").insert(row).execute()
            belief = self._from_belief_row(result.data[0])
        except Exception:
            logger.exception("Failed to insert belief for user %s", self.user_id)
            # Fall back to an in-memory-only belief so the caller still gets a dict.
            belief = {
                "id": str(uuid.uuid4()),
                "text": text,
                "category": category if category in BELIEF_CATEGORIES else "preference",
                "confidence": max(0.0, min(1.0, confidence)),
                "stability": stability,
                "durability": durability,
                "source": source,
                "source_ref": source_ref,
                "first_observed": _now_iso(),
                "last_confirmed": _now_iso(),
                "valid_from": valid_from or _today_iso(),
                "valid_until": valid_until,
                "learned_at": _now_iso(),
                "archived_at": None,
                "active": True,
                "superseded_by": None,
                "embedding": None,
                "utility": 0.0,
                "outcome_count": 0,
                "last_outcome": None,
                "outcome_history": [],
            }

        self.beliefs.append(belief)
        self.meta["updated_at"] = _now_iso()
        return belief

    def update_belief(
        self,
        belief_id: str,
        new_text: str | None = None,
        new_confidence: float | None = None,
    ) -> dict | None:
        """Update an existing active belief.  Persists changes to Supabase.

        If the text changes the embedding is regenerated.
        Returns the updated belief dict, or ``None`` if not found.
        """
        for belief in self.beliefs:
            if belief["id"] == belief_id and belief["active"]:
                now = _now_iso()
                updates: dict = {"last_confirmed": now}

                if new_text is not None:
                    belief["text"] = new_text
                    updates["text"] = new_text
                    # Regenerate embedding for updated text.
                    new_embedding = self._generate_embedding(new_text)
                    updates["embedding"] = new_embedding
                    belief["embedding"] = None  # not stored in-memory

                if new_confidence is not None:
                    clamped = max(0.0, min(1.0, new_confidence))
                    belief["confidence"] = clamped
                    updates["confidence"] = clamped

                belief["last_confirmed"] = now
                self.meta["updated_at"] = now

                try:
                    self._db.table("beliefs").update(updates).eq(
                        "id", belief_id
                    ).execute()
                except Exception:
                    logger.exception(
                        "Failed to update belief %s in Supabase", belief_id
                    )

                return belief
        return None

    def invalidate_belief(
        self,
        belief_id: str,
        superseded_by: str | None = None,
    ) -> dict | None:
        """Mark a belief as inactive (soft delete).  Persists to Supabase.

        Returns the archived belief dict, or ``None`` if not found.
        """
        for belief in self.beliefs:
            if belief["id"] == belief_id and belief["active"]:
                now = _now_iso()
                today = _today_iso()

                belief["active"] = False
                belief["archived_at"] = now
                belief["valid_until"] = today
                if superseded_by:
                    belief["superseded_by"] = superseded_by
                self.meta["updated_at"] = now

                updates = {
                    "active": False,
                    "archived_at": now,
                    "valid_until": today,
                }
                if superseded_by:
                    updates["superseded_by"] = superseded_by

                try:
                    self._db.table("beliefs").update(updates).eq(
                        "id", belief_id
                    ).execute()
                except Exception:
                    logger.exception(
                        "Failed to invalidate belief %s in Supabase", belief_id
                    )

                return belief
        return None

    def get_active_beliefs(
        self,
        category: str | None = None,
        min_confidence: float = 0.0,
    ) -> list[dict]:
        """Retrieve active beliefs from the in-memory cache, optionally filtered."""
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
        """Record a confirmed/contradicted outcome for a belief.

        Updates confidence and utility both in-memory and in Supabase.
        Returns the updated belief, or ``None`` if not found.
        """
        for belief in self.beliefs:
            if belief["id"] == belief_id and belief["active"]:
                now = _now_iso()
                today = _today_iso()

                # Ensure outcome fields exist (backwards compatibility).
                belief.setdefault("utility", 0.0)
                belief.setdefault("outcome_count", 0)
                belief.setdefault("last_outcome", None)
                belief.setdefault("outcome_history", [])

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
                    "date": today,
                    "type": outcome,
                    "detail": detail,
                })

                self.meta["updated_at"] = now

                # Persist to Supabase.
                updates = {
                    "confidence": belief["confidence"],
                    "utility": belief["utility"],
                    "outcome_count": belief["outcome_count"],
                    "last_outcome": belief["last_outcome"],
                    "last_confirmed": now,
                    "outcome_history": belief["outcome_history"],
                }
                try:
                    self._db.table("beliefs").update(updates).eq(
                        "id", belief_id
                    ).execute()
                except Exception:
                    logger.exception(
                        "Failed to persist outcome for belief %s", belief_id
                    )

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

    # ── Embedding & Similarity Search ────────────────────────────

    def _generate_embedding(self, text: str) -> list[float] | None:
        """Generate a 768-dimension embedding via Gemini Embedding API.

        Returns a plain list of floats suitable for pgvector, or ``None``
        on failure (non-fatal -- the belief is still stored without an
        embedding).
        """
        try:
            from src.agent.llm import get_client

            client = get_client()
            response = client.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=text,
            )
            embedding = list(response.embeddings[0].values)
            return embedding
        except Exception:
            logger.warning("Embedding generation failed for text: %.60s...", text)
            return None

    def embed_belief(self, belief: dict) -> list[float] | None:
        """Generate an embedding for *belief* and persist it to Supabase.

        This is the DB-backed equivalent of ``UserModel.embed_belief``.
        The embedding is NOT stored in-memory (pgvector handles search).
        Returns the embedding vector or ``None``.
        """
        embedding = self._generate_embedding(belief["text"])
        if embedding is None:
            return None

        try:
            self._db.table("beliefs").update({"embedding": embedding}).eq(
                "id", belief["id"]
            ).execute()
        except Exception:
            logger.exception(
                "Failed to persist embedding for belief %s", belief["id"]
            )

        return embedding

    def find_similar_beliefs(
        self,
        candidate_text: str,
        top_k: int = 3,
    ) -> list[tuple[dict, float]]:
        """Find the *top_k* most similar active beliefs using pgvector cosine search.

        Delegates to the ``match_beliefs()`` PostgreSQL function.  Falls
        back to returning all active beliefs (score 1.0) when fewer than
        10 beliefs exist or the embedding call fails -- matching the Mem0
        fallback pattern in the original implementation.

        Returns a list of ``(belief_dict, similarity_score)`` tuples sorted
        by descending similarity.
        """
        active = self.get_active_beliefs()

        # Fallback: if few beliefs, just return all (no embedding needed).
        if len(active) < 10:
            return [(b, 1.0) for b in active]

        embedding = self._generate_embedding(candidate_text)
        if embedding is None:
            return [(b, 1.0) for b in active]

        try:
            result = self._db.rpc(
                "match_beliefs",
                {
                    "p_user_id": self.user_id,
                    "p_embedding": embedding,
                    "p_match_count": top_k,
                    "p_min_confidence": 0.0,
                },
            ).execute()

            matches: list[tuple[dict, float]] = []
            for row in result.data or []:
                # match_beliefs returns (id, text, category, confidence, similarity).
                # Build a minimal belief-like dict for compatibility.
                belief_dict = {
                    "id": row["id"],
                    "text": row["text"],
                    "category": row["category"],
                    "confidence": row["confidence"],
                }
                # Try to enrich from in-memory cache.
                for b in self.beliefs:
                    if b["id"] == row["id"]:
                        belief_dict = b
                        break
                matches.append((belief_dict, row["similarity"]))

            return matches

        except Exception:
            logger.exception("match_beliefs RPC failed, falling back to all beliefs")
            return [(b, 1.0) for b in active]

    # ── Forget Phase ─────────────────────────────────────────────

    def prune_stale_beliefs(
        self,
        max_age_days: int = 30,
        min_confidence: float = 0.5,
    ) -> list[dict]:
        """Archive stale, low-confidence beliefs (the Forget phase).

        Archives beliefs that meet ALL of:
        - confidence < *min_confidence*
        - last_confirmed > *max_age_days* ago

        Also archives session-only beliefs.

        Returns the list of archived beliefs.
        """
        now = datetime.now()
        cutoff = now - timedelta(days=max_age_days)
        archived: list[dict] = []

        for belief in list(self.beliefs):  # iterate over a copy
            if not belief["active"]:
                continue

            should_archive = False

            # Session beliefs that survived past their session.
            if belief.get("durability") == "session":
                should_archive = True

            # Low-confidence + stale.
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

        return archived

    # ── Structured Core Updates ──────────────────────────────────

    def update_structured_core(self, field_path: str, value) -> None:
        """Update a nested field in ``structured_core`` using dot-notation.

        Example::

            model.update_structured_core("goal.target_date", "2026-10-15")

        The profile is persisted to Supabase automatically.
        """
        parts = field_path.split(".")
        target = self.structured_core
        for part in parts[:-1]:
            if part not in target or not isinstance(target[part], dict):
                target[part] = {}
            target = target[part]
        target[parts[-1]] = value
        self.meta["updated_at"] = _now_iso()

        # Auto-persist to DB so callers do not need to remember to call save().
        self.save()

    # ── User Model Summary (for prompt injection) ────────────────

    def get_model_summary(self) -> str:
        """Return a concise text summary of the user model for LLM prompt injection.

        Implements the PrefEval reminder injection pattern: active beliefs are
        formatted as COACH'S NOTES for every LLM call.
        """
        lines: list[str] = []

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

        fitness = core.get("fitness", {})
        fit_parts: list[str] = []
        if fitness.get("estimated_vo2max"):
            fit_parts.append(f"VO2max ~{fitness['estimated_vo2max']}")
        if fitness.get("threshold_pace_min_km"):
            fit_parts.append(f"threshold {fitness['threshold_pace_min_km']} min/km")
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
                by_cat[cat].append(
                    f"- {b['text']} (confidence: {b['confidence']:.1f})"
                )

            for cat in sorted(by_cat):
                lines.append(f"  [{cat.upper()}]")
                for line in by_cat[cat]:
                    lines.append(f"  {line}")

        return "\n".join(lines)

    # ── Profile Projection (backward compatibility) ──────────────

    def project_profile(self) -> dict:
        """Generate a ``profile.json``-compatible dict from ``structured_core``.

        Ensures backward compatibility with ``generate_plan()``,
        ``assess_training()``, and other Step 1-5 functions.
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
                "threshold_pace_min_km": core.get("fitness", {}).get(
                    "threshold_pace_min_km"
                ),
                "weekly_volume_km": core.get("fitness", {}).get("weekly_volume_km"),
                "trend": core.get("fitness", {}).get("trend", "unknown"),
            },
            "constraints": {
                "training_days_per_week": core.get("constraints", {}).get(
                    "training_days_per_week"
                )
                or 5,
                "max_session_minutes": core.get("constraints", {}).get(
                    "max_session_minutes"
                )
                or 90,
                "available_sports": core.get("constraints", {}).get(
                    "available_sports"
                )
                or core.get("sports")
                or [],
            },
            "created_at": self.meta.get("created_at", now),
            "updated_at": self.meta.get("updated_at", now),
        }
