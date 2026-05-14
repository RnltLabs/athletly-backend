"""Tests for the Reflexion loop (Feature 3).

Covers:
- _validate_lesson: schema validation, evidence check, PII guard.
- _safe_parse_json: tolerant JSON parsing with fence stripping.
- _evidence_in_user_blob: substring + joined-quote handling.
- _generate_and_validate_lessons: end-to-end LLM call mocked, with
  topic coercion and hallucination drop.
- run_pro_reflexion: idempotency, short-session skip, happy path.
- maybe_run_reflexion: tier routing and daily cap.
- lesson_retrieval.fetch_relevant_lessons: ranking + decay + fallback.
- lesson_retrieval.format_lessons_block: bullet rendering, cap.
- lesson_retrieval._effective_confidence: decay math.
"""

from __future__ import annotations

import asyncio
import math
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _llm_response(content: str) -> MagicMock:
    rsp = MagicMock()
    rsp.choices = [MagicMock()]
    rsp.choices[0].message.content = content
    return rsp


USER_ID = "user-reflexion-001"
SESSION_ID = "session-reflexion-001"


# ---------------------------------------------------------------------------
# _validate_lesson
# ---------------------------------------------------------------------------


class TestValidateLesson:
    def test_happy_path(self) -> None:
        from src.services.reflexion import _validate_lesson, _normalise

        blob = _normalise("Thursday tempo is never going to happen with my work schedule.")
        out = _validate_lesson(
            {
                "topic": "scheduling",
                "observation": "Athlete said Thursday tempo never works.",
                "lesson": "Move tempo runs to Wednesday by default.",
                "applicable_to": ["running"],
                "evidence": "Thursday tempo is never going to happen with my work schedule.",
                "confidence": 0.8,
            },
            user_blob=blob,
        )
        assert out is not None
        assert out["topic"] == "scheduling"
        assert out["confidence"] == 0.8

    def test_drops_when_evidence_missing(self) -> None:
        from src.services.reflexion import _validate_lesson, _normalise

        out = _validate_lesson(
            {
                "topic": "scheduling",
                "observation": "Made up something.",
                "lesson": "Move tempo runs to Wednesday by default.",
                "applicable_to": ["running"],
                "evidence": "user never actually said this hallucinated sentence",
                "confidence": 0.8,
            },
            user_blob=_normalise("the user actually said totally unrelated things"),
        )
        assert out is None

    def test_coerces_invalid_topic_to_other(self) -> None:
        from src.services.reflexion import _validate_lesson, _normalise

        blob = _normalise("I really like running on Sundays after long rides.")
        out = _validate_lesson(
            {
                "topic": "fluffy",
                "observation": "Athlete prefers Sunday runs.",
                "lesson": "Default Sunday to easy running.",
                "applicable_to": ["running"],
                "evidence": "I really like running on Sundays after long rides.",
                "confidence": 0.7,
            },
            user_blob=blob,
        )
        assert out is not None
        assert out["topic"] == "other"

    def test_drops_low_confidence(self) -> None:
        from src.services.reflexion import _validate_lesson, _normalise

        blob = _normalise("Maybe I will try tempo runs Wednesday next week, not sure.")
        out = _validate_lesson(
            {
                "topic": "scheduling",
                "observation": "obs",
                "lesson": "lesson",
                "applicable_to": [],
                "evidence": "Maybe I will try tempo runs Wednesday next week, not sure.",
                "confidence": 0.1,
            },
            user_blob=blob,
        )
        assert out is None

    def test_drops_too_long_lesson(self) -> None:
        from src.services.reflexion import _validate_lesson, _normalise

        blob = _normalise("I always run on Sundays after long rides.")
        out = _validate_lesson(
            {
                "topic": "preference",
                "observation": "obs",
                "lesson": "x" * 600,
                "applicable_to": [],
                "evidence": "I always run on Sundays after long rides.",
                "confidence": 0.9,
            },
            user_blob=blob,
        )
        assert out is None

    def test_drops_pii_email(self) -> None:
        from src.services.reflexion import _validate_lesson, _normalise

        blob = _normalise("Contact me at foo@example.com about training.")
        out = _validate_lesson(
            {
                "topic": "other",
                "observation": "obs",
                "lesson": "Reach the athlete at foo@example.com for updates.",
                "applicable_to": [],
                "evidence": "Contact me at foo@example.com about training.",
                "confidence": 0.9,
            },
            user_blob=blob,
        )
        assert out is None

    def test_handles_non_dict_input(self) -> None:
        from src.services.reflexion import _validate_lesson

        assert _validate_lesson("nope", user_blob="anything") is None
        assert _validate_lesson(None, user_blob="anything") is None
        assert _validate_lesson(123, user_blob="anything") is None


# ---------------------------------------------------------------------------
# _safe_parse_json
# ---------------------------------------------------------------------------


class TestSafeParseJson:
    def test_plain_object(self) -> None:
        from src.services.reflexion import _safe_parse_json

        out = _safe_parse_json('{"lessons": []}')
        assert out == {"lessons": []}

    def test_strips_markdown_fence(self) -> None:
        from src.services.reflexion import _safe_parse_json

        raw = '```json\n{"lessons": [{"topic": "other"}]}\n```'
        out = _safe_parse_json(raw)
        assert out is not None
        assert out["lessons"][0]["topic"] == "other"

    def test_returns_none_on_garbage(self) -> None:
        from src.services.reflexion import _safe_parse_json

        assert _safe_parse_json("") is None
        assert _safe_parse_json("not json at all") is None

    def test_returns_none_on_list_root(self) -> None:
        from src.services.reflexion import _safe_parse_json

        # We only accept top-level objects.
        assert _safe_parse_json('["not", "an", "object"]') is None


# ---------------------------------------------------------------------------
# _evidence_in_user_blob
# ---------------------------------------------------------------------------


class TestEvidenceInUserBlob:
    def test_substring(self) -> None:
        from src.services.reflexion import _evidence_in_user_blob, _normalise

        blob = _normalise("Mondays do not work for me at all.")
        assert _evidence_in_user_blob(
            "Mondays do not work for me at all.", blob,
        )

    def test_too_short_rejected(self) -> None:
        from src.services.reflexion import _evidence_in_user_blob, _normalise

        blob = _normalise("Mondays do not work for me.")
        assert not _evidence_in_user_blob("ok", blob)

    def test_joined_quote_with_ellipsis(self) -> None:
        from src.services.reflexion import _evidence_in_user_blob, _normalise

        blob = _normalise(
            "I cannot run on Mondays. I also cannot do Tuesday intervals."
        )
        ok = _evidence_in_user_blob(
            "i cannot run on mondays ... cannot do tuesday intervals", blob,
        )
        assert ok

    def test_missing_phrase(self) -> None:
        from src.services.reflexion import _evidence_in_user_blob, _normalise

        blob = _normalise("Saturdays I do long rides.")
        assert not _evidence_in_user_blob(
            "I really love midnight track sessions on Wednesdays.", blob,
        )


# ---------------------------------------------------------------------------
# _generate_and_validate_lessons (end-to-end with mocked LLM)
# ---------------------------------------------------------------------------


class TestGenerateAndValidate:
    def _run(self, transcript_messages, user_messages):
        from src.services.reflexion import _generate_and_validate_lessons

        return asyncio.run(
            _generate_and_validate_lessons(
                transcript_messages=transcript_messages,
                user_messages=user_messages,
            )
        )

    def test_full_flow_success(self) -> None:
        transcript = [
            {"role": "user", "content": "Thursday tempo is never going to happen with my work schedule."},
            {"role": "model", "content": "Got it. We can move tempo to Wednesday."},
        ]
        user_msgs = [transcript[0]]

        llm_json = (
            '{"lessons": [{"topic": "scheduling", '
            '"observation": "Athlete refused Thursday tempo.", '
            '"lesson": "Move tempo runs to Wednesday by default.", '
            '"applicable_to": ["running"], '
            '"evidence": "Thursday tempo is never going to happen with my work schedule.", '
            '"confidence": 0.85}]}'
        )

        with patch(
            "src.agent.llm.chat_completion",
            return_value=_llm_response(llm_json),
        ):
            out = self._run(transcript, user_msgs)
        assert len(out) == 1
        assert out[0]["topic"] == "scheduling"

    def test_drops_hallucinated_lesson(self) -> None:
        transcript = [
            {"role": "user", "content": "Easy run today, all good."},
            {"role": "model", "content": "Great."},
        ]
        llm_json = (
            '{"lessons": [{"topic": "injury", '
            '"observation": "Athlete mentioned knee issues.", '
            '"lesson": "Reduce mileage due to knee pain.", '
            '"applicable_to": ["running"], '
            '"evidence": "my knees hurt every time I run more than 5k", '
            '"confidence": 0.9}]}'
        )
        with patch(
            "src.agent.llm.chat_completion",
            return_value=_llm_response(llm_json),
        ):
            out = self._run(transcript, [transcript[0]])
        assert out == []

    def test_caps_to_three_lessons(self) -> None:
        long_user = (
            "I cannot run Mondays. I cannot run Tuesdays. "
            "I cannot run Wednesdays. I cannot run Thursdays. I cannot run Fridays."
        )
        transcript = [
            {"role": "user", "content": long_user},
            {"role": "model", "content": "Noted."},
        ]
        # Five legit lessons, all evidence-backed.
        lessons = [
            {
                "topic": "scheduling",
                "observation": f"Cannot run {day}",
                "lesson": f"Avoid {day} sessions.",
                "applicable_to": ["running"],
                "evidence": f"I cannot run {day}",
                "confidence": 0.8,
            }
            for day in ["Mondays", "Tuesdays", "Wednesdays", "Thursdays", "Fridays"]
        ]
        import json as _json
        with patch(
            "src.agent.llm.chat_completion",
            return_value=_llm_response(_json.dumps({"lessons": lessons})),
        ):
            out = self._run(transcript, [transcript[0]])
        assert len(out) == 3

    def test_empty_lessons_when_llm_fails(self) -> None:
        transcript = [
            {"role": "user", "content": "Hello."},
            {"role": "model", "content": "Hi."},
        ]
        with patch(
            "src.agent.llm.chat_completion",
            side_effect=Exception("LLM down"),
        ):
            out = self._run(transcript, [transcript[0]])
        assert out == []


# ---------------------------------------------------------------------------
# run_pro_reflexion: idempotency + short-session skip
# ---------------------------------------------------------------------------


class TestRunProReflexion:
    def test_skips_when_idempotent(self) -> None:
        from src.services import reflexion

        with patch("src.db.lessons_db.has_reflexion_run", return_value=True):
            out = asyncio.run(reflexion.run_pro_reflexion(USER_ID, SESSION_ID))
        assert out == 0

    def test_skips_when_session_too_short(self) -> None:
        from src.services import reflexion

        with (
            patch("src.db.lessons_db.has_reflexion_run", return_value=False),
            patch(
                "src.db.session_store_db.load_session_messages",
                return_value=[{"role": "user", "content": "Hi"}],
            ),
            patch("src.db.lessons_db.record_reflexion_run") as mock_record,
        ):
            out = asyncio.run(reflexion.run_pro_reflexion(USER_ID, SESSION_ID))
        assert out == 0
        mock_record.assert_not_called()

    def test_happy_path_records_run(self) -> None:
        from src.services import reflexion

        transcript = [
            {"role": "user", "content": "Thursday tempo never works for me."},
            {"role": "model", "content": "Got it."},
            {"role": "user", "content": "Move tempo to Wednesday please."},
            {"role": "model", "content": "Done."},
        ]
        llm_json = (
            '{"lessons": [{"topic": "scheduling", '
            '"observation": "Athlete refused Thursday tempo.", '
            '"lesson": "Move tempo runs to Wednesday by default.", '
            '"applicable_to": ["running"], '
            '"evidence": "Thursday tempo never works for me.", '
            '"confidence": 0.85}]}'
        )

        with (
            patch("src.db.lessons_db.has_reflexion_run", return_value=False),
            patch(
                "src.db.session_store_db.load_session_messages",
                return_value=transcript,
            ),
            patch(
                "src.agent.llm.chat_completion",
                return_value=_llm_response(llm_json),
            ),
            patch(
                "src.services.reflexion._generate_embedding",
                return_value=None,  # Skip the embedding RPC.
            ),
            patch(
                "src.db.lessons_db.insert_lesson",
                return_value={"id": "lesson-1"},
            ),
            patch("src.db.lessons_db.record_reflexion_run") as mock_record,
        ):
            out = asyncio.run(reflexion.run_pro_reflexion(USER_ID, SESSION_ID))
        assert out == 1
        mock_record.assert_called_once()
        kwargs = mock_record.call_args.kwargs
        assert kwargs["run_kind"] == "pro_session"
        assert kwargs["period"] == SESSION_ID
        assert kwargs["lessons_added"] == 1


# ---------------------------------------------------------------------------
# maybe_run_reflexion: tier routing + daily cap
# ---------------------------------------------------------------------------


class TestMaybeRunReflexion:
    def test_daily_cap_short_circuits(self) -> None:
        from src.services import reflexion

        with (
            patch("src.db.lessons_db.count_runs_since", return_value=5),
            patch("src.services.reflexion.run_pro_reflexion") as mock_pro,
            patch("src.services.reflexion.run_free_monthly_reflexion") as mock_free,
        ):
            asyncio.run(reflexion.maybe_run_reflexion(USER_ID, SESSION_ID))
        mock_pro.assert_not_called()
        mock_free.assert_not_called()

    def test_routes_pro(self) -> None:
        from src.services import reflexion

        async def _stub(*_a, **_k) -> int:
            return 1

        with (
            patch("src.db.lessons_db.count_runs_since", return_value=0),
            patch("src.services.reflexion._get_tier", return_value="pro"),
            patch(
                "src.services.reflexion.run_pro_reflexion",
                side_effect=_stub,
            ) as mock_pro,
            patch(
                "src.services.reflexion.run_free_monthly_reflexion",
                side_effect=_stub,
            ) as mock_free,
        ):
            asyncio.run(reflexion.maybe_run_reflexion(USER_ID, SESSION_ID))
        mock_pro.assert_called_once_with(USER_ID, SESSION_ID)
        mock_free.assert_not_called()

    def test_routes_free(self) -> None:
        from src.services import reflexion

        async def _stub(*_a, **_k) -> int:
            return 0

        with (
            patch("src.db.lessons_db.count_runs_since", return_value=0),
            patch("src.services.reflexion._get_tier", return_value="free"),
            patch(
                "src.services.reflexion.run_pro_reflexion",
                side_effect=_stub,
            ) as mock_pro,
            patch(
                "src.services.reflexion.run_free_monthly_reflexion",
                side_effect=_stub,
            ) as mock_free,
        ):
            asyncio.run(reflexion.maybe_run_reflexion(USER_ID, None))
        mock_free.assert_called_once_with(USER_ID)
        mock_pro.assert_not_called()

    def test_pro_without_prev_session_id_is_noop(self) -> None:
        from src.services import reflexion

        with (
            patch("src.db.lessons_db.count_runs_since", return_value=0),
            patch("src.services.reflexion._get_tier", return_value="pro"),
            patch("src.services.reflexion.run_pro_reflexion") as mock_pro,
            patch("src.services.reflexion.run_free_monthly_reflexion") as mock_free,
        ):
            asyncio.run(reflexion.maybe_run_reflexion(USER_ID, None))
        mock_pro.assert_not_called()
        mock_free.assert_not_called()


# ---------------------------------------------------------------------------
# lesson_retrieval
# ---------------------------------------------------------------------------


class TestEffectiveConfidence:
    def test_fresh_lesson_keeps_full_confidence(self) -> None:
        from datetime import datetime, timezone
        from src.services.lesson_retrieval import _effective_confidence

        row = {
            "confidence": 0.8,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        result = _effective_confidence(row)
        # Fresh -> decay multiplier is ~1.0.
        assert result == pytest.approx(0.8, rel=0.05)

    def test_old_lesson_decays(self) -> None:
        from datetime import datetime, timedelta, timezone
        from src.services.lesson_retrieval import _effective_confidence

        old = (datetime.now(timezone.utc) - timedelta(days=180)).isoformat()
        row = {"confidence": 0.8, "created_at": old}
        result = _effective_confidence(row)
        expected = 0.8 * math.exp(-180.0 / 90.0)
        assert result == pytest.approx(expected, rel=0.05)

    def test_no_timestamps_returns_raw(self) -> None:
        from src.services.lesson_retrieval import _effective_confidence

        assert _effective_confidence({"confidence": 0.6}) == 0.6


class TestFormatLessonsBlock:
    def test_empty_returns_empty(self) -> None:
        from src.services.lesson_retrieval import format_lessons_block

        assert format_lessons_block([]) == ""

    def test_renders_bullets(self) -> None:
        from src.services.lesson_retrieval import format_lessons_block

        block = format_lessons_block([
            {"topic": "scheduling", "lesson": "Move tempo to Wednesday."},
            {"topic": "volume", "lesson": "Cap long runs at 90 min."},
        ])
        assert "Lessons learned about this athlete" in block
        assert "[scheduling]" in block
        assert "Move tempo to Wednesday." in block
        assert "[volume]" in block

    def test_caps_block_size(self) -> None:
        from src.services.lesson_retrieval import (
            _MAX_BLOCK_CHARS,
            format_lessons_block,
        )

        many = [
            {"topic": "scheduling", "lesson": "x" * 400}
            for _ in range(20)
        ]
        block = format_lessons_block(many)
        assert len(block) <= _MAX_BLOCK_CHARS + 200


class TestFetchRelevantLessons:
    def test_returns_empty_on_failure(self) -> None:
        from src.services import lesson_retrieval

        with (
            patch(
                "src.services.lesson_retrieval._generate_embedding",
                return_value=None,
            ),
            patch(
                "src.db.lessons_db.list_active_lessons",
                side_effect=Exception("DB down"),
            ),
        ):
            out = lesson_retrieval.fetch_relevant_lessons(USER_ID, "query text")
        assert out == []

    def test_fallback_recency_when_no_embedding(self) -> None:
        from datetime import datetime, timezone
        from src.services import lesson_retrieval

        now_iso = datetime.now(timezone.utc).isoformat()
        rows = [
            {
                "id": "a",
                "topic": "scheduling",
                "lesson": "A",
                "applicable_to": ["running"],
                "confidence": 0.9,
                "created_at": now_iso,
            },
            {
                "id": "b",
                "topic": "volume",
                "lesson": "B",
                "applicable_to": ["running"],
                "confidence": 0.4,
                "created_at": now_iso,
            },
        ]
        with (
            patch(
                "src.services.lesson_retrieval._generate_embedding",
                return_value=None,
            ),
            patch(
                "src.db.lessons_db.list_active_lessons",
                return_value=rows,
            ),
        ):
            out = lesson_retrieval.fetch_relevant_lessons(USER_ID, "q", top_k=5)
        assert [l["id"] for l in out] == ["a", "b"]

    def test_embedding_path_filters_and_ranks(self) -> None:
        from datetime import datetime, timezone
        from src.services import lesson_retrieval

        now_iso = datetime.now(timezone.utc).isoformat()
        rpc_rows = [
            # High similarity + decent confidence -> kept and ranked first.
            {
                "id": "high",
                "topic": "scheduling",
                "lesson": "High match",
                "applicable_to": ["running"],
                "confidence": 0.8,
                "similarity": 0.95,
                "created_at": now_iso,
            },
            # Low similarity -> dropped.
            {
                "id": "low_sim",
                "topic": "other",
                "lesson": "Low sim",
                "applicable_to": [],
                "confidence": 0.8,
                "similarity": 0.1,
                "created_at": now_iso,
            },
            # Mid similarity, mid confidence -> kept after high.
            {
                "id": "mid",
                "topic": "volume",
                "lesson": "Mid",
                "applicable_to": ["running"],
                "confidence": 0.6,
                "similarity": 0.7,
                "created_at": now_iso,
            },
        ]
        with (
            patch(
                "src.services.lesson_retrieval._generate_embedding",
                return_value=[0.1] * 768,
            ),
            patch(
                "src.db.lessons_db.match_lessons_by_embedding",
                return_value=rpc_rows,
            ),
        ):
            out = lesson_retrieval.fetch_relevant_lessons(USER_ID, "q", top_k=5)
        ids = [l["id"] for l in out]
        assert ids[0] == "high"
        assert "low_sim" not in ids
        assert "mid" in ids


class TestBuildQueryText:
    def test_includes_all_fields(self) -> None:
        from src.services.lesson_retrieval import build_query_text_from_runtime

        text = build_query_text_from_runtime(
            athlete_name="Alex",
            sports=["running", "cycling"],
            goal_event="Marathon",
            recent_summary="3 easy runs this week, one long ride.",
        )
        assert "Alex" in text
        assert "running" in text
        assert "Marathon" in text
        assert "long ride" in text

    def test_handles_all_none(self) -> None:
        from src.services.lesson_retrieval import build_query_text_from_runtime

        assert build_query_text_from_runtime(
            athlete_name=None,
            sports=None,
            goal_event=None,
            recent_summary=None,
        ) == ""


# ---------------------------------------------------------------------------
# Constants sanity
# ---------------------------------------------------------------------------


def test_topics_constant_matches_validator() -> None:
    """The SQL CHECK constraint, the db module, and the reflexion module
    must agree on the topic taxonomy.
    """
    from src.db.lessons_db import LESSON_TOPICS
    from src.services.reflexion import _ALLOWED_TOPICS

    assert LESSON_TOPICS == _ALLOWED_TOPICS
