"""Tests for Feature 5: Episode Replay (semantic memory retrieval).

Covers:
- Embedding provider selection (Gemini / OpenAI / disabled)
- Scoring / reranking (similarity + recency + utility blend)
- Date parsing
- Formatting + token budget enforcement
- Trigger gating
- End-to-end ``build_replay_block`` with mocked Supabase + embedding
- Anti-confabulation footer presence
"""

from __future__ import annotations

import math
import os
from datetime import date, datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from src.services import embeddings, episode_retrieval


USER_ID = "00000000-0000-0000-0000-000000000001"


# ---------------------------------------------------------------------------
# Embedding provider selection
# ---------------------------------------------------------------------------


class TestEmbeddingProviderResolution:
    def setup_method(self) -> None:
        embeddings.reset_provider_cache()

    def teardown_method(self) -> None:
        embeddings.reset_provider_cache()

    def test_disabled_via_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ATHLETLY_EMBEDDINGS_DISABLED", "1")
        monkeypatch.setenv("GEMINI_API_KEY", "fake")
        assert embeddings._resolve_provider() is None

    def test_gemini_when_key_present(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ATHLETLY_EMBEDDINGS_DISABLED", raising=False)
        monkeypatch.setenv("GEMINI_API_KEY", "fake-gemini-key")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setenv("EMBEDDING_MODEL", "")
        provider = embeddings._resolve_provider()
        assert provider is not None
        assert provider.name == embeddings.PROVIDER_GEMINI

    def test_openai_when_explicit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ATHLETLY_EMBEDDINGS_DISABLED", raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "fake-openai-key")
        monkeypatch.setenv("EMBEDDING_MODEL", "openai/text-embedding-3-small")
        provider = embeddings._resolve_provider()
        assert provider is not None
        assert provider.name == embeddings.PROVIDER_OPENAI

    def test_openai_fallback_when_no_gemini(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ATHLETLY_EMBEDDINGS_DISABLED", raising=False)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "fake-openai-key")
        monkeypatch.setenv("EMBEDDING_MODEL", "")
        provider = embeddings._resolve_provider()
        assert provider is not None
        assert provider.name == embeddings.PROVIDER_OPENAI

    def test_none_when_no_keys(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ATHLETLY_EMBEDDINGS_DISABLED", raising=False)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setenv("EMBEDDING_MODEL", "")
        assert embeddings._resolve_provider() is None

    def test_unknown_provider_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ATHLETLY_EMBEDDINGS_DISABLED", raising=False)
        monkeypatch.setenv("EMBEDDING_MODEL", "cohere/embed")
        assert embeddings._resolve_provider() is None


class TestEmbedText:
    def setup_method(self) -> None:
        embeddings.reset_provider_cache()

    def teardown_method(self) -> None:
        embeddings.reset_provider_cache()

    def test_empty_returns_none(self) -> None:
        assert embeddings.embed_text("") is None
        assert embeddings.embed_text("   ") is None

    def test_provider_none_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ATHLETLY_EMBEDDINGS_DISABLED", "1")
        assert embeddings.embed_text("hello") is None

    def test_successful_call(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake = MagicMock()
        fake.name = "fake/provider"
        fake.embed.return_value = [0.1] * embeddings.EMBEDDING_DIM
        monkeypatch.delenv("ATHLETLY_EMBEDDINGS_DISABLED", raising=False)
        with patch.object(embeddings, "_resolve_provider", return_value=fake):
            embeddings.reset_provider_cache()
            vec = embeddings.embed_text("hello world")
        assert vec is not None
        assert len(vec) == embeddings.EMBEDDING_DIM
        fake.embed.assert_called_once()

    def test_retry_on_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake = MagicMock()
        fake.name = "fake/provider"
        fake.embed.side_effect = [None, None, [0.2] * embeddings.EMBEDDING_DIM]
        monkeypatch.delenv("ATHLETLY_EMBEDDINGS_DISABLED", raising=False)
        # Skip the sleep delays for speed.
        with patch.object(embeddings, "_resolve_provider", return_value=fake), \
             patch.object(embeddings.time, "sleep", lambda *_: None):
            embeddings.reset_provider_cache()
            vec = embeddings.embed_text("hello")
        assert vec is not None
        assert fake.embed.call_count == 3

    def test_gives_up_after_max_retries(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake = MagicMock()
        fake.name = "fake/provider"
        fake.embed.return_value = None
        monkeypatch.delenv("ATHLETLY_EMBEDDINGS_DISABLED", raising=False)
        with patch.object(embeddings, "_resolve_provider", return_value=fake), \
             patch.object(embeddings.time, "sleep", lambda *_: None):
            embeddings.reset_provider_cache()
            vec = embeddings.embed_text("hello")
        assert vec is None
        assert fake.embed.call_count == embeddings._MAX_RETRIES + 1


# ---------------------------------------------------------------------------
# Date parsing
# ---------------------------------------------------------------------------


class TestParseDate:
    def test_date_object(self) -> None:
        d = date(2026, 5, 14)
        assert episode_retrieval._parse_date(d) == d

    def test_datetime_object(self) -> None:
        dt = datetime(2026, 5, 14, 12, 30)
        assert episode_retrieval._parse_date(dt) == date(2026, 5, 14)

    def test_iso_date_string(self) -> None:
        assert episode_retrieval._parse_date("2026-05-14") == date(2026, 5, 14)

    def test_iso_datetime_string(self) -> None:
        assert episode_retrieval._parse_date("2026-05-14T12:30:00Z") == date(2026, 5, 14)

    def test_none_returns_none(self) -> None:
        assert episode_retrieval._parse_date(None) is None

    def test_garbage_returns_none(self) -> None:
        assert episode_retrieval._parse_date("not-a-date") is None


# ---------------------------------------------------------------------------
# Recency + reranking
# ---------------------------------------------------------------------------


class TestRecencyAndRerank:
    def test_recency_today_equals_one(self) -> None:
        today = date(2026, 5, 14)
        ep = {"period_end": today.isoformat()}
        assert episode_retrieval._recency_score(ep, today) == pytest.approx(1.0)

    def test_recency_decay_at_90_days(self) -> None:
        today = date(2026, 5, 14)
        past = (today - timedelta(days=90)).isoformat()
        ep = {"period_end": past}
        # exp(-1) is ~0.3679
        score = episode_retrieval._recency_score(ep, today)
        assert 0.30 < score < 0.40

    def test_recency_missing_date_is_zero(self) -> None:
        assert episode_retrieval._recency_score({}, date(2026, 5, 14)) == 0.0

    def test_rerank_orders_by_combined_score(self) -> None:
        today = date(2026, 5, 14)
        candidates = [
            {
                "id": "old-high-sim",
                "similarity": 0.9,
                "period_end": (today - timedelta(days=180)).isoformat(),
                "utility": 0.0,
            },
            {
                "id": "recent-mid-sim",
                "similarity": 0.7,
                "period_end": (today - timedelta(days=10)).isoformat(),
                "utility": 0.0,
            },
            {
                "id": "recent-low-sim",
                "similarity": 0.55,
                "period_end": today.isoformat(),
                "utility": 0.5,
            },
        ]
        ranked = episode_retrieval._rerank(candidates, today)
        # All three should have a final_score field.
        assert all("final_score" in c for c in ranked)
        # Old high-similarity (0.9*0.7 + 0.2*~0.135) ~ 0.66 vs
        # recent mid (0.7*0.7 + 0.2*~0.90) ~ 0.67 - both close, both above
        # the lowest one. The lowest-similarity candidate is last.
        assert ranked[-1]["id"] == "recent-low-sim"


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


class TestFormatting:
    def test_compose_bullet_uses_period_end(self) -> None:
        ep = {
            "period_end": "2026-02-20",
            "summary": "After a 5-day rest week, easy run paced 4:25/km.",
            "insights": ["5+ days rest = clear paced bounce"],
        }
        line = episode_retrieval._compose_bullet(ep)
        assert line.startswith("- 2026-02-20:")
        assert "4:25/km" in line
        assert "Insight: 5+ days rest" in line

    def test_compose_bullet_handles_missing_summary(self) -> None:
        ep = {"period_end": "2026-01-01"}
        line = episode_retrieval._compose_bullet(ep)
        assert "2026-01-01" in line

    def test_format_block_contains_header_and_footer(self) -> None:
        eps = [
            {
                "period_end": "2026-03-01",
                "summary": "Solid base week.",
                "insights": ["Volume stable"],
            },
            {
                "period_end": "2026-02-01",
                "summary": "Easy week.",
                "insights": [],
            },
        ]
        block = episode_retrieval._format_block(eps, token_budget=800)
        assert "# Past Insights (2 relevant episodes" in block
        assert "do NOT claim you just observed them" in block
        assert "2026-03-01" in block
        assert "2026-02-01" in block

    def test_format_block_empty_when_no_episodes(self) -> None:
        assert episode_retrieval._format_block([], token_budget=800) == ""

    def test_token_budget_truncates_overflow(self) -> None:
        # Build many episodes with long summaries.
        big_summary = "x " * 500
        eps = [
            {
                "period_end": f"2026-01-{i:02d}",
                "summary": big_summary,
                "insights": [],
            }
            for i in range(1, 6)
        ]
        block = episode_retrieval._format_block(eps, token_budget=200)
        # Block must respect the budget: 200 tokens * 4 chars = 800 chars
        # plus generous header/footer overhead. Allow some slop.
        assert len(block) < 1200
        # Header always present.
        assert "# Past Insights" in block


# ---------------------------------------------------------------------------
# Trigger gating
# ---------------------------------------------------------------------------


class TestShouldRetrieve:
    def test_none(self) -> None:
        assert episode_retrieval.should_retrieve(None) is False

    def test_empty(self) -> None:
        assert episode_retrieval.should_retrieve("") is False
        assert episode_retrieval.should_retrieve("   ") is False

    def test_too_short(self) -> None:
        assert episode_retrieval.should_retrieve("hi") is False
        assert episode_retrieval.should_retrieve("ok") is False

    def test_disabled_via_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ATHLETLY_EPISODE_REPLAY_DISABLED", "1")
        assert episode_retrieval.should_retrieve("how should I taper") is False

    def test_normal_message_triggers(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ATHLETLY_EPISODE_REPLAY_DISABLED", raising=False)
        assert episode_retrieval.should_retrieve("how should I taper before my race?") is True


# ---------------------------------------------------------------------------
# retrieve_relevant_episodes + build_replay_block end-to-end
# ---------------------------------------------------------------------------


def _fake_candidates() -> list[dict]:
    today = date.today()
    return [
        {
            "id": "ep-1",
            "episode_type": "weekly_reflection",
            "period_start": (today - timedelta(days=30)).isoformat(),
            "period_end": (today - timedelta(days=24)).isoformat(),
            "summary": "Tempo run at 4:05/km gave shin pain. Adjusted to 4:15.",
            "insights": ["Shin pain at fast tempo - back off pace"],
            "utility": 0.4,
            "similarity": 0.78,
        },
        {
            "id": "ep-2",
            "episode_type": "weekly_reflection",
            "period_start": (today - timedelta(days=60)).isoformat(),
            "period_end": (today - timedelta(days=54)).isoformat(),
            "summary": "Long run > 18 km in race week blew up the taper.",
            "insights": ["Cap long run at 14 km in race week"],
            "utility": 0.6,
            "similarity": 0.71,
        },
        {
            "id": "ep-3",
            "episode_type": "weekly_reflection",
            "period_start": (today - timedelta(days=10)).isoformat(),
            "period_end": (today - timedelta(days=4)).isoformat(),
            "summary": "Easy run paced 4:25/km after a 5-day rest week.",
            "insights": ["5+ days rest = clear paced bounce"],
            "utility": 0.2,
            "similarity": 0.62,
        },
    ]


class TestRetrieveRelevantEpisodes:
    def test_returns_empty_without_user_id(self) -> None:
        assert episode_retrieval.retrieve_relevant_episodes("", "query") == []

    def test_returns_empty_when_embedding_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "src.services.episode_retrieval.embed_text", lambda _t: None,
            raising=False,
        )
        # Patch the actual reference inside the function:
        with patch("src.services.embeddings.embed_text", return_value=None):
            result = episode_retrieval.retrieve_relevant_episodes(
                USER_ID, "how should I taper"
            )
        assert result == []

    def test_full_path_returns_top_k(self) -> None:
        with patch("src.services.embeddings.embed_text", return_value=[0.1] * 768), \
             patch.object(episode_retrieval, "_fetch_candidates", return_value=_fake_candidates()), \
             patch.object(episode_retrieval, "_touch_last_replayed", lambda _ids: None):
            result = episode_retrieval.retrieve_relevant_episodes(
                USER_ID, "tempo pace shin pain?", top_k=2
            )
        assert len(result) == 2
        assert all("final_score" in r for r in result)

    def test_top_k_capped_at_max(self) -> None:
        with patch("src.services.embeddings.embed_text", return_value=[0.1] * 768), \
             patch.object(episode_retrieval, "_fetch_candidates", return_value=_fake_candidates() * 3), \
             patch.object(episode_retrieval, "_touch_last_replayed", lambda _ids: None):
            result = episode_retrieval.retrieve_relevant_episodes(
                USER_ID, "tempo pace shin pain?", top_k=99
            )
        assert len(result) <= episode_retrieval.MAX_TOP_K


class TestBuildReplayBlock:
    def test_no_user_id_returns_empty(self) -> None:
        assert episode_retrieval.build_replay_block(None, "any") == ""

    def test_no_message_returns_empty(self) -> None:
        assert episode_retrieval.build_replay_block(USER_ID, "") == ""

    def test_short_message_returns_empty(self) -> None:
        assert episode_retrieval.build_replay_block(USER_ID, "ok") == ""

    def test_full_block_when_episodes_match(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ATHLETLY_EPISODE_REPLAY_DISABLED", raising=False)
        with patch("src.services.embeddings.embed_text", return_value=[0.2] * 768), \
             patch.object(episode_retrieval, "_fetch_candidates", return_value=_fake_candidates()), \
             patch.object(episode_retrieval, "_touch_last_replayed", lambda _ids: None):
            block = episode_retrieval.build_replay_block(
                USER_ID, "how should I taper before a marathon?"
            )
        assert "# Past Insights" in block
        # Anti-confabulation footer.
        assert "do NOT claim you just observed them" in block
        # Each bullet has a date prefix.
        assert block.count("- 20") >= 2

    def test_empty_block_when_no_candidates(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ATHLETLY_EPISODE_REPLAY_DISABLED", raising=False)
        with patch("src.services.embeddings.embed_text", return_value=[0.2] * 768), \
             patch.object(episode_retrieval, "_fetch_candidates", return_value=[]):
            block = episode_retrieval.build_replay_block(
                USER_ID, "how should I taper before a marathon?"
            )
        assert block == ""

    def test_disabled_returns_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ATHLETLY_EPISODE_REPLAY_DISABLED", "1")
        assert episode_retrieval.build_replay_block(USER_ID, "how should I taper") == ""


# ---------------------------------------------------------------------------
# Store-side: embedding generation in episodes_db
# ---------------------------------------------------------------------------


class TestStoreEpisodeEmbedding:
    def test_build_embedding_text_combines_summary_and_insights(self) -> None:
        from src.db.episodes_db import _build_embedding_text

        ep = {
            "summary": "Good week",
            "insights": ["pattern a", "pattern b"],
        }
        text = _build_embedding_text(ep)
        assert "Good week" in text
        assert "pattern a" in text
        assert "pattern b" in text

    def test_build_embedding_text_handles_empty(self) -> None:
        from src.db.episodes_db import _build_embedding_text

        assert _build_embedding_text({}) == ""

    def test_store_episode_attaches_embedding_when_provider_available(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from src.db import episodes_db

        captured: dict = {}

        class FakeTable:
            def insert(self, row):
                captured["row"] = row
                self._row = row
                return self

            def execute(self):
                return MagicMock(data=[{**self._row, "id": "fake-id"}])

        fake_client = MagicMock()
        fake_client.table.return_value = FakeTable()

        with patch.object(episodes_db, "get_supabase", return_value=fake_client), \
             patch("src.services.embeddings.embed_text", return_value=[0.5] * 768):
            episodes_db.store_episode(
                USER_ID,
                {"summary": "test", "insights": ["x"]},
            )
        assert "embedding" in captured["row"]
        assert len(captured["row"]["embedding"]) == 768

    def test_store_episode_omits_embedding_when_provider_unavailable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from src.db import episodes_db

        captured: dict = {}

        class FakeTable:
            def insert(self, row):
                captured["row"] = row
                self._row = row
                return self

            def execute(self):
                return MagicMock(data=[{**self._row, "id": "fake-id"}])

        fake_client = MagicMock()
        fake_client.table.return_value = FakeTable()

        with patch.object(episodes_db, "get_supabase", return_value=fake_client), \
             patch("src.services.embeddings.embed_text", return_value=None):
            episodes_db.store_episode(
                USER_ID,
                {"summary": "test"},
            )
        assert "embedding" not in captured["row"]


# ---------------------------------------------------------------------------
# Integration with build_runtime_context (smoke test)
# ---------------------------------------------------------------------------


class TestRuntimeContextIntegration:
    def test_build_runtime_context_accepts_user_message(self) -> None:
        """Smoke test that the signature accepts user_message and does not crash."""
        from src.agent.system_prompt import build_runtime_context

        # Minimal user_model stub - just enough for build_runtime_context to
        # walk through its sections without raising.
        user_model = MagicMock()
        user_model.user_id = USER_ID
        user_model.project_profile.return_value = {
            "name": "Test",
            "sports": ["running"],
            "goal": {},
            "constraints": {},
            "fitness": {},
            "preferences": {},
        }
        user_model.get_active_plan_summary.return_value = None

        # Disable embedding so we do not need keys in CI.
        with patch.dict(os.environ, {"ATHLETLY_EPISODE_REPLAY_DISABLED": "1"}):
            ctx = build_runtime_context(
                user_model=user_model,
                date="2026-05-14",
                user_message="how should I taper before a marathon?",
            )
        assert "Current Date" in ctx
        # When replay is disabled the section must not appear.
        assert "# Past Insights" not in ctx

    def test_runtime_context_injects_past_insights_when_matches(self) -> None:
        """When build_replay_block returns a block, it must appear in the runtime context."""
        from src.agent import system_prompt as sp

        user_model = MagicMock()
        user_model.user_id = USER_ID
        user_model.project_profile.return_value = {
            "name": "Test",
            "sports": ["running"],
            "goal": {},
            "constraints": {},
            "fitness": {},
            "preferences": {},
        }
        user_model.get_active_plan_summary.return_value = None

        # The retrieval module is imported lazily inside build_runtime_context.
        # Stub the module-level function on the source module so the lazy
        # import resolves to our fake.
        fake_block = (
            "# Past Insights (1 relevant episodes from your history)\n"
            "- 2026-02-20: foo\n\nUse these to inform your reasoning but do NOT claim..."
        )
        with patch.dict(os.environ, {"ATHLETLY_EPISODE_REPLAY_DISABLED": "0"}), \
             patch("src.services.episode_retrieval.build_replay_block",
                   return_value=fake_block), \
             patch("src.config.get_settings") as mock_settings:
            mock_settings.return_value.use_supabase = True
            mock_settings.return_value.agenticsports_user_id = USER_ID
            mock_settings.return_value.episode_replay_top_k = 3
            mock_settings.return_value.episode_replay_threshold = 0.55
            mock_settings.return_value.episode_replay_token_budget = 800
            ctx = sp.build_runtime_context(
                user_model=user_model,
                date="2026-05-14",
                user_message="how should I taper before a marathon?",
            )
        assert "# Past Insights" in ctx
        assert "do NOT claim" in ctx
