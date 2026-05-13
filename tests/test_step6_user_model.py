"""Tests for Step 6 Phase A: UserModel and Belief Storage."""

import json
import pytest
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from src.memory.user_model import UserModel, BELIEF_CATEGORIES


# ── Fixtures ─────────────────────────────────────────────────────


@pytest.fixture
def tmp_model_dir(tmp_path):
    """Provide a temporary directory for user model storage."""
    model_dir = tmp_path / "user_model"
    model_dir.mkdir()
    return model_dir


@pytest.fixture
def model(tmp_model_dir):
    """Create a fresh UserModel with tmp storage."""
    return UserModel(data_dir=tmp_model_dir)


@pytest.fixture
def populated_model(model):
    """Create a UserModel with structured_core and several beliefs."""
    model.update_structured_core("name", "Test Athlete")
    model.update_structured_core("sports", ["running"])
    model.update_structured_core("goal.event", "Half Marathon")
    model.update_structured_core("goal.target_date", "2026-10-15")
    model.update_structured_core("goal.target_time", "1:45:00")
    model.update_structured_core("constraints.training_days_per_week", 5)
    model.update_structured_core("constraints.max_session_minutes", 90)
    model.update_structured_core("constraints.available_sports", ["running"])

    model.add_belief("Prefers morning training before work", "scheduling", confidence=0.8)
    model.add_belief("Had knee injury 2 years ago, flares up after 15km+", "physical", confidence=0.9)
    model.add_belief("Motivated by race goals, not general fitness", "motivation", confidence=0.7)
    model.add_belief("Sleeps poorly on Sundays before work week", "constraint", confidence=0.6)

    return model


# ── UserModel Initialization ─────────────────────────────────────


class TestUserModelInit:
    def test_creates_with_empty_structured_core(self, model):
        assert model.structured_core["name"] is None
        assert model.structured_core["sports"] == []
        assert model.structured_core["goal"]["event"] is None

    def test_creates_with_empty_beliefs(self, model):
        assert model.beliefs == []

    def test_meta_has_timestamps(self, model):
        assert "created_at" in model.meta
        assert "updated_at" in model.meta
        assert model.meta["sessions_completed"] == 0


# ── Belief CRUD ──────────────────────────────────────────────────


class TestBeliefCRUD:
    def test_add_belief_returns_dict_with_all_fields(self, model):
        belief = model.add_belief(
            text="Runs best in the morning",
            category="scheduling",
            confidence=0.8,
            source="conversation",
            source_ref="session_2026-02-10",
        )

        assert belief["id"].startswith("belief_")
        assert belief["text"] == "Runs best in the morning"
        assert belief["category"] == "scheduling"
        assert belief["confidence"] == 0.8
        assert belief["source"] == "conversation"
        assert belief["source_ref"] == "session_2026-02-10"
        assert belief["stability"] == "stable"
        assert belief["durability"] == "global"
        assert belief["active"] is True
        assert belief["superseded_by"] is None
        assert belief["embedding"] is None
        assert belief["first_observed"] is not None
        assert belief["last_confirmed"] is not None
        assert belief["learned_at"] is not None
        assert belief["archived_at"] is None
        assert belief["valid_from"] is not None
        assert belief["valid_until"] is None

    def test_add_belief_stored_in_beliefs_list(self, model):
        model.add_belief("Test belief", "preference")
        assert len(model.beliefs) == 1
        assert model.beliefs[0]["text"] == "Test belief"

    def test_add_belief_clamps_confidence(self, model):
        b1 = model.add_belief("High conf", "preference", confidence=1.5)
        b2 = model.add_belief("Low conf", "preference", confidence=-0.3)
        assert b1["confidence"] == 1.0
        assert b2["confidence"] == 0.0

    def test_add_belief_invalid_category_defaults_to_preference(self, model):
        belief = model.add_belief("Test", "nonexistent_category")
        assert belief["category"] == "preference"

    def test_add_belief_session_durability(self, model):
        belief = model.add_belief("Tired today", "physical", durability="session")
        assert belief["durability"] == "session"

    def test_update_belief_text(self, model):
        belief = model.add_belief("Runs 4 days/week", "scheduling", confidence=0.7)
        updated = model.update_belief(belief["id"], new_text="Runs 5 days/week")

        assert updated is not None
        assert updated["text"] == "Runs 5 days/week"
        assert updated["embedding"] is None  # cleared for re-embedding

    def test_update_belief_confidence(self, model):
        belief = model.add_belief("Morning runner", "scheduling", confidence=0.6)
        old_confirmed = belief["last_confirmed"]

        updated = model.update_belief(belief["id"], new_confidence=0.9)
        assert updated["confidence"] == 0.9
        # last_confirmed should be bumped (or same second)
        assert updated["last_confirmed"] >= old_confirmed

    def test_update_belief_nonexistent_returns_none(self, model):
        result = model.update_belief("belief_nonexistent", new_text="test")
        assert result is None

    def test_update_inactive_belief_returns_none(self, model):
        belief = model.add_belief("Old info", "preference")
        model.invalidate_belief(belief["id"])
        result = model.update_belief(belief["id"], new_text="new info")
        assert result is None

    def test_invalidate_belief(self, model):
        belief = model.add_belief("Running at night", "scheduling")
        result = model.invalidate_belief(belief["id"])

        assert result is not None
        assert result["active"] is False
        assert result["archived_at"] is not None
        assert result["valid_until"] is not None

    def test_invalidate_belief_with_superseded_by(self, model):
        old = model.add_belief("Runs 3 days/week", "scheduling")
        new = model.add_belief("Runs 5 days/week", "scheduling")
        result = model.invalidate_belief(old["id"], superseded_by=new["id"])

        assert result["superseded_by"] == new["id"]

    def test_invalidate_nonexistent_returns_none(self, model):
        assert model.invalidate_belief("belief_nope") is None

    def test_get_active_beliefs_all(self, populated_model):
        active = populated_model.get_active_beliefs()
        assert len(active) == 4
        assert all(b["active"] for b in active)

    def test_get_active_beliefs_by_category(self, populated_model):
        physical = populated_model.get_active_beliefs(category="physical")
        assert len(physical) == 1
        assert physical[0]["category"] == "physical"

    def test_get_active_beliefs_by_min_confidence(self, populated_model):
        high_conf = populated_model.get_active_beliefs(min_confidence=0.8)
        assert len(high_conf) == 2  # 0.8 and 0.9

    def test_get_active_beliefs_excludes_inactive(self, model):
        b = model.add_belief("Old", "preference")
        model.invalidate_belief(b["id"])
        model.add_belief("Current", "preference")

        active = model.get_active_beliefs()
        assert len(active) == 1
        assert active[0]["text"] == "Current"


# ── Structured Core ──────────────────────────────────────────────


class TestStructuredCore:
    def test_update_top_level_field(self, model):
        model.update_structured_core("name", "Roman")
        assert model.structured_core["name"] == "Roman"

    def test_update_nested_field(self, model):
        model.update_structured_core("goal.event", "Marathon")
        assert model.structured_core["goal"]["event"] == "Marathon"

    def test_update_deep_nested_field(self, model):
        model.update_structured_core("fitness.estimated_vo2max", 52.5)
        assert model.structured_core["fitness"]["estimated_vo2max"] == 52.5

    def test_update_creates_intermediate_dicts(self, model):
        model.update_structured_core("new_section.sub_field", "value")
        assert model.structured_core["new_section"]["sub_field"] == "value"


# ── Model Summary ────────────────────────────────────────────────


class TestModelSummary:
    def test_summary_includes_core_info(self, populated_model):
        summary = populated_model.get_model_summary()
        assert "Test Athlete" in summary
        assert "running" in summary
        assert "Half Marathon" in summary
        assert "2026-10-15" in summary

    def test_summary_includes_beliefs(self, populated_model):
        summary = populated_model.get_model_summary()
        assert "COACH'S NOTES" in summary
        assert "morning training" in summary
        assert "knee injury" in summary

    def test_summary_excludes_low_confidence_beliefs(self, model):
        model.add_belief("Low conf note", "preference", confidence=0.3)
        model.add_belief("High conf note", "preference", confidence=0.8)

        summary = model.get_model_summary()
        assert "High conf note" in summary
        assert "Low conf note" not in summary

    def test_summary_empty_model(self, model):
        summary = model.get_model_summary()
        assert isinstance(summary, str)
        # Should not crash, just have less content


# ── Profile Projection (Backward Compat) ─────────────────────────


class TestProjectProfile:
    def test_project_profile_structure(self, populated_model):
        profile = populated_model.project_profile()

        assert profile["name"] == "Test Athlete"
        assert profile["sports"] == ["running"]
        assert profile["goal"]["event"] == "Half Marathon"
        assert profile["goal"]["target_date"] == "2026-10-15"
        assert profile["goal"]["target_time"] == "1:45:00"
        assert profile["constraints"]["training_days_per_week"] == 5
        assert profile["constraints"]["max_session_minutes"] == 90
        assert "created_at" in profile
        assert "updated_at" in profile

    def test_project_profile_defaults(self, model):
        """An empty model should still produce a valid profile with defaults."""
        profile = model.project_profile()
        assert profile["name"] == "Athlete"
        assert profile["constraints"]["training_days_per_week"] == 5
        assert profile["constraints"]["max_session_minutes"] == 90

# ── Persistence ──────────────────────────────────────────────────


class TestPersistence:
    def test_save_creates_file(self, populated_model, tmp_model_dir):
        path = populated_model.save()
        assert path.exists()
        data = json.loads(path.read_text())
        assert "structured_core" in data
        assert "beliefs" in data
        assert "meta" in data

    def test_save_load_roundtrip(self, populated_model, tmp_model_dir):
        populated_model.save()

        loaded = UserModel(data_dir=tmp_model_dir)
        loaded.load()

        assert loaded.structured_core["name"] == "Test Athlete"
        assert len(loaded.beliefs) == 4
        assert loaded.meta["sessions_completed"] == 0

    def test_load_nonexistent_raises(self, tmp_model_dir):
        model = UserModel(data_dir=tmp_model_dir / "nonexistent")
        with pytest.raises(FileNotFoundError):
            model.load()

    def test_load_or_create_existing(self, populated_model, tmp_model_dir):
        populated_model.save()
        loaded = UserModel.load_or_create(data_dir=tmp_model_dir)
        assert loaded.structured_core["name"] == "Test Athlete"

    def test_load_or_create_new(self, tmp_model_dir):
        fresh_dir = tmp_model_dir / "fresh"
        model = UserModel.load_or_create(data_dir=fresh_dir)
        assert model.structured_core["name"] is None
        assert model.beliefs == []


# ── Prune Stale Beliefs ─────────────────────────────────────────


class TestPruneStaleBeliefs:
    def test_prune_session_beliefs(self, model):
        model.add_belief("Tired today", "physical", durability="session")
        model.add_belief("Global info", "preference", durability="global")

        archived = model.prune_stale_beliefs()
        assert len(archived) == 1
        assert archived[0]["durability"] == "session"

        active = model.get_active_beliefs()
        assert len(active) == 1
        assert active[0]["text"] == "Global info"

    def test_prune_low_confidence_stale(self, model):
        belief = model.add_belief("Old uncertain info", "preference", confidence=0.3)
        # Manually backdate last_confirmed to 40 days ago
        old_date = (datetime.now() - timedelta(days=40)).isoformat(timespec="seconds")
        belief["last_confirmed"] = old_date

        archived = model.prune_stale_beliefs(max_age_days=30, min_confidence=0.5)
        assert len(archived) == 1

    def test_prune_keeps_high_confidence(self, model):
        model.add_belief("Strong belief", "preference", confidence=0.9)
        archived = model.prune_stale_beliefs()
        assert len(archived) == 0

    def test_prune_keeps_recently_confirmed(self, model):
        model.add_belief("Low but recent", "preference", confidence=0.3)
        archived = model.prune_stale_beliefs(max_age_days=30)
        assert len(archived) == 0  # just added, so recent

    def test_prune_writes_archive_file(self, model, tmp_model_dir):
        model.add_belief("To archive", "preference", durability="session")
        model.prune_stale_beliefs()

        archive_path = tmp_model_dir / "beliefs_archive.json"
        assert archive_path.exists()
        data = json.loads(archive_path.read_text())
        assert len(data) == 1
        assert data[0]["text"] == "To archive"
        # Embeddings should be stripped from archive
        assert "embedding" not in data[0]


# ── Embedding & Similarity (mocked) ─────────────────────────────


class TestEmbeddingMocked:
    def test_embed_belief_calls_api(self, model):
        belief = model.add_belief("Test belief", "preference")

        mock_response = MagicMock()
        mock_embedding = MagicMock()
        mock_embedding.values = [0.1, 0.2, 0.3, 0.4, 0.5]
        mock_response.embeddings = [mock_embedding]

        mock_client = MagicMock()
        mock_client.models.embed_content.return_value = mock_response

        with patch("src.memory.user_model.get_client", return_value=mock_client):
            result = model.embed_belief(belief)

        assert result == [0.1, 0.2, 0.3, 0.4, 0.5]
        assert belief["embedding"] == [0.1, 0.2, 0.3, 0.4, 0.5]

    def test_embed_belief_handles_api_failure(self, model):
        belief = model.add_belief("Test belief", "preference")

        with patch("src.memory.user_model.get_client", side_effect=Exception("API error")):
            result = model.embed_belief(belief)

        assert result is None
        assert belief["embedding"] is None

    def test_find_similar_fallback_for_few_beliefs(self, model):
        """When < 10 beliefs, returns all active beliefs without embeddings."""
        model.add_belief("Belief 1", "preference")
        model.add_belief("Belief 2", "scheduling")
        model.add_belief("Belief 3", "physical")

        results = model.find_similar_beliefs("some candidate text")
        assert len(results) == 3
        # All should have similarity 1.0 (fallback)
        assert all(score == 1.0 for _, score in results)

    def test_find_similar_with_embeddings(self, model):
        """When >= 10 beliefs with embeddings, uses cosine similarity."""
        # Add 10+ beliefs with embeddings
        for i in range(12):
            b = model.add_belief(f"Belief {i}", "preference")
            b["embedding"] = [float(i) / 12] * 5  # simple embedding

        mock_response = MagicMock()
        mock_embedding = MagicMock()
        mock_embedding.values = [0.5] * 5  # candidate embedding
        mock_response.embeddings = [mock_embedding]

        mock_client = MagicMock()
        mock_client.models.embed_content.return_value = mock_response

        with patch("src.memory.user_model.get_client", return_value=mock_client):
            results = model.find_similar_beliefs("candidate text", top_k=3)

        assert len(results) == 3
        # Results should be sorted by similarity descending
        sims = [score for _, score in results]
        assert sims == sorted(sims, reverse=True)

    def test_find_similar_handles_api_failure(self, model):
        """Falls back to returning all beliefs if embedding API fails."""
        for i in range(12):
            model.add_belief(f"Belief {i}", "preference")

        with patch("src.memory.user_model.get_client", side_effect=Exception("API error")):
            results = model.find_similar_beliefs("candidate text")

        assert len(results) == 12  # all active beliefs returned as fallback


# ── Belief Categories ────────────────────────────────────────────


class TestBeliefCategories:
    def test_all_expected_categories_exist(self):
        expected = {
            "preference", "constraint", "history", "motivation",
            "physical", "fitness", "scheduling", "personality", "meta",
        }
        assert BELIEF_CATEGORIES == expected
