"""Database layer for AgenticSports (Supabase/PostgreSQL).

Sub-modules
-----------
- ``client``            -- Supabase client singleton
- ``user_model_db``     -- UserModel persistence (profiles, beliefs, goals)
- ``activity_store_db`` -- Activity CRUD + import manifest
- ``session_store_db``  -- Session + message persistence
- ``episodes_db``       -- Episodic memory (training reflections)
- ``plans_db``          -- Training plan storage
"""

from src.db.client import get_supabase
from src.db.user_model_db import UserModelDB

from src.db.activity_store_db import (
    check_import_manifest,
    get_activities_summary,
    get_activity,
    get_weekly_summary,
    list_activities,
    record_import,
    store_activity,
)
from src.db.episodes_db import (
    get_episode,
    list_episodes,
    list_episodes_by_type,
    store_episode,
)
from src.db.plans_db import (
    deactivate_plan,
    get_active_plan,
    list_plans,
    store_plan,
    update_plan_evaluation,
)
from src.db.session_store_db import (
    create_session,
    get_recent_sessions,
    get_session,
    load_session_messages,
    save_message,
    update_session_summary,
)

__all__ = [
    # client
    "get_supabase",
    # user_model_db
    "UserModelDB",
    # activity_store_db
    "store_activity",
    "list_activities",
    "get_activity",
    "check_import_manifest",
    "record_import",
    "get_activities_summary",
    "get_weekly_summary",
    # session_store_db
    "create_session",
    "get_session",
    "save_message",
    "load_session_messages",
    "get_recent_sessions",
    "update_session_summary",
    # episodes_db
    "store_episode",
    "list_episodes",
    "get_episode",
    "list_episodes_by_type",
    # plans_db
    "store_plan",
    "get_active_plan",
    "list_plans",
    "update_plan_evaluation",
    "deactivate_plan",
]
