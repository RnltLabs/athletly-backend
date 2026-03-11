"""Database layer for AgenticSports (Supabase/PostgreSQL).

Sub-modules
-----------
- ``client``            -- Supabase client singleton
- ``user_model_db``     -- UserModel persistence (profiles, beliefs, goals)
- ``activity_store_db`` -- Activity CRUD + import manifest
- ``session_store_db``  -- Session + message persistence
- ``episodes_db``       -- Episodic memory (training reflections)
- ``plans_db``          -- Training plan storage
- ``pending_actions_db``-- Checkpoint / adaptive replanning actions
"""

from src.db.client import get_supabase, get_async_supabase
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
    list_episodes_for_period,
    store_episode,
)
from src.db.plans_db import (
    deactivate_plan,
    get_active_plan,
    list_plans,
    store_plan,
    update_plan_evaluation,
)
from src.db.health_data_db import (
    get_cross_source_load_summary,
    get_health_activity_summary,
    list_daily_metrics,
    list_garmin_activities,
    list_health_activities,
)
from src.db.proactive_queue_db import (
    deliver_message as deliver_proactive_message,
    expire_stale_messages as expire_proactive_messages,
    get_pending_messages as get_pending_proactive_messages,
    queue_message as queue_proactive_message,
    record_engagement as record_proactive_engagement,
)
from src.db.pending_actions_db import (
    create_pending_action,
    expire_stale_actions,
    get_pending_for_user,
    get_recently_resolved,
    resolve_pending_action,
)
from src.db.session_store_db import (
    create_session,
    get_recent_sessions,
    get_session,
    get_unsummarized_sessions,
    load_session_messages,
    save_message,
    update_session_summary,
)

__all__ = [
    # client
    "get_supabase",
    "get_async_supabase",
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
    "get_unsummarized_sessions",
    "update_session_summary",
    # episodes_db
    "store_episode",
    "list_episodes",
    "get_episode",
    "list_episodes_by_type",
    "list_episodes_for_period",
    # health_data_db
    "list_health_activities",
    "list_garmin_activities",
    "list_daily_metrics",
    "get_health_activity_summary",
    "get_cross_source_load_summary",
    # proactive_queue_db
    "queue_proactive_message",
    "get_pending_proactive_messages",
    "deliver_proactive_message",
    "record_proactive_engagement",
    "expire_proactive_messages",
    # plans_db
    "store_plan",
    "get_active_plan",
    "list_plans",
    "update_plan_evaluation",
    "deactivate_plan",
    # pending_actions_db
    "create_pending_action",
    "get_pending_for_user",
    "get_recently_resolved",
    "resolve_pending_action",
    "expire_stale_actions",
]
