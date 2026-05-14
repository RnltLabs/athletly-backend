"""LLM-driven GenUI extraction: athlete journal markdown to UI widget list.

Used by ``GET /profile/identity/widgets`` to power the "How the Coach sees
you" screen. The LLM picks the appropriate widget type for each piece of
content so the frontend never needs per-section layout logic; ANY content
the agent writes into the journal becomes a properly visualized widget.

Pipeline:
    1. Pull the markdown journal + the structured profile for the user.
    2. Hash the combined payload, look up Redis cache.
    3. On cache miss: call Claude Haiku 4.5 with a deterministic system prompt
       and parse the returned JSON.
    4. Validate every widget against its Pydantic schema; drop invalid ones.
    5. Compute countdown_days from target_date when needed.
    6. Cache the validated response for 7 days, then return.

Resilience:
    Every external call (LLM, Redis, DB) is wrapped in try/except. On total
    failure the service falls back to a deterministic widget list built from
    the structured profile + canonical journal sections, so the endpoint
    always returns 200 with at least something useful.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from datetime import date, datetime, timezone
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, ValidationError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Cache TTL: 7 days. Stale cache is impossible because the cache key includes
# a SHA-256 hash of the journal + structured profile, so any edit busts it.
_CACHE_TTL_SECONDS = 7 * 24 * 60 * 60

_CACHE_KEY_PREFIX = "identity_widgets"

# Anthropic Haiku 4.5: cheap, deterministic with temperature=0.2, supports
# prompt caching on the static system prefix when it exceeds ~4096 tokens.
_WIDGETS_MODEL = "anthropic/claude-haiku-4-5-20251001"
_WIDGETS_TEMPERATURE = 0.2

# Hard limits to keep the LLM from going crazy. The frontend handles up to
# ~16 widgets nicely; more than that is almost certainly a hallucination.
_MAX_WIDGETS = 16
_MAX_STATS_PER_GRID = 6


# ---------------------------------------------------------------------------
# Widget schema (the contract to the frontend AND the LLM)
# ---------------------------------------------------------------------------


class ProfileHeaderProps(BaseModel):
    name: str
    subtitle: str | None = None
    sport_badges: list[str] = Field(default_factory=list)


class HeroGoalProps(BaseModel):
    event: str
    target_date: str | None = None
    countdown_days: int | None = None
    target_time: str | None = None
    pace: str | None = None
    course_description: str | None = None
    source: str | None = None


class StatGridItem(BaseModel):
    label: str
    value: str
    unit: str | None = None
    icon: str | None = None


class StatGridProps(BaseModel):
    title: str
    stats: list[StatGridItem]


class TimelineItem(BaseModel):
    date: str | None = None
    label: str
    description: str | None = None
    status: Literal["done", "pending", "failed", "neutral"] = "neutral"


class TimelineProps(BaseModel):
    title: str
    items: list[TimelineItem]


class ChecklistItem(BaseModel):
    text: str
    status: Literal["open", "done", "in_progress"] = "open"


class ChecklistProps(BaseModel):
    title: str
    items: list[ChecklistItem]


class ChipListItem(BaseModel):
    label: str
    color: Literal["green", "orange", "gray", "blue"] = "gray"
    icon: str | None = None


class ChipListProps(BaseModel):
    title: str
    chips: list[ChipListItem]


class InfoCardFact(BaseModel):
    label: str
    value: str


class InfoCardProps(BaseModel):
    title: str
    icon: str | None = None
    facts: list[InfoCardFact] = Field(default_factory=list)
    description: str | None = None


class QuoteCardProps(BaseModel):
    title: str | None = None
    quote: str
    attribution: str | None = None


# Map widget type to its props validator. Used to validate the LLM output
# block by block, dropping any widget whose props fail validation rather
# than throwing the whole response away.
_PROPS_MODELS: dict[str, type[BaseModel]] = {
    "profile_header": ProfileHeaderProps,
    "hero_goal": HeroGoalProps,
    "stat_grid": StatGridProps,
    "timeline": TimelineProps,
    "checklist": ChecklistProps,
    "chip_list": ChipListProps,
    "info_card": InfoCardProps,
    "quote_card": QuoteCardProps,
}

WidgetType = Literal[
    "profile_header",
    "hero_goal",
    "stat_grid",
    "timeline",
    "checklist",
    "chip_list",
    "info_card",
    "quote_card",
]


class Widget(BaseModel):
    """Polymorphic widget. ``props`` is validated by type-specific models."""

    type: WidgetType
    props: dict[str, Any]


class WidgetsResponse(BaseModel):
    widgets: list[Widget] = Field(default_factory=list)
    generated_at: str
    cache_hit: bool = False
    edit_hint_per_widget: dict[int, str] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------


def _content_hash(markdown: str, structured: dict[str, Any]) -> str:
    """Short SHA-256 of the canonicalized markdown + structured payload."""
    payload = markdown + "\n" + json.dumps(structured, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _cache_key(user_id: str, content_hash: str) -> str:
    return f"{_CACHE_KEY_PREFIX}:{user_id}:{content_hash}"


async def _get_redis():
    """Return an async Redis client or None when Redis is unreachable.

    Mirrors the pattern in src.services.plan_review. We never raise from
    here; absent Redis just disables caching.
    """
    try:
        import redis.asyncio as aioredis

        from src.config import get_settings

        client = aioredis.from_url(
            get_settings().redis_url,
            decode_responses=True,
            socket_connect_timeout=1,
        )
        await client.ping()
        return client
    except Exception:
        logger.debug("identity_widgets: Redis unavailable, caching disabled")
        return None


async def _cache_get(redis_client, key: str) -> dict[str, Any] | None:
    if redis_client is None:
        return None
    try:
        raw = await redis_client.get(key)
    except Exception:
        logger.warning("identity_widgets: cache GET failed for %s", key, exc_info=True)
        return None
    if not raw:
        return None
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        logger.warning("identity_widgets: cache value malformed for %s", key)
        return None


async def _cache_set(redis_client, key: str, value: dict[str, Any]) -> None:
    if redis_client is None:
        return
    try:
        await redis_client.set(key, json.dumps(value), ex=_CACHE_TTL_SECONDS)
    except Exception:
        logger.warning("identity_widgets: cache SET failed for %s", key, exc_info=True)


# ---------------------------------------------------------------------------
# LLM prompt
# ---------------------------------------------------------------------------


# The system prompt is intentionally long: it includes worked examples so
# Anthropic's prompt caching can amortize the static prefix across users.
# Below ~4096 tokens (~16400 chars) caching is silently ignored by the API.
_SYSTEM_PROMPT = """You convert an athlete profile markdown document into a list of UI widgets
for a mobile app's "How the Coach sees you" screen.

OUTPUT FORMAT: Return ONLY valid JSON, no prose, no markdown fences.
Schema:
{
  "widgets": [<widget>, ...],
  "edit_hint_per_widget": {"<index>": "<draft text>", ...}
}
Each widget is shaped as:
  {"type": "<widget_type>", "props": {<props matching that widget's schema>}}

WIDGET TYPES AND WHEN TO USE:

1. profile_header
   Use ONCE at the top.
   props: {"name": str, "subtitle": str|null, "sport_badges": [str, ...]}
   subtitle example: "27 Jahre - Karlsruhe" (use hyphen, never em-dash)
   sport_badges example: ["Laufen", "Rad"]

2. hero_goal
   Use ONCE when there is a current goal/event the athlete is training for.
   props: {
     "event": str,
     "target_date": "YYYY-MM-DD" | null,
     "countdown_days": int | null,
     "target_time": "H:MM:SS" | null,
     "pace": "M:SS /km" | null,
     "course_description": str | null,
     "source": str | null
   }
   Strip leading zeros from times: "01:24:00" becomes "1:24:00".
   The server will recompute countdown_days from target_date, but emitting it
   is fine.

3. stat_grid
   Numeric metrics (VO2max, FTP, weekly km, paces, PBs). MAX 6 stats per grid.
   props: {
     "title": str,
     "stats": [
       {"label": str, "value": str, "unit": str|null, "icon": str|null},
       ...
     ]
   }
   Always stringify value. Icons are lucide names: "heart", "activity",
   "trending-up", "zap", "timer", "footprints", "bike", "mountain".

4. timeline
   Chronological history (goal changes, what we tried, milestones).
   props: {
     "title": str,
     "items": [
       {
         "date": "YYYY-MM-DD" | "Mar 2026" | null,
         "label": str,
         "description": str|null,
         "status": "done" | "pending" | "failed" | "neutral"
       },
       ...
     ]
   }

5. checklist
   Open items, follow-ups, in-progress threads.
   props: {
     "title": str,
     "items": [
       {"text": str, "status": "open" | "done" | "in_progress"},
       ...
     ]
   }

6. chip_list
   Short-form preferences, tags, traits, sports.
   props: {
     "title": str,
     "chips": [
       {"label": str, "color": "green"|"orange"|"gray"|"blue", "icon": str|null},
       ...
     ]
   }
   color hint: green for positive preferences, orange for caution/limits,
   gray for neutral, blue for sport tags.

7. info_card
   Free-form facts in label:value form that do not fit elsewhere.
   props: {
     "title": str,
     "icon": str|null,
     "facts": [{"label": str, "value": str}, ...],
     "description": str|null
   }

8. quote_card
   Standout single insight, learning, or athlete statement worth highlighting.
   props: {"title": str|null, "quote": str, "attribution": str|null}

RULES:
- Pick the widget that matches the SEMANTICS of the content, not just the
  visual layout. A bullet list of facts could be a checklist (things to
  follow up), a chip_list (traits), or an info_card (key:value pairs).
- Strip markdown markers like ** __ * from output text. Output clean plain
  text.
- ISO dates stay ISO ("2026-09-20"). The frontend reformats for display.
- Time values: strip leading zeros ("01:24:00" becomes "1:24:00",
  "0:04:10" becomes "4:10").
- One widget per logical idea. Do not duplicate the same content across
  widgets.
- Skip empty sections entirely. Do not emit a widget for an empty section.
- IMPORTANT: order widgets so the most important goal/identity info appears
  FIRST, then performance metrics, then preferences and open threads, then
  history last.
- Keep widget count under 12. Aim for 6 to 10.
- Never use em-dashes or en-dashes anywhere in your output. Use hyphens,
  colons, or restructure the sentence.

EDIT HINTS:
After the widgets array, also generate edit_hint_per_widget keyed by widget
index (as a string), where the value is a draft chat message the user would
send to update that widget. Always start with "Lass uns ... anpassen: " in
German.

Examples:
- {"0": "Lass uns mein Profil anpassen: "}
- {"1": "Lass uns mein Ziel anpassen: "}
- {"2": "Lass uns meine Trainingswerte anpassen: "}

WORKED EXAMPLE 1
Input markdown:
  # Roman
  ## Identity
  27 Jahre, Karlsruhe, Läufer und Radfahrer.
  ## Current Goal
  Karlsruher Halbmarathon am 20.09.2026. Ziel 1:24:00, Pace 3:58 /km.
  Verifiziert ueber strava-club Karlsruhe.
  ## Goal Timeline
  - 2026-03-01: wechsel von Marathon zu Halbmarathon (Knie-Reha)
  - 2026-04-15: Testlauf 21km in 1:28:30
  ## What I know about my training
  Threshold-Pace 4:10 /km, VO2max-Schaetzung 56, Wochenumfang 60 km.
  ## Preferences
  - Lange Laeufe am Sonntag
  - Keine Intervalle am Freitag
  ## Open Threads
  - Knie beobachten
  - VO2max-Test ueber Garmin nachholen
  ## What we tried
  - Drei Tempo-Einheiten pro Woche waren zuviel, eine reicht.

Output JSON:
{
  "widgets": [
    {
      "type": "profile_header",
      "props": {
        "name": "Roman",
        "subtitle": "27 Jahre - Karlsruhe",
        "sport_badges": ["Laufen", "Rad"]
      }
    },
    {
      "type": "hero_goal",
      "props": {
        "event": "Karlsruher Halbmarathon",
        "target_date": "2026-09-20",
        "countdown_days": null,
        "target_time": "1:24:00",
        "pace": "3:58 /km",
        "source": "strava-club Karlsruhe"
      }
    },
    {
      "type": "stat_grid",
      "props": {
        "title": "Performance",
        "stats": [
          {"label": "Threshold-Pace", "value": "4:10", "unit": "/km", "icon": "timer"},
          {"label": "VO2max", "value": "56", "unit": "ml/kg/min", "icon": "activity"},
          {"label": "Wochenumfang", "value": "60", "unit": "km", "icon": "footprints"}
        ]
      }
    },
    {
      "type": "chip_list",
      "props": {
        "title": "Präferenzen",
        "chips": [
          {"label": "Lange Laeufe am Sonntag", "color": "green"},
          {"label": "Keine Intervalle am Freitag", "color": "orange"}
        ]
      }
    },
    {
      "type": "checklist",
      "props": {
        "title": "Offene Themen",
        "items": [
          {"text": "Knie beobachten", "status": "in_progress"},
          {"text": "VO2max-Test ueber Garmin nachholen", "status": "open"}
        ]
      }
    },
    {
      "type": "quote_card",
      "props": {
        "title": "Was wir gelernt haben",
        "quote": "Drei Tempo-Einheiten pro Woche waren zuviel, eine reicht."
      }
    },
    {
      "type": "timeline",
      "props": {
        "title": "Ziel-Historie",
        "items": [
          {"date": "2026-03-01", "label": "Wechsel von Marathon zu Halbmarathon", "description": "Knie-Reha", "status": "neutral"},
          {"date": "2026-04-15", "label": "Testlauf 21km in 1:28:30", "status": "done"}
        ]
      }
    }
  ],
  "edit_hint_per_widget": {
    "0": "Lass uns mein Profil anpassen: ",
    "1": "Lass uns mein Ziel anpassen: ",
    "2": "Lass uns meine Trainingswerte anpassen: ",
    "3": "Lass uns meine Präferenzen anpassen: ",
    "4": "Lass uns die offenen Themen anpassen: ",
    "5": "Lass uns das Learning anpassen: ",
    "6": "Lass uns die Ziel-Historie anpassen: "
  }
}

WORKED EXAMPLE 2
Input markdown:
  # Athlete
  ## Identity
  _(not yet filled in)_
  ## Current Goal
  _(no active goal)_

Output JSON:
{
  "widgets": [
    {
      "type": "profile_header",
      "props": {"name": "Athlete", "sport_badges": []}
    },
    {
      "type": "info_card",
      "props": {
        "title": "Profil noch leer",
        "description": "Erzaehl mir, was du erreichen willst, und ich richte dir dein Profil ein."
      }
    }
  ],
  "edit_hint_per_widget": {
    "0": "Lass uns mein Profil anpassen: ",
    "1": "Lass uns mein Ziel anpassen: "
  }
}

WORKED EXAMPLE 3
Input markdown:
  # Anna
  ## Identity
  34 Jahre, Berlin, Triathletin.
  ## What I know about my training
  FTP 245 W, Schwimmschnitt 1:42 /100m, Wochenstunden 10.

Output JSON:
{
  "widgets": [
    {
      "type": "profile_header",
      "props": {
        "name": "Anna",
        "subtitle": "34 Jahre - Berlin",
        "sport_badges": ["Triathlon"]
      }
    },
    {
      "type": "stat_grid",
      "props": {
        "title": "Performance",
        "stats": [
          {"label": "FTP", "value": "245", "unit": "W", "icon": "zap"},
          {"label": "Schwimmschnitt", "value": "1:42", "unit": "/100m", "icon": "activity"},
          {"label": "Wochenstunden", "value": "10", "unit": "h", "icon": "timer"}
        ]
      }
    }
  ],
  "edit_hint_per_widget": {
    "0": "Lass uns mein Profil anpassen: ",
    "1": "Lass uns meine Trainingswerte anpassen: "
  }
}

WORKED EXAMPLE 4
Input markdown:
  # Lukas
  ## Identity
  41 Jahre, Muenchen, Hobby-Radfahrer und Bergsteiger.
  Buerojob, 2 Kinder, training meist abends.
  ## Current Goal
  Stelvio Bike Day, 02.06.2026. Tagestour, kein Wettkampf.
  ## Goal Timeline
  - 2025-09-01: Erstmals Stilfser Joch Idee
  - 2026-01-15: Plan: 8 Wochen Aufbau ab Maerz
  - 2026-03-20: Trainingsbeginn solide gestartet
  ## What I know about my training
  FTP 240 W bei 78 kg, Wochenstunden 6-8, Schwaeche an langen Anstiegen.
  ## Preferences
  - Training nach 19 Uhr
  - Lange Touren nur am Samstag
  - Indoor bei Regen ok
  - Nie zwei Tage am Stueck Tempo
  ## Open Threads
  - Hoehentraining noch nicht eingeplant
  - Bike-Fitting im April nachholen
  ## What we tried
  - Sweet-Spot 3x/Woche fuehrte zu Stagnation
  - Ein langer VO2max-Tag pro Woche bringt mehr

Output JSON:
{
  "widgets": [
    {
      "type": "profile_header",
      "props": {
        "name": "Lukas",
        "subtitle": "41 Jahre - Muenchen",
        "sport_badges": ["Rad", "Bergsteigen"]
      }
    },
    {
      "type": "hero_goal",
      "props": {
        "event": "Stelvio Bike Day",
        "target_date": "2026-06-02",
        "countdown_days": null,
        "course_description": "Tagestour, kein Wettkampf"
      }
    },
    {
      "type": "stat_grid",
      "props": {
        "title": "Performance",
        "stats": [
          {"label": "FTP", "value": "240", "unit": "W", "icon": "zap"},
          {"label": "Gewicht", "value": "78", "unit": "kg", "icon": "activity"},
          {"label": "Wochenstunden", "value": "6-8", "unit": "h", "icon": "timer"}
        ]
      }
    },
    {
      "type": "info_card",
      "props": {
        "title": "Trainingsschwerpunkt",
        "description": "Schwaeche an langen Anstiegen, Schwerpunkt auf VO2max- und Sweet-Spot-Einheiten."
      }
    },
    {
      "type": "chip_list",
      "props": {
        "title": "Präferenzen",
        "chips": [
          {"label": "Training nach 19 Uhr", "color": "green"},
          {"label": "Lange Touren nur Samstag", "color": "green"},
          {"label": "Indoor bei Regen", "color": "blue"},
          {"label": "Nie 2 Tage Tempo am Stueck", "color": "orange"}
        ]
      }
    },
    {
      "type": "checklist",
      "props": {
        "title": "Offene Themen",
        "items": [
          {"text": "Hoehentraining einplanen", "status": "open"},
          {"text": "Bike-Fitting im April nachholen", "status": "open"}
        ]
      }
    },
    {
      "type": "quote_card",
      "props": {
        "title": "Was wir gelernt haben",
        "quote": "Ein langer VO2max-Tag pro Woche bringt mehr als 3x Sweet-Spot."
      }
    },
    {
      "type": "timeline",
      "props": {
        "title": "Ziel-Historie",
        "items": [
          {"date": "2025-09-01", "label": "Idee Stilfser Joch", "status": "neutral"},
          {"date": "2026-01-15", "label": "8-Wochen-Aufbauplan", "status": "done"},
          {"date": "2026-03-20", "label": "Trainingsbeginn solide", "status": "done"}
        ]
      }
    }
  ],
  "edit_hint_per_widget": {
    "0": "Lass uns mein Profil anpassen: ",
    "1": "Lass uns mein Ziel anpassen: ",
    "2": "Lass uns meine Trainingswerte anpassen: ",
    "3": "Lass uns den Trainingsschwerpunkt anpassen: ",
    "4": "Lass uns meine Präferenzen anpassen: ",
    "5": "Lass uns die offenen Themen anpassen: ",
    "6": "Lass uns das Learning anpassen: ",
    "7": "Lass uns die Ziel-Historie anpassen: "
  }
}

EDGE CASES:
- If the journal has a heading but the body is a seed placeholder
  ("_(not yet filled in)_", "_(no active goal)_", or empty): treat that
  section as empty and skip it. Do not emit a widget that contains the
  placeholder text.
- If the structured profile carries fitness values but the markdown does
  not mention them, still emit a stat_grid using the structured values.
- If both the journal and structured profile are empty: emit only a
  profile_header (with whatever name is available, default "Athlete") and
  a single info_card nudging the athlete to start a conversation.
- If sports list is empty: omit sport_badges from profile_header (use an
  empty array, not null).
- If multiple training stats appear, group them in ONE stat_grid (up to 6
  items) rather than multiple grids.
- If "Open Threads" contains items that look like in-progress monitoring
  (e.g. "Knie beobachten", "Schlaf weiter tracken"), use status:
  "in_progress". Pure todos use status: "open".
- If "What we tried" has multiple bullet learnings, emit ONE quote_card
  for the most impactful one and roll the rest into an info_card with
  facts ("Was funktioniert" / "Was nicht funktioniert").
- Do not emit a hero_goal when the athlete has no current goal. Skip
  the widget entirely; do not emit a hero_goal with event "(no goal)".

NEVER EMIT:
- Widgets with empty arrays (e.g. stat_grid with stats: []).
- Widgets containing only the seed placeholder text.
- Duplicate widgets (same type AND same primary content).
- Em-dashes, en-dashes, or smart quotes. Use only plain ASCII punctuation.

WORKED EXAMPLE 5 (Triathlete with full data)
Input markdown:
  # Sarah
  ## Identity
  29 Jahre, Hamburg, Triathletin seit 2021.
  Vollzeit-Job in der IT, trainiert frueh morgens und abends.
  ## Current Goal
  Ironman Hamburg, 02.08.2026. Ziel: Sub 11h. Qualifikation Roth waere Bonus.
  Course: Flach, Wendepunktstrecke, Schwimm in der Aussenalster.
  ## Goal Timeline
  - 2025-10-01: Saisonpause beendet
  - 2025-11-15: Anmeldung Ironman Hamburg
  - 2026-01-10: Trainingslager Lanzarote, 25h
  - 2026-04-05: Halbmarathon-Test 1:32:14
  - 2026-05-01: Olympische Distanz Hannover, 2:14h
  ## What I know about my training
  Schwimmen: 1:32 /100m Schwellenpace, Wochenkilometer 18
  Rad: FTP 245 W bei 62 kg, Wochenstunden 8-10
  Laufen: Threshold 4:05 /km, Wochenkilometer 55, Marathon-PB 3:18
  VO2max-Schaetzung Laufen 58, Rad 64
  ## Preferences
  - Schwimmen 3x/Woche im Verein
  - Lange Radausfahrt am Samstag
  - Brick-Workouts Sonntag
  - Keine Trainings vor 6 Uhr
  - Kein Rad bei unter 5 Grad
  ## Open Threads
  - Race-Nutrition testen (neues Gel-Produkt)
  - Aero-Position Bike-Fitting buchen
  - Schlaf trackt seit 2 Wochen unruhig, beobachten
  ## What we tried
  - 4 Schwimmeinheiten waren zuviel, 3 reichen mit guten Intervallen
  - Lange Brick am Sonntag funktioniert besser als am Samstag
  - VO2max-Block im Februar hat 10W FTP gebracht

Output JSON:
{
  "widgets": [
    {
      "type": "profile_header",
      "props": {
        "name": "Sarah",
        "subtitle": "29 Jahre - Hamburg",
        "sport_badges": ["Triathlon"]
      }
    },
    {
      "type": "hero_goal",
      "props": {
        "event": "Ironman Hamburg",
        "target_date": "2026-08-02",
        "countdown_days": null,
        "target_time": "11:00:00",
        "course_description": "Flach, Wendepunktstrecke, Schwimm in der Aussenalster"
      }
    },
    {
      "type": "stat_grid",
      "props": {
        "title": "Performance",
        "stats": [
          {"label": "Schwimm-Pace", "value": "1:32", "unit": "/100m", "icon": "activity"},
          {"label": "FTP", "value": "245", "unit": "W", "icon": "zap"},
          {"label": "Laufschwelle", "value": "4:05", "unit": "/km", "icon": "timer"},
          {"label": "VO2max Lauf", "value": "58", "unit": "ml/kg/min", "icon": "trending-up"},
          {"label": "VO2max Rad", "value": "64", "unit": "ml/kg/min", "icon": "trending-up"},
          {"label": "Marathon-PB", "value": "3:18", "unit": null, "icon": "footprints"}
        ]
      }
    },
    {
      "type": "chip_list",
      "props": {
        "title": "Präferenzen",
        "chips": [
          {"label": "Schwimmen 3x im Verein", "color": "green"},
          {"label": "Lange Radausfahrt Samstag", "color": "green"},
          {"label": "Brick-Workouts Sonntag", "color": "green"},
          {"label": "Nicht vor 6 Uhr", "color": "orange"},
          {"label": "Kein Rad unter 5 Grad", "color": "orange"}
        ]
      }
    },
    {
      "type": "checklist",
      "props": {
        "title": "Offene Themen",
        "items": [
          {"text": "Race-Nutrition testen (neues Gel-Produkt)", "status": "in_progress"},
          {"text": "Aero-Position Bike-Fitting buchen", "status": "open"},
          {"text": "Unruhiger Schlaf seit 2 Wochen beobachten", "status": "in_progress"}
        ]
      }
    },
    {
      "type": "quote_card",
      "props": {
        "title": "Wichtigstes Learning",
        "quote": "VO2max-Block im Februar hat 10W FTP gebracht."
      }
    },
    {
      "type": "info_card",
      "props": {
        "title": "Was funktioniert hat",
        "facts": [
          {"label": "Schwimmen", "value": "3 Einheiten mit guten Intervallen schlagen 4 Einheiten"},
          {"label": "Brick", "value": "Sonntag funktioniert besser als Samstag"}
        ]
      }
    },
    {
      "type": "timeline",
      "props": {
        "title": "Ziel-Historie",
        "items": [
          {"date": "2025-10-01", "label": "Saisonpause beendet", "status": "done"},
          {"date": "2025-11-15", "label": "Anmeldung Ironman Hamburg", "status": "done"},
          {"date": "2026-01-10", "label": "Trainingslager Lanzarote, 25h", "status": "done"},
          {"date": "2026-04-05", "label": "Halbmarathon-Test 1:32:14", "status": "done"},
          {"date": "2026-05-01", "label": "Olympische Distanz Hannover, 2:14h", "status": "done"}
        ]
      }
    }
  ],
  "edit_hint_per_widget": {
    "0": "Lass uns mein Profil anpassen: ",
    "1": "Lass uns mein Ziel anpassen: ",
    "2": "Lass uns meine Trainingswerte anpassen: ",
    "3": "Lass uns meine Präferenzen anpassen: ",
    "4": "Lass uns die offenen Themen anpassen: ",
    "5": "Lass uns das Learning anpassen: ",
    "6": "Lass uns das anpassen: ",
    "7": "Lass uns die Ziel-Historie anpassen: "
  }
}

STYLE NOTES FOR GERMAN OUTPUT:
- Use REAL German umlauts (ä, ö, ü, ß), NOT ASCII transliteration. The
  frontend renders UTF-8 properly. NEVER write "Präferenzen", "Läufer",
  "ueberfaellig". Always write "Präferenzen", "Läufer", "überfällig",
  "Größe", "Fußball". The JSON is UTF-8 and umlauts pass through.
- Banned punctuation: NEVER use em-dash (—, U+2014) or en-dash (–, U+2013)
  in any string. Use a regular hyphen (-), colon (:), or restructure.
  These are AI-tells the user has banned globally.
- Section titles in German with proper umlauts: "Performance", "Präferenzen",
  "Offene Themen", "Ziel-Historie", "Was wir gelernt haben",
  "Trainingsschwerpunkt".
- Status semantics for checklist items:
    "open"        - todo, nothing started
    "in_progress" - currently tracking or experimenting
    "done"        - completed and resolved
- Status semantics for timeline items:
    "done"    - milestone achieved
    "pending" - planned but not yet done
    "failed"  - attempted and did not work
    "neutral" - factual change without success/failure judgment

FINAL REMINDERS:
- Output ONLY the JSON object. No prose. No code fences. No commentary.
- No em-dashes or en-dashes anywhere. Hyphens only.
- Validate every widget against the schemas above before emitting.
- Skip empty sections; do not emit empty widgets.
- The structured profile JSON is the source of truth for numeric fields
  like vo2max, FTP, weekly volume, target_date, target_time. Use it.
- The markdown is the source of truth for qualitative content like
  preferences, open threads, learnings, history. Use it.
- When both contain the same fact, prefer the structured value for the
  stat_grid and the markdown for any contextual prose.
"""


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


async def get_identity_widgets(user_id: str) -> WidgetsResponse:
    """Return the widget list for *user_id*.

    Always returns 200-worthy data: even on total LLM failure, a deterministic
    fallback built from the structured profile is returned.
    """
    markdown, structured, athlete_name = _load_journal_and_profile(user_id)

    cache_hash = _content_hash(markdown, structured)
    key = _cache_key(user_id, cache_hash)

    redis_client = await _get_redis()
    try:
        cached = await _cache_get(redis_client, key)
        if cached is not None:
            cached_response = _coerce_cached_response(cached)
            if cached_response is not None:
                return cached_response.model_copy(update={"cache_hit": True})

        response = _generate_widgets(markdown, structured, athlete_name)
        await _cache_set(redis_client, key, response.model_dump())
        return response
    finally:
        if redis_client is not None:
            try:
                await redis_client.aclose()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_journal_and_profile(
    user_id: str,
) -> tuple[str, dict[str, Any], str | None]:
    """Load markdown + structured profile + athlete name. Tolerates failures."""
    markdown = ""
    name: str | None = None
    try:
        from src.agent.athlete_journal import parse_sections, read_journal

        markdown = read_journal(user_id) or ""
        if markdown:
            parsed = parse_sections(markdown)
            name = parsed.get("_title") or None
    except Exception:
        logger.warning(
            "identity_widgets: journal load failed for user=%s",
            user_id,
            exc_info=True,
        )

    structured: dict[str, Any] = {}
    try:
        from src.db.user_model_db import UserModelDB

        model = UserModelDB.load_or_create(user_id)
        structured = model.project_profile() or {}
    except Exception:
        logger.warning(
            "identity_widgets: structured profile load failed for user=%s",
            user_id,
            exc_info=True,
        )

    # Prefer the name from the structured profile if present.
    if not name:
        candidate = structured.get("name")
        if isinstance(candidate, str) and candidate.strip():
            name = candidate.strip()

    return markdown, structured, name


# ---------------------------------------------------------------------------
# Cache coercion
# ---------------------------------------------------------------------------


def _coerce_cached_response(payload: dict[str, Any]) -> WidgetsResponse | None:
    """Validate a cached dict back into a WidgetsResponse, or None on drift."""
    try:
        # edit_hint_per_widget may be serialized with str keys -- normalize.
        hints_raw = payload.get("edit_hint_per_widget") or {}
        hints: dict[int, str] = {}
        for k, v in hints_raw.items():
            try:
                hints[int(k)] = str(v)
            except (TypeError, ValueError):
                continue
        return WidgetsResponse(
            widgets=[Widget(**w) for w in payload.get("widgets", [])],
            generated_at=str(payload.get("generated_at") or _now_iso()),
            cache_hit=False,
            edit_hint_per_widget=hints,
        )
    except (ValidationError, TypeError, ValueError):
        logger.warning("identity_widgets: cached payload failed validation")
        return None


# ---------------------------------------------------------------------------
# LLM generation
# ---------------------------------------------------------------------------


def _generate_widgets(
    markdown: str,
    structured: dict[str, Any],
    athlete_name: str | None,
) -> WidgetsResponse:
    """Call the LLM, validate, post-process, and assemble the response."""
    user_message = _build_user_message(markdown, structured)

    try:
        from src.agent.llm import chat_completion

        response = chat_completion(
            messages=[{"role": "user", "content": user_message}],
            system_prompt=_SYSTEM_PROMPT,
            temperature=_WIDGETS_TEMPERATURE,
            model=_WIDGETS_MODEL,
        )
        raw = (response.choices[0].message.content or "").strip()
        parsed = _parse_llm_output(raw)
    except Exception:
        logger.warning(
            "identity_widgets: LLM call failed, using fallback widgets",
            exc_info=True,
        )
        return _fallback_response(structured, athlete_name, markdown)

    widgets = _validate_widgets(parsed.get("widgets") or [])
    if not widgets:
        logger.info("identity_widgets: LLM returned no usable widgets, fallback")
        return _fallback_response(structured, athlete_name, markdown)

    widgets = [_post_process_widget(w) for w in widgets][:_MAX_WIDGETS]
    hints = _normalize_hints(parsed.get("edit_hint_per_widget") or {}, widgets)

    return WidgetsResponse(
        widgets=widgets,
        generated_at=_now_iso(),
        cache_hit=False,
        edit_hint_per_widget=hints,
    )


def _build_user_message(markdown: str, structured: dict[str, Any]) -> str:
    """Compose the user message: markdown + JSON of structured fields."""
    structured_json = json.dumps(structured, sort_keys=True, default=str, indent=2)
    journal = markdown.strip() or "(empty journal)"
    return (
        "ATHLETE JOURNAL (markdown):\n"
        f"{journal}\n\n"
        "STRUCTURED PROFILE (JSON):\n"
        f"{structured_json}\n\n"
        "Now produce the widgets JSON exactly as specified."
    )


def _parse_llm_output(raw: str) -> dict[str, Any]:
    """Parse the LLM response into a dict, raising on hard failure."""
    if not raw:
        raise ValueError("empty LLM response")
    from src.agent.json_utils import extract_json

    return extract_json(raw)


# ---------------------------------------------------------------------------
# Widget validation
# ---------------------------------------------------------------------------


def _validate_widgets(items: list[Any]) -> list[Widget]:
    """Return only widgets whose props pass schema validation."""
    valid: list[Widget] = []
    for raw in items:
        if not isinstance(raw, dict):
            continue
        widget_type = raw.get("type")
        props = raw.get("props")
        if widget_type not in _PROPS_MODELS or not isinstance(props, dict):
            continue
        model_cls = _PROPS_MODELS[widget_type]
        try:
            validated = model_cls(**props)
        except ValidationError:
            logger.debug(
                "identity_widgets: dropping invalid widget type=%s",
                widget_type,
            )
            continue
        valid.append(Widget(type=widget_type, props=validated.model_dump()))
    return valid


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------

_TIME_LEADING_ZERO = re.compile(r"^0+(?=\d)")


def _strip_leading_zero_time(value: str | None) -> str | None:
    """Turn "01:24:00" into "1:24:00". Returns the input unchanged otherwise."""
    if not value or not isinstance(value, str):
        return value
    return _TIME_LEADING_ZERO.sub("", value.strip())


def _post_process_widget(widget: Widget) -> Widget:
    """Apply deterministic fixes to a widget's props (immutable: returns new)."""
    if widget.type == "hero_goal":
        new_props = dict(widget.props)
        target_time = _strip_leading_zero_time(new_props.get("target_time"))
        if target_time is not None:
            new_props["target_time"] = target_time
        new_props["countdown_days"] = _compute_countdown(new_props.get("target_date"))
        return Widget(type=widget.type, props=new_props)
    return widget


def _compute_countdown(target_date: str | None) -> int | None:
    """Return days from today to *target_date*, clamped to >= 0 when past."""
    if not target_date or not isinstance(target_date, str):
        return None
    try:
        target = datetime.fromisoformat(target_date).date()
    except ValueError:
        # Tolerate plain YYYY-MM-DD only; anything else stays null.
        return None
    today = date.today()
    delta = (target - today).days
    return delta if delta >= 0 else 0


# ---------------------------------------------------------------------------
# Edit hints
# ---------------------------------------------------------------------------


def _normalize_hints(
    raw_hints: dict[Any, Any],
    widgets: list[Widget],
) -> dict[int, str]:
    """Coerce hint keys to int and clip to the actual widget index range."""
    result: dict[int, str] = {}
    for k, v in raw_hints.items():
        try:
            idx = int(k)
        except (TypeError, ValueError):
            continue
        if idx < 0 or idx >= len(widgets):
            continue
        if not isinstance(v, str) or not v.strip():
            continue
        result[idx] = v
    # Backfill any missing index with a generic German hint so the frontend
    # always has something to put into the chat composer.
    for idx, widget in enumerate(widgets):
        result.setdefault(idx, _default_hint_for(widget.type))
    return result


_DEFAULT_HINTS: dict[str, str] = {
    "profile_header": "Lass uns mein Profil anpassen: ",
    "hero_goal": "Lass uns mein Ziel anpassen: ",
    "stat_grid": "Lass uns meine Trainingswerte anpassen: ",
    "timeline": "Lass uns die Historie anpassen: ",
    "checklist": "Lass uns die offenen Themen anpassen: ",
    "chip_list": "Lass uns meine Präferenzen anpassen: ",
    "info_card": "Lass uns das anpassen: ",
    "quote_card": "Lass uns das Learning anpassen: ",
}


def _default_hint_for(widget_type: str) -> str:
    return _DEFAULT_HINTS.get(widget_type, "Lass uns das anpassen: ")


# ---------------------------------------------------------------------------
# Deterministic fallback (used when the LLM is unavailable)
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _fallback_response(
    structured: dict[str, Any],
    athlete_name: str | None,
    markdown: str,
) -> WidgetsResponse:
    """Build a minimal widget list from the structured profile + journal."""
    widgets: list[Widget] = []

    sports = list(structured.get("sports") or [])
    widgets.append(
        Widget(
            type="profile_header",
            props=ProfileHeaderProps(
                name=athlete_name or "Athlete",
                sport_badges=[s.capitalize() for s in sports if isinstance(s, str)],
            ).model_dump(),
        )
    )

    goal = structured.get("goal") or {}
    event = goal.get("event") if isinstance(goal, dict) else None
    if event:
        widgets.append(
            Widget(
                type="hero_goal",
                props=_post_process_widget(
                    Widget(
                        type="hero_goal",
                        props=HeroGoalProps(
                            event=str(event),
                            target_date=goal.get("target_date"),
                            target_time=_strip_leading_zero_time(
                                goal.get("target_time")
                            ),
                        ).model_dump(),
                    )
                ).props,
            )
        )

    fitness = structured.get("fitness") or {}
    if isinstance(fitness, dict):
        stats: list[StatGridItem] = []
        vo2 = fitness.get("estimated_vo2max")
        if vo2 is not None:
            stats.append(
                StatGridItem(
                    label="VO2max",
                    value=str(vo2),
                    unit="ml/kg/min",
                    icon="activity",
                )
            )
        threshold = fitness.get("threshold_pace_min_km")
        if threshold is not None:
            stats.append(
                StatGridItem(
                    label="Threshold-Pace",
                    value=str(threshold),
                    unit="min/km",
                    icon="timer",
                )
            )
        weekly = fitness.get("weekly_volume_km")
        if weekly is not None:
            stats.append(
                StatGridItem(
                    label="Wochenumfang",
                    value=str(weekly),
                    unit="km",
                    icon="footprints",
                )
            )
        ftp = fitness.get("ftp_watts")
        if ftp is not None:
            stats.append(
                StatGridItem(label="FTP", value=str(ftp), unit="W", icon="zap")
            )
        if stats:
            widgets.append(
                Widget(
                    type="stat_grid",
                    props=StatGridProps(
                        title="Performance",
                        stats=stats[:_MAX_STATS_PER_GRID],
                    ).model_dump(),
                )
            )

    # If the journal is empty, surface a friendly nudge as info_card.
    if not markdown.strip():
        widgets.append(
            Widget(
                type="info_card",
                props=InfoCardProps(
                    title="Profil noch leer",
                    description=(
                        "Erzaehl mir, was du erreichen willst, und ich richte "
                        "dir dein Profil ein."
                    ),
                ).model_dump(),
            )
        )

    hints = _normalize_hints({}, widgets)
    return WidgetsResponse(
        widgets=widgets,
        generated_at=_now_iso(),
        cache_hit=False,
        edit_hint_per_widget=hints,
    )


__all__ = [
    "ChecklistItem",
    "ChecklistProps",
    "ChipListItem",
    "ChipListProps",
    "HeroGoalProps",
    "InfoCardFact",
    "InfoCardProps",
    "ProfileHeaderProps",
    "QuoteCardProps",
    "StatGridItem",
    "StatGridProps",
    "TimelineItem",
    "TimelineProps",
    "Widget",
    "WidgetsResponse",
    "get_identity_widgets",
]
