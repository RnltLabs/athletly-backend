"""Tests for the complexity detector (Sprint E premium routing).

The detector decides whether a single user message gets the Sonnet +
extended-thinking treatment for the whole turn. Two things matter:

1. The Lisa-case must escalate (triathlon + race strategy + FTP math).
2. Routine turns must NOT escalate, otherwise we burn Sonnet budget.

Each test pins one direction of the decision against a concrete
message. Detector is a pure function, no fixtures needed.
"""

from __future__ import annotations

import pytest

from src.agent.complexity_detector import needs_complex_reasoning


# ---------------------------------------------------------------------------
# Triggers: messages that MUST route to complex.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("message", [
    # Lisa case shape: triathlon + race strategy + FTP math.
    "Ich bereite mich auf Roth vor. Wie pace ich die Langdistanz "
    "bei FTP 240 W?",
    # Multi-sport keyword alone triggers (Group A).
    "Mein erster Triathlon kommt im August, wie sehe ich aus?",
    # FTP + number-comparison.
    "Mein FTP ist 240 W, kann ich 280 W NP halten ueber 1h?",
    # 16-week marathon plan = long-horizon plan.
    "Plane mir einen 16-Wochen Marathonaufbau bitte.",
    # Triathlon + nutrition (B + A).
    "Triathlon Ernaehrung waehrend Race?",
    # CSS calculation question (Group C only with arithmetic question).
    "Wie berechne ich CSS aus 400m und 200m Zeit?",
    # Hyrox pacing (A + B).
    "Hyrox Pacing Strategie fuer den ersten Wettkampf?",
    # Ironman (A) alone.
    "Ich will einen Ironman in 18 Monaten machen.",
    # VDOT mentioned (C) + planning context (D).
    "VDOT 50, baue mir einen 12 Wochen Aufbauplan.",
    # 70.3 (A) alone.
    "Was ist eine realistische Zeit fuer mein erstes 70.3?",
    # Karvonen mentioned.
    "Berechne meine Karvonen Zonen, Ruhepuls 50, Max 190.",
])
def test_trigger_cases_route_to_complex(message: str) -> None:
    is_complex, matched = needs_complex_reasoning(message)
    assert is_complex, f"expected complex routing for: {message!r} (matched={matched})"
    assert matched, "matched keyword list must be non-empty when triggered"


def test_has_recent_long_plan_request_forces_complex() -> None:
    """The plan-history flag is an explicit caller override."""
    msg = "Wie geht es mir heute?"
    is_complex, matched = needs_complex_reasoning(
        msg, has_recent_long_plan_request=True,
    )
    assert is_complex
    assert "recent_long_plan_request" in matched


# ---------------------------------------------------------------------------
# Non-triggers: messages that MUST stay routine.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("message", [
    "Wie fuehle ich mich heute? Ich bin etwas muede.",
    "Hallo, mein Name ist Lisa und ich freue mich auf das Coaching.",
    "Ich war gestern laufen, 8 km easy in der Mittagspause.",
    "Zeig mir meine letzten Aktivitaeten bitte.",
    "Ich bin heute ueberhaupt nicht motiviert.",
    "Mein Knie zwickt seit gestern Abend ein bisschen.",
    "Wann ist mein naechster Termin?",
    "Hey, alles gut bei dir?",
    "Habe heute morgen 30 Minuten Krafttraining gemacht.",
    # Pace mention alone must NOT escalate (false-positive guard).
    "Ich lief heute 5:00 pro km, schoen gleichmaessig.",
])
def test_non_trigger_cases_stay_routine(message: str) -> None:
    is_complex, _matched = needs_complex_reasoning(message)
    assert not is_complex, f"expected routine routing for: {message!r}"


def test_short_messages_skip_detection() -> None:
    """Messages under the min length threshold never escalate."""
    is_complex, matched = needs_complex_reasoning("FTP?")
    assert not is_complex
    assert matched == []


def test_empty_string_does_not_escalate() -> None:
    assert needs_complex_reasoning("") == (False, [])


def test_non_string_does_not_crash() -> None:
    is_complex, matched = needs_complex_reasoning(None)  # type: ignore[arg-type]
    assert not is_complex
    assert matched == []


# ---------------------------------------------------------------------------
# Audit trail: matched keywords must reflect what fired.
# ---------------------------------------------------------------------------


def test_matched_keywords_contain_sport_token() -> None:
    _, matched = needs_complex_reasoning(
        "Triathlon Saisonplanung mit Ernaehrungsstrategie",
    )
    joined = " ".join(matched)
    assert "triathlon" in joined.lower()


def test_matched_keywords_contain_ftp_symbol() -> None:
    _, matched = needs_complex_reasoning(
        "Mein FTP ist 240 W, was sind meine Zonen?",
    )
    joined = " ".join(matched).lower()
    assert "ftp" in joined


def test_matched_keywords_deterministic_order() -> None:
    """Order is A -> B -> C -> D -> E for stable log lines."""
    _, matched = needs_complex_reasoning(
        "Triathlon Ernaehrung mit FTP 240 W, 12 weeks Aufbau, wie schnell?"
    )
    # Sport token (Group A) must come before sport-math (C) which must
    # come before long-horizon (D). We can't assert exact strings but
    # the relative positions are testable.
    lower_matched = [m.lower() for m in matched]
    assert "triathlon" in lower_matched
    assert any("ftp" in m for m in lower_matched)
    triathlon_idx = lower_matched.index("triathlon")
    ftp_idx = next(i for i, m in enumerate(lower_matched) if "ftp" in m)
    assert triathlon_idx < ftp_idx


# ---------------------------------------------------------------------------
# Cost guard: tight false-positive rate on routine sport chatter.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("message", [
    "Heute 60 Minuten geradelt, schoen entspannt.",
    "Mein Plan fuer naechste Woche steht bereits.",
    "Habe heute 4 Wiederholungen gemacht, war hart.",
    "Mein langer Lauf am Sonntag war 18 km.",
    "Bin seit Montag krank, kein Training.",
])
def test_routine_sport_chatter_stays_routine(message: str) -> None:
    """Casual training updates without sport-math / strategy MUST not escalate."""
    is_complex, _matched = needs_complex_reasoning(message)
    assert not is_complex, f"expected routine routing for: {message!r}"
