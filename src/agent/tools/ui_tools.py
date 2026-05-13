"""Generative UI tools - request the frontend to render inline input widgets.

Generative UI lets the agent surface structured input components (choice
chips, number steppers, date pickers, confirm buttons) directly in the chat
instead of asking free-form questions. The user taps an option and the
frontend sends the choice back as a regular user message; the agent then
continues the conversation as if the user had typed the answer.

Convention: any tool that wants to render a UI component returns a dict
like::

    {
        "status": "rendered",
        "_ui_component": {
            "type": "choice_single",
            "id": "<uuid>",
            "props": {"question": "...", "options": [...]},
        },
    }

The agent loop inspects ``_ui_component`` after the tool runs and emits a
matching ``ui_component`` SSE event via ``on_progress``. The marker is
stripped from the dict that the model sees so internal plumbing does not
pollute the chat history. The model only sees ``{"status": "rendered"}``,
confirming the component has been surfaced.

These tools are pure UI surfaces. They carry no coaching logic and impose
no rules. The agent decides when and how to use them.
"""

from __future__ import annotations

import uuid

from src.agent.tools.registry import Tool, ToolRegistry


def _new_ui_id() -> str:
    """Generate a fresh hex UUID for a UI component instance."""
    return uuid.uuid4().hex


def register_ui_tools(registry: ToolRegistry, user_model=None) -> None:
    """Register generative UI tools into the registry."""
    del user_model  # currently unused; kept for signature parity

    # -- ask_choice ---------------------------------------------------------

    def ask_choice(
        question: str,
        options: list[str],
        multi: bool = False,
        submit_label: str | None = None,
    ) -> dict:
        """Render a choice card (single or multi-select chips) in the chat."""
        component_type = "choice_multi" if multi else "choice_single"
        props: dict = {
            "question": question,
            "options": list(options),
        }
        # Multi-select always needs a submit label; default to "Weiter".
        # Single-select can omit it (chip tap submits immediately).
        if multi:
            props["submit_label"] = submit_label or "Weiter"
        elif submit_label:
            props["submit_label"] = submit_label

        return {
            "status": "rendered",
            "_ui_component": {
                "type": component_type,
                "id": _new_ui_id(),
                "props": props,
            },
        }

    registry.register(Tool(
        name="ask_choice",
        description=(
            "Render a choice card with chip buttons in the chat for the "
            "user to tap. Use this for structured questions where the set "
            "of valid answers is known and small enough to display as "
            "chips (sport selection, training days, fixed enums). Set "
            "multi=true to allow multiple selections. DO NOT use this for "
            "free-text answers like name or goal description - those need "
            "a plain prose question instead. The user's tap will arrive "
            "as a normal user message in the next turn."
        ),
        handler=ask_choice,
        parameters={
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The question shown above the chips.",
                },
                "options": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Chip labels offered to the user.",
                },
                "multi": {
                    "type": "boolean",
                    "description": (
                        "If true, the user can pick multiple options and "
                        "submits via a button. If false (default), tapping "
                        "a chip submits immediately."
                    ),
                },
                "submit_label": {
                    "type": "string",
                    "description": (
                        "Label for the submit button. Defaults to 'Weiter' "
                        "for multi-select; omitted for single-select."
                    ),
                },
            },
            "required": ["question", "options"],
        },
        category="meta",
    ))

    # -- ask_number ---------------------------------------------------------

    def ask_number(
        question: str,
        min: int,
        max: int,
        step: int = 1,
        initial: int | None = None,
        unit: str | None = None,
        submit_label: str = "Weiter",
    ) -> dict:
        """Render a number stepper/slider for picking an integer in a range."""
        props: dict = {
            "question": question,
            "min": min,
            "max": max,
            "step": step,
            "initial": initial if initial is not None else min,
            "submit_label": submit_label,
        }
        if unit:
            props["unit"] = unit

        return {
            "status": "rendered",
            "_ui_component": {
                "type": "number_stepper",
                "id": _new_ui_id(),
                "props": props,
            },
        }

    registry.register(Tool(
        name="ask_number",
        description=(
            "Render a number stepper/slider in the chat for the user to "
            "pick an integer in a range. Use this for bounded numeric "
            "questions where the user should not have to type a number "
            "manually. Examples: 'Wie viele Trainingstage pro Woche?' "
            "(min=1, max=7), 'Max. Session-Dauer?' (min=20, max=240, "
            "step=10, unit='Min'). The user's choice will arrive as a "
            "normal user message in the next turn."
        ),
        handler=ask_number,
        parameters={
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The question shown above the stepper.",
                },
                "min": {
                    "type": "integer",
                    "description": "Smallest valid value (inclusive).",
                },
                "max": {
                    "type": "integer",
                    "description": "Largest valid value (inclusive).",
                },
                "step": {
                    "type": "integer",
                    "description": "Increment between values. Defaults to 1.",
                },
                "initial": {
                    "type": "integer",
                    "description": (
                        "Pre-selected value when the stepper appears. "
                        "Defaults to min."
                    ),
                },
                "unit": {
                    "type": "string",
                    "description": (
                        "Unit label shown next to the number, e.g. 'Min' "
                        "or 'Tage'. Optional."
                    ),
                },
                "submit_label": {
                    "type": "string",
                    "description": "Submit button label. Defaults to 'Weiter'.",
                },
            },
            "required": ["question", "min", "max"],
        },
        category="meta",
    ))

    # -- ask_date -----------------------------------------------------------

    def ask_date(
        question: str,
        min_date: str | None = None,
        max_date: str | None = None,
        initial_date: str | None = None,
        submit_label: str = "Weiter",
    ) -> dict:
        """Render a date picker for selecting a single ISO date."""
        props: dict = {
            "question": question,
            "submit_label": submit_label,
        }
        if min_date:
            props["min_date"] = min_date
        if max_date:
            props["max_date"] = max_date
        if initial_date:
            props["initial_date"] = initial_date

        return {
            "status": "rendered",
            "_ui_component": {
                "type": "date_picker",
                "id": _new_ui_id(),
                "props": props,
            },
        }

    registry.register(Tool(
        name="ask_date",
        description=(
            "Render a date picker in the chat for the user to select a "
            "single date. Use this for any date question (target race, "
            "start date, deadline). Pass min_date as today's ISO date "
            "(YYYY-MM-DD) when the answer must be in the future. The "
            "user's choice will arrive as a normal user message in the "
            "next turn."
        ),
        handler=ask_date,
        parameters={
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The question shown above the date picker.",
                },
                "min_date": {
                    "type": "string",
                    "description": (
                        "Earliest selectable date as ISO YYYY-MM-DD. "
                        "Optional."
                    ),
                },
                "max_date": {
                    "type": "string",
                    "description": (
                        "Latest selectable date as ISO YYYY-MM-DD. Optional."
                    ),
                },
                "initial_date": {
                    "type": "string",
                    "description": (
                        "Pre-selected date as ISO YYYY-MM-DD. Optional."
                    ),
                },
                "submit_label": {
                    "type": "string",
                    "description": "Submit button label. Defaults to 'Weiter'.",
                },
            },
            "required": ["question"],
        },
        category="meta",
    ))

    # -- ask_confirm --------------------------------------------------------

    def ask_confirm(
        question: str,
        confirm_label: str = "Ja",
        cancel_label: str = "Nein",
    ) -> dict:
        """Render a Ja/Nein confirm card with two action buttons."""
        props = {
            "question": question,
            "confirm_label": confirm_label,
            "cancel_label": cancel_label,
        }
        return {
            "status": "rendered",
            "_ui_component": {
                "type": "confirm",
                "id": _new_ui_id(),
                "props": props,
            },
        }

    registry.register(Tool(
        name="ask_confirm",
        description=(
            "Render a Ja/Nein confirm card with two buttons in the chat. "
            "Use this for binary yes/no questions where you want a clean "
            "tap response instead of free text. The user's choice will "
            "arrive as a normal user message in the next turn."
        ),
        handler=ask_confirm,
        parameters={
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The question shown above the buttons.",
                },
                "confirm_label": {
                    "type": "string",
                    "description": "Label for the positive button. Defaults to 'Ja'.",
                },
                "cancel_label": {
                    "type": "string",
                    "description": "Label for the negative button. Defaults to 'Nein'.",
                },
            },
            "required": ["question"],
        },
        category="meta",
    ))

    # -- ask_text_input -----------------------------------------------------

    def ask_text_input(
        question: str,
        fields: list[dict],
        submit_label: str = "Weiter",
    ) -> dict:
        """Render a multi-field free-text form inline in the chat."""
        # Normalize each field dict, applying defaults and dropping any
        # unexpected keys so the SSE payload stays tight.
        normalized_fields: list[dict] = []
        for field_def in fields:
            normalized: dict = {
                "name": field_def["name"],
                "label": field_def["label"],
                "type": field_def.get("type", "text"),
            }
            placeholder = field_def.get("placeholder")
            if placeholder is not None:
                normalized["placeholder"] = placeholder
            is_password = field_def.get("isPassword")
            if is_password is not None:
                normalized["isPassword"] = bool(is_password)
            normalized_fields.append(normalized)

        props = {
            "question": question,
            "fields": normalized_fields,
            "submit_label": submit_label,
        }

        return {
            "status": "rendered",
            "_ui_component": {
                "type": "text_input",
                "id": _new_ui_id(),
                "props": props,
            },
        }

    registry.register(Tool(
        name="ask_text_input",
        description=(
            "Render a multi-field form inline in the chat. Use this when "
            "you need structured free-text answers (e.g. 'Tell me your age "
            "and city') or when single-field input is more natural than "
            "chips. The user's answer comes back as a normal chat message "
            "in the form 'name=value, name=value, ...'. Do NOT use for "
            "Garmin connect (a dedicated tool handles that). Do NOT use "
            "for signup/login (frontend handles those)."
        ),
        handler=ask_text_input,
        parameters={
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The form heading shown above the fields.",
                },
                "fields": {
                    "type": "array",
                    "description": (
                        "One entry per input field. Each entry has 'name' "
                        "(key returned to the agent), 'label' (visible "
                        "label), optional 'placeholder', optional 'type' "
                        "(text|email|password, default text), and optional "
                        "'isPassword' (bool, default false)."
                    ),
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "label": {"type": "string"},
                            "placeholder": {"type": "string"},
                            "type": {
                                "type": "string",
                                "enum": ["text", "email", "password"],
                            },
                            "isPassword": {"type": "boolean"},
                        },
                        "required": ["name", "label"],
                    },
                },
                "submit_label": {
                    "type": "string",
                    "description": "Submit button label. Defaults to 'Weiter'.",
                },
            },
            "required": ["question", "fields"],
        },
        category="meta",
    ))
