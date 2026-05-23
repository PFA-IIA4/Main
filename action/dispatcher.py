"""
Action dispatcher: maps intent + entities to robot actions.
"""

import datetime
import os
from typing import Dict, Optional

# Simple in-memory session state
_session = {
    "active": False,
    "on_break": False,
    "start_time": None,
    "breaks": 0,
    "total_study_seconds": 0,
    "current_session_seconds": 0,
}

SESSION_STARTED_MESSAGE = (
    "Session started. Say 'break' to pause or 'stop session' to end it."
)
SESSION_ALREADY_ACTIVE_MESSAGE = "Robot is currently in session. Say 'stop session' to end it."
SESSION_ON_BREAK_MESSAGE = "Session is currently on break. Say 'resume session' to continue."
RAG_SESSION_REQUIRED_MESSAGE = (
    "Robot is currently not in session. Start a session first to ask RAG questions."
)

RAG_BASE_URL = os.getenv("RAG_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
RAG_ASK_PATH = os.getenv("RAG_ASK_PATH", "/ask")


def _get_rag_timeout_seconds(default: float = 40.0) -> float:
    raw_value = os.getenv("RAG_TIMEOUT_SECONDS", str(default)).strip()
    try:
        timeout = float(raw_value)
        return timeout if timeout > 0 else default
    except ValueError:
        return default


def _build_rag_ask_url() -> str:
    ask_path = RAG_ASK_PATH if RAG_ASK_PATH.startswith("/") else f"/{RAG_ASK_PATH}"
    return f"{RAG_BASE_URL}{ask_path}"


def _extract_answer_from_rag_payload(data: object) -> Optional[str]:
    if not isinstance(data, dict):
        return None

    for key in ("answer", "response", "result", "output", "text"):
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def dispatch(intent: str, entities: Optional[Dict] = None, text: str = "") -> str:
    """
    Execute an action based on intent and extracted entities.

    Parameters
    ----------
    intent : str
        Classified intent label.
    entities : dict, optional
        Extracted entities (distance, angle) for NAVIGATE.
    text : str
        Original recognized text (used for chatbot fallback).

    Returns
    -------
    str
        Human-readable action result message.
    """
    handlers = {
        "START_SESSION": _handle_start_session,
        "STOP_SESSION": _handle_stop_session,
        "RESUME_SESSION": _handle_resume_session,
        "GET_STATS": _handle_get_stats,
        "SMALL_TALK": _handle_small_talk,
        "BREAK": _handle_break,
        "NAVIGATE": _handle_navigate,
        "RAG_QUERY": _handle_rag_query,
        "UNKNOWN": _handle_unknown,
    }

    handler = handlers.get(intent, _handle_unknown)
    return handler(entities=entities, text=text)


def _handle_start_session(**kwargs) -> str:
    if _session["active"]:
        if _session["on_break"]:
            return SESSION_ON_BREAK_MESSAGE
        return SESSION_ALREADY_ACTIVE_MESSAGE
    _session["active"] = True
    _session["on_break"] = False
    _session["start_time"] = datetime.datetime.now()
    _session["breaks"] = 0
    _session["current_session_seconds"] = 0
    return SESSION_STARTED_MESSAGE


def _handle_stop_session(**kwargs) -> str:
    if not _session["active"]:
        return "No active session to stop."
    elapsed = 0.0
    if _session["start_time"]:
        elapsed = (datetime.datetime.now() - _session["start_time"]).total_seconds()
        _session["total_study_seconds"] += elapsed
        _session["current_session_seconds"] += elapsed
    session_duration = _session["current_session_seconds"]
    _session["active"] = False
    _session["on_break"] = False
    _session["start_time"] = None
    _session["current_session_seconds"] = 0
    return f"Session ended. Duration: {session_duration:.0f}s."


def _handle_get_stats(**kwargs) -> str:
    current = 0
    if _session["active"] and _session["start_time"]:
        current = (datetime.datetime.now() - _session["start_time"]).total_seconds()
    total = _session["total_study_seconds"] + current
    return (
        f"Total study time: {total:.0f}s | "
        f"Breaks taken: {_session['breaks']} | "
        f"Session active: {_session['active']} | "
        f"On break: {_session['on_break']}"
    )


def _handle_break(**kwargs) -> str:
    if not _session["active"]:
        return "No active session. Start a session first."
    if _session["on_break"]:
        return "Break already in progress. Say 'resume session' when you're ready."
    if _session["start_time"]:
        elapsed = (datetime.datetime.now() - _session["start_time"]).total_seconds()
        _session["total_study_seconds"] += elapsed
        _session["current_session_seconds"] += elapsed
        _session["start_time"] = None
    _session["breaks"] += 1
    _session["on_break"] = True
    return f"Break #{_session['breaks']}. Say 'resume session' when you're ready."


def _handle_resume_session(**kwargs) -> str:
    if not _session["active"]:
        return "No active session to resume. Start a session first."
    if not _session["on_break"]:
        return "Session already active."
    _session["on_break"] = False
    _session["start_time"] = datetime.datetime.now()
    return "Break ended. Back to work!"


def _handle_navigate(entities: Optional[Dict] = None, **kwargs) -> str:
    if entities is None:
        return "Navigation failed: no entities provided."
    distance = entities.get("distance")
    angle = entities.get("angle")
    parts = []
    if distance is not None:
        parts.append(f"Moving {distance}m forward")
    if angle is not None:
        parts.append(f"Turning {angle}°")
    if not parts:
        return "Navigation failed: missing distance and angle."
    # In production this would send commands to the ESP32
    return " | ".join(parts) + " [command sent to ESP32]"


def handle_rag_query(text: str) -> str:
    """Forward question text to the external RAG backend over HTTP."""
    if not _session["active"]:
        return RAG_SESSION_REQUIRED_MESSAGE
    if not text.strip():
        return "No RAG query provided."

    try:
        import requests
    except ImportError:
        return "Error contacting RAG system: requests package is not installed."

    ask_url = _build_rag_ask_url()
    timeout_seconds = _get_rag_timeout_seconds()

    def _parse_response(response):
        response.raise_for_status()
        data = response.json()

        answer = _extract_answer_from_rag_payload(data)
        if answer:
            return answer

        if isinstance(data, dict) and data.get("error"):
            return f"RAG system error: {data['error']}"

        return "No answer returned from RAG system."

    try:
        response = requests.post(
            ask_url,
            json={"query": text},
            timeout=timeout_seconds,
        )

        # Some backends accept only query params for /ask.
        if response.status_code in (405, 422):
            response = requests.post(
                ask_url,
                params={"query": text},
                timeout=timeout_seconds,
            )

        # Some backends expose GET /ask?query=... only.
        if response.status_code == 405:
            response = requests.get(
                ask_url,
                params={"query": text},
                timeout=timeout_seconds,
            )

        return _parse_response(response)
    except requests.RequestException as e:
        return f"Error contacting RAG system: {str(e)}"
    except ValueError:
        return "Error contacting RAG system: invalid JSON response."


def _handle_rag_query(text: str = "", **kwargs) -> str:
    return handle_rag_query(text)


def _handle_small_talk(text: str = "", **kwargs) -> str:
    # Route conversational phrases to chatbot, similar to UNKNOWN.
    return "CHATBOT_FALLBACK"


def _handle_unknown(text: str = "", **kwargs) -> str:
    # Caller (main.py) will invoke chatbot for UNKNOWN intent
    return "CHATBOT_FALLBACK"
