"""
Action dispatcher: maps intent + entities to robot actions.
"""

import datetime
from typing import Dict, Optional

# Simple in-memory session state
_session = {
    "active": False,
    "start_time": None,
    "breaks": 0,
    "total_study_seconds": 0,
}


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
        "GET_STATS": _handle_get_stats,
        "BREAK": _handle_break,
        "NAVIGATE": _handle_navigate,
        "RAG_QUERY": _handle_rag_query,
        "UNKNOWN": _handle_unknown,
    }

    handler = handlers.get(intent, _handle_unknown)
    return handler(entities=entities, text=text)


def _handle_start_session(**kwargs) -> str:
    if _session["active"]:
        return "Session is already active."
    _session["active"] = True
    _session["start_time"] = datetime.datetime.now()
    _session["breaks"] = 0
    return "Session started. Good luck studying!"


def _handle_stop_session(**kwargs) -> str:
    if not _session["active"]:
        return "No active session to stop."
    elapsed = (datetime.datetime.now() - _session["start_time"]).total_seconds()
    _session["total_study_seconds"] += elapsed
    _session["active"] = False
    _session["start_time"] = None
    return f"Session ended. Duration: {elapsed:.0f}s."


def _handle_get_stats(**kwargs) -> str:
    current = 0
    if _session["active"] and _session["start_time"]:
        current = (datetime.datetime.now() - _session["start_time"]).total_seconds()
    total = _session["total_study_seconds"] + current
    return (
        f"Total study time: {total:.0f}s | "
        f"Breaks taken: {_session['breaks']} | "
        f"Session active: {_session['active']}"
    )


def _handle_break(**kwargs) -> str:
    if not _session["active"]:
        return "No active session. Start a session first."
    _session["breaks"] += 1
    return f"Break #{_session['breaks']}. Take your time!"


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
    if not text.strip():
        return "No RAG query provided."

    try:
        import requests
    except ImportError:
        return "Error contacting RAG system: requests package is not installed."

    try:
        response = requests.post(
            "http://127.0.0.1:8000/ask",
            json={"query": text},
            timeout=10,
        )

        # The current backend may expect a query parameter instead of JSON.
        if response.status_code == 422:
            response = requests.post(
                "http://127.0.0.1:8000/ask",
                params={"query": text},
                timeout=10,
            )

        response.raise_for_status()

        data = response.json()
        if isinstance(data, dict):
            if data.get("answer"):
                return str(data["answer"])
            if data.get("error"):
                return f"RAG system error: {data['error']}"

        return "No answer returned from RAG system."
    except requests.RequestException as e:
        return f"Error contacting RAG system: {str(e)}"
    except ValueError:
        return "Error contacting RAG system: invalid JSON response."


def _handle_rag_query(text: str = "", **kwargs) -> str:
    return handle_rag_query(text)


def _handle_unknown(text: str = "", **kwargs) -> str:
    # Caller (main.py) will invoke chatbot for UNKNOWN intent
    return "CHATBOT_FALLBACK"
