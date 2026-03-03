"""
Chat history and conversation summary management.

Handles the rolling window of recent messages and the LLM-generated
summary of older turns that have been evicted from the window.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

# Maximum recent turns to keep in full before summarising older ones.
# Each "turn" is one user + one assistant message.
MAX_RECENT_TURNS = 4  # 4 pairs = 8 messages


def init_session_state() -> None:
    """Initialise session state keys on first load."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "graph_name" not in st.session_state:
        st.session_state.graph_name = "rag_graph"
    if "conversation_summary" not in st.session_state:
        st.session_state.conversation_summary = ""


def build_chat_state() -> dict[str, Any]:
    """
    Build the chat_history and conversation_summary fields from
    Streamlit session state for injection into the graph.

    Policy:
    - Keep the last ``MAX_RECENT_TURNS`` user+assistant pairs verbatim.
    - Older turns are represented only by ``conversation_summary``
      (which is updated each time we evict turns).

    Returns:
        Dict with ``chat_history`` and ``conversation_summary`` keys.
    """
    messages = st.session_state.get("messages", [])

    # Extract user/assistant pairs
    chat_turns: list[dict[str, str]] = []
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role in ("user", "assistant") and content:
            chat_turns.append({"role": role, "content": content})

    # No history → first turn
    if not chat_turns:
        return {"chat_history": [], "conversation_summary": ""}

    # Split into "old" (to summarise) and "recent" (to keep verbatim)
    max_messages = MAX_RECENT_TURNS * 2  # pairs → individual messages
    recent = chat_turns[-max_messages:]

    # Keep the existing rolling summary
    conversation_summary = st.session_state.get("conversation_summary", "")

    return {
        "chat_history": recent,
        "conversation_summary": conversation_summary,
    }


def update_conversation_summary(model, cfg) -> None:
    """
    If the conversation has grown beyond the recent-turn window,
    generate a rolling summary of the older turns.

    This is called after each assistant response is added.
    """
    messages = st.session_state.get("messages", [])
    max_messages = MAX_RECENT_TURNS * 2

    # Only summarise when we have more turns than the window
    if len(messages) <= max_messages + 2:
        return

    # Gather the turns that are about to be evicted from the window
    evicted = messages[-(max_messages + 2) : -max_messages]
    if not evicted:
        return

    existing_summary = st.session_state.get("conversation_summary", "")

    # Build a simple summary prompt
    evicted_text = "\n".join(
        f"{m['role'].capitalize()}: {m['content'][:300]}" for m in evicted
    )

    summary_prompt = (
        "Summarise the following conversation turns into 2-3 sentences. "
        "Preserve key topics, entities, and conclusions.\n\n"
    )
    if existing_summary:
        summary_prompt += f"Previous summary: {existing_summary}\n\n"
    summary_prompt += f"New turns:\n{evicted_text}\n\nUpdated summary:"

    try:
        response = model.generate(
            [{"role": "user", "content": summary_prompt}],
            temperature=0.2,
            max_tokens=200,
        )
        st.session_state["conversation_summary"] = response.text.strip()
    except Exception:
        # Summary failure is non-critical — keep existing summary
        pass
