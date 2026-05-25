"""
modules/chat_manager.py — Module 10: Multi-Session Chat Manager
────────────────────────────────────────────────────────────────
Responsibilities:
  - Create, switch, rename, and delete named chat sessions
  - Persist chat history per session in Streamlit session_state
  - Track which documents are associated with each chat session
  - Provide a clean API consumed by app.py

Data model stored in st.session_state["chat_sessions"]:
  {
    "<session_id>": {
      "id": str,
      "name": str,
      "created_at": float (epoch),
      "messages": [{"role": str, "content": str, "citations": list}, ...],
      "pinned_sources": list[str],   # documents scoped to this chat
    },
    ...
  }

  st.session_state["active_chat_id"] = "<session_id>"
"""

import uuid
import time
import logging
from typing import List, Optional, Dict, Any

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────

SESSIONS_KEY = "chat_sessions"
ACTIVE_KEY = "active_chat_id"
DEFAULT_CHAT_NAME = "New Chat"
MAX_SESSIONS = 20  # Guard against unbounded growth


# ──────────────────────────────────────────────
# ChatManager
# ──────────────────────────────────────────────

class ChatManager:
    """
    Manages multiple named chat sessions inside Streamlit's session_state.

    All methods operate on st.session_state directly so state persists
    across Streamlit reruns without any external database.

    Usage (inside app.py):
        from modules.chat_manager import ChatManager
        cm = ChatManager()
        cm.ensure_default_session()

        # Create a new chat
        sid = cm.new_session("Research Chat")
        cm.set_active(sid)

        # Add a message
        cm.add_message(role="user", content="Hello")
        cm.add_message(role="assistant", content="Hi!", citations=[...])

        # Read history
        history = cm.active_messages()
    """

    def __init__(self):
        # Import here to avoid issues if used outside Streamlit context in tests
        import streamlit as st
        self._st = st
        self._ensure_state()

    # ──────────────────────────────────────────
    # Bootstrapping
    # ──────────────────────────────────────────

    def _ensure_state(self):
        """Initialize session_state keys if they don't exist yet."""
        if SESSIONS_KEY not in self._st.session_state:
            self._st.session_state[SESSIONS_KEY] = {}
        if ACTIVE_KEY not in self._st.session_state:
            self._st.session_state[ACTIVE_KEY] = None

    def ensure_default_session(self) -> str:
        """
        Guarantee at least one chat session exists and one is active.
        Call this once at the top of app.py on every rerun.

        Returns:
            Active session ID.
        """
        sessions = self._sessions()
        if not sessions:
            sid = self._create_session(DEFAULT_CHAT_NAME)
            self._set_active(sid)
        elif self._st.session_state[ACTIVE_KEY] not in sessions:
            # Active ID is stale (e.g. after deletion)
            self._set_active(next(iter(sessions)))
        return self._st.session_state[ACTIVE_KEY]

    # ──────────────────────────────────────────
    # Session CRUD
    # ──────────────────────────────────────────

    def new_session(self, name: Optional[str] = None) -> str:
        """
        Create a new empty chat session and make it active.

        Args:
            name: Display name. Defaults to 'New Chat N'.

        Returns:
            New session ID.
        """
        sessions = self._sessions()
        if len(sessions) >= MAX_SESSIONS:
            # Remove oldest session to stay within limit
            oldest_id = min(sessions, key=lambda k: sessions[k]["created_at"])
            self.delete_session(oldest_id)

        if not name:
            n = len(self._sessions()) + 1
            name = f"Chat {n}"

        sid = self._create_session(name)
        self._set_active(sid)
        logger.info(f"New chat session created: {sid!r} ({name!r})")
        return sid

    def delete_session(self, session_id: str) -> bool:
        """
        Delete a session. Switches active to another session if needed.

        Returns:
            True if deleted, False if not found.
        """
        sessions = self._sessions()
        if session_id not in sessions:
            return False

        del sessions[session_id]
        logger.info(f"Session {session_id!r} deleted.")

        # Reassign active if needed
        if self._st.session_state[ACTIVE_KEY] == session_id:
            if sessions:
                self._set_active(next(iter(sessions)))
            else:
                sid = self._create_session(DEFAULT_CHAT_NAME)
                self._set_active(sid)

        return True

    def rename_session(self, session_id: str, new_name: str) -> bool:
        """
        Rename a chat session.

        Returns:
            True if renamed, False if session not found.
        """
        sessions = self._sessions()
        if session_id not in sessions:
            return False
        sessions[session_id]["name"] = new_name.strip() or DEFAULT_CHAT_NAME
        return True

    def set_active(self, session_id: str) -> bool:
        """
        Switch the active chat window.

        Returns:
            True if switched, False if session_id unknown.
        """
        if session_id not in self._sessions():
            return False
        self._set_active(session_id)
        return True

    def clear_session(self, session_id: Optional[str] = None):
        """
        Clear all messages in a session (keep the session itself).

        Args:
            session_id: Defaults to the active session.
        """
        sid = session_id or self.active_id()
        session = self._get_session(sid)
        if session:
            session["messages"] = []

    # ──────────────────────────────────────────
    # Messages
    # ──────────────────────────────────────────

    def add_message(
        self,
        role: str,
        content: str,
        citations: Optional[List[dict]] = None,
        session_id: Optional[str] = None,
    ):
        """
        Append a message to the session's history.

        Args:
            role: 'user' or 'assistant'.
            content: Message text.
            citations: Optional list of citation dicts.
            session_id: Defaults to the active session.
        """
        sid = session_id or self.active_id()
        session = self._get_session(sid)
        if session is None:
            logger.warning(f"add_message: session {sid!r} not found.")
            return

        msg: Dict[str, Any] = {"role": role, "content": content}
        if citations:
            msg["citations"] = citations
        session["messages"].append(msg)

    def active_messages(self) -> List[dict]:
        """Return messages for the currently active chat."""
        session = self._get_session(self.active_id())
        return session["messages"] if session else []

    def session_messages(self, session_id: str) -> List[dict]:
        """Return messages for a specific session."""
        session = self._get_session(session_id)
        return session["messages"] if session else []

    # ──────────────────────────────────────────
    # Document pinning (per-chat source filter)
    # ──────────────────────────────────────────

    def pin_source(self, source: str, session_id: Optional[str] = None):
        """Restrict this chat session to a specific document."""
        sid = session_id or self.active_id()
        session = self._get_session(sid)
        if session and source not in session["pinned_sources"]:
            session["pinned_sources"].append(source)

    def unpin_source(self, source: str, session_id: Optional[str] = None):
        """Remove a pinned source from this chat session."""
        sid = session_id or self.active_id()
        session = self._get_session(sid)
        if session and source in session["pinned_sources"]:
            session["pinned_sources"].remove(source)

    def pinned_sources(self, session_id: Optional[str] = None) -> List[str]:
        """Return the list of pinned sources for a session."""
        sid = session_id or self.active_id()
        session = self._get_session(sid)
        return session["pinned_sources"] if session else []

    # ──────────────────────────────────────────
    # Read-only accessors
    # ──────────────────────────────────────────

    def active_id(self) -> Optional[str]:
        """Return the active session ID."""
        return self._st.session_state.get(ACTIVE_KEY)

    def active_session(self) -> Optional[dict]:
        """Return the full active session dict."""
        return self._get_session(self.active_id())

    def all_sessions(self) -> Dict[str, dict]:
        """Return all sessions as {id: session_dict}."""
        return dict(self._sessions())

    def session_list(self) -> List[dict]:
        """
        Return sessions as a list sorted by creation time (newest first).
        Each entry: {id, name, message_count, created_at, pinned_sources}
        """
        sessions = self._sessions()
        result = []
        for sid, s in sessions.items():
            result.append({
                "id": sid,
                "name": s["name"],
                "message_count": len(s["messages"]),
                "created_at": s["created_at"],
                "pinned_sources": s["pinned_sources"],
            })
        result.sort(key=lambda x: x["created_at"], reverse=True)
        return result

    def session_count(self) -> int:
        return len(self._sessions())

    # ──────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────

    def _sessions(self) -> Dict[str, dict]:
        return self._st.session_state[SESSIONS_KEY]

    def _get_session(self, session_id: Optional[str]) -> Optional[dict]:
        if not session_id:
            return None
        return self._sessions().get(session_id)

    def _create_session(self, name: str) -> str:
        sid = str(uuid.uuid4())[:8]
        self._sessions()[sid] = {
            "id": sid,
            "name": name,
            "created_at": time.time(),
            "messages": [],
            "pinned_sources": [],
        }
        return sid

    def _set_active(self, session_id: str):
        self._st.session_state[ACTIVE_KEY] = session_id
