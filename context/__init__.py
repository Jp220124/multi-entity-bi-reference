"""Shared context layer.

Every agent reads from and writes to the context layer through the
``ContextStore`` interface. The default implementation is SQLite for
zero-setup local development; a Supabase implementation is a drop-in
replacement.
"""

from .store import ContextStore, SQLiteContextStore

__all__ = ["ContextStore", "SQLiteContextStore"]
