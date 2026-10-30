"""SQLite connection helpers for short-lived evolution operations."""

from __future__ import annotations

import sqlite3


class ClosingSQLiteConnection(sqlite3.Connection):
    """Make SQLite's context-manager usage also release its file handle."""

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            return super().__exit__(exc_type, exc_value, traceback)
        finally:
            self.close()
