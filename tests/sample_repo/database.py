import sqlite3
from typing import Any, List, Optional

DATABASE_URL = "sqlite:///./app.db"


def create_connection(db_path: str = "./app.db") -> sqlite3.Connection:
    """Create and return a raw SQLite database connection."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


class DatabaseManager:
    """Connection manager supporting queries and transactions."""

    def __init__(self, db_path: str = "./app.db"):
        self.db_path = db_path
        self._connection: Optional[sqlite3.Connection] = None

    def connect(self) -> None:
        """Establish database connection pool."""
        if not self._connection:
            self._connection = create_connection(self.db_path)

    def execute_query(self, query: str, params: tuple = ()) -> List[Any]:
        """Execute a parameterized query and return fetched rows."""
        self.connect()
        cursor = self._connection.cursor()
        cursor.execute(query, params)
        return cursor.fetchall()

    def disconnect(self) -> None:
        """Close active database connection."""
        if self._connection:
            self._connection.close()
            self._connection = None
