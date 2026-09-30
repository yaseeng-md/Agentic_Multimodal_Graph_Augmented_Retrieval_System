from pathlib import Path
import sqlite3
from datetime import datetime, timezone

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Registry:
    """Small local registry for documents, versions and page hashes."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path))
        self.conn.row_factory = sqlite3.Row
        self._create_tables()

    def close(self) -> None:
        self.conn.close()

    def _create_tables(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS documents (
                document_id TEXT PRIMARY KEY,
                display_name TEXT NOT NULL,
                current_version_id TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS versions (
                version_id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL,
                version_number INTEGER NOT NULL,
                file_hash TEXT NOT NULL UNIQUE,
                original_filename TEXT NOT NULL,
                stored_pdf_path TEXT NOT NULL,
                page_count INTEGER,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                completed_at TEXT,
                FOREIGN KEY(document_id) REFERENCES documents(document_id)
            );

            CREATE TABLE IF NOT EXISTS pages (
                version_id TEXT NOT NULL,
                page_number INTEGER NOT NULL,
                page_hash TEXT NOT NULL,
                point_id TEXT NOT NULL,
                PRIMARY KEY(version_id, page_number),
                FOREIGN KEY(version_id) REFERENCES versions(version_id)
            );

            CREATE INDEX IF NOT EXISTS idx_versions_file_hash
                ON versions(file_hash);

            CREATE INDEX IF NOT EXISTS idx_pages_page_hash
                ON pages(page_hash);
            """
        )
        self.conn.commit()

    def get_by_file_hash(self, file_hash: str) -> sqlite3.Row | None:
        return self.conn.execute(
            "SELECT * FROM versions WHERE file_hash = ? LIMIT 1",
            (file_hash,),
        ).fetchone()

    def get_document(self, document_id: str) -> sqlite3.Row | None:
        return self.conn.execute(
            "SELECT * FROM documents WHERE document_id = ?",
            (document_id,),
        ).fetchone()

    def get_versions(self, document_id: str) -> list[sqlite3.Row]:
        return list(
            self.conn.execute(
                "SELECT * FROM versions WHERE document_id = ? ORDER BY version_number",
                (document_id,),
            ).fetchall()
        )

    def get_current_version(self, document_id: str) -> sqlite3.Row | None:
        return self.conn.execute(
            """
            SELECT v.*
            FROM versions v
            JOIN documents d ON d.current_version_id = v.version_id
            WHERE d.document_id = ?
            """,
            (document_id,),
        ).fetchone()

    def get_pages(self, version_id: str) -> dict[str, sqlite3.Row]:
        rows = self.conn.execute(
            "SELECT * FROM pages WHERE version_id = ?",
            (version_id,),
        ).fetchall()
        return {row["page_hash"]: row for row in rows}

    def create_document(self, document_id: str, display_name: str) -> None:
        now = utc_now()
        self.conn.execute(
            "INSERT INTO documents(document_id, display_name, created_at, updated_at) VALUES (?, ?, ?, ?)",
            (document_id, display_name, now, now),
        )
        self.conn.commit()

    def create_version(
        self,
        *,
        version_id: str,
        document_id: str,
        version_number: int,
        file_hash: str,
        original_filename: str,
        stored_pdf_path: str,
        status: str = "processing",
    ) -> None:
        self.conn.execute(
            """
            INSERT INTO versions(
                version_id, document_id, version_number, file_hash,
                original_filename, stored_pdf_path, status, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                version_id,
                document_id,
                version_number,
                file_hash,
                original_filename,
                stored_pdf_path,
                status,
                utc_now(),
            ),
        )
        self.conn.commit()

    def update_version(
        self,
        version_id: str,
        *,
        page_count: int | None = None,
        status: str | None = None,
        completed_at: str | None = None,
    ) -> None:
        fields = []
        values = []
        if page_count is not None:
            fields.append("page_count = ?")
            values.append(page_count)
        if status is not None:
            fields.append("status = ?")
            values.append(status)
        if completed_at is not None:
            fields.append("completed_at = ?")
            values.append(completed_at)
        if not fields:
            return
        values.append(version_id)
        self.conn.execute(
            f"UPDATE versions SET {', '.join(fields)} WHERE version_id = ?",
            values,
        )
        self.conn.commit()

    def set_current_version(self, document_id: str, version_id: str) -> None:
        self.conn.execute(
            "UPDATE documents SET current_version_id = ?, updated_at = ? WHERE document_id = ?",
            (version_id, utc_now(), document_id),
        )
        self.conn.commit()

    def upsert_page(
        self,
        version_id: str,
        page_number: int,
        page_hash: str,
        point_id: str,
    ) -> None:
        self.conn.execute(
            """
            INSERT INTO pages(version_id, page_number, page_hash, point_id)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(version_id, page_number)
            DO UPDATE SET page_hash=excluded.page_hash, point_id=excluded.point_id
            """,
            (version_id, page_number, page_hash, point_id),
        )
        self.conn.commit()

    def reset(self) -> None:
        self.conn.executescript(
            """
            DROP TABLE IF EXISTS pages;
            DROP TABLE IF EXISTS versions;
            DROP TABLE IF EXISTS documents;
            """
        )
        self.conn.commit()
        self._create_tables()

    def list_document_versions(self) -> list[sqlite3.Row]:
        """Return one row for every document/version pair."""
        return list(
            self.conn.execute(
                """
                SELECT
                    v.original_filename AS filename,
                    v.document_id AS document_id,
                    v.version_id AS version_id,
                    v.version_number AS version_number,
                    CASE
                        WHEN d.current_version_id = v.version_id
                        THEN 1
                        ELSE 0
                    END AS is_current
                FROM versions v
                JOIN documents d
                    ON d.document_id = v.document_id
                ORDER BY v.document_id, v.version_number
                """
            ).fetchall()
        )


