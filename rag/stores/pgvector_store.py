"""Postgres + pgvector. Run `make db` first. Same HNSW algorithm, but now with SQL:
filters, joins, transactions, and the ops tooling you already have for Postgres.
"""
import os

import numpy as np
import psycopg
from pgvector.psycopg import register_vector


class PgVectorStore:
    def __init__(self, dim: int, table: str = "chunks", dsn: str | None = None, reset: bool = True):
        self.table = table
        self.conn = psycopg.connect(dsn or os.environ["DATABASE_URL"], autocommit=True)
        self.conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(self.conn)
        if reset:
            self.conn.execute(f"DROP TABLE IF EXISTS {table}")
        self.conn.execute(
            f"CREATE TABLE IF NOT EXISTS {table} (id text PRIMARY KEY, embedding vector({dim}))"
        )

    def add(self, ids, vectors, batch: int = 1000):
        with self.conn.cursor() as cur:
            for s in range(0, len(ids), batch):
                with cur.copy(f"COPY {self.table} (id, embedding) FROM STDIN WITH (FORMAT BINARY)") as cp:
                    cp.set_types(["text", "vector"])
                    for i, v in zip(ids[s:s + batch], vectors[s:s + batch]):
                        cp.write_row([i, v.astype(np.float32)])

    def build_index(self, m: int = 16, ef_construction: int = 64):
        self.conn.execute(
            f"CREATE INDEX IF NOT EXISTS {self.table}_hnsw ON {self.table} "
            f"USING hnsw (embedding vector_cosine_ops) WITH (m={m}, ef_construction={ef_construction})"
        )

    def set_ef_search(self, ef: int):
        self.conn.execute(f"SET hnsw.ef_search = {int(ef)}")

    def search(self, query, k):
        rows = self.conn.execute(
            f"SELECT id, 1 - (embedding <=> %s) AS score FROM {self.table} "
            f"ORDER BY embedding <=> %s LIMIT %s",
            (query.astype(np.float32), query.astype(np.float32), k),
        ).fetchall()
        return [(r[0], float(r[1])) for r in rows]

    def search_batch(self, queries, k):
        return [self.search(q, k) for q in queries]
