import hashlib
import json
import sqlite3
from decimal import Decimal

from .models import canonical_json


class RequestCache:
    def __init__(self, path):
        self.connection = sqlite3.connect(path)
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS responses "
            "(key TEXT PRIMARY KEY, response TEXT NOT NULL, cost TEXT)"
        )

    @staticmethod
    def key(namespace, exact_input):
        return hashlib.sha256(
            canonical_json([namespace, exact_input]).encode()
        ).hexdigest()

    def get(self, key):
        row = self.connection.execute(
            "SELECT response, cost FROM responses WHERE key = ?", (key,)
        ).fetchone()
        if row is None:
            return None
        return json.loads(row[0]), Decimal(row[1]) if row[1] is not None else None

    def put(self, key, result, cost):
        with self.connection:
            self.connection.execute(
                "INSERT OR IGNORE INTO responses VALUES (?, ?, ?)",
                (key, canonical_json(result), str(cost) if cost is not None else None),
            )

    def close(self):
        self.connection.close()
