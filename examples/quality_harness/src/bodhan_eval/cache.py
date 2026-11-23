import hashlib
import json
import sqlite3
import threading
import uuid
from decimal import Decimal

from .models import canonical_json


class RequestCache:
    def __init__(self, path):
        self.lock = threading.RLock()
        self.connection = sqlite3.connect(path, check_same_thread=False)
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS responses "
            "(key TEXT PRIMARY KEY, response TEXT NOT NULL, cost TEXT)"
        )
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS failures "
            "(key TEXT PRIMARY KEY, reason TEXT NOT NULL)"
        )
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS runs "
            "(id TEXT PRIMARY KEY, plan TEXT NOT NULL, complete INTEGER NOT NULL DEFAULT 0)"
        )
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS decisions "
            "(run_id TEXT NOT NULL, position INTEGER NOT NULL, key TEXT NOT NULL, value TEXT NOT NULL, "
            "PRIMARY KEY (run_id, position))"
        )

    @staticmethod
    def key(namespace, exact_input):
        return hashlib.sha256(
            canonical_json([namespace, exact_input]).encode()
        ).hexdigest()

    def get(self, key):
        with self.lock:
            row = self.connection.execute(
                "SELECT response, cost FROM responses WHERE key = ?", (key,)
            ).fetchone()
        if row is None:
            return None
        return json.loads(row[0]), Decimal(row[1]) if row[1] is not None else None

    def put(self, key, result, cost):
        with self.lock, self.connection:
            self.connection.execute(
                "INSERT OR IGNORE INTO responses VALUES (?, ?, ?)",
                (key, canonical_json(result), str(cost) if cost is not None else None),
            )
            self.connection.execute("DELETE FROM failures WHERE key = ?", (key,))

    def get_failure(self, key):
        with self.lock:
            row = self.connection.execute(
                "SELECT reason FROM failures WHERE key = ?", (key,)
            ).fetchone()
        return row[0] if row else None

    def put_failure(self, key, reason):
        # Only safe Unavailable messages belong here, never raw provider bodies.
        with self.lock, self.connection:
            self.connection.execute(
                "INSERT OR REPLACE INTO failures VALUES (?, ?)", (key, reason)
            )

    def close(self):
        with self.lock:
            self.connection.close()

    def start_run(self, plan):
        run_id = uuid.uuid4().hex
        with self.lock, self.connection:
            self.connection.execute("INSERT INTO runs (id, plan) VALUES (?, ?)", (run_id, plan))
        return run_id

    def find_run(self, plan, run_id=None):
        with self.lock:
            row = self.connection.execute(
                "SELECT id FROM runs WHERE plan = ? " + ("AND id = ?" if run_id else "ORDER BY rowid DESC LIMIT 1"),
                (plan, run_id) if run_id else (plan,),
            ).fetchone()
        return row[0] if row else None

    def finish_run(self, run_id):
        with self.lock, self.connection:
            self.connection.execute("UPDATE runs SET complete = 1 WHERE id = ?", (run_id,))

    def put_decision(self, run_id, position, key, value):
        # Commit before dispatch too: interrupted outcomes remain unknown, never clean.
        with self.lock, self.connection:
            self.connection.execute("INSERT OR REPLACE INTO decisions VALUES (?, ?, ?, ?)",
                                    (run_id, position, key, canonical_json(value)))

    def get_decision(self, run_id, position, key):
        with self.lock:
            row = self.connection.execute("SELECT key, value FROM decisions WHERE run_id = ? AND position = ?",
                                          (run_id, position)).fetchone()
        if row is None:
            return None
        if row[0] != key:
            raise ValueError("replay request differs from the saved run decision")
        return json.loads(row[1])
