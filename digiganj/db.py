"""پایگاه‌دادهٔ SQLite: ایندکس محصولات/تصاویر، چرخه‌ها، هشدارها و شواهد.

چرا SQLite؟ بدون سرور، قابل بازرسی با هر ابزاری، و برای «شاهد» بودن مناسب است:
هر ردیف زمانِ مشاهدهٔ اولیه/آخرین مشاهده را نگه می‌دارد و هیچ‌چیز بازنویسی
نمی‌شود. همهٔ زمان‌ها یونیکس (UTC) ذخیره می‌شوند و هنگام نمایش به تهران/جلالی
تبدیل می‌گردند.

نکتهٔ مهم برای صحت شواهد: `first_seen_ts` زمانِ **مشاهدهٔ ما** است، نه لزوماً
زمان انتشار. زمان انتشار قابل‌استناد، مُهر زمانی داخل نام فایل روی CDN است
(`stamp_ts`) که جداگانه ذخیره می‌شود.
"""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path
from typing import Any, Iterable, Sequence

from . import utils

SCHEMA = """
CREATE TABLE IF NOT EXISTS products (
    id            TEXT PRIMARY KEY,
    name          TEXT NOT NULL DEFAULT '',
    uri           TEXT NOT NULL DEFAULT '',
    price         INTEGER NOT NULL DEFAULT 0,
    page          INTEGER NOT NULL DEFAULT 0,
    rank          INTEGER NOT NULL DEFAULT 0,
    first_seen_ts REAL NOT NULL DEFAULT 0,
    last_seen_ts  REAL NOT NULL DEFAULT 0,
    gallery_ts    REAL NOT NULL DEFAULT 0,
    gallery_source TEXT NOT NULL DEFAULT '',
    gallery_errors INTEGER NOT NULL DEFAULT 0,
    gone_ts       REAL NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_products_page ON products(page);
CREATE INDEX IF NOT EXISTS idx_products_gallery ON products(gallery_ts);

CREATE TABLE IF NOT EXISTS images (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    product_id     TEXT NOT NULL,
    url            TEXT NOT NULL UNIQUE,
    filename       TEXT NOT NULL DEFAULT '',
    stamp_ts       REAL,
    stamp_kind     TEXT NOT NULL DEFAULT '',
    first_seen_ts  REAL NOT NULL DEFAULT 0,
    last_seen_ts   REAL NOT NULL DEFAULT 0,
    cycle_first    INTEGER NOT NULL DEFAULT 0,
    score          REAL NOT NULL DEFAULT 0,
    score_reasons  TEXT NOT NULL DEFAULT '',
    kind           TEXT NOT NULL DEFAULT '',
    alert_ts       REAL NOT NULL DEFAULT 0,
    downloaded_ts  REAL NOT NULL DEFAULT 0,
    file_path      TEXT NOT NULL DEFAULT '',
    sha1           TEXT NOT NULL DEFAULT '',
    bytes          INTEGER NOT NULL DEFAULT 0,
    ocr_text       TEXT NOT NULL DEFAULT '',
    ocr_codes      TEXT NOT NULL DEFAULT '',
    ocr_engine     TEXT NOT NULL DEFAULT '',
    ocr_ts         REAL NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_images_product ON images(product_id);
CREATE INDEX IF NOT EXISTS idx_images_stamp ON images(stamp_ts);
CREATE INDEX IF NOT EXISTS idx_images_first ON images(first_seen_ts);
CREATE INDEX IF NOT EXISTS idx_images_score ON images(score);

CREATE TABLE IF NOT EXISTS cycles (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    challenge_id TEXT NOT NULL DEFAULT '',
    started_ts   REAL NOT NULL,
    ended_ts     REAL NOT NULL DEFAULT 0,
    products     INTEGER NOT NULL DEFAULT 0,
    galleries    INTEGER NOT NULL DEFAULT 0,
    images       INTEGER NOT NULL DEFAULT 0,
    new_images   INTEGER NOT NULL DEFAULT 0,
    alerts       INTEGER NOT NULL DEFAULT 0,
    requests     INTEGER NOT NULL DEFAULT 0,
    blocked      INTEGER NOT NULL DEFAULT 0,
    note         TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS alerts (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    ts         REAL NOT NULL,
    image_id   INTEGER,
    product_id TEXT NOT NULL DEFAULT '',
    score      REAL NOT NULL DEFAULT 0,
    kind       TEXT NOT NULL DEFAULT '',
    reason     TEXT NOT NULL DEFAULT '',
    codes      TEXT NOT NULL DEFAULT '',
    acked      INTEGER NOT NULL DEFAULT 0,
    notified   INTEGER NOT NULL DEFAULT 0,
    note       TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_alerts_ts ON alerts(ts);

CREATE TABLE IF NOT EXISTS codes (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          REAL NOT NULL,
    code        TEXT NOT NULL UNIQUE,
    image_id    INTEGER,
    product_id  TEXT NOT NULL DEFAULT '',
    source      TEXT NOT NULL DEFAULT '',
    status      TEXT NOT NULL DEFAULT 'unknown',
    note        TEXT NOT NULL DEFAULT '',
    checked_ts  REAL NOT NULL DEFAULT 0,
    priority    REAL NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS verifications (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    ts         REAL NOT NULL,
    code       TEXT NOT NULL,
    product_id TEXT NOT NULL DEFAULT '',
    status     INTEGER NOT NULL DEFAULT 0,
    valid      INTEGER,
    response   TEXT NOT NULL DEFAULT '',
    elapsed_ms INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS events (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    ts      REAL NOT NULL,
    kind    TEXT NOT NULL DEFAULT 'note',
    message TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts);

CREATE TABLE IF NOT EXISTS kv (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL DEFAULT ''
);
"""


class Store:
    """لایهٔ پایگاه‌داده با دسترسی thread-safe.

    یک اتصال مشترک + قفل؛ برای این حجم داده (چند هزار ردیف) کافی و سریع است.
    """

    def __init__(self, path: str | Path):
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._db = sqlite3.connect(self.path, check_same_thread=False, timeout=30)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=NORMAL")
        self._db.execute("PRAGMA foreign_keys=ON")
        with self._lock:
            self._db.executescript(SCHEMA)
            self._db.commit()

    # ------------------------------------------------------------------ #
    # پایه
    # ------------------------------------------------------------------ #
    def close(self) -> None:
        with self._lock:
            try:
                self._db.commit()
                self._db.close()
            except Exception:
                pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _exec(self, sql: str, params: Sequence[Any] = ()) -> sqlite3.Cursor:
        with self._lock:
            cur = self._db.execute(sql, tuple(params))
            self._db.commit()
            return cur

    def _query(self, sql: str, params: Sequence[Any] = ()) -> list[sqlite3.Row]:
        with self._lock:
            return list(self._db.execute(sql, tuple(params)).fetchall())

    def _query_one(self, sql: str, params: Sequence[Any] = ()) -> sqlite3.Row | None:
        rows = self._query(sql, params)
        return rows[0] if rows else None

    def execute_script(self, script: str) -> None:
        with self._lock:
            self._db.executescript(script)
            self._db.commit()

    # ------------------------------------------------------------------ #
    # کلید/مقدار
    # ------------------------------------------------------------------ #
    def get_kv(self, key: str, default: str = "") -> str:
        row = self._query_one("SELECT value FROM kv WHERE key = ?", (key,))
        return row["value"] if row else default

    def set_kv(self, key: str, value: Any) -> None:
        self._exec(
            "INSERT INTO kv(key, value) VALUES(?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, str(value)),
        )

    # ------------------------------------------------------------------ #
    # رویدادها
    # ------------------------------------------------------------------ #
    def event(self, kind: str, message: str, ts: float | None = None) -> None:
        self._exec(
            "INSERT INTO events(ts, kind, message) VALUES(?, ?, ?)",
            (ts if ts is not None else utils.now_ts(), kind, message),
        )

    def events(self, limit: int = 200, kind: str | None = None) -> list[dict[str, Any]]:
        if kind:
            rows = self._query(
                "SELECT * FROM events WHERE kind = ? ORDER BY id DESC LIMIT ?", (kind, limit)
            )
        else:
            rows = self._query("SELECT * FROM events ORDER BY id DESC LIMIT ?", (limit,))
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------ #
    # محصولات
    # ------------------------------------------------------------------ #
    def upsert_products(self, products: Iterable, page: int = 0,
                        now: float | None = None) -> tuple[int, int]:
        """درج/به‌روزرسانی محصولات؛ برمی‌گرداند (تعداد جدید، تعداد کل)."""
        now = utils.now_ts() if now is None else now
        new_count = 0
        total = 0
        with self._lock:
            for p in products:
                pid = str(p.id)
                if not pid:
                    continue
                row = self._query_one("SELECT id FROM products WHERE id = ?", (pid,))
                if row:
                    self._db.execute(
                        "UPDATE products SET name=?, uri=?, price=?, page=?, rank=?, "
                        "last_seen_ts=?, gone_ts=0 WHERE id=?",
                        (p.name, p.uri, p.price, page or p.page, p.rank, now, pid),
                    )
                else:
                    self._db.execute(
                        "INSERT INTO products(id, name, uri, price, page, rank, "
                        "first_seen_ts, last_seen_ts) VALUES(?,?,?,?,?,?,?,?)",
                        (pid, p.name, p.uri, p.price, page or p.page, p.rank, now, now),
                    )
                    new_count += 1
                total += 1
            self._db.commit()
        return new_count, total

    def products(self, include_gone: bool = False) -> list[dict[str, Any]]:
        sql = "SELECT * FROM products"
        if not include_gone:
            sql += " WHERE gone_ts = 0"
        sql += " ORDER BY page ASC, rank ASC, id ASC"
        return [dict(r) for r in self._query(sql)]

    def product(self, pid: str) -> dict[str, Any] | None:
        row = self._query_one("SELECT * FROM products WHERE id = ?", (str(pid),))
        return dict(row) if row else None

    def mark_gone(self, since_ts: float, now: float | None = None) -> int:
        """محصولاتی که از فهرست حذف شده‌اند علامت‌گذاری می‌شوند."""
        now = utils.now_ts() if now is None else now
        cur = self._exec(
            "UPDATE products SET gone_ts = ? WHERE last_seen_ts < ? AND gone_ts = 0",
            (now, since_ts),
        )
        return cur.rowcount

    def mark_gallery(self, pid: str, source: str, ts: float | None = None,
                     ok: bool = True) -> None:
        now = utils.now_ts() if ts is None else ts
        if ok:
            self._exec(
                "UPDATE products SET gallery_ts=?, gallery_source=?, gallery_errors=0 "
                "WHERE id=?",
                (now, source, str(pid)),
            )
        else:
            self._exec(
                "UPDATE products SET gallery_errors = gallery_errors + 1 WHERE id = ?",
                (str(pid),),
            )

    def due_products(self, resync_after: float, limit: int = 0,
                     watch_ids: Sequence[str] = ()) -> list[dict[str, Any]]:
        """محصولاتی که گالری‌شان قدیمی‌تر از آستانه است؛ اولویت با لیست تماشا."""
        cutoff = utils.now_ts() - resync_after
        sql = (
            "SELECT * FROM products WHERE gone_ts = 0 AND (gallery_ts = 0 OR gallery_ts < ?) "
            "ORDER BY gallery_ts ASC, page ASC"
        )
        rows = [dict(r) for r in self._query(sql, (cutoff,))]
        if watch_ids:
            watch = {str(w) for w in watch_ids}
            head = [r for r in rows if r["id"] in watch]
            tail = [r for r in rows if r["id"] not in watch]
            rows = head + tail
        return rows[:limit] if limit else rows

    # ------------------------------------------------------------------ #
    # تصاویر
    # ------------------------------------------------------------------ #
    def upsert_images(self, items: Iterable, now: float | None = None,
                      cycle_id: int = 0) -> list[dict[str, Any]]:
        """درج تصاویر تازه؛ برمی‌گرداند فقط تصاویر **جدید** با شناسهٔ پایگاه‌داده."""
        now = utils.now_ts() if now is None else now
        fresh: list[dict[str, Any]] = []
        with self._lock:
            for item in items:
                row = self._query_one("SELECT * FROM images WHERE url = ?", (item.url,))
                if row:
                    self._db.execute(
                        "UPDATE images SET last_seen_ts=?, stamp_ts=COALESCE(stamp_ts, ?), "
                        "stamp_kind=CASE WHEN stamp_kind='' THEN ? ELSE stamp_kind END "
                        "WHERE id=?",
                        (now, item.stamp_ts, item.stamp_kind, row["id"]),
                    )
                    continue
                cur = self._db.execute(
                    "INSERT INTO images(product_id, url, filename, stamp_ts, stamp_kind, "
                    "first_seen_ts, last_seen_ts, cycle_first) VALUES(?,?,?,?,?,?,?,?)",
                    (
                        str(item.product_id),
                        item.url,
                        item.filename,
                        item.stamp_ts,
                        item.stamp_kind,
                        now,
                        now,
                        cycle_id,
                    ),
                )
                fresh.append(
                    {
                        "id": cur.lastrowid,
                        "product_id": str(item.product_id),
                        "url": item.url,
                        "filename": item.filename,
                        "stamp_ts": item.stamp_ts,
                        "stamp_kind": item.stamp_kind,
                        "first_seen_ts": now,
                        "cycle_first": cycle_id,
                    }
                )
            self._db.commit()
        return fresh

    def images_for_product(self, pid: str) -> list[dict[str, Any]]:
        return [
            dict(r)
            for r in self._query(
                "SELECT * FROM images WHERE product_id = ? ORDER BY id ASC", (str(pid),)
            )
        ]

    def image(self, image_id: int) -> dict[str, Any] | None:
        row = self._query_one("SELECT * FROM images WHERE id = ?", (image_id,))
        return dict(row) if row else None

    def image_by_url(self, url: str) -> dict[str, Any] | None:
        row = self._query_one("SELECT * FROM images WHERE url = ?", (url,))
        return dict(row) if row else None

    def set_score(self, image_id: int, score: float, reasons: Sequence[str],
                  kind: str) -> None:
        self._exec(
            "UPDATE images SET score=?, score_reasons=?, kind=? WHERE id=?",
            (float(score), "\n".join(reasons), kind, image_id),
        )

    def set_download(self, image_id: int, file_path: str, sha1: str, size: int,
                     ts: float | None = None) -> None:
        self._exec(
            "UPDATE images SET downloaded_ts=?, file_path=?, sha1=?, bytes=? WHERE id=?",
            (utils.now_ts() if ts is None else ts, file_path, sha1, size, image_id),
        )

    def set_ocr(self, image_id: int, text: str, codes: Sequence[str], engine: str) -> None:
        self._exec(
            "UPDATE images SET ocr_text=?, ocr_codes=?, ocr_engine=?, ocr_ts=? WHERE id=?",
            (text, ",".join(codes), engine, utils.now_ts(), image_id),
        )

    def candidates(self, min_score: float = 0.0, limit: int = 200,
                   only_new: bool = False) -> list[dict[str, Any]]:
        sql = (
            "SELECT i.*, p.name AS product_name, p.uri AS product_uri "
            "FROM images i LEFT JOIN products p ON p.id = i.product_id "
            "WHERE i.score >= ?"
        )
        params: list[Any] = [min_score]
        if only_new:
            sql += " AND i.alert_ts = 0"
        sql += " ORDER BY i.score DESC, i.stamp_ts DESC, i.id ASC LIMIT ?"
        params.append(limit)
        return [dict(r) for r in self._query(sql, params)]

    def product_scan_stats(self, product_id: str) -> dict[str, int]:
        row = self._query_one(
            "SELECT COUNT(*) AS n, COALESCE(SUM(alert_ts > 0), 0) AS alerted, "
            "COALESCE(SUM(downloaded_ts > 0), 0) AS downloaded "
            "FROM images WHERE product_id = ?",
            (str(product_id),),
        )
        return dict(row) if row else {"n": 0, "alerted": 0, "downloaded": 0}

    # ------------------------------------------------------------------ #
    # چرخه‌ها و هشدارها
    # ------------------------------------------------------------------ #
    def start_cycle(self, challenge_id: str, ts: float | None = None) -> int:
        cur = self._exec(
            "INSERT INTO cycles(challenge_id, started_ts) VALUES(?, ?)",
            (challenge_id, utils.now_ts() if ts is None else ts),
        )
        return int(cur.lastrowid or 0)

    def finish_cycle(self, cycle_id: int, **counts: Any) -> None:
        fields = {
            "ended_ts": utils.now_ts(),
            "products": counts.get("products", 0),
            "galleries": counts.get("galleries", 0),
            "images": counts.get("images", 0),
            "new_images": counts.get("new_images", 0),
            "alerts": counts.get("alerts", 0),
            "requests": counts.get("requests", 0),
            "blocked": counts.get("blocked", 0),
            "note": str(counts.get("note", "")),
        }
        self._exec(
            "UPDATE cycles SET ended_ts=:ended_ts, products=:products, galleries=:galleries, "
            "images=:images, new_images=:new_images, alerts=:alerts, requests=:requests, "
            "blocked=:blocked, note=:note WHERE id=:id",
            {**fields, "id": cycle_id},
        )

    def cycles(self, limit: int = 100) -> list[dict[str, Any]]:
        return [
            dict(r)
            for r in self._query("SELECT * FROM cycles ORDER BY id DESC LIMIT ?", (limit,))
        ]

    def last_alert_ts(self, image_id: int) -> float:
        row = self._query_one(
            "SELECT ts FROM alerts WHERE image_id = ? ORDER BY id DESC LIMIT 1", (image_id,)
        )
        return float(row["ts"]) if row else 0.0

    def add_alert(self, image_id: int | None, product_id: str, score: float, kind: str,
                  reason: str, codes: Sequence[str] = ()) -> int:
        cur = self._exec(
            "INSERT INTO alerts(ts, image_id, product_id, score, kind, reason, codes) "
            "VALUES(?,?,?,?,?,?,?)",
            (utils.now_ts(), image_id, str(product_id), float(score), kind, reason,
             ",".join(codes)),
        )
        if image_id:
            self._exec("UPDATE images SET alert_ts=? WHERE id=?",
                       (utils.now_ts(), image_id))
        return int(cur.lastrowid or 0)

    def alerts(self, limit: int = 100, min_score: float = 0.0) -> list[dict[str, Any]]:
        rows = self._query(
            "SELECT a.*, i.url AS image_url, i.filename, i.stamp_ts, i.file_path, "
            "i.score AS image_score, i.ocr_codes, p.name AS product_name, p.uri AS product_uri "
            "FROM alerts a LEFT JOIN images i ON i.id = a.image_id "
            "LEFT JOIN products p ON p.id = a.product_id "
            "WHERE a.score >= ? ORDER BY a.score DESC, a.id DESC LIMIT ?",
            (min_score, limit),
        )
        return [dict(r) for r in rows]

    def mark_notified(self, alert_id: int) -> None:
        self._exec("UPDATE alerts SET notified = 1 WHERE id = ?", (alert_id,))

    def unverified_alerts(self, limit: int = 20) -> list[dict[str, Any]]:
        return [dict(r) for r in self._query(
            "SELECT * FROM alerts WHERE codes != '' AND note = '' ORDER BY score DESC LIMIT ?",
            (limit,),
        )]

    # ------------------------------------------------------------------ #
    # کدها و بررسی‌ها
    # ------------------------------------------------------------------ #
    def add_code(self, code: str, image_id: int | None = None, product_id: str = "",
                 source: str = "ocr", note: str = "", priority: float = 0.0) -> bool:
        """ثبت کد جدید؛ False اگر قبلاً ثبت شده باشد."""
        code = code.strip()
        if not code:
            return False
        existing = self._query_one("SELECT id FROM codes WHERE code = ?", (code,))
        if existing:
            return False
        self._exec(
            "INSERT INTO codes(ts, code, image_id, product_id, source, note, priority) "
            "VALUES(?,?,?,?,?,?,?)",
            (utils.now_ts(), code, image_id, str(product_id), source, note, float(priority)),
        )
        return True

    def codes(self, limit: int = 500, status: str | None = None) -> list[dict[str, Any]]:
        if status:
            rows = self._query(
                "SELECT * FROM codes WHERE status = ? ORDER BY id DESC LIMIT ?",
                (status, limit),
            )
        else:
            rows = self._query("SELECT * FROM codes ORDER BY id DESC LIMIT ?", (limit,))
        return [dict(r) for r in rows]

    def pending_codes(self, limit: int = 50) -> list[dict[str, Any]]:
        return [
            dict(r)
            for r in self._query(
                "SELECT * FROM codes WHERE status = 'unknown' AND note = '' "
                "ORDER BY priority DESC, id ASC LIMIT ?",
                (limit,),
            )
        ]

    def set_code_status(self, code: str, status: str, note: str = "") -> None:
        self._exec(
            "UPDATE codes SET status=?, note=?, checked_ts=? WHERE code=?",
            (status, note, utils.now_ts(), code),
        )

    def add_verification(self, code: str, product_id: str, status: int,
                         valid: bool | None, response: str, elapsed_ms: int) -> int:
        cur = self._exec(
            "INSERT INTO verifications(ts, code, product_id, status, valid, response, "
            "elapsed_ms) VALUES(?,?,?,?,?,?,?)",
            (
                utils.now_ts(),
                code,
                str(product_id),
                int(status),
                None if valid is None else int(bool(valid)),
                utils.shorten(response, 2000),
                int(elapsed_ms),
            ),
        )
        return int(cur.lastrowid or 0)

    def verifications(self, limit: int = 200) -> list[dict[str, Any]]:
        return [
            dict(r)
            for r in self._query(
                "SELECT * FROM verifications ORDER BY id DESC LIMIT ?", (limit,)
            )
        ]

    # ------------------------------------------------------------------ #
    # آمار
    # ------------------------------------------------------------------ #
    def stats(self) -> dict[str, Any]:
        def one(sql: str, params: Sequence[Any] = ()) -> int:
            row = self._query_one(sql, params)
            return int(list(row)[0]) if row else 0

        stamps = self._query_one(
            "SELECT COUNT(*) AS n, MIN(COALESCE(stamp_ts, first_seen_ts)) AS first, "
            "MAX(COALESCE(stamp_ts, first_seen_ts)) AS last FROM images"
        )
        return {
            "products": one("SELECT COUNT(*) FROM products"),
            "products_active": one("SELECT COUNT(*) FROM products WHERE gone_ts = 0"),
            "images": one("SELECT COUNT(*) FROM images"),
            "images_stamped": one("SELECT COUNT(*) FROM images WHERE stamp_ts IS NOT NULL"),
            "images_downloaded": one("SELECT COUNT(*) FROM images WHERE downloaded_ts > 0"),
            "images_ocred": one("SELECT COUNT(*) FROM images WHERE ocr_ts > 0"),
            "alerts": one("SELECT COUNT(*) FROM alerts"),
            "codes": one("SELECT COUNT(*) FROM codes"),
            "cycles": one("SELECT COUNT(*) FROM cycles"),
            "verifications": one("SELECT COUNT(*) FROM verifications"),
            "earliest_image_ts": stamps["first"] if stamps else None,
            "latest_image_ts": stamps["last"] if stamps else None,
        }

    def export_json(self, path: str | Path) -> Path:
        data = {
            "generated_at": utils.iso_utc(utils.now_ts()),
            "stats": self.stats(),
            "products": self.products(include_gone=True),
            "images": [dict(r) for r in self._query(
                "SELECT id, product_id, url, filename, stamp_ts, stamp_kind, first_seen_ts, "
                "last_seen_ts, score, kind, alert_ts, ocr_codes FROM images ORDER BY id"
            )],
            "alerts": self.alerts(limit=1000),
            "codes": self.codes(limit=1000),
            "verifications": self.verifications(limit=1000),
            "cycles": self.cycles(limit=500),
            "events": self.events(limit=1000),
        }
        utils.write_json(path, data)
        return Path(path)

    def vacuum(self) -> None:
        with self._lock:
            self._db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            self._db.execute("VACUUM")
            self._db.commit()

    # ---- برای داشبورد ----
    def state_payload(self, challenge=None) -> dict[str, Any]:
        stats = self.stats()
        last_cycles = self.cycles(limit=6)
        top = self.alerts(limit=10)
        recent_new = [
            dict(r)
            for r in self._query(
                "SELECT i.*, p.name AS product_name FROM images i "
                "LEFT JOIN products p ON p.id = i.product_id "
                "ORDER BY i.first_seen_ts DESC, i.id DESC LIMIT 25"
            )
        ]
        events = self.events(limit=40)
        payload = {
            "now": utils.now_ts(),
            "stats": stats,
            "cycles": last_cycles,
            "alerts": top,
            "recent_images": recent_new,
            "events": events,
        }
        if challenge is not None:
            payload["challenge"] = {
                "id": challenge.id,
                "title": challenge.title,
                "starts_at": challenge.starts_at,
                "ends_at": challenge.ends_at,
                "start_ts": challenge.start_ts,
                "end_ts": challenge.end_ts,
                "phase": challenge.phase(),
            }
        return payload


def open_store(path: str | Path) -> Store:
    return Store(path)
