"""لایهٔ شبکه: نشست پایدار، تلاش مجدد هوشمند، احترام به محدودیت نرخ.

فلسفهٔ طراحی
------------
دیجی‌کالا برای هر کاربر محدودیت نرخ و محافظت ضدربات دارد. «سریع‌ترین» ابزار،
ابزاری است که **کمترین تعداد درخواست** را بفرستد و در برابر خطا شکننده نباشد:

* کش ETag/Last-Modified برای صفحه‌های فهرست (اگر سرور پشتیبانی کند، ۳۰۴).
* عقب‌نشینی نمایی (exponential backoff) با نوسان تصادفی، به‌جای هجوم دوباره.
* فاصلهٔ تطبیقی بین درخواست‌ها: با هر 429/503 فاصله دو برابر می‌شود و به‌آرامی
  پایین می‌آید. یعنی در صورت فشار، ابزار **کندتر** می‌شود، نه پرتهاجم‌تر.
* چرخش هویت (UA/پروفایل/کوکی/پروکسی) فقط پس از شکست‌های پیاپی و همراه با
  خنک‌سازی اجباری انجام می‌شود؛ برای «دور زدن محدودیت» با نرخ بالا نیست.

پشتیبانی از TLS واقعی مرورگر
----------------------------
اگر بستهٔ ``curl_cffi`` نصب باشد، درخواست‌ها با اثر انگشت TLS/HTTP2 کروم ارسال
می‌شوند؛ در غیر این صورت ``httpx`` با HTTP/2 استفاده می‌شود.
"""

from __future__ import annotations

import json
import random
import threading
import time
import zlib
from dataclasses import dataclass, field
from typing import Any, Callable

from . import utils
from .console import Console, Logger

# --------------------------------------------------------------------------- #
# خطاها
# --------------------------------------------------------------------------- #


class NetError(Exception):
    """خطای پایهٔ شبکه."""

    def __init__(self, message: str, url: str = "", status: int | None = None):
        super().__init__(message)
        self.url = url
        self.status = status


class TransientError(NetError):
    """خطای موقت: می‌توان دوباره تلاش کرد."""


class BlockedError(NetError):
    """بلوکهٔ موقت (۴۰۳/۴۲۹/چالش امنیتی): باید عقب نشست و هویت را نو کرد."""

    def __init__(self, message: str, url: str = "", status: int | None = None,
                 retry_after: float = 0.0):
        super().__init__(message, url, status)
        self.retry_after = retry_after


class NotFoundError(NetError):
    """۴۰۴: تلاش مجدد بی‌فایده است."""


class CircuitOpenError(NetError):
    """مدار حفاظتی باز است؛ مدتی هیچ درخواستی فرستاده نمی‌شود."""


# --------------------------------------------------------------------------- #
# آمار و محدودساز نرخ
# --------------------------------------------------------------------------- #


@dataclass
class Stats:
    requests: int = 0
    ok: int = 0
    cached: int = 0
    not_modified: int = 0
    failed: int = 0
    blocked: int = 0
    retries: int = 0
    bytes_down: int = 0
    total_seconds: float = 0.0
    status_counts: dict[int, int] = field(default_factory=dict)
    by_purpose: dict[str, int] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def bump(self, field_name: str, n: int = 1) -> None:
        with self._lock:
            setattr(self, field_name, getattr(self, field_name) + n)

    def status(self, code: int) -> None:
        with self._lock:
            self.status_counts[code] = self.status_counts.get(code, 0) + 1

    def purpose(self, name: str) -> None:
        with self._lock:
            self.by_purpose[name] = self.by_purpose.get(name, 0) + 1

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "requests": self.requests,
                "ok": self.ok,
                "cached": self.cached,
                "not_modified": self.not_modified,
                "failed": self.failed,
                "blocked": self.blocked,
                "retries": self.retries,
                "bytes_down": self.bytes_down,
                "total_seconds": round(self.total_seconds, 2),
                "status_counts": dict(self.status_counts),
                "by_purpose": dict(self.by_purpose),
            }


class RateLimiter:
    """فاصله‌گذار تطبیقی بین درخواست‌ها (thread-safe).

    * ``min_interval`` کف فاصله است؛ هیچ‌وقت از آن کمتر نمی‌شود.
    * در خطای 429/503، فاصله دو برابر (تا سقف) می‌شود.
    * در موفقیت‌های پیاپی، فاصله به‌آرامی به کف نزدیک می‌شود.
    """

    def __init__(self, min_interval: float = 0.06, max_interval: float = 8.0):
        self.min_interval = max(0.0, min_interval)
        self.max_interval = max(self.min_interval, max_interval)
        self._interval = self.min_interval
        self._next = 0.0
        self._lock = threading.Lock()
        self._success_streak = 0

    @property
    def interval(self) -> float:
        with self._lock:
            return self._interval

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            delay = max(0.0, self._next - now)
            if delay:
                self._next += self._interval
            else:
                self._next = now + self._interval
        if delay:
            time.sleep(delay)

    def note_success(self) -> None:
        with self._lock:
            self._success_streak += 1
            if self._success_streak >= 8 and self._interval > self.min_interval:
                self._interval = max(self.min_interval, self._interval * 0.85)
                self._success_streak = 0

    def note_block(self, retry_after: float = 0.0) -> None:
        with self._lock:
            self._interval = min(self.max_interval, max(self._interval * 2, 0.5))
            self._success_streak = 0
            self._next = max(self._next, time.monotonic() + retry_after)

    def note_error(self) -> None:
        with self._lock:
            self._interval = min(self.max_interval, self._interval * 1.4)
            self._success_streak = 0


class CircuitBreaker:
    """مدار حفاظتی: پس از شکست‌های پیاپی، جریان را موقتاً قطع می‌کند."""

    def __init__(self, failures: int = 25, cooldown: float = 30.0):
        self.failures = max(1, failures)
        self.cooldown = max(1.0, cooldown)
        self._count = 0
        self._opened_at = 0.0
        self._lock = threading.Lock()

    @property
    def is_open(self) -> bool:
        with self._lock:
            if not self._opened_at:
                return False
            elapsed = time.monotonic() - self._opened_at
            if elapsed >= self.cooldown:
                # دورهٔ خنک‌سازی تمام شد → مدار می‌بندد (نیمه‌باز)
                self._opened_at = 0.0
                self._count = 0
                return False
            return True

    def remaining(self) -> float:
        with self._lock:
            if not self._opened_at:
                return 0.0
            return max(0.0, self.cooldown - (time.monotonic() - self._opened_at))

    def note_success(self) -> None:
        with self._lock:
            if self._count:
                self._count = max(0, self._count - 1)

    def note_failure(self) -> bool:
        """True اگر همین حالا مدار باز شد."""
        with self._lock:
            self._count += 1
            if self._count >= self.failures and not self._opened_at:
                self._opened_at = time.monotonic()
                return True
            return False

    def trip(self) -> None:
        with self._lock:
            self._opened_at = time.monotonic()
            self._count = self.failures


# --------------------------------------------------------------------------- #
# پاسخ
# --------------------------------------------------------------------------- #


@dataclass
class Response:
    url: str
    status: int
    headers: dict[str, str]
    content: bytes
    elapsed: float
    from_cache: bool = False
    purpose: str = ""

    @property
    def text(self) -> str:
        enc = ""
        ctype = self.headers.get("content-type", "")
        for part in ctype.split(";"):
            part = part.strip()
            if part.lower().startswith("charset="):
                enc = part.split("=", 1)[1].strip()
        if not enc:
            enc = "utf-8"
        try:
            return self.content.decode(enc, errors="replace")
        except LookupError:
            return self.content.decode("utf-8", errors="replace")

    def json(self) -> Any:
        # برخی پاسخ‌ها gzip خام برمی‌گردانند (وقتی خودمان Accept-Encoding می‌فرستیم)
        data = self.content
        if data[:2] == b"\x1f\x8b":
            try:
                data = zlib.decompress(data, 16 + zlib.MAX_WBITS)
            except zlib.error:
                pass
        return json.loads(data.decode("utf-8", errors="replace"))

    @property
    def ok(self) -> bool:
        return 200 <= self.status < 300


# --------------------------------------------------------------------------- #
# هویت مرورگر
# --------------------------------------------------------------------------- #

_SEC_CH_UA_CHROME = (
    '"Chromium";v="131", "Not_A Brand";v="24", "Google Chrome";v="131"'
)
_SEC_CH_UA_FIREFOX = '"Firefox";v="133", "Not A(Brand";v="24"'


@dataclass
class Identity:
    """اثر انگشت مرورگر یک نشست.

    هدف: درخواست‌های ابزار شبیه یک مرورگر واقعی باشد (چون سرور برای کلاینت
    غیرمرورگری محافظت سخت‌گیرانه‌تری اعمال می‌کند) و در طول یک نشست **ثابت**
    بماند؛ تغییر مداوم UA خودش نشانهٔ ربات است.
    """

    user_agent: str
    profile: dict[str, str]
    accept_language: str = "fa-IR,fa;q=0.9,en-US;q=0.8,en;q=0.7"

    @classmethod
    def random(cls, cfg) -> "Identity":
        uas = cfg.user_agents or [utils.__dict__.get("x", "")]  # noqa: F841
        profile = random.choice(cfg.browser_profiles) if cfg.browser_profiles else {
            "name": "chrome",
            "platform": '"Windows"',
            "mobile": "?0",
            "brands": "chrome",
        }
        ua = random.choice(uas)
        if "Firefox" in ua:
            profile = {
                "name": "firefox",
                "platform": '"Windows"',
                "mobile": "?0",
                "brands": "firefox",
            }
        return cls(user_agent=ua, profile=profile)

    @property
    def is_firefox(self) -> bool:
        return "Firefox" in self.user_agent

    def headers(self, referer: str | None = None) -> dict[str, str]:
        h = {
            "User-Agent": self.user_agent,
            "Accept": (
                "text/html,application/xhtml+xml,application/xml;q=0.9,"
                "image/avif,image/webp,image/apng,*/*;q=0.8,"
                "application/signed-exchange;v=b3;q=0.7"
            ),
            "Accept-Language": self.accept_language,
            "Accept-Encoding": "gzip, deflate, br",
            "Upgrade-Insecure-Requests": "1",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "none",
            "Sec-Fetch-User": "?1",
            "Cache-Control": "max-age=0",
            "Connection": "keep-alive",
            "DNT": "1",
        }
        if self.is_firefox:
            h["TE"] = "trailers"
        else:
            h["sec-ch-ua"] = _SEC_CH_UA_CHROME
            h["sec-ch-ua-mobile"] = self.profile.get("mobile", "?0")
            h["sec-ch-ua-platform"] = self.profile.get("platform", '"Windows"')
            h["Sec-Fetch-Site"] = "same-origin"
        if referer:
            h["Referer"] = referer
            h["Sec-Fetch-Site"] = "same-origin"
        return h

    def xhr_headers(self, referer: str) -> dict[str, str]:
        h = self.headers(referer)
        h.update(
            {
                "Accept": "application/json, text/javascript, */*; q=0.01",
                "X-Requested-With": "XMLHttpRequest",
                "Sec-Fetch-Dest": "empty",
                "Sec-Fetch-Mode": "cors",
                "Sec-Fetch-Site": "same-origin",
            }
        )
        return h

    def api_headers(self, referer: str) -> dict[str, str]:
        h = self.headers(referer)
        h.update(
            {
                "Accept": "application/json, text/plain, */*",
                "Origin": "https://www.digikala.com",
                "Sec-Fetch-Dest": "empty",
                "Sec-Fetch-Mode": "cors",
                "Sec-Fetch-Site": "same-site",
            }
        )
        return h


# --------------------------------------------------------------------------- #
# پشتانه‌های HTTP
# --------------------------------------------------------------------------- #


class _HttpxBackend:
    name = "httpx"

    def __init__(self, cfg, proxy: str | None = None):
        import httpx  # وارد کردن دیرهنگام

        self._httpx = httpx
        kwargs: dict[str, Any] = {
            "timeout": httpx.Timeout(cfg.timeout, connect=cfg.connect_timeout),
            "follow_redirects": True,
            "verify": cfg.verify_tls,
            "http2": bool(cfg.http2),
            "limits": httpx.Limits(
                max_connections=max(8, cfg.concurrency * 2),
                max_keepalive_connections=max(4, cfg.concurrency),
            ),
            "trust_env": False,
        }
        if proxy:
            kwargs["proxy"] = proxy
        self._client = httpx.Client(**kwargs)

    def request(self, method: str, url: str, headers: dict[str, str], stream: bool = False):
        req = self._client.build_request(method, url, headers=headers)
        if stream:
            return self._client.send(req, stream=True)
        return self._client.send(req)

    def close(self) -> None:
        try:
            self._client.close()
        except Exception:
            pass


class _CurlBackend:
    name = "curl_cffi"

    def __init__(self, cfg, proxy: str | None = None):
        from curl_cffi import requests as crequests  # وارد کردن دیرهنگام

        self._cr = crequests
        kwargs: dict[str, Any] = {
            "impersonate": cfg.impersonate,
            "timeout": max(cfg.timeout, cfg.connect_timeout),
            "verify": cfg.verify_tls,
            "allow_redirects": True,
        }
        if proxy:
            kwargs["proxies"] = {"http": proxy, "https": proxy}
        self._session = crequests.Session(**kwargs)

    def request(self, method: str, url: str, headers: dict[str, str], stream: bool = False):
        return self._session.request(method, url, headers=headers, stream=stream)

    def close(self) -> None:
        try:
            self._session.close()
        except Exception:
            pass


def _load_backend(cfg, proxy: str | None):
    """انتخاب پشتانه بر اساس تنظیمات و بسته‌های نصب‌شده."""
    mode = (cfg.transport or "auto").lower()
    if mode in ("curl_cffi", "curl", "auto"):
        try:
            return _CurlBackend(cfg, proxy)
        except Exception:
            if mode != "auto":
                raise
    return _HttpxBackend(cfg, proxy)


def _proxy_ok(proxy: str, cfg) -> bool:
    """بررسی سریع سلامت پروکسی (بدون هزینهٔ زیاد)."""
    try:
        backend = _load_backend(cfg, proxy)
        try:
            resp = backend.request(
                "HEAD",
                cfg.proxy_health_url,
                {"User-Agent": random.choice(cfg.user_agents)},
            )
            code = getattr(resp, "status_code", 0)
            return 200 <= int(code) < 500
        finally:
            backend.close()
    except Exception:
        return False


class ProxyPool:
    """گردانندهٔ پروکسی با بررسی سلامت و کنارگذاری موقت."""

    def __init__(self, cfg, logger: Logger | None = None):
        self.cfg = cfg
        self.logger = logger
        self._all = list(cfg.proxies or [])
        self._healthy: list[str] = []
        self._dead: dict[str, float] = {}
        self._lock = threading.Lock()
        self._index = 0

    def prepare(self) -> None:
        if not self._all:
            return
        if not self.cfg.proxy_health_check:
            self._healthy = list(self._all)
            return
        healthy = []
        for p in self._all:
            ok = _proxy_ok(p, self.cfg)
            if ok:
                healthy.append(p)
            elif self.logger:
                self.logger.warn(f"پروکسی ناسالم کنار گذاشته شد: {utils.shorten(p, 40)}")
        with self._lock:
            self._healthy = healthy or []
        if self.logger and self._all:
            self.logger.info(
                f"پروکسی‌های سالم: {utils.fa_num(len(self._healthy))} از "
                f"{utils.fa_num(len(self._all))}"
            )

    @property
    def enabled(self) -> bool:
        return bool(self._all)

    def next(self) -> str | None:
        with self._lock:
            now = time.monotonic()
            alive = [p for p in self._healthy if self._dead.get(p, 0) < now]
            if not alive:
                # همهٔ پروکسی‌ها موقتاً مرده‌اند؛ صبر می‌کنیم تا یکی آزاد شود
                if not self._healthy:
                    return None
                wait = min(v for v in self._dead.values() if v > now) - now
                time.sleep(max(0.0, min(wait, 5.0)))
                alive = [p for p in self._healthy if self._dead.get(p, 0) < now]
                if not alive:
                    return None
            p = alive[self._index % len(alive)]
            self._index += 1
            return p

    def penalize(self, proxy: str | None, seconds: float = 60.0) -> None:
        if not proxy:
            return
        with self._lock:
            self._dead[proxy] = time.monotonic() + seconds

    @property
    def count(self) -> int:
        return len(self._healthy)


# --------------------------------------------------------------------------- #
# کلاینت اصلی
# --------------------------------------------------------------------------- #


class HttpClient:
    """کلاینت HTTP با تلاش مجدد، کش و آمار.

    همهٔ متدها thread-safe هستند؛ یک نمونه بین چند رشته به اشتراک گذاشته
    می‌شود. هر رشته درخواست خود را از طریق یک backend مستقل (نشست) می‌فرستد
    تا کوکی/اتصال‌ها در هم تنیده نشوند.

    پارامترها
    ---------
    min_interval: فاصلهٔ حداقلی بین درخواست‌های هر نشست (ثانیه).
    on_block: تابع اختیاری برای گزارش بلوکه‌شدن (برای لاگ/UI).
    """

    def __init__(
        self,
        cfg,
        logger: Logger | None = None,
        console: Console | None = None,
        min_interval: float | None = None,
        on_block: Callable[[str, float], None] | None = None,
        identity: Identity | None = None,
        proxies: ProxyPool | None = None,
    ):
        self.cfg = cfg
        self.logger = logger
        self.console = console or Console()
        self.stats = Stats()
        self.on_block = on_block
        self.identity = identity or Identity.random(cfg)
        self.proxies = proxies or ProxyPool(cfg, logger)
        # فاصلهٔ پایه از پیکربندی خوانده می‌شود (اگر تعریف شده باشد)
        base_interval = (
            min_interval
            if min_interval is not None
            else float(getattr(cfg, "min_request_interval", 0.06) or 0.06)
        )
        self.limiter = RateLimiter(
            min_interval=max(0.0, base_interval),
            max_interval=max(2.0, cfg.backoff_cap),
        )
        self.breaker = CircuitBreaker(cfg.circuit_failures, cfg.circuit_cooldown)
        self._tls = threading.local()
        self._local_backends: list[Any] = []
        self._lock = threading.Lock()
        self._etag_cache: dict[str, tuple[str, str, bytes, float, int]] = {}
        self._sem = threading.BoundedSemaphore(max(1, cfg.concurrency))
        self._block_streak = 0
        self._last_rotate = 0.0
        self._session_requests = 0

    # ---- مدیریت backend هر رشته ----
    def _backend(self):
        b = getattr(self._tls, "backend", None)
        if b is None:
            proxy = self.proxies.next() if self.proxies.enabled else None
            b = _load_backend(self.cfg, proxy)
            self._tls.backend = b
            self._tls.proxy = proxy
            with self._lock:
                self._local_backends.append(b)
        return b

    def clear_thread_backend(self) -> None:
        b = getattr(self._tls, "backend", None)
        if b is not None:
            try:
                b.close()
            except Exception:
                pass
            self._tls.backend = None

    def close(self) -> None:
        with self._lock:
            for b in self._local_backends:
                try:
                    b.close()
                except Exception:
                    pass
            self._local_backends = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # ---- چرخش هویت ----
    def rotate_identity(self, reason: str = "") -> None:
        """نشست تازه با هویت/پروکسی/کوکی نو، پس از شکست‌های پیاپی.

        فقط برای بازیابی از حالت بلوکه است؛ نرخ درخواست را افزایش نمی‌دهد
        (محدودساز نرخ همچنان اعمال می‌شود).
        """
        now = time.monotonic()
        with self._lock:
            if now - self._last_rotate < 5.0:
                return
            self._last_rotate = now
            self._session_requests = 0
        old = getattr(self._tls, "proxy", None)
        if old:
            self.proxies.penalize(old, 45.0)
        self.identity = Identity.random(self.cfg)
        self.clear_thread_backend()
        if self.logger and reason:
            self.logger.warn(f"نشست تازه ساخته شد ({reason}).")

    # ---- هستهٔ درخواست ----
    def request(
        self,
        method: str,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        purpose: str = "generic",
        retries: int | None = None,
        expect_json: bool = False,
        use_cache: bool | None = None,
        max_bytes: int | None = None,
        stream: bool = False,
    ) -> Response:
        retries = self.cfg.max_retries if retries is None else retries
        use_cache = self.cfg.respect_cache if use_cache is None else use_cache
        last_exc: Exception | None = None

        if self.breaker.is_open:
            raise CircuitOpenError(
                f"مدار حفاظتی باز است؛ {utils.fa_num(round(self.breaker.remaining()))} ثانیه صبر کنید.",
                url,
            )

        # درخواست شرطی از کش
        cached = self._etag_cache.get(url) if use_cache else None
        req_headers = dict(headers or {})
        req_headers.setdefault("User-Agent", self.identity.user_agent)
        if cached:
            etag, last_modified, _body, _ts, _status = cached
            if etag:
                req_headers["If-None-Match"] = etag
            if last_modified:
                req_headers["If-Modified-Since"] = last_modified

        for attempt in range(retries + 1):
            if self.breaker.is_open:
                raise CircuitOpenError("مدار حفاظتی حین تلاش باز شد.", url)
            self.limiter.wait()
            started = time.monotonic()
            with self._sem:
                backend = self._backend()
                try:
                    raw = backend.request(method, url, req_headers, stream=stream)
                    status = int(getattr(raw, "status_code", 0))
                    resp_headers = {
                        str(k).lower(): str(v)
                        for k, v in dict(getattr(raw, "headers", {}) or {}).items()
                    }
                    content = self._read_body(raw, max_bytes)
                    elapsed = time.monotonic() - started
                except Exception as exc:  # خطای شبکه/زمان
                    elapsed = time.monotonic() - started
                    self.stats.bump("requests")
                    self.stats.bump("total_seconds", elapsed)
                    self.stats.purpose(purpose)
                    self.stats.bump("failed")
                    self.limiter.note_error()
                    last_exc = TransientError(str(exc), url)
                    if attempt < retries:
                        self.stats.bump("retries")
                        self._sleep_backoff(attempt, reason=str(exc)[:60])
                        continue
                    if self.breaker.note_failure() and self.logger:
                        self.logger.warn("مدار حفاظتی باز شد؛ توقف موقت درخواست‌ها.")
                    raise last_exc from exc

            self.stats.bump("requests")
            self.stats.bump("total_seconds", elapsed)
            self.stats.purpose(purpose)
            self.stats.status(status)
            try:
                body_len = len(content)
                self.stats.bump("bytes_down", body_len)
            except Exception:
                pass

            # ۳۰۴: از کش
            if status == 304 and cached:
                etag, last_modified, body, ts, orig_status = cached
                self.stats.bump("not_modified")
                self.stats.bump("ok")
                self.limiter.note_success()
                self.breaker.note_success()
                return Response(
                    url=url,
                    status=orig_status,
                    headers={"content-type": "application/json", "x-from-cache": "1"},
                    content=body,
                    elapsed=elapsed,
                    from_cache=True,
                    purpose=purpose,
                )

            if 200 <= status < 300:
                resp = Response(url, status, resp_headers, content, elapsed, False, purpose)
                if expect_json:
                    try:
                        resp.json()
                    except Exception:
                        # احتمالاً چالش امنیتی با کد ۲۰۰
                        if self._looks_blocked(resp):
                            self._handle_block(resp, purpose, attempt, retries)
                            continue
                        if attempt < retries:
                            self.stats.bump("retries")
                            self._sleep_backoff(attempt, "پاسخ JSON نامعتبر")
                            continue
                        ctype = (resp_headers.get("content-type") or "").lower()
                        if "html" in ctype:
                            raise TransientError(
                                "به‌جای JSON صفحهٔ HTML برگشت — احتمالاً آدرس فهرست "
                                "برای این دوره درست نیست. `digiganj challenge list` را "
                                "ببینید و مسیر/دسته‌بندی‌ها را به‌روز کنید.",
                                url,
                                status,
                            )
                        raise TransientError(
                            f"پاسخ JSON نامعتبر بود (نوع محتوا: {ctype or 'نامشخص'}).",
                            url,
                            status,
                        )
                self.stats.bump("ok")
                self.limiter.note_success()
                self.breaker.note_success()
                self._block_streak = 0
                with self._lock:
                    self._session_requests += 1
                    if use_cache and len(self._etag_cache) < 5000:
                        etag = resp_headers.get("etag", "")
                        lm = resp_headers.get("last-modified", "")
                        if etag or lm:
                            self._etag_cache[url] = (etag, lm, content, time.time(), status)
                return resp

            if status == 404:
                self.stats.bump("failed")
                self.breaker.note_success()
                raise NotFoundError("منبع پیدا نشد (۴۰۴).", url, status)

            if status in (403, 429) or (status == 503 and self._looks_blocked(
                Response(url, status, resp_headers, content, elapsed)
            )):
                self._handle_block(
                    Response(url, status, resp_headers, content, elapsed), purpose, attempt, retries
                )
                last_exc = BlockedError(
                    f"دسترسی موقتاً محدود شد (HTTP {status}).", url, status
                )
                continue

            if status in (500, 502, 503, 504, 522, 524):
                self.stats.bump("failed")
                self.limiter.note_error()
                last_exc = TransientError(f"خطای سرور (HTTP {status}).", url, status)
                if attempt < retries:
                    self.stats.bump("retries")
                    self._sleep_backoff(attempt, f"HTTP {status}")
                    continue
                if self.breaker.note_failure() and self.logger:
                    self.logger.warn("مدار حفاظتی باز شد؛ توقف موقت درخواست‌ها.")
                raise last_exc

            # بقیهٔ کدها
            self.stats.bump("failed")
            last_exc = NetError(f"پاسخ غیرمنتظره (HTTP {status}).", url, status)
            if attempt < retries:
                self.stats.bump("retries")
                self._sleep_backoff(attempt, f"HTTP {status}")
                continue
            raise last_exc

        raise last_exc or NetError("درخواست ناموفق بود.", url)

    # ---- کمکی‌های درخواست ----
    def _handle_block(
        self, resp: Response, purpose: str, attempt: int, retries: int
    ) -> None:
        self.stats.bump("blocked")
        self.stats.bump("failed")
        retry_after = 0.0
        try:
            retry_after = float(resp.headers.get("retry-after", "0") or 0)
        except ValueError:
            retry_after = 0.0
        self.limiter.note_block(retry_after)
        self._block_streak += 1
        if self.on_block:
            try:
                self.on_block(f"HTTP {resp.status} در {purpose}", retry_after)
            except Exception:
                pass
        if self.logger:
            level = self.logger.warn if self._block_streak <= 2 else self.logger.debug
            level(
                f"محدودیت دسترسی (HTTP {resp.status}) در «{purpose}»؛ "
                f"فاصلهٔ درخواست‌ها به {utils.fa_num(f'{self.limiter.interval:.2f}')} ثانیه رسید."
            )
        # چند بلوکهٔ پیاپی → نشست تازه + خنک‌سازی
        if self._block_streak >= 3:
            self.rotate_identity("بلوکهٔ پیاپی")
            time.sleep(min(10.0, 1.5 * self._block_streak))
        if attempt < retries:
            self.stats.bump("retries")
            self._sleep_backoff(attempt, f"HTTP {resp.status}", base=2.0)

    def _sleep_backoff(self, attempt: int, reason: str = "", base: float | None = None) -> None:
        base = self.cfg.backoff_base if base is None else base
        delay = min(self.cfg.backoff_cap, base * (2 ** attempt)) + random.uniform(0, 0.4)
        if self.logger and attempt > 0:
            self.logger.debug(
                f"تلاش مجدد پس از {utils.fa_num(f'{delay:.1f}')} ثانیه ({reason})."
            )
        time.sleep(delay)

    def _read_body(self, raw, max_bytes: int | None) -> bytes:
        content = getattr(raw, "content", None)
        if content is not None:
            data = content
            if max_bytes is not None and len(data) > max_bytes:
                data = data[:max_bytes]
            return data
        # استریم
        chunks: list[bytes] = []
        total = 0
        try:
            it = raw.iter_bytes()
        except AttributeError:
            try:
                it = raw.iter_content(65536)
            except AttributeError:
                it = iter([b""])
        for chunk in it:
            if not chunk:
                continue
            chunks.append(chunk)
            total += len(chunk)
            if max_bytes is not None and total >= max_bytes:
                break
        data = b"".join(chunks)
        return data[:max_bytes] if max_bytes is not None else data

    @staticmethod
    def _looks_blocked(resp: Response) -> bool:
        body = resp.content[:200000].decode("utf-8", errors="ignore").lower()
        markers = (
            "captcha",
            "security",
            "cf-challenge",
            "به نظر می‌رسد",
            "دسترسی شما",
            "درخواست شما",
            "are you a robot",
            "access denied",
            "too many requests",
            "unusual traffic",
        )
        return any(m in body for m in markers)

    # ---- میان‌بُرها ----
    def get(self, url: str, **kwargs) -> Response:
        return self.request("GET", url, **kwargs)

    def head(self, url: str, **kwargs) -> Response:
        return self.request("HEAD", url, **kwargs)

    def get_json(self, url: str, **kwargs) -> Any:
        kwargs.setdefault("expect_json", True)
        kwargs.setdefault("use_cache", True)
        return self.request("GET", url, **kwargs).json()

    def get_limited(self, url: str, max_bytes: int, **kwargs) -> bytes:
        kwargs.setdefault("max_bytes", max_bytes)
        return self.request("GET", url, **kwargs).content

    def probe(self, url: str, **kwargs) -> Response:
        """کم‌هزینه‌ترین راه برای دیدن «آیا تصویر عوض شده؟»

        اگر سرور ETag/Last-Modified بدهد، HEAD + درخواست شرطی کافی است و
        حجم صفر منتقل می‌شود. وگرنه GET با کیفیت پایین.
        """
        mode = self.cfg.probe_mode
        if mode in ("auto", "head"):
            try:
                resp = self.head(url, **kwargs)
                if resp.ok and (
                    resp.headers.get("etag") or resp.headers.get("content-length")
                ):
                    return resp
                if mode == "head":
                    return resp
            except (BlockedError, CircuitOpenError):
                raise
            except NetError:
                if mode == "head":
                    raise
        kwargs.setdefault("max_bytes", 2 * 1024 * 1024)
        return self.get(url, **kwargs)

    # ---- گزارش ----
    def stats_line(self) -> str:
        s = self.stats
        return (
            f"درخواست: {utils.fa_num(s.requests)} | موفق: {utils.fa_num(s.ok)} | "
            f"کش: {utils.fa_num(s.cached)} | بلوکه: {utils.fa_num(s.blocked)} | "
            f"ناموفق: {utils.fa_num(s.failed)} | "
            f"حجم: {utils.human_bytes(s.bytes_down)} | "
            f"فاصله: {utils.fa_num(f'{self.limiter.interval:.2f}')}s"
        )


def make_client(cfg, logger: Logger | None = None, console: Console | None = None,
                on_block: Callable[[str, float], None] | None = None) -> HttpClient:
    pool = ProxyPool(cfg, logger)
    pool.prepare()
    client = HttpClient(cfg, logger=logger, console=console, on_block=on_block, proxies=pool)
    if logger:
        logger.info(
            f"پشتانهٔ شبکه: {getattr(client._backend(), 'name', 'httpx')}"
            + (f" | پروکسی سالم: {utils.fa_num(pool.count)}" if pool.enabled else "")
        )
    return client
