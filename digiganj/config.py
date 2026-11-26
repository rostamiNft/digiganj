"""پیکربندی ابزار: مسیرها، پارامترهای شبکه، تقویم چالش‌ها و تنظیمات OCR.

پیکربندی به‌صورت JSON نگهداری می‌شود (خوانا، بدون وابستگی اضافه). کلیدهای
ناشناخته نادیده گرفته می‌شوند تا نسخه‌های قدیمی/جدید فایل سازگار بمانند.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import utils

DEFAULT_CONFIG_NAME = "config.json"

# نمونهٔ UAهای واقعی کروم/فایرفاکس دسکتاپ. برای هر نشست یکی انتخاب و حفظ می‌شود؛
# چرخش UA فقط هنگام برخورد با خطا انجام می‌شود تا اثر انگشت نشست پایدار بماند.
DEFAULT_USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/131.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/130.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:133.0) Gecko/20100101 Firefox/133.0",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/129.0.0.0 Safari/537.36",
]

DEFAULT_BROWSER_PROFILES = [
    {"name": "chrome-win", "platform": '"Windows"', "mobile": "?0", "brands": "chrome"},
    {"name": "chrome-mac", "platform": '"macOS"', "mobile": "?0", "brands": "chrome"},
    {"name": "firefox", "platform": '"Linux"', "mobile": "?0", "brands": "firefox"},
]


@dataclass
class Challenge:
    """یک دورهٔ مسابقهٔ شکار گنج."""

    id: str
    title: str
    starts_at: str
    ends_at: str
    prize: str = ""
    note: str = ""
    tags: list[str] = field(default_factory=list)
    list_path: str = "/treasure-hunt/products/"
    era: str = ""  # مثل «۱۴۰۰» یا «۱۴۰۳»
    # ---- مسیر جستجوی محصولات (از دورهٔ ۱۴۰۵ به بعد) ----
    # در دوره‌های جدید، فهرست محصولات یک «جستجوی دسته‌بندی» است که خودِ
    # صفحهٔ ایونت اعلام می‌کند؛ مثلاً:
    #   /search/?categories[0]=10942&categories[1]=5604&sort=7
    # اگر خالی باشد، ``list_path`` استفاده می‌شود (رفتار دوره‌های قدیم).
    search_path: str = ""
    # شناسهٔ دسته‌بندی‌هایی که کدها میان عکس‌هایشان پنهان است
    category_ids: list[int] = field(default_factory=list)
    # آیا کد گنج در این دوره «DK + حروف» است
    code_format: str = ""  # "" = خودکار از روی CODE_FORMATS
    enabled: bool = True

    # ---- زمان‌ها (محاسبه‌شده) ----
    @property
    def start_ts(self) -> float:
        return utils.dt_to_ts(utils.parse_dt(self.starts_at))

    @property
    def end_ts(self) -> float:
        return utils.dt_to_ts(utils.parse_dt(self.ends_at))

    @property
    def start_jalali(self) -> str:
        return utils.jalali_pretty(self.start_ts)

    @property
    def end_jalali(self) -> str:
        return utils.jalali_pretty(self.end_ts)

    @property
    def window_seconds(self) -> float:
        return max(0.0, self.end_ts - self.start_ts)

    def contains(self, ts: float) -> bool:
        return self.start_ts <= ts <= self.end_ts

    def phase(self, ts: float | None = None) -> str:
        ts = utils.now_ts() if ts is None else ts
        if ts < self.start_ts:
            return "upcoming"
        if ts > self.end_ts:
            return "finished"
        return "active"

    def phase_fa(self, ts: float | None = None) -> str:
        return {"upcoming": "پیش‌رو", "active": "در جریان", "finished": "پایان‌یافته"}[
            self.phase(ts)
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "starts_at": self.starts_at,
            "ends_at": self.ends_at,
            "prize": self.prize,
            "note": self.note,
            "tags": list(self.tags),
            "list_path": self.list_path,
            "search_path": self.search_path,
            "category_ids": list(self.category_ids),
            "code_format": self.code_format,
            "era": self.era,
            "enabled": self.enabled,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Challenge":
        return cls(
            id=str(d["id"]),
            title=str(d.get("title") or d["id"]),
            starts_at=str(d["starts_at"]),
            ends_at=str(d["ends_at"]),
            prize=str(d.get("prize", "")),
            note=str(d.get("note", "")),
            tags=list(d.get("tags", []) or []),
            list_path=str(d.get("list_path", "/treasure-hunt/products/")),
            search_path=str(d.get("search_path", "") or ""),
            category_ids=[int(x) for x in (d.get("category_ids") or [])],
            code_format=str(d.get("code_format", "") or ""),
            era=str(d.get("era", "")),
            enabled=bool(d.get("enabled", True)),
        )


def _default_challenges() -> list[Challenge]:
    """چالش‌های نمونه.

    زمان‌ها برای «تمرین» هستند تا کاربر بتواند کل چرخهٔ ابزار را همین حالا
    بیازماید. برای ایونت واقعی، `digiganj config init` را اجرا و زمان‌های
    واقعی را در config.json وارد کنید. فرمت زمان: ISO-8601 با منطقهٔ تهران.
    """
    return [
        Challenge(
            id="rehearsal",
            title="تمرین محلی (غیرفعال — برای آزمون فعالش کنید)",
            starts_at="2020-01-01T00:00:00+03:30",
            ends_at="2035-01-01T00:00:00+03:30",
            prize="—",
            note="برای آزمودن جریان کار روی دادهٔ نمونه. چون این بازه همیشه "
                 "«جاری» است، به‌طور پیش‌فرض غیرفعال است تا چالش واقعی را پنهان نکند.",
            tags=["تمرین"],
            era="تمرین",
            enabled=False,
        ),
        Challenge(
            id="bf-1400",
            title="شکار گنج بلک‌فرایدی ۱۴۰۰",
            starts_at="2021-11-24T00:00:00+03:30",
            ends_at="2021-11-30T23:59:59+03:30",
            prize="کد تخفیف ۹۹٪ (آیفون ۱۳)",
            note="دورهٔ تاریخی؛ برای بازپخش/پژوهش.",
            tags=["تاریخی", "بلک‌فرایدی"],
            era="۱۴۰۰",
        ),
        Challenge(
            id="bf-1403",
            title="شکار گنج بلک‌فرایدی ۱۴۰۳",
            starts_at="2024-11-20T00:00:00+03:30",
            ends_at="2024-11-30T23:59:59+03:30",
            prize="کد تخفیف ۹۹٪ / طلای دیجیتال",
            note="دورهٔ تاریخی؛ برای بازپخش/پژوهش.",
            tags=["تاریخی", "بلک‌فرایدی"],
            era="۱۴۰۳",
        ),
        Challenge(
            id="bf-1405",
            title="جاکدی / شکار گنج ۱۴۰۵",
            starts_at="2026-11-24T00:00:00+03:30",
            ends_at="2026-12-06T23:59:59+03:30",
            prize="۹۹٪ تخفیف یکی از کالاهای جایزه + طلای دیجیتال",
            note=(
                "قالب ۱۴۰۵: کد گنج «DK + ۳ یا ۴ حرف» (۵ یا ۶ کاراکتر، فقط حروف) "
                "و در گالری عکس کالاهای دسته‌بندی‌های اعلام‌شده پنهان می‌شود. "
                "دسته‌بندی‌ها هر ساعت در صفحهٔ /landings/prize-catcher/?tab=hunt "
                "اعلام می‌شوند؛ شناسه‌ها را از لینک «دیدن کالاها» همان صفحه بردارید."
            ),
            tags=["جاکدی", "بلک‌فرایدی", "۱۴۰۵"],
            era="۱۴۰۵",
            search_path=(
                "/search/?categories%5B0%5D=10942&categories%5B1%5D=5604"
                "&categories%5B2%5D=11225&categories%5B3%5D=2402"
                "&categories%5B4%5D=10476&categories%5B5%5D=11112"
                "&categories%5B6%5D=11586&sort=7"
            ),
            category_ids=[10942, 5604, 11225, 2402, 10476, 11112, 11586],
            code_format="dk-hunt",
        ),
    ]


@dataclass
class Config:
    # ---- مسیرها ----
    base_url: str = "https://www.digikala.com"
    api_base_url: str = "https://api.digikala.com"
    data_dir: str = "data"
    scans_dir: str = "scans"
    reports_dir: str = "reports"
    logs_dir: str = "logs"
    db_path: str = "data/digiganj.sqlite"
    log_file: str = "logs/digiganj.log"
    codes_file: str = "data/codes.txt"
    cookies_file: str = ""  # اختیاری: کوکی‌های نشست خودتان (فقط برای بررسی دستی)
    cookie_header: str = ""  # جایگزین ساده: رشتهٔ کامل کوکی

    # ---- آدرس‌ها ----
    products_path: str = "/ajax/treasure-hunt/products/"
    gallery_api_path: str = "/v1/product/dkp-{id}/"
    gallery_html_path: str = "/product/dkp-{id}/"
    gallery_source: str = "auto"  # auto | api | html
    small_image_suffix: str = "?x-oss-process=image/resize,h_160/quality,q_40"
    probe_image_suffix: str = "?x-oss-process=image/resize,h_240/quality,q_25"
    full_image_suffix: str = "?x-oss-process=image/resize,w_1200/quality,q_85"

    # ---- شبکه ----
    concurrency: int = 10
    timeout: float = 15.0
    connect_timeout: float = 8.0
    max_retries: int = 4
    backoff_base: float = 0.7
    backoff_cap: float = 20.0
    http2: bool = True
    transport: str = "auto"  # auto | httpx | curl_cffi
    impersonate: str = "chrome"  # فقط برای curl_cffi
    verify_tls: bool = True
    proxies: list[str] = field(default_factory=list)
    proxy_health_check: bool = True
    proxy_health_url: str = "https://www.digikala.com/"
    circuit_failures: int = 25
    circuit_cooldown: float = 30.0
    list_page_batch: int = 6
    gallery_resync_every: float = 45.0
    respect_cache: bool = True
    page_size_hint: int = 25
    user_agents: list[str] = field(default_factory=lambda: list(DEFAULT_USER_AGENTS))
    browser_profiles: list[dict[str, str]] = field(
        default_factory=lambda: copy.deepcopy(DEFAULT_BROWSER_PROFILES)
    )

    # ---- بودجه/محدودیت (برای رفتار مؤدبانه و پایدار) ----
    max_products: int = 0  # 0 = بدون سقف
    max_images_per_product: int = 40
    max_download_bytes: int = 4 * 1024 * 1024
    cycle_pause: float = 0.4
    status_every: float = 20.0
    probe_mode: str = "auto"  # auto | head | get
    max_new_images_per_cycle: int = 400

    # ---- تحلیل ----
    hero_strategy: str = "auto"  # auto | content | epoch
    epoch_min: int = 1_500_000_000
    epoch_max: int = 4_102_444_800
    epoch_window_before: float = 600.0
    epoch_window_after: float = 1800.0
    alert_score: int = 55
    alert_cooldown: float = 600.0  # فاصلهٔ حداقلی بین دو هشدار برای یک تصویر

    # ---- OCR ----
    ocr_enabled: bool = True
    ocr_lang: str = "eng"
    ocr_psm: int = 7
    ocr_min_score: float = 7.0
    ocr_max_pixels: int = 2_000_000
    ocr_min_interval: float = 0.35
    ocr_try_raw: bool = True
    code_regex: str = r"(?=[A-Za-z0-9]*[A-Za-z])(?=[A-Za-z0-9]*\d)[A-Za-z0-9]{8,24}"
    ocr_max_candidates: int = 12

    # ---- هشدار/گزارش ----
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""
    webhook_url: str = ""
    report_top_alerts: int = 40
    report_embed_images: bool = False
    report_thumb_px: int = 420

    # ---- عمومی ----
    log_level: str = "INFO"
    challenges: list[Challenge] = field(default_factory=_default_challenges)

    # ------------------------------------------------------------------ #
    # ساخت/بارگذاری/ذخیره
    # ------------------------------------------------------------------ #

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Config":
        data = dict(data or {})
        raw_challenges = data.pop("challenges", None)
        known = {f for f in cls.__dataclass_fields__ if f != "challenges"}
        kwargs = {k: v for k, v in data.items() if k in known}
        cfg = cls(**kwargs)
        if raw_challenges is not None:
            cfg.challenges = [Challenge.from_dict(c) for c in raw_challenges]
        cfg.validate()
        return cfg

    @classmethod
    def load(cls, path: str | Path = DEFAULT_CONFIG_NAME) -> "Config":
        p = Path(path)
        if not p.exists():
            return cls()
        with open(p, "r", encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if name == "challenges":
                out[name] = [c.to_dict() for c in value]
            else:
                out[name] = copy.deepcopy(value)
        return out

    def save(self, path: str | Path = DEFAULT_CONFIG_NAME) -> Path:
        p = Path(path)
        utils.write_json(p, self.to_dict())
        return p

    # ------------------------------------------------------------------ #
    # اعتبارسنجی و کمکی‌ها
    # ------------------------------------------------------------------ #

    def validate(self) -> None:
        problems: list[str] = []
        if not self.challenges:
            problems.append("هیچ چالشی تعریف نشده است.")
        seen: set[str] = set()
        for c in self.challenges:
            if c.id in seen:
                problems.append(f"شناسهٔ تکراری چالش: {c.id}")
            seen.add(c.id)
            try:
                if c.end_ts <= c.start_ts:
                    problems.append(f"پایان چالش {c.id} پیش از شروع آن است.")
            except Exception as exc:  # pragma: no cover - ورودی کاربر
                problems.append(f"زمان چالش {c.id} نامعتبر است: {exc}")
        if self.concurrency < 1:
            problems.append("concurrency باید حداقل ۱ باشد.")
        if self.probe_mode not in {"auto", "head", "get"}:
            problems.append("probe_mode باید auto/head/get باشد.")
        if self.gallery_source not in {"auto", "api", "html"}:
            problems.append("gallery_source باید auto/api/html باشد.")
        if problems:
            raise ValueError("پیکربندی نامعتبر است:\n - " + "\n - ".join(problems))

    # ---- مسیرها ----
    def path(self, *parts: str) -> Path:
        """ساخت مسیر داخل ریشهٔ داده/گزارش."""
        return Path(parts[0], *parts[1:]) if parts else Path(".")

    def ensure_dirs(self) -> None:
        for d in (
            self.data_dir,
            self.scans_dir,
            self.reports_dir,
            self.logs_dir,
            str(Path(self.db_path).parent),
        ):
            utils.ensure_dir(d)

    # ---- چالش‌ها ----
    def enabled_challenges(self) -> list[Challenge]:
        return [c for c in self.challenges if c.enabled]

    def challenge(self, cid: str) -> Challenge:
        for c in self.challenges:
            if c.id == cid:
                return c
        raise KeyError(f"چالش «{cid}» پیدا نشد.")

    def current_challenge(self, ts: float | None = None) -> Challenge | None:
        ts = utils.now_ts() if ts is None else ts
        active = [c for c in self.enabled_challenges() if c.contains(ts)]
        if active:
            return min(active, key=lambda c: c.end_ts - c.start_ts)
        return None

    def next_challenge(self, ts: float | None = None) -> Challenge | None:
        ts = utils.now_ts() if ts is None else ts
        future = [c for c in self.enabled_challenges() if c.start_ts > ts]
        return min(future, key=lambda c: c.start_ts) if future else None

    def resolve_challenge(
        self, cid: str | None = None, ts: float | None = None, allow_finished: bool = True
    ) -> Challenge:
        """انتخاب چالش: شناسهٔ صریح، چالش جاری، بعدی، یا آخرین چالش فعال‌شده."""
        if cid and cid != "auto":
            return self.challenge(cid)
        ts = utils.now_ts() if ts is None else ts
        current = self.current_challenge(ts)
        if current:
            return current
        if not allow_finished:
            nxt = self.next_challenge(ts)
            if nxt:
                return nxt
        finished = [c for c in self.enabled_challenges() if c.end_ts < ts]
        if finished:
            return max(finished, key=lambda c: c.end_ts)
        nxt = self.next_challenge(ts)
        if nxt:
            return nxt
        if self.challenges:
            return self.challenges[0]
        raise ValueError("هیچ چالشی تعریف نشده است.")

    # ---- آدرس‌ها ----
    def url(self, path: str) -> str:
        if path.startswith("http"):
            return path
        return self.base_url.rstrip("/") + "/" + path.lstrip("/")

    def api_url(self, path: str) -> str:
        if path.startswith("http"):
            return path
        return self.api_base_url.rstrip("/") + "/" + path.lstrip("/")

    def challenge_url(self, challenge: Challenge) -> str:
        """آدرسی که فهرست محصولات دوره از آن خوانده می‌شود.

        اگر دوره مسیر جستجوی دسته‌بندی داشته باشد، همان مقدم است: در
        دوره‌های جدید، دسته‌بندی‌ها هر ساعت عوض می‌شوند و در خود صفحهٔ
        ایونت اعلام می‌شوند.
        """
        if challenge.search_path:
            return self.url(challenge.search_path)
        if challenge.category_ids:
            return self.search_url(challenge.category_ids)
        return self.url(challenge.list_path)

    def search_url(self, category_ids: "list[int]", sort: int = 7) -> str:
        """ساخت آدرس جستجوی دسته‌بندی، هم‌شکل با لینک صفحهٔ ایونت."""
        from urllib.parse import quote

        parts = "&".join(
            quote(f"categories[{i}]") + f"={int(cid)}"
            for i, cid in enumerate(category_ids)
        )
        return self.url(f"/search/?{parts}&sort={int(sort)}")

    def gallery_api_url(self, product_id: str | int) -> str:
        """آدرس API گالری محصول: ``/v1/product/dkp-{id}/``."""
        return self.api_url(self.gallery_api_path.format(id=product_id))

    def product_url(self, product_id: str | int) -> str:
        """آدرس عمومی صفحهٔ محصول: ``/product/dkp-{id}/``."""
        return self.url(self.gallery_html_path.format(id=product_id))
