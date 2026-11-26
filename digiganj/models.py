"""مدل‌های داده و تحلیل نام تصویر.

نکتهٔ کلیدی که دوستان در ۱۴۰۰ کشف کردند (و در کد مرجع هم هست): فایل تصویر
رمز روی CDN دیجی‌کالا نامی دارد که به زمان آپلود گره خورده است؛ پس **بدون
دانلود تصویر** می‌توان فهمید کدام تصویر تازه است. این ابزار همان تحلیل را
منظم‌تر انجام می‌دهد: چند الگوی زمانی رایج (یونیکس ثانیه/میلی‌ثانیه،
«YYYYMMDDHHMMSS»، «YYYY_MM_DD…»، هش SHA-1 خالص) پشتیبانی می‌شود.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from . import utils

# --------------------------------------------------------------------------- #
# استخراج و اجزای URL تصویر
# --------------------------------------------------------------------------- #

_SIZE_PARAMS = re.compile(
    r"(?:image/resize|im/resize)[^/]*|(?:x-oss-process=[^&]*)", re.IGNORECASE
)


def normalize_image_url(url: str) -> str:
    """آدرس را برای مقایسه/کلید پایگاه‌داده یکدست می‌کند (پارامترهای پردازش حذف)."""
    url = url.strip()
    if url.startswith("//"):
        url = "https:" + url
    base = url.split("?", 1)[0]
    return base


def image_filename(url: str) -> str:
    base = normalize_image_url(url)
    return base.rsplit("/", 1)[-1]


def image_dir_name(url: str) -> str:
    """پوشهٔ والد روی CDN، مثل «digikala-products» یا «digikala-product-images»."""
    base = normalize_image_url(url)
    parts = base.split("/")
    if len(parts) < 2:
        return ""
    return parts[-2]


def with_suffix(url: str, suffix: str) -> str:
    base = normalize_image_url(url)
    if not suffix:
        return base
    return base + suffix


def host_of(url: str) -> str:
    try:
        from urllib.parse import urlsplit

        return urlsplit(url).netloc.lower()
    except Exception:
        return ""


# --------------------------------------------------------------------------- #
# زمان‌خوانی از نام فایل
# --------------------------------------------------------------------------- #

_EPOCH_MS = re.compile(r"(?<!\d)(1[67]\d{11})(?!\d)")
_EPOCH_S = re.compile(r"(?<!\d)(1[5-9]\d{8}|2[0-1]\d{8})(?!\d)")
_COMPACT = re.compile(r"(?<!\d)(20\d{2})(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])"
                      r"([01]\d|2[0-3])([0-5]\d)([0-5]\d)(?!\d)")
_DATE_UNDERSCORE = re.compile(
    r"(?<!\d)(20\d{2})[_\-.]?(0[1-9]|1[0-2])[_\-.]?(0[1-9]|[12]\d|3[01])"
    r"(?:[_\-.T\s]?([01]\d|2[0-3])[_\-.:]?([0-5]\d)[_\-.:]?([0-5]\d)?)?(?!\d)"
)
_HEX40 = re.compile(r"(?<![0-9a-f])[0-9a-f]{40}(?![0-9a-f])")


def parse_stamp(url: str, epoch_min: int = 1_500_000_000, epoch_max: int = 4_102_444_800):
    """اگر نام فایل زمان داشته باشد، آن را برمی‌گرداند: ``(ts, kind)`` وگرنه None."""
    name = image_filename(url)
    stem = name.rsplit(".", 1)[0]

    m = _EPOCH_MS.search(stem)
    if m:
        ts = int(m.group(1)) / 1000.0
        if epoch_min <= ts <= epoch_max:
            return ts, "epoch_ms"

    m = _EPOCH_S.search(stem)
    if m:
        ts = float(m.group(1))
        if epoch_min <= ts <= epoch_max:
            return ts, "epoch_s"

    m = _COMPACT.search(stem)
    if m:
        y, mo, d, h, mi, s = (int(x) for x in m.groups())
        try:
            return utils.dt_to_ts(utils.datetime(y, mo, d, h, mi, s)), "datetime"
        except ValueError:
            pass

    m = _DATE_UNDERSCORE.search(stem)
    if m:
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        h = int(m.group(4) or 0)
        mi = int(m.group(5) or 0)
        s = int(m.group(6) or 0)
        try:
            return utils.dt_to_ts(utils.datetime(y, mo, d, h, mi, s)), "datetime"
        except ValueError:
            pass

    if _HEX40.search(stem):
        return None, "sha1"
    return None


def looks_sha1_name(url: str) -> bool:
    name = image_filename(url)
    stem = name.rsplit(".", 1)[0]
    return bool(_HEX40.fullmatch(stem) or _HEX40.search(stem))


# --------------------------------------------------------------------------- #
# محصول و گالری
# --------------------------------------------------------------------------- #


@dataclass
class Product:
    id: str
    name: str = ""
    url: str = ""
    price: int = 0
    uri: str = ""  # مثل "/product/dkp-123/نام"
    rank: int = 0
    page: int = 0
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_api(cls, obj: dict[str, Any], page: int = 0, rank: int = 0) -> "Product":
        pid = obj.get("id") or obj.get("product_id") or obj.get("dkp")
        name = obj.get("name") or obj.get("title") or ""
        uri = obj.get("uri") or obj.get("product_url") or obj.get("url") or ""
        if uri and not uri.startswith("http"):
            uri = "/" + uri.lstrip("/")
        elif uri.startswith("http"):
            uri = "/" + uri.split("//", 1)[-1].split("/", 1)[-1]
        price = 0
        pr = obj.get("price") or {}
        if isinstance(pr, dict):
            price = int(pr.get("selling_price") or pr.get("rrp_price") or 0)
        elif isinstance(pr, (int, float)):
            price = int(pr)
        if not price:
            dp = obj.get("default_variant") or {}
            if isinstance(dp, dict):
                price = int(dp.get("price", {}).get("selling_price") or 0) if isinstance(
                    dp.get("price"), dict
                ) else 0
        return cls(
            id=str(pid or ""),
            name=str(name),
            uri=uri,
            price=price,
            rank=rank,
            page=page,
            raw=obj,
        )

    @property
    def dkp(self) -> str:
        return f"dkp-{self.id}"

    def url_for(self, base_url: str) -> str:
        if self.uri and self.uri.startswith("/"):
            return base_url.rstrip("/") + self.uri
        return f"{base_url.rstrip('/')}/product/dkp-{self.id}/"

    def short_name(self, limit: int = 48) -> str:
        return utils.shorten(self.name or self.dkp, limit)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "uri": self.uri,
            "price": self.price,
            "page": self.page,
            "rank": self.rank,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Product":
        return cls(
            id=str(d["id"]),
            name=str(d.get("name", "")),
            uri=str(d.get("uri", "")),
            price=int(d.get("price") or 0),
            page=int(d.get("page") or 0),
            rank=int(d.get("rank") or 0),
        )


@dataclass
class GalleryItem:
    """یک تصویر از گالری محصول، همراه با تحلیل زمان نام فایل."""

    product_id: str
    url: str
    stamp_ts: float | None = None
    stamp_kind: str = ""
    source: str = ""  # api | html

    @classmethod
    def build(cls, product_id: str, url: str, cfg, source: str = "") -> "GalleryItem":
        ts, kind = parse_stamp(url, cfg.epoch_min, cfg.epoch_max)
        return cls(
            product_id=str(product_id),
            url=normalize_image_url(url),
            stamp_ts=ts,
            stamp_kind=kind,
            source=source,
        )

    @property
    def filename(self) -> str:
        return image_filename(self.url)

    def novelty(self, challenge, cfg) -> float | None:
        """چقدر تصویر «تازه» است: ثانیه‌های پس از شروع چالش (منفی = قبل از شروع).

        فایل‌های با زمان نامعتبر/بدون زمان، None برمی‌گردانند؛ فایل‌های قدیمی
        مقدار منفیِ بزرگ می‌گیرند و در رتبه‌بندی پایین می‌آیند.
        """
        if self.stamp_ts is None:
            return None
        return self.stamp_ts - challenge.start_ts

    def in_window(self, challenge, cfg) -> bool:
        if self.stamp_ts is None:
            return False
        return (
            challenge.start_ts - cfg.epoch_window_before
            <= self.stamp_ts
            <= challenge.end_ts + cfg.epoch_window_after
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "product_id": self.product_id,
            "url": self.url,
            "stamp_ts": self.stamp_ts,
            "stamp_kind": self.stamp_kind,
            "source": self.source,
        }
