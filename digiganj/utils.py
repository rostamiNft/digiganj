"""ابزارهای پایه: زمان، تقویم جلالی، اعداد فارسی، هش و فایل.

همهٔ خروجی‌های این ماژول برای نمایش به کاربر فارسی‌سازی می‌شوند؛ اما داده‌های
ذخیره‌شده (پایگاه‌داده، JSON) همیشه لاتین و ماشین‌خوان می‌مانند تا پردازش و
مقایسه دقیق باشد.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

try:  # منطقهٔ زمانی تهران؛ در نبود tzdata از آفست ثابت استفاده می‌کنیم.
    from zoneinfo import ZoneInfo

    TEHRAN = ZoneInfo("Asia/Tehran")
except Exception:  # pragma: no cover - فقط روی سیستم‌های بدون tzdata
    TEHRAN = timezone(timedelta(hours=3, minutes=30))

UTC = timezone.utc

# --------------------------------------------------------------------------- #
# زمان
# --------------------------------------------------------------------------- #


def now_utc() -> datetime:
    """زمان کنونی به‌صورت aware در UTC."""
    return datetime.now(UTC)


def now_ts() -> float:
    """زمان کنونی به ثانیه (یونیکس)."""
    return time.time()


def to_tehran(dt: datetime) -> datetime:
    """تبدیل هر زمان aware به منطقهٔ زمانی تهران."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=TEHRAN)
    return dt.astimezone(TEHRAN)


def parse_dt(text: str) -> datetime:
    """تجزیهٔ زمان ISO؛ زمان بدون منطقه، تهران فرض می‌شود."""
    text = text.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=TEHRAN)
    return dt


def ts_to_dt(ts: float) -> datetime:
    return datetime.fromtimestamp(ts, UTC)


def dt_to_ts(dt: datetime) -> float:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=TEHRAN)
    return dt.timestamp()


def iso_utc(ts: float | datetime) -> str:
    dt = ts_to_dt(ts) if isinstance(ts, (int, float)) else ts.astimezone(UTC)
    return dt.isoformat(timespec="seconds")


def time_only(ts: float, fa: bool = True) -> str:
    """ساعت:دقیقه:ثانیه به وقت تهران."""
    s = to_tehran(ts_to_dt(ts)).strftime("%H:%M:%S")
    return fa_num(s) if fa else s


def sleep_until(ts: float) -> None:
    """خواب دقیق تا یک لحظهٔ مشخص (حساس به تغییر ساعت سیستم)."""
    while True:
        remaining = ts - time.time()
        if remaining <= 0:
            return
        time.sleep(min(remaining, 0.25))


# --------------------------------------------------------------------------- #
# تقویم جلالی (پیاده‌سازی مستقل از وابستگی خارجی)
# --------------------------------------------------------------------------- #

_JALALI_MONTHS = (
    "فروردین",
    "اردیبهشت",
    "خرداد",
    "تیر",
    "مرداد",
    "شهریور",
    "مهر",
    "آبان",
    "آذر",
    "دی",
    "بهمن",
    "اسفند",
)


def gregorian_to_jalali(gy: int, gm: int, gd: int) -> tuple[int, int, int]:
    """تبدیل تاریخ میلادی به جلالی (الگوریتم استاندارد jdf)."""
    g_d_m = (0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334)
    gy2 = gy + 1 if gm > 2 else gy
    days = (
        355666
        + 365 * gy
        + (gy2 + 3) // 4
        - (gy2 + 99) // 100
        + (gy2 + 399) // 400
        + gd
        + g_d_m[gm - 1]
    )
    jy = -1595 + 33 * (days // 12053)
    days %= 12053
    jy += 4 * (days // 1461)
    days %= 1461
    if days > 365:
        jy += (days - 1) // 365
        days = (days - 1) % 365
    if days < 186:
        jm = 1 + days // 31
        jd = 1 + days % 31
    else:
        jm = 7 + (days - 186) // 30
        jd = 1 + (days - 186) % 30
    return jy, jm, jd


def jalali_to_gregorian(jy: int, jm: int, jd: int) -> tuple[int, int, int]:
    """تبدیل تاریخ جلالی به میلادی (الگوریتم استاندارد jdf)."""
    jy += 1595
    days = -355668 + 365 * jy + (jy // 33) * 8 + ((jy % 33) + 3) // 4 + jd
    days += (jm - 1) * 31 if jm < 7 else (jm - 7) * 30 + 186
    gy = 400 * (days // 146097)
    days %= 146097
    if days > 36524:
        days -= 1
        gy += 100 * (days // 36524)
        days %= 36524
        if days >= 365:
            days += 1
    gy += 4 * (days // 1461)
    days %= 1461
    if days > 365:
        gy += (days - 1) // 365
        days = (days - 1) % 365
    gd = days + 1
    sal_a = (
        0,
        31,
        29 if (gy % 4 == 0 and gy % 100 != 0) or gy % 400 == 0 else 28,
        31,
        30,
        31,
        30,
        31,
        31,
        30,
        31,
        30,
        31,
    )
    gm = 0
    while gm < 13 and gd > sal_a[gm]:
        gd -= sal_a[gm]
        gm += 1
    return gy, gm, gd


def jalali_tuple(ts: float | datetime) -> tuple[int, int, int, int, int, int]:
    dt = to_tehran(ts_to_dt(ts) if isinstance(ts, (int, float)) else ts)
    jy, jm, jd = gregorian_to_jalali(dt.year, dt.month, dt.day)
    return jy, jm, jd, dt.hour, dt.minute, dt.second


def jalali_date(ts: float | datetime, fa: bool = True) -> str:
    jy, jm, jd, _, _, _ = jalali_tuple(ts)
    s = f"{jy:04d}/{jm:02d}/{jd:02d}"
    return fa_num(s) if fa else s


def jalali_datetime(ts: float | datetime, fa: bool = True) -> str:
    jy, jm, jd, h, m, s = jalali_tuple(ts)
    text = f"{jy:04d}/{jm:02d}/{jd:02d} - {h:02d}:{m:02d}:{s:02d}"
    return fa_num(text) if fa else text


def jalali_short(ts: float | datetime, fa: bool = True) -> str:
    """تاریخ جلالی کوتاه: «۱۴۰۵/۰۹/۰۵ ۱۲:۳۴» (بدون ثانیه)."""
    jy, jm, jd, h, m, _s = jalali_tuple(ts)
    text = f"{jy:04d}/{jm:02d}/{jd:02d} {h:02d}:{m:02d}"
    return fa_num(text) if fa else text


def jalali_pretty(ts: float | datetime) -> str:
    """مثل «۵ آذر ۱۴۰۵، ساعت ۱۰:۰۰:۰۰»."""
    jy, jm, jd, h, mi, s = jalali_tuple(ts)
    return (
        f"{fa_num(str(jd))} {_JALALI_MONTHS[jm - 1]} {fa_num(str(jy))}، "
        f"ساعت {fa_num(f'{h:02d}:{mi:02d}:{s:02d}')}"
    )


def filename_stamp(ts: float | None = None) -> str:
    """مُهر زمان مناسب نام فایل (میلادی، مرتب‌شدنی)."""
    dt = ts_to_dt(ts if ts is not None else time.time())
    return to_tehran(dt).strftime("%Y%m%d-%H%M%S")


# --------------------------------------------------------------------------- #
# نمایش فارسی
# --------------------------------------------------------------------------- #

_FA_DIGITS = str.maketrans("0123456789", "۰۱۲۳۴۵۶۷۸۹")


def fa_num(value: Any) -> str:
    """ارقام لاتین را فارسی می‌کند؛ بقیهٔ متن دست‌نخورده می‌ماند."""
    return str(value).translate(_FA_DIGITS)


def human_duration(seconds: float, precision: int = 1) -> str:
    """مدت‌زمان خوانا: «۲.۴ ثانیه»، «۳ دقیقه و ۱۲ ثانیه»."""
    seconds = max(0.0, float(seconds))
    if seconds < 1:
        return f"{fa_num(f'{seconds * 1000:.0f}')} میلی‌ثانیه"
    if seconds < 60:
        return f"{fa_num(f'{seconds:.{precision}f}')} ثانیه"
    minutes, sec = divmod(int(round(seconds)), 60)
    if minutes < 60:
        return f"{fa_num(minutes)} دقیقه و {fa_num(sec)} ثانیه"
    hours, minutes = divmod(minutes, 60)
    return f"{fa_num(hours)} ساعت و {fa_num(minutes)} دقیقه"


def human_bytes(n: int | float) -> str:
    units = ("بایت", "کیلوبایت", "مگابایت", "گیگابایت")
    value = float(n)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{fa_num(f'{value:.1f}')} {unit}"
        value /= 1024
    return f"{fa_num(f'{value:.1f}')} بایت"


_DURATION_RE = re.compile(r"(\d+(?:\.\d+)?)\s*([smhdSMHD]?)")


def parse_duration(text: str | int | float) -> float:
    """«۳۰s»، «2m»، «۱.۵h» یا عدد خام (ثانیه) را به ثانیه تبدیل می‌کند."""
    if isinstance(text, (int, float)):
        return float(text)
    text = str(text).strip()
    total = 0.0
    found = False
    for value, unit in _DURATION_RE.findall(text):
        found = True
        factor = {"": 1, "s": 1, "m": 60, "h": 3600, "d": 86400}[unit.lower()]
        total += float(value) * factor
    if not found:
        raise ValueError(f"مدت‌زمان نامعتبر: {text!r}")
    return total


# --------------------------------------------------------------------------- #
# هش و فایل
# --------------------------------------------------------------------------- #


def sha1_hex(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def md5_hex(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def file_sha1(path: str | Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def atomic_write(path: str | Path, data: bytes) -> None:
    """نوشتن اتمیک تا فایل نیمه‌کاره باقی نماند."""
    p = Path(path)
    ensure_dir(p.parent)
    fd, tmp = tempfile.mkstemp(dir=str(p.parent), prefix=".tmp-")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        os.replace(tmp, p)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def write_json_str(obj: Any, indent: int | None = None) -> str:
    """سریال‌سازی JSON به رشته (برای ذخیره در پایگاه‌داده)."""
    return json.dumps(obj, ensure_ascii=False, indent=indent, default=str)


def write_json(path: str | Path, obj: Any, indent: int = 2) -> None:
    data = json.dumps(obj, ensure_ascii=False, indent=indent, default=str)
    atomic_write(path, data.encode("utf-8"))


def read_json(path: str | Path, default: Any = None) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return default


def read_text(path: str | Path, default: str = "") -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            return fh.read()
    except OSError:
        return default


def safe_filename(name: str, max_len: int = 120) -> str:
    """نام فایل امن با حفظ حروف فارسی/لاتین و ارقام."""
    name = re.sub(r"[^\w\u0600-\u06FF.\-]+", "_", name, flags=re.UNICODE)
    return name[:max_len].strip("._") or "file"


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


# --------------------------------------------------------------------------- #
# ریاضی/تصادفی
# --------------------------------------------------------------------------- #


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def jitter(low: float = 0.0, high: float = 0.25) -> float:
    return random.uniform(low, high)


def pct(part: float, whole: float, digits: int = 1) -> str:
    if not whole:
        return fa_num("0%")
    return f"{fa_num(f'{100 * part / whole:.{digits}f}')}٪"


def shorten(text: str, limit: int = 80) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def bidi_ltr(text: str) -> str:
    """متن لاتین را برای نمایش درست داخل جملهٔ فارسی می‌پیچد."""
    return f"\u2068{text}\u2069"
