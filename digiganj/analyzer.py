"""تحلیل: امتیازدهی و انتخاب «کاندیدای رمز» از میان تصاویر.

سه سیگنال، هم‌راستا با کشف دوستان در ۱۴۰۰:

۱) **زمانِ نام فایل** (``stamp_ts``) — قوی‌ترین سیگنال، و رایگان: هر تصویری که
   زمانش در پنجرهٔ مجازِ چالش باشد، اولویت بالایی می‌گیرد. همین است که
   «بدون دانلود تصویر» تشخیص را ممکن می‌کند.
۲) **نو بودن** (اولین مشاهدهٔ ما در این چرخه) — هر تصویری که تازه در گالری
   ظاهر شود، حتی اگر نامش زمان نداشته باشد.
۳) **محتوای تصویر** — فقط برای چند کاندیدای برتر، با OCR (tesseract یا موتور
   داخلی). به‌علاوه تحلیل سادهٔ «ناحیهٔ متن/کادر» که بدون OCR هم می‌تواند
   نشانه بدهد.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from . import models, utils


# --------------------------------------------------------------------------- #
# امتیازدهی
# --------------------------------------------------------------------------- #


@dataclass
class Score:
    value: float
    reasons: list[str] = field(default_factory=list)
    kind: str = "normal"  # normal | epoch | new | ocr | strong

    def add(self, amount: float, reason: str) -> None:
        self.value += amount
        self.reasons.append(reason)


def score_image(image: dict[str, Any], challenge, cfg, *,
                is_new: bool = False, hero_product: str = "",
                ocr_codes: Iterable[str] = (), ocr_score: float = 0.0) -> Score:
    """امتیاز یک تصویر را با دلایل فارسی برمی‌گرداند.

    «پنجرهٔ تمرکز» = ۳۰ دقیقه قبل تا ۳۰ دقیقه بعد از شروع چالش. این بازه
    همان زمانی است که رمز منتشر می‌شود. برای ایونت‌های چندروزه که هر روز
    یک رمز منتشر می‌کنند، کاربر می‌تواند پنجره را در تنظیمات بزرگ‌تر کند.
    """
    sc = Score(0.0)
    stamp = image.get("stamp_ts")
    pid = str(image.get("product_id", ""))

    if stamp is not None:
        try:
            delta = float(stamp) - challenge.start_ts
        except (TypeError, ValueError):
            delta = None
        if delta is not None:
            if -cfg.epoch_window_before <= delta <= cfg.epoch_window_after:
                # ← سیگنال اصلی: «تازه منتشر شده»
                sc.add(
                    45.0,
                    f"زمان فایل در پنجرهٔ مجاز ({utils.human_duration(max(0, delta))} پس از شروع)",
                )
                sc.kind = "epoch"
            elif delta < 0:
                # تصویر قدیمی (پیش از شروع) — هرچه قدیمی‌تر کم‌ارزش‌تر
                days = abs(delta) / 86400.0
                penalty = min(20.0, days)
                sc.add(
                    -penalty,
                    f"زمان فایل {utils.human_duration(abs(delta))} پیش از شروع است",
                )
            else:
                # تصویر پس از شروع ولی خارج از پنجرهٔ تمرکز (چالش ادامه دارد)
                sc.add(-3.0,
                       f"زمان فایل {utils.human_duration(delta)} پس از شروع، "
                       "خارج از پنجرهٔ تمرکز است")
        if image.get("stamp_kind") == "sha1":
            sc.add(3.0, "نام فایل هش SHA-1 خالص است (الگوی معمول فایل‌های قدیمی)")

    if is_new:
        sc.add(8.0, "نخستین بار در گالری دیده شد")
    if hero_product and pid == str(hero_product):
        sc.add(30.0, "محصول «گنج» (HERO) است")
    if ocr_codes:
        sc.add(60.0 + min(20.0, 20.0 * ocr_score), "متن شبیه رمز در تصویر شناسایی شد")
        sc.kind = "strong" if sc.value >= cfg.alert_score else sc.kind
    elif ocr_score > 0:
        sc.add(min(12.0, 12.0 * ocr_score), "ناحیهٔ متن‌مانند در تصویر دیده شد")

    sc.value = max(0.0, min(100.0, sc.value))  # سقف ۱۰۰ برای قابل‌فهم بودن
    return sc


# --------------------------------------------------------------------------- #
# برنامه‌ریزی کار
# --------------------------------------------------------------------------- #


@dataclass
class ScanPlan:
    """چه کاری، به چه ترتیبی، برای کدام محصولات."""

    hero_product: str = ""
    rank_products: list[str] = field(default_factory=list)
    rest_products: list[str] = field(default_factory=list)
    watch_products: list[str] = field(default_factory=list)
    total: int = 0


def plan_products(products: list[dict[str, Any]], cfg, new_ids: Iterable[str] = (),
                  hero_id: str = "", max_n: int = 0) -> ScanPlan:
    """اولویت‌بندی محصولات: HERO، سپس تازه‌ها، سپس بقیه.

    «HERO» محصولی است که گالری‌اش در پنجرهٔ چالش بارگذاری شده (مثلاً
    محصول خطای صفحه/بنر). اگر معلوم نباشد، خالی می‌ماند.
    """
    by_id = {str(p["id"]): p for p in products}
    order: list[str] = []
    hero = str(hero_id) if hero_id else ""
    if hero and hero in by_id:
        order.append(hero)
    new_set = {str(i) for i in new_ids}
    for pid in sorted(new_set, key=lambda p: by_id[p]["gallery_ts"] if p in by_id else 0):
        if pid in by_id and pid not in order:
            order.append(pid)
    for pid in sorted(by_id, key=lambda p: (
        by_id[p].get("gallery_ts", 0) or 0,
        by_id[p].get("page", 0) or 0,
    )):
        if pid not in order:
            order.append(pid)
    if max_n:
        order = order[:max_n]
    return ScanPlan(
        hero_product=hero,
        rank_products=order,
        rest_products=[],
        watch_products=[p for p in order if p in new_set],
        total=len(order),
    )


# --------------------------------------------------------------------------- #
# خلاصه برای گزارش
# --------------------------------------------------------------------------- #


def summarize_candidates(candidates: list[dict[str, Any]], challenge, cfg) -> dict[str, Any]:
    """خلاصهٔ آماری نامزدها برای چاپ/گزارش."""
    total = len(candidates)
    epochs = [c for c in candidates if c.get("stamp_ts")]
    new = [c for c in candidates if c.get("alert_ts")]
    by_product: dict[str, int] = {}
    for c in candidates:
        by_product[str(c.get("product_id"))] = by_product.get(str(c.get("product_id")), 0) + 1
    return {
        "total": total,
        "timed": len(epochs),
        "alerted": len(new),
        "products": len(by_product),
        "by_product": dict(sorted(by_product.items(), key=lambda kv: -kv[1])[:20]),
    }


def build_evidence_rows(candidates: list[dict[str, Any]], challenge, cfg,
                        limit: int = 40) -> list[dict[str, Any]]:
    """ردیف‌های شاهد با زمان جلالی و فاصله از شروع چالش."""
    rows: list[dict[str, Any]] = []
    for c in candidates[:limit]:
        stamp = c.get("stamp_ts")
        delta = ""
        if stamp:
            delta = utils.human_duration(max(0.0, float(stamp) - challenge.start_ts))
        rows.append(
            {
                "id": c.get("id"),
                "product": c.get("product_id"),
                "product_name": c.get("product_name", ""),
                "stamp": utils.jalali_datetime(float(stamp)) if stamp else "—",
                "stamp_ts": stamp,
                "delta": delta,
                "score": round(float(c.get("score", 0)), 1),
                "kind": c.get("kind", ""),
                "url": c.get("url", ""),
                "codes": c.get("ocr_codes", ""),
                "alert_ts": c.get("alert_ts", 0),
                "thumb": image_url_thumb(c.get("url", ""), 160),
            }
        )
    return rows


IMG_THUMB_SUFFIX = "?x-oss-process=image/resize,w_{w}/quality,q_60"
IMG_LARGE_SUFFIX = "?x-oss-process=image/resize,w_1200/quality,q_85"


def image_url_thumb(url: str, width: int = 200) -> str:
    if not url:
        return ""
    base = models.normalize_image_url(url)
    return base + IMG_THUMB_SUFFIX.format(w=width)


def image_url_large(url: str) -> str:
    if not url:
        return ""
    base = models.normalize_image_url(url)
    return base + IMG_LARGE_SUFFIX


# --------------------------------------------------------------------------- #
# تحلیل سریع محتوا (بدون OCR)
# --------------------------------------------------------------------------- #


def quick_content_hint(img_path: str) -> float:
    """نشانهٔ «ناحیهٔ متن/کادر» در تصویر، بدون OCR (۰..۱).

    هدف: صرفه‌جویی در OCR. اگر تصویر هیچ بافت متنی/هندسی ندارد، احتمال
    وجود رمز کم است. اگر بافت متن‌مانند پیدا شد، OCR ارزش هزینه را دارد.
    """
    try:
        from PIL import Image
        import numpy as np
    except Exception:
        return 0.0
    try:
        img = Image.open(img_path).convert("L")
    except Exception:
        return 0.0
    if img.width < 16 or img.height < 16:
        return 0.0
    # کوچک‌سازی برای سرعت
    w = min(480, img.width)
    ratio = w / img.width
    img = img.resize((w, max(8, int(img.height * ratio))))
    arr = np.asarray(img, dtype=np.float32)
    # لبه‌ها (گرادیان ساده). برای هم‌شکل شدن، هر گرادیان فقط تا مرز مشترک بریده می‌شود.
    h, w = arr.shape
    gx = np.abs(np.diff(arr, axis=1))[: h - 1, : w - 1]
    gy = np.abs(np.diff(arr, axis=0))[: h - 1, : w - 1]
    edges = (gx + gy) > 40
    edge_density = float(edges.mean())
    # کنتراست
    contrast = float(arr.max() - arr.min())
    score = 0.0
    if 0.006 < edge_density < 0.22:
        score += 0.6
    if contrast > 110:
        score += 0.4
    return min(1.0, score)


# --------------------------------------------------------------------------- #
# خلاصهٔ رمزها برای نمایش
# --------------------------------------------------------------------------- #


def codes_summary(codes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    status_fa = {
        "unknown": "بررسی‌نشده",
        "likely": "محتمل",
        "invalid": "نامعتبر",
        "used": "استفاده‌شده",
        "expired": "منقضی",
        "valid": "معتبر",
        "error": "خطا",
    }
    for c in codes:
        rows.append(
            {
                "code": c.get("code", ""),
                "product": c.get("product_id", ""),
                "status": status_fa.get(c.get("status", "unknown"), c.get("status", "")),
                "source": c.get("source", ""),
                "ts": utils.jalali_datetime(float(c["ts"])) if c.get("ts") else "",
                "note": c.get("note", ""),
            }
        )
    return rows


def verify_plan(codes: list[dict[str, Any]], cfg) -> list[tuple[int, str, list[str]]]:
    """برنامهٔ بررسی کدها: (image_id، رمز اصلی، نسخه‌های نزدیک).

    نسخه‌های «مشتبه» فقط برای وقتی مفیدند که می‌خواهید کد را با سرور
    اعتبارسنجی کنید؛ در این نسخه، بررسی خودکار روی سرور دیجی‌کالا انجام
    نمی‌شود (جزئیات در README و بخش اخلاق).
    """
    from .ocr import code_variants

    plan = []
    for c in codes:
        code = c.get("code", "")
        if not code:
            continue
        plan.append(
            (
                int(c.get("image_id") or 0),
                code,
                code_variants(code, max_variants=6),
            )
        )
    return plan
