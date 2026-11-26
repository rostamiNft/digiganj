"""موتور اسکن: چرخه‌های منظم و هدفمند.

جریان یک چرخه
------------
۱) فهرست محصولات چالش (کش‌شده؛ فقط در صورت کهنه بودن دوباره خوانده می‌شود).
۲) گالری هر محصول (فقط محصولاتی که گالری‌شان قدیمی است یا در لیست تماشاست).
۳) ثبت تصاویر تازه در پایگاه‌داده و امتیازدهی «بدون دانلود» (زمان نام فایل).
۴) برای نامزدهای برتر: دانلود، نشانهٔ محتوا، OCR و (در صورت yافتن رمز) هشدار.
۵) پایان چرخه + گزارش اختیاری + صدور آمار فارسی.

حالت «آماده‌باش» (prewarm) پیش از شروع چالش: کل ایندکس ساخته می‌شود تا در
لحظهٔ شروع، چرخه‌های بعدی فقط «تفاوت» را ببینند و بسیار سریع باشند. این
همان ترفندی است که دوستان ۱۴۰۰ استفاده کردند، منظم‌تر و مؤدبانه‌تر.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

from . import analyzer, digikala, models, utils
from .config import Challenge, Config
from .console import Console, Logger
from .db import Store
from .net import BlockedError, CircuitOpenError, HttpClient, NetError
from . import ocr as ocr_mod
from .ocr import OcrRunner

# --------------------------------------------------------------------------- #
# جدول نویسه (ساخت خودکار)
# --------------------------------------------------------------------------- #

_BUILTIN_FONT_DIRS = (
    "/usr/share/fonts/truetype/dejavu",
    "/usr/share/fonts/truetype/liberation",
    "/usr/share/fonts/truetype/noto",
    "/usr/share/fonts/TTF",
    "/Library/Fonts",
    "C:/Windows/Fonts",
)


def _default_glyph_path() -> Path:
    root = Path(__file__).resolve().parent.parent
    for cand in (Path("glyphs.npz"), Path("data") / "glyphs.npz",
                 root / "glyphs.npz", root / "data" / "glyphs.npz"):
        if cand.exists():
            return cand
    # مسیر پیش‌فرض برای ساخت: کنار داده
    return Path("data") / "glyphs.npz"


def _find_font_dir() -> str:
    for d in _BUILTIN_FONT_DIRS:
        if Path(d).exists() and any(Path(d).glob("*.ttf")):
            return d
    return ""


def resolve_glyphs_path(cfg=None, auto_build: bool = True) -> str:
    """مسیر جدول نویسهٔ OCR؛ در صورت نبود، با فونت‌های موجود خودکار می‌سازد.

    چرا خودکار؟ کاربر نباید برای «اولین اجرا» دستور اضافه بزند. ساخت جدول
    فقط چند ثانیه طول می‌کشد و نتیجه در ``data/glyphs.npz`` ذخیره می‌شود.
    """
    existing = _default_glyph_path()
    if existing.exists():
        return str(existing)
    if not auto_build:
        return ""
    font_dir = _find_font_dir()
    if not font_dir:
        return ""
    try:
        from .ocr import train_glyph_table

        target = existing
        target.parent.mkdir(parents=True, exist_ok=True)
        train_glyph_table(target, fonts_dir=font_dir, scales=(16, 22, 30, 44),
                          thickness=(False,))
        return str(target)
    except Exception:
        return ""


# --------------------------------------------------------------------------- #
# نتیجهٔ چرخه
# --------------------------------------------------------------------------- #


@dataclass
class CycleResult:
    cycle_id: int = 0
    started: float = 0.0
    ended: float = 0.0
    products: int = 0
    galleries: int = 0
    images: int = 0
    new_images: int = 0
    alerts: int = 0
    codes: int = 0
    ocr_runs: int = 0
    downloads: int = 0
    errors: int = 0
    blocked: int = 0
    note: str = ""
    watch_ids: list[str] = field(default_factory=list)

    @property
    def elapsed(self) -> float:
        return max(0.0, (self.ended or utils.now_ts()) - self.started)

    def as_dict(self) -> dict[str, Any]:
        return {
            "cycle_id": self.cycle_id,
            "started": self.started,
            "ended": self.ended,
            "elapsed": round(self.elapsed, 2),
            "products": self.products,
            "galleries": self.galleries,
            "images": self.images,
            "new_images": self.new_images,
            "alerts": self.alerts,
            "codes": self.codes,
            "ocr_runs": self.ocr_runs,
            "downloads": self.downloads,
            "errors": self.errors,
            "blocked": self.blocked,
            "note": self.note,
            "watch_ids": self.watch_ids,
        }


# --------------------------------------------------------------------------- #
# اسکنر
# --------------------------------------------------------------------------- #


class Scanner:
    def __init__(self, cfg: Config, http: HttpClient, store: Store,
                 client: digikala.DigikalaClient,
                 logger: Logger | None = None, console: Console | None = None,
                 ocr: OcrRunner | None = None,
                 notifier=None):
        self.cfg = cfg
        self.http = http
        self.store = store
        self.dk = client
        self.logger = logger or Logger(console=console)
        self.console = console or Console()
        self.ocr = ocr or OcrRunner(cfg, glyphs_path=resolve_glyphs_path(cfg))
        self.notifier = notifier
        self.images_dir = utils.ensure_dir(Path(cfg.data_dir) / "images")
        self._stop = threading.Event()
        self._on_alert: Callable[[dict[str, Any]], None] | None = None
        self._last_status = 0.0
        self._last_ocr = 0.0
        self._alert_times: dict[int, float] = {}
        self._seen_listing_ts = 0.0

    # ------------------------------------------------------------------ #
    # آماده‌سازی
    # ------------------------------------------------------------------ #
    def _glyph_path(self) -> str:
        return resolve_glyphs_path(self.cfg)

    def on_alert(self, fn: Callable[[dict[str, Any]], None]) -> None:
        self._on_alert = fn

    def stop(self) -> None:
        self._stop.set()

    def reset_stop(self) -> None:
        self._stop.clear()

    @property
    def stopped(self) -> bool:
        return self._stop.is_set()

    # ------------------------------------------------------------------ #
    # فهرست
    # ------------------------------------------------------------------ #
    def ensure_listing(self, challenge: Challenge, force: bool = False,
                       progress: bool = True) -> list[dict[str, Any]]:
        age = utils.now_ts() - self._seen_listing_ts
        existing = self.store.products()
        if not force and existing and age < 60 * 30:
            return existing
        self.logger.info(f"خواندن فهرست محصولات چالش «{challenge.title}»…")
        products, pages = self.dk.fetch_listing(
            challenge,
            max_products=self.cfg.max_products,
            progress=(
                (lambda n, p: self.console.hint(
                    f"فهرست: {utils.fa_num(n)} محصول از {utils.fa_num(p)} صفحه…"))
                if progress else None
            ),
        )
        if not products:
            self.logger.warn("فهرست محصولات خالی برگشت؛ از ایندکس قبلی استفاده می‌شود.")
            return existing
        new_n, total = self.store.upsert_products(products)
        self.store.event(
            "listing",
            f"فهرست به‌روز شد: {utils.fa_num(new_n)} محصول جدید، "
            f"{utils.fa_num(total)} محصول در {utils.fa_num(pages)} صفحه",
        )
        self._seen_listing_ts = utils.now_ts()
        self.logger.good(
            f"فهرست آماده: {utils.fa_num(total)} محصول"
            + (f" ({utils.fa_num(new_n)} تازه)" if new_n else "")
        )
        return self.store.products()

    # ------------------------------------------------------------------ #
    # گالری
    # ------------------------------------------------------------------ #
    def sync_gallery(self, row: dict[str, Any], cycle_id: int,
                     challenge: Challenge) -> tuple[int, int]:
        """گالری یک محصول را می‌گیرد؛ برمی‌گرداند (تعداد تصویر، تعداد تازه)."""
        product = models.Product.from_dict(row)
        items, source = self.dk.fetch_gallery(product)
        if not items:
            self.store.mark_gallery(str(product.id), source, ok=False)
            return 0, 0
        fresh = self.store.upsert_images(items, cycle_id=cycle_id)
        self.store.mark_gallery(str(product.id), source, ok=True)
        return len(items), len(fresh)

    def _process_gallery_batch(
        self, rows: list[dict[str, Any]], cycle_id: int, challenge: Challenge,
        result: CycleResult, progress_cb: Callable[[int, int], None] | None = None,
    ) -> list[dict[str, Any]]:
        """گالری‌ها را موازی می‌خواند و تصاویر تازه را برمی‌گرداند."""
        fresh_all: list[dict[str, Any]] = []
        done = 0
        total = len(rows)
        if not rows:
            return fresh_all
        workers = max(2, min(self.cfg.concurrency, 12))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(self.sync_gallery, row, cycle_id, challenge): row
                       for row in rows}
            for fut in as_completed(futures):
                row = futures[fut]
                done += 1
                try:
                    n_img, n_new = fut.result()
                    result.galleries += 1
                    result.images += n_img
                    result.new_images += n_new
                except CircuitOpenError:
                    self.logger.warn("مدار حفاظتی باز شد؛ ادامهٔ این دسته متوقف شد.")
                    result.blocked += 1
                    break
                except BlockedError as exc:
                    result.blocked += 1
                    self.logger.warn(f"گالری dkp-{row['id']} بلوکه شد: {utils.shorten(exc, 100)}")
                except NetError as exc:
                    result.errors += 1
                    self.logger.debug(f"گالری dkp-{row['id']} ناموفق: {exc}")
                if progress_cb:
                    progress_cb(done, total)
        images = []
        for row in rows:
            images.extend(self.store.images_for_product(str(row["id"])))
        return images

    # ------------------------------------------------------------------ #
    # امتیازدهی و کاندیدها
    # ------------------------------------------------------------------ #
    def score_new_images(self, fresh: list[dict[str, Any]], challenge: Challenge,
                         hero_id: str = "") -> list[dict[str, Any]]:
        """امتیازدهی سریع (بدون دانلود) به تصاویر تازه؛ برمی‌گرداند برترین‌ها."""
        scored: list[dict[str, Any]] = []
        for img in fresh:
            sc = analyzer.score_image(img, challenge, self.cfg, is_new=True,
                                      hero_product=hero_id)
            self.store.set_score(int(img["id"]), sc.value, sc.reasons, sc.kind)
            img = dict(img)
            img["score"] = sc.value
            img["score_reasons"] = sc.reasons
            img["kind"] = sc.kind
            scored.append(img)
        scored.sort(key=lambda i: -float(i.get("score", 0)))
        return scored

    # ------------------------------------------------------------------ #
    # دانلود و OCR
    # ------------------------------------------------------------------ #
    def _download(self, image: dict[str, Any]) -> tuple[bytes, str] | None:
        try:
            data, sha1 = self.dk.download_image(
                image["url"], None, max_bytes=self.cfg.max_download_bytes
            )
        except (BlockedError, CircuitOpenError):
            raise
        except NetError as exc:
            self.logger.debug(f"دانلود ناموفق {image.get('filename')}: {exc}")
            return None
        name = utils.safe_filename(image.get("filename") or f"img-{image.get('id')}")
        dest = self.images_dir / str(image.get("product_id")) / name
        utils.atomic_write(dest, data)
        self.store.set_download(int(image["id"]), str(dest), sha1, len(data))
        return data, str(dest)

    def _ocr_image(self, image: dict[str, Any], path: str) -> tuple[list[str], float]:
        interval = self.cfg.ocr_min_interval
        wait = interval - (utils.now_ts() - self._last_ocr)
        if wait > 0:
            time.sleep(wait)
        self._last_ocr = utils.now_ts()
        res = self.ocr.run(path)
        return res.codes, res.score

    def evaluate_candidates(self, candidates: list[dict[str, Any]], challenge: Challenge,
                           cycle_id: int, result: CycleResult, limit: int = 0) -> int:
        """دانلود/OCR نامزدها و ثبت هشدار. برمی‌گرداند تعداد هشدار جدید."""
        if not self.cfg.ocr_enabled and not self.cfg.ocr_try_raw:
            return 0
        if not self.ocr.available:
            if self.ocr.warnings and self.logger:
                for w in self.ocr.warnings:
                    self.logger.debug(w)
            return 0
        limit = limit or self.cfg.ocr_max_candidates
        alerts = 0
        for cand in candidates[:limit]:
            if self.stopped:
                break
            image = self.store.image(int(cand["id"]))
            if not image:
                continue
            # قبلاً هشدار داده شده و هم‌چنان در دورهٔ خنک‌سازی است؟
            last = self._alert_times.get(int(image["id"]), 0.0)
            if last and utils.now_ts() - last < self.cfg.alert_cooldown:
                continue
            path = image.get("file_path") or ""
            if not path or not Path(path).exists():
                try:
                    got = self._download(image)
                except (BlockedError, CircuitOpenError) as exc:
                    result.blocked += 1
                    self.logger.warn(f"دانلود متوقف شد: {utils.shorten(exc, 100)}")
                    break
                if not got:
                    continue
                result.downloads += 1
                path = got[1]
            hint = analyzer.quick_content_hint(path)
            if hint <= 0.05:
                continue
            try:
                codes, score = self._ocr_image(image, path)
            except (BlockedError, CircuitOpenError):
                raise
            except Exception as exc:
                self.logger.debug(f"OCR ناموفق: {exc}")
                continue
            result.ocr_runs += 1
            # برش کادر رمز: سریع‌ترین راه «تأیید با چشم» برای کاربر
            try:
                card = ocr_mod.extract_code_card(path)
                if card:
                    self.store.set_kv(f"card:{int(image['id'])}", card)
            except Exception:
                card = ""
            if codes:
                self.store.set_ocr(int(image["id"]), " ".join(codes), codes,
                                   self.ocr.engines[0] if self.ocr.engines else "")
                new_codes = 0
                # خوانش خام خودِ موتور با منبع «ocr» ثبت می‌شود؛ گونه‌های
                # ساختگی با منبع «variant» تا در رتبه‌بندی، خوانش واقعی همیشه
                # بالاتر از حدس‌های ترکیبی بماند.
                for code in dict.fromkeys(codes):
                    if self.store.add_code(code, int(image["id"]),
                                           str(image["product_id"]), source="ocr"):
                        new_codes += 1
                        result.codes += 1
                guesses: list[str] = []
                for code in codes:
                    try:
                        guesses.extend(
                            ocr_mod.code_variants(code, max_variants=4, min_risk=2)
                        )
                    except Exception:
                        pass
                for code in dict.fromkeys(guesses):
                    if code in codes:
                        continue
                    if self.store.add_code(code, int(image["id"]),
                                           str(image["product_id"]), source="variant"):
                        new_codes += 1
                        result.codes += 1
                sc = analyzer.score_image(image, challenge, self.cfg,
                                          ocr_codes=codes, ocr_score=score)
                self.store.set_score(int(image["id"]), sc.value, sc.reasons, sc.kind)
                alert_id = self._emit_alert(image, sc, codes, challenge)
                if alert_id:
                    alerts += 1
                    result.alerts += 1
                    self._alert_times[int(image["id"])] = utils.now_ts()
        return alerts

    def _emit_alert(self, image: dict[str, Any], sc: analyzer.Score,
                    codes: Iterable[str], challenge: Challenge) -> int:
        pid = str(image.get("product_id", ""))
        product = self.store.product(pid) or {}
        product_url = digikala.build_product_url(
            self.cfg, pid, product.get("uri", "")
        )
        reason = "؛ ".join(r for r in sc.reasons if r)
        alert_id = self.store.add_alert(
            int(image["id"]) if image.get("id") else None,
            pid,
            float(sc.value),
            sc.kind,
            reason,
            list(codes),
        )
        info = dict(image)
        info["product_name"] = product.get("name", "")
        info["score"] = sc.value
        info["codes"] = ", ".join(codes)
        info["alert_id"] = alert_id
        self.store.event("alert", f"هشدار برای dkp-{pid} (امتیاز {sc.value:.0f})")
        if self.logger:
            self.logger.good(
                f"نشانهٔ رمز! dkp-{pid} — امتیاز {utils.fa_num(round(sc.value))}"
                + (f" — رمز: {', '.join(codes)}" if codes else "")
            )
        if self.notifier:
            try:
                results = self.notifier.alert(info, challenge, product_url)
                if results.get("telegram") or results.get("webhook"):
                    self.store.mark_notified(alert_id)
            except Exception as exc:
                self.logger.debug(f"ارسال هشدار ناموفق: {exc}")
        if self._on_alert:
            try:
                self._on_alert(info)
            except Exception:
                pass
        return alert_id

    # ------------------------------------------------------------------ #
    # چرخهٔ کامل
    # ------------------------------------------------------------------ #
    def run_cycle(self, challenge: Challenge, *, force_listing: bool = False,
                  max_products: int = 0, evaluate: bool = True) -> CycleResult:
        result = CycleResult(started=utils.now_ts())
        result.cycle_id = self.store.start_cycle(challenge.id, result.started)
        self.store.event("cycle-start", f"چرخهٔ اسکن #{result.cycle_id} آغاز شد.")
        # ثبت چالش جاری تا گزارش/نامزدها بدون تنظیم دستی همان داده را ببینند
        try:
            self.store.set_kv("last_challenge", challenge.id)
            self.store.set_kv("last_challenge_meta", utils.write_json_str({
                "title": challenge.title,
                "start_ts": challenge.start_ts,
                "end_ts": challenge.end_ts,
                "list_path": challenge.list_path,
            }))
        except Exception:
            pass
        t0 = time.time()

        products = self.ensure_listing(challenge, force=force_listing)
        if max_products:
            products = products[:max_products]
        result.products = len(products)
        if not products:
            result.ended = utils.now_ts()
            result.note = "فهرست خالی بود."
            self.store.finish_cycle(result.cycle_id, **result.as_dict())
            return result

        due = {str(p["id"]) for p in self.store.due_products(self.cfg.gallery_resync_every)}
        hero_id = self._detect_hero(products, challenge)
        if hero_id:
            self.logger.info(f"محصول گنج (HERO) شناسایی شد: dkp-{hero_id}")
        watch = set(self.store.get_kv("watch_ids", "").split(",")) - {""}
        plan_ids = [
            p["id"] for p in products
            if str(p["id"]) in due or str(p["id"]) == hero_id or str(p["id"]) in watch
        ]
        if not plan_ids:
            plan_ids = [p["id"] for p in products[: max(1, len(products) // 4)]]
        rows = [p for p in products if p["id"] in set(plan_ids)]
        rows.sort(key=lambda p: (p["id"] != hero_id, p.get("page", 0) or 0))
        if watch:
            rows.sort(key=lambda p: (str(p["id"]) not in watch, p["id"] != hero_id))

        self.logger.info(
            f"چرخه #{utils.fa_num(result.cycle_id)}: {utils.fa_num(len(rows))} گالری از "
            f"{utils.fa_num(len(products))} محصول…"
        )

        def progress(done: int, total: int) -> None:
            now = utils.now_ts()
            if now - self._last_status < 3.0 and done < total:
                return
            self._last_status = now
            utils_pct = utils.pct(done, total)
            self.console.hint(
                f"گالری {utils.fa_num(done)}/{utils.fa_num(total)} ({utils_pct}) — "
                f"{utils.fa_num(result.new_images)} تصویر تازه"
            )

        # دسته‌بندی برای به‌روزرسانی تدریجی: هر دسته، بلافاصله بررسی می‌شود
        batch = max(8, self.cfg.list_page_batch * 4)
        for i in range(0, len(rows), batch):
            if self.stopped:
                result.note = "متوقف شد."
                break
            chunk = rows[i: i + batch]
            chunk_fresh = self._process_gallery_batch(
                chunk, result.cycle_id, challenge, result, progress
            )
            if chunk_fresh:
                scored = self.score_new_images(chunk_fresh, challenge, hero_id)
                if evaluate:
                    self.evaluate_candidates(scored, challenge, result.cycle_id, result)
            if self.cfg.cycle_pause:
                time.sleep(self.cfg.cycle_pause)

        result.ended = utils.now_ts()
        self.store.finish_cycle(
            result.cycle_id,
            products=result.products,
            galleries=result.galleries,
            images=result.images,
            new_images=result.new_images,
            alerts=result.alerts,
            requests=self.http.stats.requests,
            blocked=result.blocked,
            note=result.note,
        )
        self.store.event(
            "cycle-finish",
            f"چرخه #{result.cycle_id} پایان یافت: "
            f"{utils.fa_num(result.galleries)} گالری، "
            f"{utils.fa_num(result.new_images)} تصویر تازه، "
            f"{utils.fa_num(result.alerts)} هشدار، "
            f"{utils.human_duration(result.elapsed)}",
        )
        self.logger.good(
            f"چرخه #{utils.fa_num(result.cycle_id)} پایان: "
            f"{utils.fa_num(result.galleries)} گالری، "
            f"{utils.fa_num(result.new_images)} تازه، "
            f"{utils.fa_num(result.alerts)} هشدار، "
            f"{utils.human_duration(result.elapsed)}"
            + (f" — {utils.pct(result.galleries, len(rows))} تکمیل" if rows else "")
        )
        if self._last_status and time.time() - t0 > 1:
            self.console.hint(self.http.stats_line())
        return result

    def _detect_hero(self, products: list[dict[str, Any]], challenge: Challenge) -> str:
        """تشخیص «محصول گنج»: محصولی که گالری‌اش نزدیک شروع چالش به‌روز شده.

        این روش مبتنی بر مشاهده است، نه ادعای قطعی. اگر پیدا نشود، رشتهٔ خالی
        برمی‌گردد. (به‌عنوان تنظیم دستی هم قابل تعیین است.)
        """
        manual = self.store.get_kv("hero_product", "")
        if manual:
            return manual
        best = ""
        best_score = -1.0
        for p in products:
            ts = float(p.get("gallery_ts") or 0)
            if not ts:
                continue
            if challenge.start_ts - 3600 <= ts <= challenge.start_ts + 3600:
                if ts > best_score:
                    best_score = ts
                    best = str(p["id"])
        return best

    # ------------------------------------------------------------------ #
    # آماده‌باش پیش از چالش
    # ------------------------------------------------------------------ #
    def prewarm(self, challenge: Challenge, max_products: int = 0,
                progress_cb: Callable[[int, int], None] | None = None) -> dict[str, int]:
        """ساخت ایندکس کامل پیش از شروع (بدون OCR)."""
        self.logger.info("حالت آماده‌باش: ساخت ایندکس کامل، بدون OCR…")
        result = self.run_cycle(challenge, force_listing=True,
                                max_products=max_products, evaluate=False)
        self.store.set_kv("prewarm_ts", utils.now_ts())
        return {
            "galleries": result.galleries,
            "images": result.images,
            "new_images": result.new_images,
            "elapsed": int(result.elapsed),
        }

    # ------------------------------------------------------------------ #
    # حلقهٔ بلندمدت با زمان‌بندی هوشمند
    # ------------------------------------------------------------------ #
    def run_forever(
        self,
        challenge: Challenge,
        *,
        until_end: bool = True,
        prewarm_minutes: float = 0.0,
        interval: float = 45.0,
        max_cycles: int = 0,
        on_cycle: Callable[[CycleResult], None] | None = None,
        on_report: Callable[[], None] | None = None,
        report_every: float = 900.0,
    ) -> None:
        """حلقهٔ اصلی با زمان‌بندی متناسب با چالش.

        * پیش از شروع: خواب تا (شروع − prewarm_minutes)؛ سپس prewarm و
          انتظار دقیق تا شروع.
        * در حین چالش: چرخه‌های سریع با فاصلهٔ interval.
        * پس از پایان: توقف (یا ادامه تا وقتی متوقف نشده‌اید).
        """
        self.reset_stop()
        if prewarm_minutes and utils.now_ts() < challenge.start_ts - prewarm_minutes * 60:
            target = challenge.start_ts - prewarm_minutes * 60
            self.logger.info(
                f"انتظار تا {utils.jalali_pretty(target)} برای آماده‌باش "
                f"({utils.human_duration(target - utils.now_ts())})…"
            )
            while not self.stopped and utils.now_ts() < target:
                time.sleep(min(5.0, max(0.2, target - utils.now_ts())))
        if prewarm_minutes or utils.now_ts() >= challenge.start_ts - 600:
            if utils.now_ts() < challenge.start_ts:
                self.prewarm(challenge)
            if self.stopped:
                return
        # انتظار دقیق تا شروع
        if utils.now_ts() < challenge.start_ts:
            delta = challenge.start_ts - utils.now_ts()
            if delta > 0:
                self.logger.info(
                    f"شروع چالش در {utils.jalali_pretty(challenge.start_ts)} "
                    f"({utils.human_duration(delta)} دیگر)…"
                )
                while not self.stopped and utils.now_ts() < challenge.start_ts:
                    time.sleep(min(2.0, max(0.05, challenge.start_ts - utils.now_ts())))
        if self.stopped:
            return

        cycles = 0
        last_report = 0.0
        while not self.stopped:
            now = utils.now_ts()
            if until_end and now > challenge.end_ts:
                self.logger.info("چالش به پایان رسید؛ حلقه متوقف شد.")
                break
            cycles += 1
            try:
                res = self.run_cycle(challenge)
                if on_cycle:
                    on_cycle(res)
            except CircuitOpenError as exc:
                self.logger.warn(f"توقف حفاظتی: {exc}")
            except Exception as exc:
                self.logger.error(f"خطای چرخه: {utils.shorten(exc, 200)}")
            if max_cycles and cycles >= max_cycles:
                break
            if on_report and self._should_report(report_every, last_report):
                last_report = utils.now_ts()
                try:
                    on_report()
                except Exception as exc:
                    self.logger.debug(f"گزارش دوره‌ای ناموفق: {exc}")
            # خواب هوشمند: تا شروع چرخهٔ بعد
            if not self.stopped and (not until_end or utils.now_ts() < challenge.end_ts):
                self._sleep_interval(interval)
        self.logger.info("حلقهٔ اسکن پایان یافت.")

    def _should_report(self, every: float, last: float) -> bool:
        return every > 0 and utils.now_ts() - last >= every

    def _sleep_interval(self, interval: float) -> None:
        end = utils.now_ts() + max(1.0, interval)
        while not self.stopped and utils.now_ts() < end:
            time.sleep(min(0.5, max(0.02, end - utils.now_ts())))

    # ------------------------------------------------------------------ #
    # دیدن یک محصول خاص (watch)
    # ------------------------------------------------------------------ #
    def watch_ids(self, ids: Iterable[str]) -> None:
        old = set(self.store.get_kv("watch_ids", "").split(",")) - {""}
        old.update(str(i) for i in ids)
        self.store.set_kv("watch_ids", ",".join(sorted(old)))

    def unwatch_ids(self, ids: Iterable[str]) -> None:
        old = set(self.store.get_kv("watch_ids", "").split(",")) - {""}
        old -= {str(i) for i in ids}
        self.store.set_kv("watch_ids", ",".join(sorted(old)))

    def set_hero(self, pid: str) -> None:
        self.store.set_kv("hero_product", str(pid))

    # ------------------------------------------------------------------ #
    # گزارش زندهٔ چرخه
    # ------------------------------------------------------------------ #
    def status_line(self) -> str:
        s = self.store.stats()
        return (
            f"محصول: {utils.fa_num(s['products'])} | تصویر: {utils.fa_num(s['images'])} | "
            f"هشدار: {utils.fa_num(s['alerts'])} | رمز: {utils.fa_num(s['codes'])} | "
            f"{self.http.stats_line()}"
        )
