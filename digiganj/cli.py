"""رابط خط فرمان digiganj — همهٔ دستورها به فارسی.

نمونه‌ها::

    digiganj doctor                     # بررسی سلامت محیط و دسترسی‌ها
    digiganj demo                       # اجرای دموی محلی (بدون تماس با دیجی‌کالا)
    digiganj config init                # ساخت config.json
    digiganj config show                # نمایش پیکربندی فعلی
    digiganj challenge list             # فهرست چالش‌ها (جاری/بعدی/گذشته)
    digiganj list                       # خواندن و نمایش فهرست محصولات چالش
    digiganj scan --once                # یک چرخهٔ اسکن کامل
    digiganj scan --watch --prewarm 10  # حلقهٔ بلندمدت با آماده‌باش
    digiganj scan --offline             # اسکن از روی دادهٔ نمونه (بدون شبکه)
    digiganj candidates --top 20        # نامزدهای نشانه‌دار
    digiganj codes                      # رمزهای استخراج‌شده
    digiganj report                     # ساخت گزارش HTML
    digiganj evidence                   # بستهٔ شواهد
    digiganj image <image_id>           # نمایش/ذخیرهٔ یک تصویر از پایگاه‌داده
    digiganj ocr-train --fonts DIR      # ساخت جدول نویسه برای موتور OCR داخلی
    digiganj demo-scan                  # دمو + اسکن کامل، در یک فرمان
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

from . import __version__, analyzer, config as config_mod, db, digikala, net, notify
from . import ocr as ocr_mod
from . import report as report_mod
from . import scanner as scanner_mod
from . import utils
from .console import Console, Logger

# ثابت‌های رنگ (برای استفاده در f-string بدون بک‌اسلش)
YELLOW_B = "\033[1;33m"
GREEN = "\033[32m"
RED = "\033[31m"
GRAY = "\033[90m"
BOLD = "\033[1m"

# رنگ‌های ساده برای پیام‌ها (Console خودش هوشمند است)
C = Console()


# --------------------------------------------------------------------------- #
# راه‌اندازی مشترک
# --------------------------------------------------------------------------- #


def load_config(path: str | None, require: bool = False) -> config_mod.Config:
    p = path or config_mod.DEFAULT_CONFIG_NAME
    if not Path(p).exists():
        if require:
            raise SystemExit(
                f"فایل پیکربندی «{p}» پیدا نشد. ابتدا `digiganj config init` را اجرا کنید."
            )
        return config_mod.Config()
    return config_mod.Config.load(p)


def build_components(cfg: config_mod.Config, level: str | None = None, quiet: bool = False):
    cfg.ensure_dirs()
    con = Console(verbose=not quiet)
    log = Logger(cfg.log_file, level=level or cfg.log_level, console=con)
    http = net.HttpClient(cfg, logger=log, console=con)
    store = db.Store(resolve_db_path(cfg))
    dk = digikala.DigikalaClient(cfg, http, logger=log)
    ocr_runner = ocr_mod.OcrRunner(cfg, glyphs_path=find_glyphs_path())
    notifier = notify.Notifier(cfg, logger=log, console=con)
    sc = scanner_mod.Scanner(cfg, http, store, dk, logger=log, console=con,
                             ocr=ocr_runner, notifier=notifier)
    return con, log, http, store, dk, sc, ocr_runner, notifier


def resolve_db_path(cfg: config_mod.Config) -> str:
    """مسیر پایگاه‌داده با جستجوی خودکار.

    اگر مسیر پیش‌فرض وجود دارد، همان. اگر نه، نزدیک‌ترین ``*.sqlite`` در
    زیرپوشه‌های ``data``، ``demo*``، ``scans`` یا خود پوشهٔ جاری را پیدا می‌کند.
    این کار باعث می‌شود پس از ``demo-scan``، دستورهای گزارش/آمار بدون تنظیم
    دستی مسیر هم کار کنند.
    """
    env = os.environ.get("DIGIGANJ_DB")
    if env and Path(env).exists():
        return env
    p = Path(cfg.db_path)
    candidates: list[Path] = [p] if p.exists() else []
    for base in (Path("."), Path("data"), *sorted(Path(".").glob("demo*"))):
        if base.is_dir():
            candidates.extend(base.glob("*.sqlite"))
    if not candidates:
        return str(p)
    # پایگاه‌داده‌ای که بیشترین تصویر را دارد برنده است؛ اگر همه خالی بودند،
    # تازه‌ترین فایل.
    def _weight(q: Path) -> tuple[int, float]:
        try:
            import sqlite3

            with sqlite3.connect(f"file:{q}?mode=ro", uri=True) as conn:
                n = conn.execute("SELECT COUNT(*) FROM images").fetchone()[0]
                n += conn.execute("SELECT COUNT(*) FROM alerts").fetchone()[0] * 5
        except Exception:
            n = 0
        try:
            mtime = q.stat().st_mtime
        except OSError:
            mtime = 0.0
        return (n, mtime)

    return str(max(candidates, key=_weight))


def find_glyphs_path() -> str:
    """مسیر جدول نویسه؛ اگر نبود، با فونت‌های سیستم ساخته می‌شود."""
    return scanner_mod.resolve_glyphs_path(None, auto_build=True)


def parse_challenge_arg(args, cfg: config_mod.Config, store=None):
    """چالش درخواستی کاربر یا «خودکار».

    در حالت ``auto`` اگر پایگاه‌داده چالشی ثبت‌شده داشته باشد (مثلاً از
    ``demo-scan``)، همان برگردانده می‌شود تا گزارش با دادهٔ واقعاً اسکن‌شده
    هم‌خوان باشد؛ در غیر این صورت چالش جاری/آخرین چالش پیکربندی.
    """
    cid = getattr(args, "challenge", None) or "auto"
    if cid in ("auto", "", None) and store is not None:
        try:
            seen = store.get_kv("last_challenge", "")
            if seen:
                try:
                    return cfg.challenge(seen)
                except KeyError:
                    pass
                # چالش دمو در پیکربندی نیست؛ یک شیء موقت از داده‌های پایگاه‌داده
                raw_meta = store.get_kv("last_challenge_meta", "")
                try:
                    meta = json.loads(raw_meta) if raw_meta else {}
                except Exception:
                    meta = {}
                if isinstance(meta, dict) and meta.get("start_ts"):
                    return config_mod.Challenge(
                        id=seen,
                        title=meta.get("title") or seen,
                        starts_at=utils.iso_utc(float(meta["start_ts"])),
                        ends_at=utils.iso_utc(float(meta.get("end_ts") or meta["start_ts"] + 86400)),
                        list_path=meta.get("list_path") or cfg.products_path,
                    )
        except Exception:
            pass
    return cfg.resolve_challenge(cid, allow_finished=True)


# --------------------------------------------------------------------------- #
# دستور: doctor
# --------------------------------------------------------------------------- #


def cmd_doctor(args) -> int:
    cfg = load_config(args.config)
    C.banner("digiganj — بررسی سلامت محیط", f"نسخهٔ {__version__}")
    rows = []
    # پایتون
    rows.append(["پایتون", sys.version.split()[0], C.c("✔", GREEN), ""])
    # وابستگی‌ها
    for pkg, label, required in [
        ("httpx", "httpx", True),
        ("numpy", "numpy", True),
        ("PIL", "Pillow", True),
        ("curl_cffi", "curl_cffi (TLS مرورگر)", False),
    ]:
        try:
            module = __import__(pkg)
            ver = getattr(module, "__version__", "?")
            rows.append([label, ver, C.c("✔", GREEN), ""])
        except Exception:
            mark = C.c("✖", RED) if required else C.c("—", GRAY)
            rows.append([label, "نصب نیست", mark, "pip install " + pkg])
    # tesseract
    tesseract = ocr_mod.tesseract_available()
    rows.append(["tesseract", "نصب است" if tesseract else "نصب نیست",
                 C.c("✔", GREEN) if tesseract else C.c("—", GRAY),
                 "" if tesseract else "اختیاری: sudo apt install tesseract-ocr"])
    # جدول نویسه
    gp = find_glyphs_path()
    rows.append(["جدول نویسهٔ OCR", Path(gp).name if gp else "یافت نشد",
                 C.c("✔", GREEN) if gp else C.c("—", GRAY),
                 "" if gp else "digiganj ocr-train --fonts /usr/share/fonts/truetype/dejavu"])
    # مسیرهای داده
    try:
        cfg.ensure_dirs()
        writable = os.access(cfg.data_dir, os.W_OK)
        rows.append(["پوشهٔ داده", cfg.data_dir,
                     C.c("✔", GREEN) if writable else C.c("✖", RED), ""])
    except Exception as exc:
        rows.append(["پوشهٔ داده", utils.shorten(exc, 60), C.c("✖", RED), ""])
    C.table(["مورد", "وضعیت", "بررسی", "یادداشت"], rows)

    # شبکه
    C.title("بررسی دسترسی شبکه (اختیاری)")
    try:
        con = Console()
        http = net.HttpClient(cfg, logger=Logger(None, level="ERROR", console=con), console=con)
        results = digikala.check_availability(http, cfg)
        http.close()
        nrows = []
        for name, info in results.items():
            mark = C.c("✔", GREEN) if info["ok"] else C.c("✖", RED)
            nrows.append([name, str(info["status"]), mark, utils.shorten(info["detail"], 60)])
        C.table(["نقطهٔ دسترسی", "کد", "وضعیت", "توضیح"], nrows)
    except Exception as exc:
        C.warn(f"بررسی شبکه انجام نشد: {utils.shorten(exc, 160)}")

    # قابلیت‌های OCR
    status = ocr_mod.engine_status(cfg)
    C.title("موتور OCR")
    C.kv([
        ("فعال", "بله" if status["enabled"] else "خیر"),
        ("tesseract", "دارد" if status["tesseract_binary"] else "ندارد"),
        ("موتورها", "، ".join(status["engines"]) or "—"),
        ("آماده", "بله" if status["ready"] else "خیر"),
    ])
    for w in status["warnings"]:
        C.warn(w)
    return 0


# --------------------------------------------------------------------------- #
# دستور: config
# --------------------------------------------------------------------------- #


def cmd_config(args) -> int:
    action = args.action
    cfg = load_config(args.config)
    if action == "init":
        target = Path(args.out or cfg.config_path if hasattr(cfg, "config_path") else
                      (args.out or config_mod.DEFAULT_CONFIG_NAME))
        if target.exists() and not args.force:
            C.warn(f"فایل «{target}» از قبل هست. برای بازنویسی --force بدهید.")
            return 1
        cfg.save(target)
        C.good(f"پیکربندی در «{target}» ساخته شد.")
        C.hint("زمان‌های چالش را متناسب با ایونت واقعی ویرایش کنید.")
        return 0
    if action == "show":
        data = cfg.to_dict()
        # توکن/کوکی‌ها را نمایش نده
        for secret in ("telegram_bot_token", "cookie_header"):
            if data.get(secret):
                data[secret] = "***"
        C.write(json.dumps(data, ensure_ascii=False, indent=2))
        return 0
    if action == "set":
        if not args.key or args.value is None:
            C.err("مثال: digiganj config set concurrency 8")
            return 1
        data = cfg.to_dict()
        if args.key not in data:
            C.err(f"کلید «{args.key}» در پیکربندی نیست.")
            return 1
        try:
            val: Any = json.loads(args.value)
        except Exception:
            val = args.value
        data[args.key] = val
        new_cfg = config_mod.Config.from_dict(data)
        new_cfg.save(args.config or config_mod.DEFAULT_CONFIG_NAME)
        C.good(f"{args.key} = {val}")
        return 0
    if action == "path":
        C.write(str(Path(args.config or config_mod.DEFAULT_CONFIG_NAME).resolve()))
        return 0
    C.err("عملیات ناشناخته. از init/show/set/path استفاده کنید.")
    return 1


# --------------------------------------------------------------------------- #
# دستور: challenge
# --------------------------------------------------------------------------- #


def cmd_challenge(args) -> int:
    cfg = load_config(args.config)
    action = getattr(args, "action", None) or "list"
    if action in ("list", "ls"):
        rows = []
        for c in cfg.challenges:
            rows.append([
                c.id,
                utils.shorten(c.title, 34),
                utils.jalali_short(c.start_ts)[:10],
                utils.jalali_short(c.end_ts)[:10],
                c.phase_fa(),
                "بله" if c.enabled else "خیر",
            ])
        C.table(["شناسه", "عنوان", "شروع", "پایان", "وضعیت", "فعال"], rows)
        cur = cfg.current_challenge()
        if cur:
            C.info(f"چالش جاری: {cur.title} ({cur.id})")
        else:
            nxt = cfg.next_challenge()
            if nxt:
                C.info(f"چالش جاری نیست. بعدی: {nxt.title} — "
                       f"{utils.jalali_pretty(nxt.start_ts)}")
        return 0
    if action == "add":
        if not (args.id and args.start and args.end):
            C.err("نیاز: --id --start --end (ISO-8601).")
            return 1
        cfg.challenges.append(
            config_mod.Challenge(
                id=args.id,
                title=args.title or args.id,
                starts_at=args.start,
                ends_at=args.end,
                prize=args.prize or "",
                note=args.note or "",
                list_path=args.list_path or "/treasure-hunt/products/",
                era=args.era or "",
            )
        )
        cfg.validate()
        cfg.save(args.config or config_mod.DEFAULT_CONFIG_NAME)
        C.good(f"چالش «{args.id}» اضافه شد.")
        return 0
    if action == "rm":
        pid = args.id
        before = len(cfg.challenges)
        cfg.challenges = [c for c in cfg.challenges if c.id != pid]
        if len(cfg.challenges) == before:
            C.err(f"چالش «{pid}» پیدا نشد.")
            return 1
        cfg.validate()
        cfg.save(args.config or config_mod.DEFAULT_CONFIG_NAME)
        C.good(f"چالش «{pid}» حذف شد.")
        return 0
    C.err("عملیات ناشناخته: list / add / rm")
    return 1


# --------------------------------------------------------------------------- #
# دستور: doctor / auth
# --------------------------------------------------------------------------- #


def cmd_auth(args) -> int:
    """ذخیرهٔ کوکی نشست خودتان (اختیاری) برای بررسی دستی.

    این ابزار هرگز به حساب شما وارد نمی‌شود و کد تخفیف را روی حساب ثبت
    نمی‌کند. کوکی فقط برای این است که درخواست‌ها، در صورت لزوم، مثل مرورگر
    واردشده شما دیده شوند.
    """
    cfg = load_config(args.config)
    if args.clear:
        cfg.cookie_header = ""
        cfg.save(args.config or config_mod.DEFAULT_CONFIG_NAME)
        C.good("کوکی پاک شد.")
        return 0
    raw = args.cookie
    if not raw:
        C.hint("کوکی را از DevTools مرورگر (Application → Cookies) کپی و این‌جا بچسبانید.")
        try:
            raw = input("کوکی: ").strip()
        except EOFError:
            raw = ""
    if not raw:
        C.err("کوکی خالی است.")
        return 1
    if ";" not in raw and "=" in raw:
        C.warn("کوکی معمولاً چند جفت کلید=مقدار دارد؛ مطمئن شوید کامل کپی شده باشد.")
    cfg.cookie_header = raw
    cfg.save(args.config or config_mod.DEFAULT_CONFIG_NAME)
    C.good("کوکی ذخیره شد (در config.json). این فایل را به اشتراک نگذارید.")
    return 0


# --------------------------------------------------------------------------- #
# دستور: list
# --------------------------------------------------------------------------- #


def cmd_list(args) -> int:
    cfg = load_config(args.config)
    ch = parse_challenge_arg(args, cfg)
    con, log, http, store, dk, sc, _ocr, _n = build_components(cfg)
    try:
        products, pages = dk.fetch_listing(ch, max_products=args.limit or 0)
        if not products:
            C.err("هیچ محصولی دریافت نشد.")
            return 1
        rows = [
            [p.id, utils.shorten(p.name, 46), utils.fa_num(p.price or "—"), utils.fa_num(p.page)]
            for p in products[: args.show]
        ]
        C.table(["dkp", "نام", "قیمت (تومان)", "صفحه"], rows)
        C.info(f"جمع: {utils.fa_num(len(products))} محصول در {utils.fa_num(pages or '؟')} صفحه")
        return 0
    finally:
        http.close()
        store.close()


# --------------------------------------------------------------------------- #
# دستور: scan
# --------------------------------------------------------------------------- #


def cmd_scan(args) -> int:
    cfg = load_config(args.config)
    if args.offline:
        cfg.base_url = "http://127.0.0.1:8099"
        cfg.api_base_url = cfg.base_url
    if args.no_ocr:
        cfg.ocr_enabled = False
    if args.concurrency:
        cfg.concurrency = args.concurrency
    con, log, http, store, dk, sc, ocr_runner, notifier = build_components(cfg)
    ch = parse_challenge_arg(args, cfg, store)
    C.banner("digiganj — اسکن شکار گنج", ch.title)
    C.kv([
        ("چالش", ch.id),
        ("وضعیت", ch.phase_fa()),
        ("شروع", utils.jalali_pretty(ch.start_ts)),
        ("پایان", utils.jalali_pretty(ch.end_ts)),
        ("پیش از شروع", utils.human_duration(max(0, ch.start_ts - utils.now_ts()))),
        ("OCR", "روشن" if cfg.ocr_enabled else "خاموش"),
        ("هشدار", "، ".join(notifier.channels())),
    ])
    if not ocr_runner.available and cfg.ocr_enabled:
        C.warn("هیچ موتور OCR آماده نیست؛ تشخیص فقط بر اساس زمان نام فایل انجام می‌شود.")
        for w in ocr_runner.warnings:
            C.hint(w)

    interrupted = False
    try:
        if args.once or not args.watch:
            sc.run_cycle(ch, force_listing=args.refresh, max_products=args.limit or 0,
                         evaluate=not args.no_ocr)
        else:
            sc.run_forever(
                ch,
                until_end=not args.forever,
                prewarm_minutes=args.prewarm,
                interval=args.interval,
                max_cycles=args.cycles or 0,
                on_cycle=(
                    (lambda res: _auto_report(res, store, ch, cfg, args))
                    if args.report_every else None
                ),
                report_every=args.report_every,
                on_report=(
                    (lambda: _write_cycle_report(store, ch, cfg))
                    if args.report_every else None
                ),
            )
    except KeyboardInterrupt:
        interrupted = True
        sc.stop()
        C.warn("با Ctrl+C متوقف شد؛ جمع‌بندی…")
    finally:
        if args.report or interrupted:
            try:
                p = _write_cycle_report(store, ch, cfg)
                C.good(f"گزارش: {p}")
            except Exception as exc:
                C.warn(f"نوشتن گزارش ناموفق: {utils.shorten(exc, 120)}")
        stats = store.stats()
        C.title("جمع‌بندی")
        C.kv([
            ("محصولات", utils.fa_num(stats["products"])),
            ("تصاویر", utils.fa_num(stats["images"])),
            ("هشدارها", utils.fa_num(stats["alerts"])),
            ("رمزها", utils.fa_num(stats["codes"])),
            ("درخواست‌ها", utils.fa_num(http.stats.requests)),
            ("بلوکه", utils.fa_num(http.stats.blocked)),
            ("حجم", utils.human_bytes(http.stats.bytes_down)),
        ])
        if stats["codes"]:
            C.good("رمزهای یافت‌شده:")
            for c in store.codes(limit=8):
                C.write(f"   {C.c(c['code'], YELLOW_B)}  (dkp-{c['product_id']}, {c['source']})")
        http.close()
        store.close()
    return 0 if not interrupted else 130


def _write_cycle_report(store, ch, cfg) -> Path:
    import time as _t
    data = {
        "challenge": ch,
        "generated_at": _t.time(),
        "candidates": store.candidates(limit=200),
        "alerts": store.alerts(limit=200),
        "codes": store.codes(limit=200),
        "stats": store.stats(),
        "products_count": store.stats().get("products", 0),
        "scans": {"cycles": store.cycles(limit=20)},
    }
    stamp = utils.filename_stamp()
    p = Path(cfg.reports_dir) / f"report-{stamp}.html"
    return report_mod.write_report(p, data, cfg)


def _auto_report(res, store, ch, cfg, args) -> None:
    _write_cycle_report(store, ch, cfg)


# --------------------------------------------------------------------------- #
# دستور: candidates / codes / report / evidence
# --------------------------------------------------------------------------- #


def cmd_candidates(args) -> int:
    cfg = load_config(args.config)
    store = db.Store(resolve_db_path(cfg))
    try:
        cands = store.candidates(min_score=args.min_score, limit=args.top)
        if not cands:
            C.warn("نامزدی ثبت نشده. ابتدا `digiganj scan` را اجرا کنید.")
            return 0
        C.banner("نامزدهای نشانه‌دار", f"{utils.fa_num(len(cands))} تصویر")
        rows = []
        for c in cands:
            rows.append([
                str(c["id"]),
                f"dkp-{c['product_id']}",
                utils.jalali_short(c["stamp_ts"]) if c["stamp_ts"] else "—",
                utils.fa_num(round(float(c["score"]))),
                c["kind"] or "—",
                utils.shorten(c.get("ocr_codes") or "—", 30),
                utils.shorten(c.get("product_name") or "", 28),
            ])
        C.table(["#", "محصول", "زمان فایل", "امتیاز", "نوع", "رمزها", "نام"], rows)
        if args.json:
            out = Path(args.json)
            utils.write_json(out, cands)
            C.good(f"JSON: {out}")
        return 0
    finally:
        store.close()


def _lev(a: str, b: str) -> int:
    """فاصلهٔ ویرایشی برای خوشه‌بندی واریانت‌های OCR."""
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _code_quality(code: str) -> tuple:
    """امتیاز شباهت به کد واقعی: طول ۱۰، شروع با ۲ حرف کوچک، ۸ کاراکتر بعدی مخلوط."""
    score = 0
    if len(code) == 10:
        score += 3
    elif 9 <= len(code) <= 11:
        score += 1
    if len(code) >= 3 and code[:2].islower():
        score += 2
    if any(ch.isupper() for ch in code[2:]) and any(ch.isdigit() for ch in code[2:]):
        score += 2
    if code.endswith(("l", "I", "1")):
        score -= 2  # دنبالهٔ «l» اغلب چسبیدن لبهٔ کارت است
    if code.count("O") + code.count("0") > 1:
        score -= 1
    return (-score, -len(code), code)


def _group_codes(codes: list[dict]) -> list[list[dict]]:
    """کدهای نزدیک را کنار هم می‌گذارد؛ سرِ هر خوشه «بهترین» گونه است."""
    order = sorted(codes, key=lambda c: _code_quality(c["code"]))
    groups: list[list[dict]] = []
    for c in order:
        for g in groups:
            if _lev(g[0]["code"], c["code"]) <= 2:
                g.append(c)
                break
        else:
            groups.append([c])
    return groups


def cmd_codes(args) -> int:
    cfg = load_config(args.config)
    store = db.Store(resolve_db_path(cfg))
    try:
        codes = store.codes(limit=args.top, status=args.status)
        if not codes:
            C.warn("رمز ثبت‌شده‌ای نیست.")
            return 0
        # خوشه‌بندی واریانت‌ها: نسخه‌های نزدیک (فاصلهٔ لِوِنشتاین کوچک) کنار هم
        groups = _group_codes(codes)
        C.banner("رمزهای استخراج‌شده", f"{utils.fa_num(len(codes))} واریانت در {utils.fa_num(len(groups))} خوشه")
        rows = []
        for g in groups:
            head = g[0]
            variants = [c["code"] for c in g[1:]]
            rows.append([
                head["code"],
                f"dkp-{head['product_id']}",
                head["source"],
                str(len(g)),
                utils.jalali_short(head["ts"]),
                "، ".join(variants[:3]) + ("…" if len(variants) > 3 else ""),
            ])
        C.table(["رمز پیشنهادی", "محصول", "منبع", "واریانت", "زمان", "گونه‌های نزدیک"], rows)
        C.hint("«رمز پیشنهادی» بر پایهٔ شکل کد (حروف بزرگ/کوچک و کاراکترهای مشابه) رتبه‌بندی شده است؛ "
               "همهٔ گونه‌ها را می‌توانید به‌ترتیب در فرم رسمی بیازمایید.")
        C.hint("ستون «وضعیت» به‌صورت خودکار پر نمی‌شود؛ اعتبارسنجی رمز فقط با آزمون دستی خودتان انجام می‌شود.")
        return 0
    finally:
        store.close()


def cmd_report(args) -> int:
    cfg = load_config(args.config)
    store = db.Store(resolve_db_path(cfg))
    ch = parse_challenge_arg(args, cfg, store)
    try:
        stamp = utils.filename_stamp()
        out = Path(args.out or f"{cfg.reports_dir}/report-{stamp}.html")
        if args.embed:
            cfg.report_embed_images = True
        data = {
            "challenge": ch,
            "generated_at": utils.now_ts(),
            "candidates": store.candidates(limit=args.top),
            "alerts": store.alerts(limit=args.top),
            "codes": store.codes(limit=args.top),
            "stats": store.stats(),
            "products_count": store.stats().get("products", 0),
            "scans": {"cycles": store.cycles(limit=30)},
        }
        p = report_mod.write_report(out, data, cfg)
        C.good(f"گزارش نوشته شد: {p}")
        C.hint(f"حجم: {utils.human_bytes(p.stat().st_size)}")
        if args.json:
            jp = Path(args.json)
            utils.write_json(jp, data)
            C.good(f"JSON: {jp}")
        return 0
    finally:
        store.close()


def cmd_evidence(args) -> int:
    cfg = load_config(args.config)
    store = db.Store(resolve_db_path(cfg))
    ch = parse_challenge_arg(args, cfg, store)
    try:
        p = report_mod.write_evidence_pack(cfg.reports_dir, store, cfg, ch)
        C.good(f"بستهٔ شواهد: {p}")
        for f in sorted(Path(p).iterdir()):
            C.write(f"   {f.name} ({utils.human_bytes(f.stat().st_size)})")
        return 0
    finally:
        store.close()


def cmd_image(args) -> int:
    cfg = load_config(args.config)
    store = db.Store(resolve_db_path(cfg))
    try:
        img = store.image(args.image_id)
        if not img:
            C.err(f"تصویر #{args.image_id} در پایگاه‌داده نیست.")
            return 1
        C.kv([
            ("محصول", f"dkp-{img['product_id']}"),
            ("فایل", img["filename"]),
            ("زمان نام فایل", utils.jalali_pretty(img["stamp_ts"]) if img["stamp_ts"] else "—"),
            ("امتیاز", utils.fa_num(round(float(img["score"])))),
            ("وضعیت", img["kind"] or "—"),
            ("رمزها", img.get("ocr_codes") or "—"),
            ("مسیر محلی", img.get("file_path") or "(دانلود نشده)"),
        ])
        C.write("")
        C.write(C.c("پیوند تصویر: ", GRAY) + analyzer.image_url_large(img["url"]))
        if img.get("ocr_text"):
            C.write(C.c("متن OCR: ", GRAY) + utils.shorten(img["ocr_text"], 300))
        if args.save:
            if img.get("file_path") and Path(img["file_path"]).exists():
                shutil.copy(img["file_path"], args.save)
                C.good(f"کپی شد: {args.save}")
            else:
                C.warn("نسخهٔ محلی موجود نیست؛ ابتدا scan با OCR اجرا کنید.")
        return 0
    finally:
        store.close()


# --------------------------------------------------------------------------- #
# دستور: demo
# --------------------------------------------------------------------------- #


def cmd_demo(args) -> int:
    from . import demo
    srv = demo.DemoServer(
        port=args.port,
        products=args.products,
        treasure_product=args.treasure_product,
        reveal_after=args.reveal_after,
        treasure_code=args.code,
    )
    C.banner("digiganj — سرور دمو", f"http://127.0.0.1:{args.port}")
    C.hint(f"محصول گنج: dkp-{args.treasure_product}؛ پس از {utils.fa_num(args.reveal_after)} ثانیه منتشر می‌شود.")
    C.hint("برای آزمون کامل: در ترمینال دیگر `digiganj demo-scan --port "
           + str(args.port) + "` را اجرا کنید.")
    try:
        srv.httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.httpd.server_close()
    return 0


def cmd_demo_scan(args) -> int:
    """دمو + اسکن در یک فرمان: برای آزمون سریع کل جریان."""
    from . import demo

    srv = demo.DemoServer(
        port=args.port,
        products=args.products,
        treasure_product=args.treasure_product,
        reveal_after=0.0,
        treasure_code=args.code,
    ).start()
    time.sleep(0.3)
    cfg = config_mod.Config()
    cfg.base_url = srv.base_url
    cfg.api_base_url = srv.base_url
    cfg.data_dir = str(Path(args.workdir) / "data")
    cfg.db_path = str(Path(args.workdir) / "data" / "digiganj.sqlite")
    cfg.logs_dir = str(Path(args.workdir) / "logs")
    cfg.log_file = str(Path(args.workdir) / "logs" / "digiganj.log")
    cfg.reports_dir = str(Path(args.workdir) / "reports")
    cfg.scans_dir = str(Path(args.workdir) / "scans")
    cfg.gallery_source = "api"
    cfg.concurrency = 4
    cfg.ensure_dirs()

    now = time.time()
    ch = config_mod.Challenge(
        id="demo",
        title="دموی محلی",
        starts_at=utils.iso_utc(now - 120),
        ends_at=utils.iso_utc(now + 3600 * 6),
        list_path="/ajax/treasure-hunt/products/",
    )
    cfg.challenges = [ch]
    cfg.validate()

    C.banner("digiganj — آزمون کامل روی دمو", f"داده در {args.workdir}")
    con, log, http, store, dk, sc, ocr_runner, notifier = build_components(cfg)
    try:
        C.info(f"OCR آماده: {'، '.join(ocr_runner.engines) or 'خیر'}")
        if not ocr_runner.engines:
            C.warn("برای آزمون OCR، ابتدا جدول نویسه بسازید:")
            C.hint("digiganj ocr-train --fonts /usr/share/fonts/truetype/dejavu")
        res = sc.run_cycle(ch, force_listing=True)
        C.title("نتیجه")
        C.kv([
            ("گالری", utils.fa_num(res.galleries)),
            ("تصویر تازه", utils.fa_num(res.new_images)),
            ("دانلود", utils.fa_num(res.downloads)),
            ("OCR", utils.fa_num(res.ocr_runs)),
            ("هشدار", utils.fa_num(res.alerts)),
            ("رمز", utils.fa_num(res.codes)),
            ("زمان", utils.human_duration(res.elapsed)),
        ])
        alerts = store.alerts(limit=5)
        if alerts:
            C.good("نشانهٔ رمز پیدا شد:")
            for a in alerts[:3]:
                C.write(f"   dkp-{a['product_id']} — امتیاز {utils.fa_num(round(a['score']))}")
                if a.get("codes"):
                    C.write(f"   رمزها: {C.c(a['codes'], YELLOW_B)}")
        p = _write_cycle_report(store, ch, cfg)
        C.good(f"گزارش: {p}")
        return 0 if alerts else 2
    finally:
        http.close()
        store.close()
        srv.stop()


# --------------------------------------------------------------------------- #
# دستور: ocr-train / ocr-test
# --------------------------------------------------------------------------- #


def cmd_ocr_train(args) -> int:
    load_config(args.config)  # فقط برای سازگاری مسیر؛ جدول مستقل از پیکربندی است
    C.banner("digiganj — آموزش جدول نویسه", "برای موتور OCR داخلی")
    fonts = args.fonts
    if not fonts:
        for cand in (
            "/usr/share/fonts/truetype/dejavu",
            "/usr/share/fonts/TTF",
            "/Library/Fonts",
            "C:/Windows/Fonts",
        ):
            if Path(cand).exists():
                fonts = cand
                break
    if fonts and not Path(fonts).exists():
        C.err(f"مسیر فونت‌ها پیدا نشد: {fonts}")
        return 1
    scales = [int(s) for s in args.scales.split(",")] if args.scales else (16, 22, 30, 44)
    thickness = (False, True) if args.bold else (False,)
    out = Path(args.out or "glyphs.npz")
    info = ocr_mod.train_glyph_table(
        out, fonts_dir=fonts, scales=scales, thickness=thickness
    )
    C.good(
        f"جدول ساخته شد: {utils.fa_num(info['glyphs'])} نمونه از "
        f"{utils.fa_num(info['classes'])} نویسه → {info['path']}"
    )
    if not fonts:
        C.warn("بدون فونت TrueType، دقت موتور داخلی کم است؛ یک پوشهٔ فونت بدهید.")
    return 0


def cmd_ocr_test(args) -> int:
    cfg = load_config(args.config)
    runner = ocr_mod.OcrRunner(cfg, glyphs_path=find_glyphs_path())
    if not runner.available:
        C.err("هیچ موتور OCR آماده نیست.")
        return 1
    targets = list(args.files or [])
    if not targets and args.image_id:
        store = db.Store(resolve_db_path(cfg))
        img = store.image(args.image_id)
        store.close()
        if img and img.get("file_path"):
            targets = [img["file_path"]]
    if not targets:
        C.err("یک تصویر بدهید: digiganj ocr-test IMAGE")
        return 1
    C.info(f"موتورها: {'، '.join(runner.engines)}")
    for t in targets:
        p = Path(t)
        if not p.exists():
            C.warn(f"فایل نیست: {p}")
            continue
        res = runner.run(str(p))
        C.write("")
        C.write(C.c(p.name, BOLD))
        C.kv([
            ("تشخیص", res.engine),
            ("امتیاز", utils.fa_num(f"{res.score:.2f}")),
            ("زمان", utils.fa_num(f"{res.elapsed * 1000:.0f}") + " میلی‌ثانیه"),
        ])
        if res.texts:
            C.write(C.c("متن‌ها: ", GRAY) + " | ".join(res.texts[:4]))
        if res.codes:
            C.good("رمزهای محتمل: " + "، ".join(res.codes[:8]))
        elif res.error:
            C.warn(res.error)
    return 0


# --------------------------------------------------------------------------- #
# دستور: demo
# --------------------------------------------------------------------------- #


def cmd_stats(args) -> int:
    cfg = load_config(args.config)
    store = db.Store(resolve_db_path(cfg))
    try:
        stats = store.stats()
        C.banner("digiganj — آمار پایگاه‌داده")
        rows = [[k, utils.fa_num(v) if isinstance(v, (int, float)) else str(v or "—")]
                for k, v in stats.items()]
        C.table(["شمارنده", "مقدار"], rows)
        if args.cycles:
            C.title("دوره‌های اخیر")
            rows2 = []
            for c in store.cycles(limit=10):
                rows2.append([
                    str(c["id"]), c["challenge_id"],
                    utils.jalali_short(c["started_ts"]),
                    utils.fa_num(c["galleries"]),
                    utils.fa_num(c["new_images"]),
                    utils.fa_num(c["alerts"]),
                    utils.human_duration(c["ended_ts"] - c["started_ts"])
                    if c["ended_ts"] else "—",
                ])
            C.table(["#", "چالش", "شروع", "گالری", "تازه", "هشدار", "مدت"], rows2)
        return 0
    finally:
        store.close()


def cmd_events(args) -> int:
    cfg = load_config(args.config)
    store = db.Store(resolve_db_path(cfg))
    try:
        evs = store.events(limit=args.top)
        if not evs:
            C.warn("رویدادی ثبت نشده.")
            return 0
        for e in evs:
            C.write(f"{C.c(utils.jalali_datetime(e['ts']), GRAY)} "
                    f"[{e['kind']}] {e['message']}")
        return 0
    finally:
        store.close()


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="digiganj",
        description="ابزار پژوهشی «شکار گنج» دیجی‌کالا — یافتن نشانهٔ رمز، ثبت شواهد و گزارش شفافیت.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--version", action="version", version=f"digiganj {__version__}")
    ap.add_argument("--config", help="مسیر فایل پیکربندی (پیش‌فرض: config.json)")
    sub = ap.add_subparsers(dest="cmd")

    p = sub.add_parser("doctor", help="بررسی سلامت محیط و دسترسی شبکه")
    p.add_argument("--config", default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    p.set_defaults(func=cmd_doctor)

    p = sub.add_parser("config", help="مدیریت پیکربندی")
    p.add_argument("action", nargs="?", default="show",
                   choices=["init", "show", "set", "path"])
    p.add_argument("key", nargs="?")
    p.add_argument("value", nargs="?")
    p.add_argument("--out", help="مسیر فایل خروجی برای init")
    p.add_argument("--force", action="store_true", help="بازنویسی فایل موجود")
    p.add_argument("--config", default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    p.set_defaults(func=cmd_config)

    p = sub.add_parser("challenge", help="مدیریت چالش‌ها")
    p.add_argument("action", nargs="?", default="list", choices=["list", "ls", "add", "rm"])
    p.add_argument("--id")
    p.add_argument("--title")
    p.add_argument("--start")
    p.add_argument("--end")
    p.add_argument("--prize")
    p.add_argument("--note")
    p.add_argument("--era")
    p.add_argument("--list-path")
    p.add_argument("--config", default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    p.set_defaults(func=cmd_challenge)

    p = sub.add_parser("auth", help="ذخیرهٔ کوکی نشست خودتان (اختیاری)")
    p.add_argument("--cookie", help="رشتهٔ کامل کوکی")
    p.add_argument("--clear", action="store_true")
    p.add_argument("--config", default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    p.set_defaults(func=cmd_auth)

    p = sub.add_parser("list", help="خواندن فهرست محصولات چالش")
    p.add_argument("--challenge", default="auto")
    p.add_argument("--limit", type=int, default=0, help="حداکثر تعداد محصول (0=همه)")
    p.add_argument("--show", type=int, default=30)
    p.add_argument("--config", default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    p.set_defaults(func=cmd_list)

    p = sub.add_parser("scan", help="اسکن گالری و یافتن نشانهٔ رمز")
    p.add_argument("--challenge", default="auto")
    p.add_argument("--once", action="store_true", help="فقط یک چرخه")
    p.add_argument("--watch", action="store_true", help="حلقهٔ بلندمدت")
    p.add_argument("--forever", action="store_true",
                   help="پس از پایان چالش هم ادامه بده (پیش‌فرض: توقف)")
    p.add_argument("--prewarm", type=float, default=0.0,
                   help="دقیقه‌ها پیش از شروع، ایندکس کامل بساز")
    p.add_argument("--interval", type=float, default=45.0,
                   help="فاصلهٔ چرخه‌ها در حالت watch (ثانیه)")
    p.add_argument("--cycles", type=int, default=0, help="حداکثر چرخه (0=بی‌نهایت)")
    p.add_argument("--limit", type=int, default=0, help="حداکثر محصول در چرخه")
    p.add_argument("--concurrency", type=int, default=0)
    p.add_argument("--refresh", action="store_true", help="فهرست را از نو بخوان")
    p.add_argument("--no-ocr", action="store_true", help="OCR را خاموش کن")
    p.add_argument("--report", action="store_true", help="گزارش پس از پایان")
    p.add_argument("--report-every", type=float, default=0.0,
                   help="هر N ثانیه گزارش خودکار بساز (0=خاموش)")
    p.add_argument("--offline", action="store_true",
                   help="اتصال به سرور دموی محلی (localhost:8099)")
    p.add_argument("--config", default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    p.set_defaults(func=cmd_scan)

    p = sub.add_parser("candidates", help="نمایش نامزدهای نشانه‌دار")
    p.add_argument("--top", type=int, default=30)
    p.add_argument("--min-score", type=float, default=25.0)
    p.add_argument("--json", help="ذخیرهٔ JSON در این مسیر")
    p.add_argument("--config", default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    p.set_defaults(func=cmd_candidates)

    p = sub.add_parser("codes", help="نمایش رمزهای استخراج‌شده")
    p.add_argument("--top", type=int, default=50)
    p.add_argument("--status", choices=["unknown", "likely", "invalid", "used", "valid"])
    p.add_argument("--config", default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    p.set_defaults(func=cmd_codes)

    p = sub.add_parser("report", help="ساخت گزارش HTML")
    p.add_argument("--challenge", default="auto")
    p.add_argument("--out", help="مسیر فایل خروجی")
    p.add_argument("--top", type=int, default=40)
    p.add_argument("--embed", action="store_true", help="جاسازی تصاویر (خودبسنده)")
    p.add_argument("--json", help="خروجی JSON مکمل")
    p.add_argument("--config", default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("evidence", help="ساخت بستهٔ شواهد (HTML+JSON)")
    p.add_argument("--challenge", default="auto")
    p.add_argument("--config", default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    p.set_defaults(func=cmd_evidence)

    p = sub.add_parser("image", help="نمایش جزئیات یک تصویر")
    p.add_argument("image_id", type=int)
    p.add_argument("--save", help="کپی فایل محلی به این مسیر")
    p.add_argument("--config", default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    p.set_defaults(func=cmd_image)

    p = sub.add_parser("stats", help="آمار پایگاه‌داده")
    p.add_argument("--cycles", action="store_true", help="نمایش دوره‌های اخیر")
    p.add_argument("--config", default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    p.set_defaults(func=cmd_stats)

    p = sub.add_parser("events", help="رویدادهای ثبت‌شده")
    p.add_argument("--top", type=int, default=50)
    p.add_argument("--config", default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    p.set_defaults(func=cmd_events)

    p = sub.add_parser("ocr-train", help="ساخت جدول نویسه برای موتور OCR داخلی")
    p.add_argument("--fonts", help="پوشهٔ فونت‌های TTF/OTF")
    p.add_argument("--out", default="glyphs.npz")
    p.add_argument("--scales", help="اندازه‌های رندر، مثل 16,22,30,44")
    p.add_argument("--bold", action="store_true", help="شامل وزن ضخیم هم بشود")
    p.add_argument("--config", default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    p.set_defaults(func=cmd_ocr_train)

    p = sub.add_parser("ocr-test", help="آزمایش OCR روی یک تصویر")
    p.add_argument("files", nargs="*")
    p.add_argument("--image-id", type=int)
    p.add_argument("--config", default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    p.set_defaults(func=cmd_ocr_test)

    p = sub.add_parser("demo", help="اجرای سرور دموی محلی (بدون تماس با دیجی‌کالا)")
    p.add_argument("--port", type=int, default=8099)
    p.add_argument("--products", type=int, default=60)
    p.add_argument("--treasure-product", type=int, default=1007)
    p.add_argument("--reveal-after", type=float, default=12.0)
    p.add_argument("--code", default="fvKA4jd9DP")
    p.add_argument("--config", default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    p.set_defaults(func=cmd_demo)

    p = sub.add_parser("demo-scan", help="دمو + اسکن کامل در یک فرمان")
    p.add_argument("--port", type=int, default=8099)
    p.add_argument("--products", type=int, default=25)
    p.add_argument("--treasure-product", type=int, default=1007)
    p.add_argument("--code", default="fvKA4jd9DP")
    p.add_argument("--workdir", default="demo-run")
    p.add_argument("--config", default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    p.set_defaults(func=cmd_demo_scan)
    return ap


def main(argv: list[str] | None = None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)
    if not getattr(args, "func", None):
        ap.print_help()
        return 0
    try:
        return int(args.func(args) or 0)
    except KeyboardInterrupt:
        C.warn("متوقف شد.")
        return 130
    except SystemExit:
        raise
    except Exception as exc:
        import traceback

        C.err(f"خطا: {utils.shorten(exc, 300)}")
        if os.environ.get("DIGIGANJ_DEBUG"):
            traceback.print_exc()
        else:
            C.hint("برای جزئیات بیشتر: DIGIGANJ_DEBUG=1 ... اجرا کنید.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
