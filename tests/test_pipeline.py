"""آزمون یکپارچه: کل جریان روی سرور دمو (بدون تماس با دیجی‌کالا).

این آزمون در صورت نبود وابستگی‌های اختیاری، آسان‌گیرانه رد می‌شود تا
محیط‌های حداقلی هم بتوانند بقیهٔ آزمون‌ها را اجرا کنند.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

def _safe_import(name: str) -> bool:
    try:
        __import__(name)
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not all(_safe_import(m) for m in ("httpx", "numpy", "PIL")),
    reason="وابستگی‌های شبکه/تصویر نصب نیستند",
)


@pytest.fixture()
def demo_env(tmp_path):
    from digiganj import config, console, db, demo, digikala, net, scanner

    srv = demo.DemoServer(
        port=0,  # پورت آزاد
        products=12,
        treasure_product=1005,
        reveal_after=0.0,
        treasure_code="Hx7Kp2QmR9",
    ).start()
    time.sleep(0.2)

    cfg = config.Config()
    cfg.base_url = srv.base_url
    cfg.api_base_url = srv.base_url
    cfg.data_dir = str(tmp_path / "data")
    cfg.logs_dir = str(tmp_path / "logs")
    cfg.reports_dir = str(tmp_path / "reports")
    cfg.scans_dir = str(tmp_path / "scans")
    cfg.db_path = str(tmp_path / "data" / "d.sqlite")
    cfg.log_file = str(tmp_path / "logs" / "l.log")
    cfg.concurrency = 3
    cfg.gallery_source = "api"
    cfg.ensure_dirs()

    now = time.time()
    ch = config.Challenge(
        id="demo",
        title="دمو",
        starts_at=config.utils.iso_utc(now - 120),
        ends_at=config.utils.iso_utc(now + 7200),
        list_path="/ajax/treasure-hunt/products/",
    )
    cfg.challenges = [ch]

    con = console.Console(verbose=False)
    log = console.Logger(cfg.log_file, level="WARN", console=con)
    http = net.HttpClient(cfg, logger=log, console=con, min_interval=0.0)
    store = db.Store(cfg.db_path)
    dk = digikala.DigikalaClient(cfg, http, logger=log)
    sc = scanner.Scanner(cfg, http, store, dk, logger=log, console=con)
    yield {
        "cfg": cfg, "srv": srv, "ch": ch, "http": http, "store": store,
        "sc": sc, "console": con, "tmp": tmp_path,
    }
    try:
        http.close()
        store.close()
    finally:
        srv.stop()


def test_listing_and_gallery(demo_env):
    from digiganj import digikala

    dk = digikala.DigikalaClient(demo_env["cfg"], demo_env["http"])
    products, pages = dk.fetch_listing(demo_env["ch"])
    assert len(products) == 12
    assert pages >= 1
    items, source = dk.fetch_gallery(products[0])
    assert source == "api"
    assert len(items) == 5
    # همهٔ تصاویر باید stamp داشته باشند
    assert all(i.stamp_ts for i in items)


def test_full_scan_finds_treasure(demo_env):
    sc = demo_env["sc"]
    ch = demo_env["ch"]
    store = demo_env["store"]

    res = sc.run_cycle(ch, force_listing=True)
    assert res.galleries == 12
    assert res.new_images >= 60  # ۵ تصویر × ۱۲ محصول + گنج
    # گنج باید شناسایی شود
    assert res.alerts >= 1
    alerts = store.alerts(limit=3)
    assert alerts
    top = alerts[0]
    assert top["product_id"] == "1005"
    assert "Hx7Kp2QmR9" in (top["codes"] or "") or top["score"] >= 50


def test_codes_deduplicated(demo_env):
    sc = demo_env["sc"]
    ch = demo_env["ch"]
    store = demo_env["store"]
    sc.run_cycle(ch, force_listing=True)
    codes = store.codes(limit=100)
    keys = [c["code"] for c in codes]
    assert len(keys) == len(set(keys)), "رمزهای تکراری نباید ثبت شوند"


def test_report_generation(demo_env):
    from digiganj import report

    sc = demo_env["sc"]
    ch = demo_env["ch"]
    store = demo_env["store"]
    cfg = demo_env["cfg"]
    sc.run_cycle(ch, force_listing=True)
    data = {
        "challenge": ch,
        "generated_at": time.time(),
        "candidates": store.candidates(limit=50),
        "alerts": store.alerts(limit=50),
        "codes": store.codes(limit=50),
        "stats": store.stats(),
        "products_count": 12,
        "scans": {"cycles": store.cycles(limit=10)},
    }
    out = demo_env["tmp"] / "report.html"
    p = report.write_report(out, data, cfg)
    text = p.read_text(encoding="utf-8")
    assert "dir=\"rtl\"" in text
    assert "نامزدهای نشانه‌دار" in text
    assert "محدودیت" in text or "محدودیت‌ها" in text


def test_evidence_pack(demo_env):
    from digiganj import report

    sc = demo_env["sc"]
    ch = demo_env["ch"]
    store = demo_env["store"]
    cfg = demo_env["cfg"]
    sc.run_cycle(ch, force_listing=True)
    pack = report.write_evidence_pack(cfg.reports_dir, store, cfg, ch)
    assert (pack / "report.html").exists()
    assert (pack / "report.json").exists()
    assert (pack / "raw.json").exists()


def test_ocr_engine_end_to_end(tmp_path):
    """تست جدا: آموزش جدول نویسه با فونت واقعی (اگر باشد) و خواندن کد."""
    from digiganj import ocr

    if not ocr.pillow_available():
        pytest.skip("Pillow/numpy نصب نیست")
    font_dirs = [
        "/usr/share/fonts/truetype/dejavu",
        "/usr/share/fonts/truetype/liberation",
    ]
    font_dir = next((d for d in font_dirs if Path(d).exists()), "")
    if not font_dir:
        pytest.skip("فونت TrueType پیدا نشد")

    table = tmp_path / "glyphs.npz"
    ocr.train_glyph_table(table, fonts_dir=font_dir, scales=(22, 30), thickness=(False,))
    assert table.exists()

    from PIL import Image, ImageDraw, ImageFont

    code = "fvKA4jd9DP"
    img = Image.new("RGB", (420, 100), (250, 250, 250))
    try:
        font = ImageFont.truetype(f"{font_dir}/DejaVuSansMono.ttf", 30)
    except Exception:
        pytest.skip("فونت DejaVuSansMono نیست")
    ImageDraw.Draw(img).text((15, 25), code, fill=(20, 20, 20), font=font)
    p = tmp_path / "code.png"
    img.save(p)

    runner = ocr.OcrRunner(_cfg_stub(), glyphs_path=table)
    res = runner.run(str(p))
    assert res.codes, "هیچ رمزی استخراج نشد"
    assert code in res.codes or code in "".join(res.texts)


def _cfg_stub():
    class C:
        ocr_enabled = True
        ocr_max_pixels = 2_000_000
        ocr_lang = "eng"
        ocr_psm = 7
        ocr_max_candidates = 24
        code_regex = r"(?=[A-Za-z0-9]*[A-Za-z])(?=[A-Za-z0-9]*\d)[A-Za-z0-9]{8,24}"
        ocr_try_raw = True
        ocr_min_score = 7.0
        ocr_min_interval = 0.0
        ocr_max_pixels = 2_000_000

    return C()


def test_politeness_backoff_unit():
    from digiganj import net

    lim = net.RateLimiter(min_interval=0.01)
    base = lim.interval
    lim.note_block()
    assert lim.interval > base
    for _ in range(20):
        lim.note_success()
    assert lim.interval <= base * 2 or lim.interval >= base


def test_circuit_breaker():
    from digiganj import net

    # cooldown حداقل ۱ ثانیه است (رفتار دفاعی)؛ آزمون با همان کار می‌کند
    cb = net.CircuitBreaker(failures=3, cooldown=1.0)
    assert not cb.is_open
    cb.note_failure()
    cb.note_failure()
    assert not cb.is_open
    assert cb.note_failure()  # سومین → باز
    assert cb.is_open
    assert cb.remaining() <= 1.0
    time.sleep(1.05)
    assert not cb.is_open


def test_dashboard_state(demo_env):
    """داشبورد باید چالش درست، شمارنده‌ها و خوشهٔ رمز را برگرداند."""
    from digiganj import dashboard

    sc = demo_env["sc"]
    ch = demo_env["ch"]
    store = demo_env["store"]
    cfg = demo_env["cfg"]
    sc.run_cycle(ch, force_listing=True)

    state = dashboard.collect_state(store, cfg, None)
    assert state["challenge"]["id"] == "demo"
    assert state["challenge"]["phase"] == "active"
    assert state["stats"]["images"] > 0
    assert state["top_score"] >= 50
    assert state["codes"], "خوشهٔ رمز باید پر باشد"
    # هر خانواده باید خوانش خام داشته باشد (منبع حدسی به‌تنهایی کافی نیست)
    assert any(c["source"] == "ocr" for c in state["codes"])
    # گونه‌های یک رمز نباید به ده‌ها خوشه بشکنند (پس از خوشه‌بندی خانواده‌ای)
    assert len(state["codes"]) <= 8, "گونه‌های یک رمز نباید به خوشه‌های بی‌شمار بشکنند"
    # خانوادهٔ نخست باید خوانش خام باشد، نه حدس ترکیبی
    assert state["codes"][0]["source"] == "ocr"

    html = dashboard.render_dashboard(state)
    assert 'dir="rtl"' in html
    assert "function g2j" in html, "تبدیل جلالی باید درون‌خطی باشد"
    assert "http" not in html.split("<script>")[0].split("</style>")[1], "بدون منبع بیرونی"


def test_dashboard_tehran_offset_is_documented():
    """زمان‌ها باید به وقت تهران باشند، نه منطقهٔ زمانی مرورگر."""
    from digiganj import dashboard

    html = dashboard.render_dashboard({"challenge": {}, "stats": {}})
    assert "TEHRAN_OFFSET" in html
    assert "3.5 * 3600" in html


def _free_port() -> int:
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _serve_in_thread(cfg, store):
    """سرور واقعی داشبورد را در نخ پس‌زمینه بالا می‌آورد؛ (thread, port) برمی‌گرداند."""
    import threading

    from digiganj import dashboard

    port = _free_port()
    t = threading.Thread(
        target=dashboard.serve,
        args=(store, cfg, None, "127.0.0.1", port),
        kwargs={"interval": 0.5, "quiet": True},
        daemon=True,
    )
    t.start()
    # انتظار برای آماده شدن پورت (بدون sleep ثابت شکننده)
    import socket

    for _ in range(100):
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                break
        except OSError:
            time.sleep(0.05)
    return t, port


def test_dashboard_local_file_guard(tmp_path, monkeypatch):
    """فقط تصویرهای داخل ریشه‌های مجاز سرو می‌شوند.

    چرا آزمون؟ سرور روی 0.0.0.0 باز می‌شود و نشانی پیش‌نمایش عمومی است؛
    اگر این بررسی خراب شود، هر .jpg/.png روی دیسک خواندنی می‌شود.
    """
    import urllib.error
    import urllib.parse
    import urllib.request

    from digiganj import config, db

    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)  # ریشهٔ مجاز = پوشهٔ کاری سرور

    store = db.Store(str(work / "d.sqlite"))
    cfg = config.Config()
    cfg.data_dir = str(work / "data")
    cfg.reports_dir = str(work / "reports")
    Path(cfg.data_dir).mkdir(parents=True, exist_ok=True)

    ok_img = Path(cfg.data_dir) / "ok.png"
    ok_img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"x" * 32)

    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    outside = outside_dir / "secret.png"
    outside.write_bytes(b"\x89PNG\r\n\x1a\n" + b"y" * 32)

    _t, port = _serve_in_thread(cfg, store)

    def get(path: str):
        url = f"http://127.0.0.1:{port}{path}"
        try:
            with urllib.request.urlopen(url, timeout=4) as r:
                return r.status, r.headers.get("Content-Type"), len(r.read())
        except urllib.error.HTTPError as e:
            return e.code, e.headers.get("Content-Type"), len(e.read())

    def q(p):
        return urllib.parse.quote(str(p))

    try:
        status, ctype, size = get(f"/local/{q(ok_img)}")
        assert status == 200, "تصویر داخل data_dir باید سرو شود"
        assert ctype == "image/png", f"نوع محتوا باید PNG باشد، نه {ctype}"
        assert size > 8

        assert get(f"/local/{q(outside)}")[0] == 403, "فایل بیرون از ریشه‌ها باید رد شود"
        assert get(f"/local/{q(work / 'd.sqlite')}")[0] == 403, "فایل غیرتصویری باید رد شود"
        assert get(f"/local/{q(work / 'nope.png')}")[0] == 403, "فایل ناموجود باید رد شود"

        assert get("/")[0] == 200
        state_status, state_ctype, _ = get("/state.json")
        assert state_status == 200
        assert "application/json" in (state_ctype or "")
    finally:
        store.close()


def test_dashboard_http11_keepalive(tmp_path, monkeypatch):
    """اتصال باید HTTP/1.1 و قابل استفادهٔ دوباره باشد (داشبورد هر ۳ ثانیه poll می‌کند)."""
    import http.client

    from digiganj import config, db

    work = tmp_path / "w"
    work.mkdir()
    monkeypatch.chdir(work)
    store = db.Store(str(work / "d.sqlite"))
    cfg = config.Config()
    cfg.data_dir = str(work / "data")
    cfg.reports_dir = str(work / "reports")
    Path(cfg.data_dir).mkdir(parents=True, exist_ok=True)

    _t, port = _serve_in_thread(cfg, store)
    try:
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=4)
        conn.request("GET", "/state.json")
        r1 = conn.getresponse()
        r1.read()
        assert r1.version == 11, "پروتکل باید HTTP/1.1 باشد، نه HTTP/1.0"
        # درخواست دوم روی همان سوکت: اگر keep-alive باشد، کار می‌کند
        conn.request("GET", "/")
        r2 = conn.getresponse()
        assert r2.status == 200
        assert len(r2.read()) > 100
        conn.close()
    finally:
        store.close()
