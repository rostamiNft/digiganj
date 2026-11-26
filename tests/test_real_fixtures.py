"""آزمون سازگاری با دادهٔ واقعی ۱۴۰۰ (از ریپوی مرجع hosni).

این فیکسچرها همان پاسخ‌های واقعی دیجی‌کالا در نوامبر ۲۰۲۱ هستند؛ اگر روزی
ساختار API عوض شود، این آزمون‌ها زودتر از همه خبر می‌دهند.

منبع:
- listing: ``/ajax/treasure-hunt/products/?pageno=1`` (۳۶ محصول، ۴۶ صفحه)
- gallery: ``api.digikala.com/v1/product/dkp-*/`` با ``image_list`` رشته‌ای
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

FIX = Path(__file__).resolve().parent / "fixtures"


def _load(name: str):
    with open(FIX / name, "r", encoding="utf-8") as fh:
        return json.load(fh)


def test_listing_2021_shape():
    from digiganj import digikala

    payload = _load("listing_2021_mini.json")
    page = digikala.parse_listing(payload, page=1)
    assert len(page.products) == 3
    assert page.pages == 46, "تعداد صفحه‌ها باید از trackerData خوانده شود"
    first = page.products[0]
    assert first.id == "4958276"
    assert "POCO" in first.name or "شیائومی" in first.name
    assert first.price == 66990000
    assert first.page == 1
    assert first.rank == 1


def test_gallery_2021_shape():
    from digiganj import digikala, models

    payload = _load("product_2021.json")
    urls = digikala.extract_gallery_from_json(payload, "https://api.digikala.com")
    # ۶ تصویر image_list + ۱ main
    assert len(urls) == 7
    assert all(u.startswith("https://dkstatics-public.digikala.com/") for u in urls)

    stamps = [models.parse_stamp(u) for u in urls]
    assert all(kind == "epoch_s" for _, kind in stamps), "همه باید epoch ثانیه باشند"
    assert all(ts is not None and 1.6e9 < ts < 1.7e9 for ts, _ in stamps)


def test_gallery_real_url_has_upload_epoch():
    """درس اصلی پژوهش ۱۴۰۰: epoch در نام فایل، نشانهٔ لحظهٔ بارگذاری است."""
    from digiganj import models

    url = (
        "https://dkstatics-public.digikala.com/digikala-products/"
        "9437718e809537ca2d4d2dcf7fa6bcbfa42fae1b_1632197880.jpg"
        "?x-oss-process=image/resize,m_lfit,h_800,w_800/quality,q_60"
    )
    ts, kind = models.parse_stamp(url)
    assert kind == "epoch_s"
    assert ts == 1632197880.0
    # همان روز چالش ۱۴۰۰ (۲۴ نوامبر ۲۰۲۱)
    from digiganj import utils

    assert utils.gregorian_to_jalali(2021, 11, 24) == (1400, 9, 3)


def test_listing_html_fallback():
    """اگر API جواب نداد، HTML فهرست هم باید تجزیه شود."""
    from digiganj import digikala

    html = """
    <html><body>
      <div class="c-listing">
        <a class="c-product-box__img" href="/product/dkp-1234/phone/">
          <img src="//dkstatics-public.digikala.com/x/abc_1637930000.jpg">
        </a>
        <a class="c-product-box__img" href="/product/dkp-5678/case/">
          <img data-src="https://dkstatics-public.digikala.com/x/def_1637930100.jpg">
        </a>
      </div>
    </body></html>
    """
    products = digikala.parse_listing_html(html, page=1)
    assert len(products) == 2
    assert {p.id for p in products} == {"1234", "5678"}


def test_api_product_url_variants():
    from digiganj import config

    cfg = config.Config()
    assert cfg.gallery_api_url("1007") == "https://api.digikala.com/v1/product/dkp-1007/"
    assert cfg.product_url("42").endswith("/product/dkp-42/")


# --------------------------------------------------------------------------- #
# مسیر جستجوی دسته‌بندی (قالب ۱۴۰۵)
# --------------------------------------------------------------------------- #

# صفحهٔ واقعی جستجو (ساختار، بدون داده‌های حجیم)
_SEARCH_HTML = """
<html><body>
<div class="c-listing">
  <a href="https://www.digikala.com/product/dkp-21345476/body-splash/?x=1">x</a>
  <a href="https://www.digikala.com/product/dkp-22063450/coin/">x</a>
  <a href="https://www.digikala.com/product/dkp-21979346/mist/">x</a>
</div>
<span>۱۲,۹۷۶ کالا</span>
</body></html>
"""


def _client():
    from digiganj import config, digikala

    return digikala.DigikalaClient(config.Config(), None, None), config.Config()


def test_search_pagination_uses_page_param():
    """جستجو با `&page=N` صفحه‌بندی می‌شود، نه `pageno`."""
    client, cfg = _client()
    ch = cfg.challenge("bf-1405")
    assert client.listing_url(1, ch) == cfg.challenge_url(ch)
    p2 = client.listing_url(2, ch)
    assert "page=2" in p2
    assert "pageno" not in p2
    assert p2.startswith(cfg.challenge_url(ch))


def test_legacy_pagination_unchanged():
    """فهرست ایونت ۱۴۰۰ همان `pageno`+`sortby` را دارد."""
    client, cfg = _client()
    old = cfg.challenge("bf-1400")
    u = client.listing_url(3, old)
    assert "pageno=3" in u
    assert "sortby=4" in u


def test_total_from_html_persian_digits():
    client, _ = _client()
    assert client._total_from_html(_SEARCH_HTML) == 12976
    assert client._total_from_html("<span>1,234 کالا</span>") == 1234
    assert client._total_from_html("بدون عدد") == 0


def test_search_pages_estimation_sane():
    client, _ = _client()
    pages = client._estimate_search_pages(_SEARCH_HTML, 1)
    assert pages == 541, f"۱۲٬۹۷۶ کالا / ۲۴ = ۵۴۱ صفحه، نه {pages}"
    # بدون عدد: فقط یک صفحهٔ بعد
    assert client._estimate_search_pages("<html></html>", 1) == 2
    # سقف امن
    assert client._estimate_search_pages("<span>۹۹۹٬۹۹۹ کالا</span>", 1) <= 800


def test_search_html_parsed_into_products():
    client, cfg = _client()
    prods = client._parse_listing_html(_SEARCH_HTML, 1)
    assert [p.id for p in prods] == ["21345476", "22063450", "21979346"]
    assert all(p.page == 1 for p in prods)


def test_is_search_path_detection():
    client, cfg = _client()
    assert client._is_search_path(cfg.challenge("bf-1405"))
    assert not client._is_search_path(cfg.challenge("bf-1400"))
    # چالشی که فقط دسته‌بندی دارد
    from digiganj import config

    ch = config.Challenge(
        id="x", title="x",
        starts_at="2026-11-26T00:00:00+03:30",
        ends_at="2026-11-27T00:00:00+03:30",
        category_ids=[10942],
    )
    assert client._is_search_path(ch)


# --------------------------------------------------------------------------- #
# صفحهٔ جستجو باید با هدر «سند» خواسته شود، نه هدر «XHR/JSON»
# --------------------------------------------------------------------------- #

_SHELL_HTML = "<HTML><body><div id='app'>Loading...</div></body></HTML>"


def _full_search_html(n=20, total="۱۲,۹۷۶"):
    links = "".join(
        f'<a href="https://www.digikala.com/product/dkp-{21345470 + i}/x/">p</a>'
        for i in range(n)
    )
    return f"<html><body>{links}<span>{total} کالا</span></body></html>"


class _FakeResp:
    def __init__(self, text, status=200):
        self.text = text
        self.status = status
        self.content = text.encode()
        self.ok = status == 200


class _MwHttp:
    """دیجی‌کالا را شبیه‌سازی می‌کند: درخواست JSON ⇒ پوستهٔ خالی، سند ⇒ صفحهٔ کامل."""

    def __init__(self, full=None):
        from digiganj import net

        self.identity = net.Identity(user_agent="UA", profile={})
        self.calls: list[tuple[str, str]] = []
        self.full = full if full is not None else _full_search_html()

    def request(self, method, url, headers=None, **kw):
        h = {k.lower(): v for k, v in (headers or {}).items()}
        wants_json = "x-requested-with" in h or "json" in h.get("accept", "")
        self.calls.append((url, "XHR" if wants_json else "DOCUMENT"))
        return _FakeResp(_SHELL_HTML if wants_json else self.full)


def test_search_listing_uses_document_headers():
    """اگر با هدر XHR برویم، پوستهٔ خالی می‌گیریم و اسکن می‌شکند."""
    from digiganj import config, digikala

    cfg = config.Config()
    http = _MwHttp()
    client = digikala.DigikalaClient(cfg, http, None)
    lp = client.fetch_listing_page(1, cfg.challenge("bf-1405"))

    assert len(lp.products) == 20
    assert lp.pages == 541
    assert [kind for _, kind in http.calls] == ["DOCUMENT"], http.calls
    assert "application/json" not in str(http.calls)


def test_search_listing_falls_back_to_bare_url():
    """اگر آدرس با page=N خالی بود، آدرس بدون آن دوباره امتحان می‌شود."""
    from digiganj import config, digikala

    class PickyHttp(_MwHttp):
        def request(self, method, url, headers=None, **kw):
            self.calls.append((url, "?"))
            # صفحهٔ ۲ فقط بدون پارامتر page رندر می‌شود
            if "&page=" in url:
                return _FakeResp(_SHELL_HTML)
            return _FakeResp(self.full)

    cfg = config.Config()
    http = PickyHttp()
    client = digikala.DigikalaClient(cfg, http, None)
    lp = client.fetch_listing_page(2, cfg.challenge("bf-1405"))

    assert len(lp.products) == 20
    assert len(http.calls) == 2, http.calls
    assert "&page=" in http.calls[0][0]
    assert "&page=" not in http.calls[1][0]


def test_dead_search_page_error_is_actionable():
    """پیام خطا باید علت (پوستهٔ خالی) را بگوید، نه ادعای «ساختار عوض شده»."""
    import pytest

    from digiganj import config, digikala, net

    class DeadHttp(_MwHttp):
        def request(self, method, url, headers=None, **kw):
            self.calls.append((url, "?"))
            return _FakeResp(_SHELL_HTML)

    cfg = config.Config()
    client = digikala.DigikalaClient(cfg, DeadHttp(), None)
    with pytest.raises(net.NetError) as info:
        client.fetch_listing_page(1, cfg.challenge("bf-1405"))

    msg = str(info.value)
    assert "پوستهٔ خالی" in msg
    assert "doctor" in msg
    # فهرست قدیم این دوره نیست — پیام نباید مسیر مرده را پیشنهاد کند
    assert "treasure-hunt" not in msg


def test_empty_listing_hint_distinguishes_captcha():
    """صفحهٔ کپچا باید متفاوت از پوستهٔ خالی گزارش شود."""
    from digiganj import digikala

    captcha = "<html><body>" + "x" * 5000 + " captcha form</body></html>"
    hint = digikala.DigikalaClient._empty_listing_hint(_FakeResp(captcha))
    assert "امنیتی" in hint
