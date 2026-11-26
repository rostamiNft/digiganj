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
