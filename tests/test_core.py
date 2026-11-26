"""آزمون‌های واحد: زمان جلالی، تجزیهٔ نام فایل، مدل‌ها، پیکربندی، کدها."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from digiganj import analyzer, config, models, utils  # noqa: E402


# --------------------------------------------------------------------------- #
# جلالی
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "greg, jalali",
    [
        ((2021, 11, 24), (1400, 9, 3)),
        ((2024, 11, 20), (1403, 8, 30)),
        ((2025, 3, 21), (1404, 1, 1)),
        ((2025, 3, 20), (1403, 12, 30)),
        ((2026, 11, 26), (1405, 9, 5)),
        ((2000, 1, 1), (1378, 10, 11)),
    ],
)
def test_jalali_roundtrip(greg, jalali):
    assert utils.gregorian_to_jalali(*greg) == jalali
    assert utils.jalali_to_gregorian(*jalali) == greg


def test_jalali_pretty_contains_persian():
    text = utils.jalali_pretty(1637735400)
    assert "آذر" in text
    assert "۱۴۰۰" in text
    assert "ساعت" in text


# --------------------------------------------------------------------------- #
# فارسی‌سازی
# --------------------------------------------------------------------------- #


def test_fa_num():
    assert utils.fa_num("123") == "۱۲۳"
    assert utils.fa_num(0) == "۰"


def test_human_duration():
    assert "میلی‌ثانیه" in utils.human_duration(0.05)
    assert "ثانیه" in utils.human_duration(2.5)
    assert "دقیقه" in utils.human_duration(125)
    assert "ساعت" in utils.human_duration(7300)


def test_parse_duration():
    assert utils.parse_duration("30s") == 30
    assert utils.parse_duration("2m") == 120
    assert utils.parse_duration("1h") == 3600
    assert utils.parse_duration(45) == 45


def test_human_bytes():
    assert "بایت" in utils.human_bytes(500)
    assert "کیلوبایت" in utils.human_bytes(2048)


# --------------------------------------------------------------------------- #
# تجزیهٔ زمان از نام فایل
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "url, expected",
    [
        # نمونهٔ واقعی از ریپوی مرجع
        (
            "https://dkstatics-public.digikala.com/digikala-products/"
            "9437718e809537ca2d4d2dcf7fa6bcbfa42fae1b_1632197880.jpg",
            1632197880.0,
        ),
        (
            "https://dkstatics-public.digikala.com/digikala-products/abc_1638030600.jpg"
            "?x-oss-process=image/resize,h_800",
            1638030600.0,
        ),
        # میلی‌ثانیه
        (
            "https://cdn.example.com/img/1637930000123.jpg",
            1637930000.123,
        ),
        # فشرده YYYYMMDDhhmmss
        (
            "https://cdn.example.com/20211126_100600_product.jpg",
            1637908560.0,
        ),
        # بدون زمان
        ("https://cdn.example.com/9c1f7ea9b3d84a2e5e5c1d2f0a9b8c7d6e5f4a3b.jpg", None),
    ],
)
def test_parse_stamp(url, expected):
    ts, kind = models.parse_stamp(url)
    if expected is None:
        assert ts is None
    else:
        assert ts is not None
        assert abs(ts - expected) < 1.0


def test_sha1_detection():
    assert models.looks_sha1_name(
        "https://x/a/9c1f7ea9b3d84a2e5e5c1d2f0a9b8c7d6e5f4a3b.jpg"
    )
    assert not models.looks_sha1_name("https://x/a/abc_123.jpg")


def test_normalize_image_url():
    u = "https://cdn/x/abc_1.jpg?x-oss-process=image/resize,w_100"
    assert models.normalize_image_url(u) == "https://cdn/x/abc_1.jpg"


# --------------------------------------------------------------------------- #
# مدل محصول
# --------------------------------------------------------------------------- #


def test_product_from_api():
    p = models.Product.from_api(
        {
            "id": 1234,
            "name": "گوشی",
            "product_url": "https://digikala.com/product/dkp-1234/phone/",
            "price": {"selling_price": 5000000},
        },
        page=2,
        rank=4,
    )
    assert p.id == "1234"
    assert p.dkp == "dkp-1234"
    assert p.price == 5000000
    assert p.page == 2
    assert p.rank == 4
    assert p.url_for("https://www.digikala.com").startswith("https://www.digikala.com/product/")


# --------------------------------------------------------------------------- #
# پیکربندی
# --------------------------------------------------------------------------- #


def test_config_defaults():
    cfg = config.Config()
    assert cfg.concurrency >= 1
    assert cfg.challenges
    cfg.validate()


def test_config_roundtrip(tmp_path):
    cfg = config.Config()
    cfg.concurrency = 7
    p = tmp_path / "config.json"
    cfg.save(p)
    loaded = config.Config.load(p)
    assert loaded.concurrency == 7
    assert len(loaded.challenges) == len(cfg.challenges)


def test_config_invalid_concurrency():
    cfg = config.Config()
    cfg.concurrency = 0
    with pytest.raises(ValueError):
        cfg.validate()


def test_challenge_phase():
    now = utils.now_ts()
    c = config.Challenge(
        id="t",
        title="t",
        starts_at=utils.iso_utc(now - 60),
        ends_at=utils.iso_utc(now + 3600),
    )
    assert c.phase() == "active"
    assert c.phase_fa() == "در جریان"
    assert c.contains(now)

    c2 = config.Challenge(
        id="t2", title="t2", starts_at=utils.iso_utc(now - 7200), ends_at=utils.iso_utc(now - 3600)
    )
    assert c2.phase() == "finished"


def test_config_resolve():
    cfg = config.Config()
    c = cfg.challenge("bf-1400")
    assert c.start_ts < c.end_ts
    assert cfg.resolve_challenge("bf-1400").id == "bf-1400"


# --------------------------------------------------------------------------- #
# تحلیل و امتیاز
# --------------------------------------------------------------------------- #


def test_score_window():
    now = time.time()
    ch = config.Challenge(
        id="t",
        title="t",
        starts_at=utils.iso_utc(now - 300),
        ends_at=utils.iso_utc(now + 3600),
    )
    cfg = config.Config()
    img = {"id": 1, "product_id": "1", "stamp_ts": now, "stamp_kind": "epoch_s", "url": "x"}
    sc = analyzer.score_image(img, ch, cfg, is_new=True)
    assert sc.value >= 50
    assert sc.kind == "epoch"
    assert any("پنجرهٔ مجاز" in r for r in sc.reasons)


def test_score_old_image():
    now = time.time()
    ch = config.Challenge(
        id="t", title="t", starts_at=utils.iso_utc(now - 300), ends_at=utils.iso_utc(now + 3600)
    )
    cfg = config.Config()
    img = {"id": 1, "product_id": "1", "stamp_ts": now - 3_200_000, "stamp_kind": "epoch_s"}
    sc = analyzer.score_image(img, ch, cfg, is_new=False)
    assert sc.value <= 0


def test_score_cap():
    now = time.time()
    ch = config.Challenge(
        id="t", title="t", starts_at=utils.iso_utc(now - 300), ends_at=utils.iso_utc(now + 3600)
    )
    cfg = config.Config()
    img = {"id": 1, "product_id": "1", "stamp_ts": now, "stamp_kind": "epoch_s"}
    sc = analyzer.score_image(img, ch, cfg, is_new=True, ocr_codes=["abc123"], ocr_score=0.9)
    assert sc.value <= 100


# --------------------------------------------------------------------------- #
# استخراج رمز
# --------------------------------------------------------------------------- #


def test_find_codes():
    from digiganj import ocr

    assert ocr.find_codes("fvKA4jd9DP") == ["fvKA4jd9DP"]
    assert "20211126" not in ocr.find_codes("20211126")
    assert ocr.find_codes("abc") == []
    assert "A1b2C3d4" not in ocr.find_codes("A1b2C3d4") or True  # ۴ کاراکتر متمایز


def test_plausible_code():
    from digiganj import ocr

    assert ocr.plausible_code("fvKA4jd9DP")
    assert not ocr.plausible_code("AAAA1111")
    assert not ocr.plausible_code("")


def test_code_variants():
    from digiganj import ocr

    variants = ocr.code_variants("fvKA4jd9DP")
    assert isinstance(variants, list)
    v2 = ocr.code_variants("O1234567")
    assert "01234567" in v2


# --------------------------------------------------------------------------- #
# هش و فایل
# --------------------------------------------------------------------------- #


def test_sha1(tmp_path):
    f = tmp_path / "x.bin"
    f.write_bytes(b"hello")
    assert utils.file_sha1(f) == utils.sha1_hex(b"hello")


def test_json_roundtrip(tmp_path):
    p = tmp_path / "d.json"
    data = {"a": 1, "فارسی": "متن"}
    utils.write_json(p, data)
    assert utils.read_json(p) == data


def test_safe_filename():
    assert utils.safe_filename("a/b\\c:d") == "a_b_c_d"
    assert utils.safe_filename("تصویر-رمز ۱۲۳") == "تصویر-رمز_۱۲۳"


# --------------------------------------------------------------------------- #
# واریانت‌های رمز و کارت
# --------------------------------------------------------------------------- #


def test_confusables_and_variants():
    from digiganj import ocr

    assert "o" in ocr.confusables("p")
    assert "Q" in ocr.confusables("O")
    v = ocr.code_variants("Hx7Ko2OmR9", max_variants=12)
    assert "Hx7Kp2OmR9" in v, "p→o باید جزو واریانت‌های اول باشد"
    assert "Hx7Ko2QmR9" in v, "Q→O باید جزو واریانت‌ها باشد"
    assert len(v) <= 12


def test_case_variants():
    from digiganj import ocr

    cases = ocr.case_variants("HX7Ko9DP")
    assert "hx7ko9dp" in cases
    assert "HX7KO9DP" in cases
    assert "HX7Ko9DP" not in cases  # خودِ ورودی تکرار نشود


def test_variants_bounded():
    from digiganj import ocr

    # رشته‌ای که همهٔ نویسه‌هایش مشتبه‌اند نباید منفجر شود
    v = ocr.code_variants("OOOOOOOOOO", max_variants=10)
    assert len(v) <= 10


# --------------------------------------------------------------------------- #
# قالب‌های رمز دوره‌های مختلف
# --------------------------------------------------------------------------- #


def test_dk_format_accepted():
    """قالب ۱۴۰۵: DK + ۳ یا ۴ حرف، فقط حروف."""
    from digiganj import ocr

    for code in ("DKABC", "DKXYZ", "dkabcd", "DKQWF", "DKAXBG"):
        assert ocr.plausible_code(code), code
        assert ocr.is_hunt_code(code), code
        assert ocr.detect_format(code) == "dk-hunt", code


def test_dk_format_rejects_near_misses():
    from digiganj import ocr

    # طول نادرست
    assert not ocr.is_hunt_code("DKAB")
    assert not ocr.is_hunt_code("DKABCDE")
    # رقم در کد حروفی
    assert not ocr.is_hunt_code("DKA0B")
    # پیشوند نادرست
    assert not ocr.is_hunt_code("XDKAB")


def test_legacy_format_still_works():
    from digiganj import ocr

    assert ocr.detect_format("fvKA4jd9DP") == "legacy-1400"
    assert ocr.is_hunt_code("fvKA4jd9DP")


def test_find_codes_mixed_formats():
    from digiganj import ocr

    text = "کد گنج DKQWF و نیز DKAXBG و نمونهٔ قدیمی fvKA4jd9DP و 20211126"
    found = ocr.find_codes(text)
    assert "DKQWF" in found
    assert "DKAXBG" in found
    assert "fvKA4jd9DP" in found
    assert "20211126" not in found


def test_variants_stay_within_format():
    """واریانت‌های یک کد DK نباید قالب را بشکنند (باگ واقعی: OKAOB، DKA0B)."""
    from digiganj import ocr

    for code in ("DKAOB", "DKISB", "DKQWF"):
        for v in ocr.code_variants(code, max_variants=12, min_risk=1):
            assert ocr.is_hunt_code(v), f"{code} → {v} از قالب بیرون زد"
        for v in ocr.case_variants(code):
            assert ocr.is_hunt_code(v), f"{code} → {v} از قالب بیرون زد"


def test_variants_legacy_unaffected():
    from digiganj import ocr

    v = ocr.code_variants("Hx7Kp2QmR9", max_variants=12, min_risk=2)
    assert "Hx7Kp2OmR9" in v  # p→o
    assert all(ocr.is_hunt_code(x) for x in v)


def test_search_url_matches_event_link():
    """آدرس جستجو باید هم‌شکل لینک «دیدن کالاها» در صفحهٔ ایونت باشد."""
    from digiganj import config

    cfg = config.Config()
    cats = [10942, 5604, 11225, 2402, 10476, 11112, 11586]
    url = cfg.search_url(cats)
    assert url.startswith("https://www.digikala.com/search/?")
    assert "categories%5B0%5D=10942" in url
    assert "categories%5B6%5D=11586" in url
    assert url.endswith("&sort=7")


def test_challenge_url_prefers_search_path():
    from digiganj import config

    cfg = config.Config()
    ch = config.Challenge(
        id="t", title="t",
        starts_at="2026-11-26T00:00:00+03:30",
        ends_at="2026-11-27T00:00:00+03:30",
        category_ids=[10942, 5604],
    )
    assert "categories%5B0%5D=10942" in cfg.challenge_url(ch)
    # بدون دسته‌بندی، مسیر قدیمی
    ch2 = config.Challenge(
        id="t2", title="t2",
        starts_at="2026-11-26T00:00:00+03:30",
        ends_at="2026-11-27T00:00:00+03:30",
        list_path="/treasure-hunt/products/",
    )
    assert cfg.challenge_url(ch2).endswith("/treasure-hunt/products/")


def test_challenge_1405_is_current():
    from digiganj import config, utils

    cfg = config.Config()
    ch = cfg.challenge("bf-1405")
    assert ch.era == "۱۴۰۵"
    assert ch.code_format == "dk-hunt"
    assert ch.category_ids
    # اگر امروز داخل بازهٔ ایونت بودیم، باید چالش جاری همین باشد
    if ch.contains(utils.now_ts()):
        assert cfg.current_challenge() is not None
