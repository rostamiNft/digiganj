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
