"""گزارش HTML (فارسی/راست‌به‌چپ) و بستهٔ شواهد.

گزارش برای «نشان دادن» است، نه فقط برای بایگانی:

* هر نامزد با تصویر بندانگشتی، زمان جلالی، فاصله از شروع چالش و امتیاز.
* هم‌زمان، زمان دقیق اولین مشاهدهٔ ما و زمان نام فایل روی CDN جدا نمایش
  داده می‌شود تا «شاهد» از «ادعا» تفکیک شود.
* یک جدول «گاه‌شمار» از انتشار تصاویر (بر اساس زمان نام فایل) تا الگوی
  انتشار — پیش/پس از شروع رسمی — دیده شود.
* خروجی تک‌فایلی و خودبسنده (بدون نیاز به اینترنت برای باز شدن)؛ تصاویر
  به‌صورت اختیاری جاسازی می‌شوند.
"""

from __future__ import annotations

import base64
import html
from pathlib import Path
from typing import Any

from . import analyzer, utils

CSS = """
:root {
  --bg: #0f1117; --card: #161a23; --line: #242a37; --txt: #e7eaf0; --dim: #8d96a8;
  --accent: #ffd400; --good: #37d07a; --bad: #ff5c70; --warn: #ffab40;
}
* { box-sizing: border-box; }
body {
  margin: 0; padding: 28px 20px 60px; background: var(--bg); color: var(--txt);
  font-family: "Vazirmatn", "IRANSans", "Segoe UI", Tahoma, sans-serif; direction: rtl;
}
.wrap { max-width: 1120px; margin: 0 auto; }
h1 { font-size: 26px; margin: 0 0 6px; }
h2 { font-size: 19px; margin: 34px 0 12px; color: var(--accent); }
h3 { font-size: 16px; margin: 20px 0 8px; }
p, li { line-height: 1.9; color: #cdd3df; }
.sub { color: var(--dim); margin: 0 0 20px; font-size: 14px; }
.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(240px, 1fr)); gap: 14px; }
.card { background: var(--card); border: 1px solid var(--line); border-radius: 14px; overflow: hidden; }
.card img { width: 100%; height: 166px; object-fit: cover; background: #0b0d12; display: block; }
.card .body { padding: 12px 14px; }
.card .name { font-size: 13px; font-weight: 700; margin: 0 0 4px; }
.card .meta { font-size: 12px; color: var(--dim); margin: 2px 0; }
.badge { display: inline-block; padding: 2px 8px; border-radius: 999px; font-size: 11px;
  background: rgba(255,212,0,.14); color: var(--accent); margin-inline-start: 4px; }
.badge.bad { background: rgba(255,92,112,.15); color: var(--bad); }
.badge.good { background: rgba(55,208,122,.15); color: var(--good); }
table { width: 100%; border-collapse: collapse; background: var(--card);
  border: 1px solid var(--line); border-radius: 12px; overflow: hidden; }
th, td { padding: 9px 12px; text-align: right; font-size: 13px; border-bottom: 1px solid var(--line); }
th { color: var(--dim); font-weight: 600; font-size: 12px; }
tr:last-child td { border-bottom: 0; }
code { font-family: "JetBrains Mono", "Fira Code", Consolas, monospace; font-size: 13px;
  background: #0b0d12; padding: 2px 6px; border-radius: 6px; direction: ltr; display: inline-block; }
.hist { display: flex; flex-direction: column; gap: 6px; margin: 10px 0 6px; }
.hrow { display: flex; align-items: center; gap: 10px; direction: ltr; }
.hlab { width: 54px; text-align: right; flex: none; }
.hbar { height: 14px; border-radius: 7px; background: linear-gradient(90deg,#ffd400,#ff9f1c); display: block; }
.hval { color: #8d96a8; font-size: 13px; }
.kpi { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 10px; }
.kpi .box { background: var(--card); border: 1px solid var(--line); border-radius: 12px; padding: 14px; }
.kpi .v { font-size: 22px; font-weight: 800; }
.kpi .l { font-size: 12px; color: var(--dim); }
.note { border-right: 3px solid var(--accent); background: rgba(255,212,0,.06);
  padding: 12px 16px; border-radius: 10px; margin: 16px 0; }
.note.warn { border-color: var(--warn); background: rgba(255,171,64,.08); }
.small { font-size: 12px; color: var(--dim); }
a { color: #7cc4ff; text-decoration: none; }
footer { margin-top: 40px; color: var(--dim); font-size: 12px; }
"""


def _esc(x: Any) -> str:
    return html.escape(str(x if x is not None else ""))


class _ChallengeView:
    """نمای یکنواخت برای چالش (از JSON یا شیء Challenge)."""

    def __init__(self, data: dict[str, Any]):
        self._d = data
        self.id = data.get("id", "")
        self.title = data.get("title", "")
        start = data.get("start_ts")
        end = data.get("end_ts")
        if start is None and data.get("starts_at"):
            start = utils.dt_to_ts(utils.parse_dt(str(data["starts_at"])))
        if end is None and data.get("ends_at"):
            end = utils.dt_to_ts(utils.parse_dt(str(data["ends_at"])))
        self.start_ts = float(start or 0)
        self.end_ts = float(end or 0)

    def phase(self) -> str:
        now = utils.now_ts()
        if now < self.start_ts:
            return "upcoming"
        if now > self.end_ts:
            return "finished"
        return "active"

    def to_dict(self) -> dict[str, Any]:
        return dict(self._d)


def _embed_image(path: str | Path, max_px: int = 420) -> str:
    """تصویر را به data-URI تبدیل می‌کند (اختیاری، نسخهٔ خودبسنده)."""
    try:
        from PIL import Image
    except Exception:
        return ""
    try:
        p = Path(path)
        if not p.exists():
            return ""
        img = Image.open(p)
        img.thumbnail((max_px, max_px))
        import io

        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=78)
        mime = "image/jpeg"
        data = base64.b64encode(buf.getvalue()).decode()
        return f"data:{mime};base64,{data}"
    except Exception:
        return ""


def _fmt_ts(ts: Any) -> str:
    if not ts:
        return "—"
    try:
        return utils.jalali_pretty(float(ts))
    except Exception:
        return "—"


def _cluster_histogram(candidates: list[dict[str, Any]], challenge, cfg,
                       hours: int = 6) -> dict[str, int]:
    """توزیع ساعتی زمان نام فایل تصاویر (نسبت به شروع چالش).

    تصاویر بسیار قدیمی (بیش از ۴ روز فاصله) حذف می‌شوند تا مقیاس نمودار
    معنادار بماند؛ پنجرهٔ نمایش حول شروع چالش است.
    """
    hist: dict[str, int] = {}
    for c in candidates:
        ts = c.get("stamp_ts")
        if not ts:
            continue
        delta = float(ts) - challenge.start_ts
        if abs(delta) > hours * 3600 * 4:
            continue
        bucket = int(delta // 3600)
        label = f"{bucket:+d}"
        hist[label] = hist.get(label, 0) + 1
    if not hist:
        return {}
    items = sorted(hist.items(), key=lambda kv: int(kv[0]))
    # اگر سطل‌ها پراکنده‌اند، فقط بازه‌های اطراف شروع و انتها + پرتکرارترین‌ها
    if len(items) > 14:
        keys = {k for k, _ in items[:4]} | {k for k, _ in items[-4:]}
        keys |= {k for k, _ in sorted(items, key=lambda kv: -kv[1])[:8]}
        items = [(k, v) for k, v in items if k in keys]
    return dict(items)


def _thumb_or_placeholder(c: dict[str, Any], cfg, embed: bool) -> str:
    path = c.get("file_path", "")
    if embed and path:
        data = _embed_image(path, cfg.report_thumb_px)
        if data:
            return f'<img src="{data}" alt="">'
        return '<img src="" alt="">'
    url = analyzer.image_url_thumb(c.get("url", ""), cfg.report_thumb_px)
    if not url:
        return ""
    return f'<img loading="lazy" src="{_esc(url)}" alt="">'


def render_html(data: dict[str, Any], cfg, embed: bool | None = None) -> str:
    """تولید HTML گزارش.

    ``data`` کلیدهای: challenge, generated_at, candidates, alert_rows, codes,
    products_count, scans, stats, alerts, evidence_notes (اختیاری).
    """
    emb = cfg.report_embed_images if embed is None else embed
    ch = data.get("challenge")
    # چالش می‌تواند شیء Challenge یا دیکشنری (از JSON) باشد؛ برای فیلدهای
    # مشترک، نمای یکنواخت می‌سازیم.
    if isinstance(ch, dict):
        ch = _ChallengeView(ch)
    generated = data.get("generated_at") or utils.now_ts()
    cands = data.get("candidates") or []
    alerts = data.get("alerts") or []
    codes = data.get("codes") or []
    stats = data.get("stats") or {}
    scans = data.get("scans") or {}
    prod_count = data.get("products_count") or len({c.get("product_id") for c in cands})
    notes = data.get("evidence_notes") or []
    hist = _cluster_histogram(cands, ch, cfg) if ch else {}

    title = f"گزارش شکار گنج — {ch.title}" if ch else "گزارش شکار گنج"
    phase = ch.phase() if ch else "finished"
    phase_fa = {"upcoming": "پیش‌رو", "active": "در جریان", "finished": "پایان‌یافته"}.get(phase, "—")
    sep = " &nbsp;|&nbsp; "
    meta_bits = [
        f"چالش: {ch.title}" if ch else "",
        f"وضعیت: {phase_fa}",
        f"بازه: {_fmt_ts(ch.start_ts) if ch else '—'} تا {_fmt_ts(ch.end_ts) if ch else '—'}",
        f"تولید گزارش: {_fmt_ts(generated)}",
    ]

    parts: list[str] = []
    parts.append(f"""<!doctype html>
<html lang="fa" dir="rtl"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{_esc(title)}</title>
<style>{CSS}</style></head>
<body><div class="wrap">
<h1>🎁 {_esc(title)}</h1>
<p class="sub">{sep.join(_esc(b) for b in meta_bits if b)}</p>
""")

    # هشدار محتوا
    parts.append("""<div class="note warn">
<strong>قبل از استناد بخوانید:</strong> این گزارش شواهد «مشاهده»ٔ ابزار است، نه ادعای
اثبات‌شده. زمان نام فایل روی CDN (اگر وجود داشته باشد) قابل‌اتکاترین نشانهٔ زمان
انتشار است؛ «اولین مشاهده» زمان دیدن ابزار شماست و ممکن است دیرتر باشد.
برای ادعای عمومی، گزارش خام و زمان‌های خود را حفظ کنید.
</div>""")

    # KPI
    parts.append(f"""<div class="kpi">
  <div class="box"><div class="v">{utils.fa_num(prod_count)}</div><div class="l">محصول بررسی‌شده</div></div>
  <div class="box"><div class="v">{utils.fa_num(stats.get('images', 0))}</div><div class="l">تصویر ثبت‌شده</div></div>
  <div class="box"><div class="v">{utils.fa_num(len(cands))}</div><div class="l">نامزد نشانه‌دار</div></div>
  <div class="box"><div class="v">{utils.fa_num(len(alerts))}</div><div class="l">هشدار ثبت‌شده</div></div>
  <div class="box"><div class="v">{utils.fa_num(len(codes))}</div><div class="l">رمز استخراج‌شده</div></div>
  <div class="box"><div class="v">{utils.fa_num(len(scans.get('cycles', []) if isinstance(scans, dict) else []))}</div><div class="l">دورهٔ اسکن</div></div>
</div>""")

    # تصاویر نامزد
    meaningful = [c for c in cands if float(c.get("score") or 0) > 0]
    parts.append("<h2>نامزدهای نشانه‌دار</h2>")
    if meaningful:
        parts.append('<div class="grid">')
        for c in meaningful[: cfg.report_top_alerts]:
            name = _esc(c.get("product_name") or f"dkp-{c.get('product_id')}")
            badge_score = int(round(float(c.get("score", 0))))
            stamp = _fmt_ts(c.get("stamp_ts"))
            delta = ""
            if ch and c.get("stamp_ts"):
                d = float(c["stamp_ts"]) - ch.start_ts
                if d >= 0:
                    delta = utils.human_duration(d) + " پس از شروع"
                else:
                    delta = utils.human_duration(abs(d)) + " پیش از شروع"
            codes_txt = c.get("ocr_codes") or ""
            prod_url = f"https://www.digikala.com/product/dkp-{c.get('product_id')}/"
            img = _thumb_or_placeholder(c, cfg, emb)
            parts.append(f"""<div class="card">
  {img}
  <div class="body">
    <p class="name">{name}<span class="badge">{utils.fa_num(badge_score)} امتیاز</span></p>
    <p class="meta">زمان فایل: {_esc(stamp)}</p>
    <p class="meta">{_esc(delta) if delta else '—'}</p>
    {'<p class="meta">رمزهای خوانده‌شده: <code>' + _esc(codes_txt) + '</code></p>' if codes_txt else ''}
    <p class="meta"><a href="{_esc(prod_url)}">صفحهٔ محصول</a>
       | <a href="{_esc(analyzer.image_url_large(c.get('url','')))}">تصویر کامل</a></p>
  </div>
</div>""")
        parts.append("</div>")
        hidden = len(cands) - len(meaningful)
        if hidden > 0:
            parts.append(
                f'<p class="small">+ {utils.fa_num(hidden)} تصویر دیگر با امتیاز صفر/منفی '
                "(تصاویر قدیمی) در پایگاه‌داده نگه‌داری می‌شوند و در این‌جا نمایش داده نشده‌اند.</p>"
            )
    else:
        parts.append("<p>هنوز هیچ نامزدی با امتیاز مثبت پیدا نشده است. یک چرخهٔ اسکن اجرا کنید یا آستانه را در تنظیمات پایین آورید.</p>")

    # گاه‌شمار ساعتی
    if hist:
        parts.append("<h2>گاه‌شمار انتشار (بر پایهٔ زمان نام فایل)</h2>")
        parts.append("<table><tr><th>ساعت نسبت به شروع چالش</th><th>تعداد تصویر</th></tr>")
        for k, v in hist.items():
            parts.append(f"<tr><td><code>{_esc(k)}h</code></td><td>{utils.fa_num(v)}</td></tr>")
        parts.append("</table>")
        mx = max(hist.values()) or 1
        bars = []
        for k, v in hist.items():
            pct = max(4, round(100 * v / mx))
            bars.append(
                f'<div class="hrow"><span class="hlab"><code>{_esc(k)}h</code></span>'
                f'<span class="hbar" style="width:{pct}%"></span>'
                f'<span class="hval">{utils.fa_num(v)}</span></div>'
            )
        parts.append('<div class="hist">' + "".join(bars) + "</div>")
        parts.append("<p class='small'>ساعت ۰ همان لحظهٔ شروع رسمی است؛ اعداد مثبت پس از شروع، منفی پیش از آن.</p>")

    # جدول هشدارها
    if alerts:
        parts.append("<h2>جدول هشدارها</h2>")
        parts.append("<table><tr><th>زمان (جلالی)</th><th>محصول</th><th>امتیاز</th><th>دلیل</th><th>رمز</th></tr>")
        for a in alerts[: cfg.report_top_alerts]:
            parts.append(
                "<tr>"
                f"<td>{_esc(_fmt_ts(a.get('ts')))}</td>"
                f"<td>dkp-{_esc(a.get('product_id'))}</td>"
                f"<td>{utils.fa_num(round(float(a.get('score') or 0)))}</td>"
                f"<td>{_esc(utils.shorten(a.get('reason', ''), 90))}</td>"
                f"<td>{('<code>' + _esc(a.get('codes')) + '</code>') if a.get('codes') else '—'}</td>"
                "</tr>"
            )
        parts.append("</table>")

    # رمزها
    if codes:
        parts.append("<h2>رمزهای استخراج‌شده</h2>")
        parts.append("<table><tr><th>رمز</th><th>محصول</th><th>منبع</th><th>وضعیت</th><th>زمان</th></tr>")
        for c in codes:
            parts.append(
                "<tr>"
                f"<td><code>{_esc(c.get('code'))}</code></td>"
                f"<td>dkp-{_esc(c.get('product_id'))}</td>"
                f"<td>{_esc(c.get('source'))}</td>"
                f"<td>{_esc(c.get('status'))}</td>"
                f"<td>{_esc(_fmt_ts(c.get('ts')))}</td>"
                "</tr>"
            )
        parts.append("</table>")

    # روش و محدودیت‌ها
    parts.append("""<h2>روش و محدودیت‌ها</h2>
<div class="note">
<strong>روش:</strong> ابزار فهرست محصولات چالش را می‌خواند، گالری هر محصول را با
API عمومی می‌گیرد و نام فایل هر تصویر را تحلیل می‌کند. تصویرهایی که زمانشان در
پنجرهٔ چالش است یا تازه‌اند، رتبه می‌گیرند. برای چند نامزد برتر، در صورت فعال
بودن، OCR اجرا و متن شبیه رمز استخراج می‌شود. هرگز کد تخفیف به‌صورت خودکار روی
حساب کاربری ثبت نمی‌شود.
</div>
<div class="note warn">
<strong>محدودیت‌ها:</strong> (۱) زمان نام فایل ممکن است توسط تیم فنی پاک/جعل شود؛
(۲) عدم مشاهدهٔ نشانه، اثبات تقلب نیست؛ (۳) صحت رمز فقط با آزمون در فرم رسمی
(در بازهٔ مجاز خودتان) معلوم می‌شود؛ (۴) ابزار به سرعت، محدودیت نرخ را رعایت
می‌کند و در صورت فشار، کندتر می‌شود.
</div>""")
    if notes:
        parts.append("<h2>یادداشت‌های شاهد</h2><ul>")
        for n in notes:
            parts.append(f"<li>{_esc(n)}</li>")
        parts.append("</ul>")

    parts.append("""<footer>
digiganj — گزارش خودکار. این فایل شامل دادهٔ خام نیست؛ برای بایگانی، همراه
«data/digiganj.sqlite» و فایل JSON خروجی نگه‌داری کنید.
</footer></div></body></html>""")
    return "\n".join(parts)


def write_report(out_path: str | Path, data: dict[str, Any], cfg,
                 embed: bool | None = None) -> Path:
    text = render_html(data, cfg, embed=embed)
    p = Path(out_path)
    utils.atomic_write(p, text.encode("utf-8"))
    return p


def write_report_bundle(out_dir: str | Path, data: dict[str, Any], cfg) -> dict[str, Path]:
    """گزارش HTML + JSON + تصاویر جاسازی‌نشده (نسخهٔ قابل بایگانی)."""
    base = Path(out_dir)
    utils.ensure_dir(base)
    stamp = utils.filename_stamp()
    html_path = base / f"report-{stamp}.html"
    json_path = base / f"report-{stamp}.json"
    write_report(html_path, data, cfg)
    utils.write_json(json_path, data)
    return {"html": html_path, "json": json_path}


def write_evidence_pack(out_dir: str | Path, store, cfg, challenge) -> Path:
    """بستهٔ شواهد: JSON خام پایگاه‌داده + نمایهٔ تصاویر + گزارش HTML.

    خروجی یک پوشهٔ زمان‌دار است تا شواهد هر دوره جدا بماند.
    """
    stamp = utils.filename_stamp()
    base = Path(out_dir) / f"evidence-{stamp}"
    utils.ensure_dir(base)
    store.export_json(base / "raw.json")
    stats = store.stats()
    cands = store.candidates(min_score=0, limit=2000)
    alerts = store.alerts(limit=500)
    codes = store.codes(limit=500)
    cycles = store.cycles(limit=100)
    payload = {
        "challenge": challenge.to_dict() if challenge else None,
        "generated_at": utils.now_ts(),
        "stats": stats,
        "candidates": cands,
        "alerts": alerts,
        "codes": codes,
        "scans": {"cycles": cycles},
        "products_count": stats.get("products", 0),
    }
    write_report(base / "report.html", payload, cfg)
    utils.write_json(base / "report.json", payload)
    return base
