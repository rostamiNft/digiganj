"""داشبورد زندهٔ اسکن: صفحهٔ HTML خودبسنده که وضعیت پایگاه‌داده را نشان می‌دهد.

طرح
----
دو حالت دارد:

1. **حالت فایل (پیش‌فرض)**: یک HTML تک‌فایلی که وضعیت را در لحظهٔ ساخت
   «عکس می‌گیرد». مناسب بایگانی یا فرستادن به دیگران.
2. **حالت زنده**: همان صفحه با یک نقطهٔ پایانی JSON (``--serve``) هر چند
   ثانیه وضعیت تازه می‌گیرد و بدون بارگذاری مجدد به‌روز می‌شود.

هیچ CDN یا فونت بیرونی استفاده نمی‌شود؛ همه‌چیز درون فایل است تا در شبکهٔ
محدود هم باز شود. رنگ و چیدمان از همان پوستهٔ گزارش پیروی می‌کند.
"""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from . import db, utils
from .console import Console

# --------------------------------------------------------------------------- #
# جمع‌آوری داده
# --------------------------------------------------------------------------- #


def collect_state(store: db.Store, cfg=None, challenge=None, limit: int = 12) -> dict[str, Any]:
    """عکس فوری از وضعیت برای داشبورد (همه‌چیز JSON-پذیر)."""
    now = utils.now_ts()
    ch = challenge
    if ch is None and cfg is not None:
        cid = store.get_kv("last_challenge", "")
        if cid:
            try:
                ch = cfg.challenge(cid)
            except KeyError:
                # چالشی که در پیکربندی نیست (مثل دمو): از متادیتای پایگاه‌داده
                raw = store.get_kv("last_challenge_meta", "")
                try:
                    meta = json.loads(raw) if raw else {}
                except Exception:
                    meta = {}
                if isinstance(meta, dict) and meta.get("start_ts"):
                    from .config import Challenge

                    ch = Challenge(
                        id=cid,
                        title=meta.get("title") or cid,
                        starts_at=utils.iso_utc(float(meta["start_ts"])),
                        ends_at=utils.iso_utc(
                            float(meta.get("end_ts") or meta["start_ts"] + 86400)
                        ),
                        list_path=meta.get("list_path") or "/",
                    )
        if ch is None:
            ch = cfg.current_challenge() or (
                cfg.challenges[0] if cfg.challenges else None
            )

    stats = store.stats()
    cycles = store.cycles(limit=8)
    alerts = store.alerts(limit=limit)
    cands = store.candidates(limit=40)
    # همهٔ واریانت‌ها برای خوشه‌بندی دقیق خوانده می‌شوند، سپس نمایش محدود می‌شود
    codes_raw = store.codes(limit=max(400, limit * 20))

    # «خانوادهٔ رمز» با شمارش آرا (همهٔ گونه‌ها، نه یک رمز قطعی)
    groups = _cluster_readings(codes_raw, limit)

    start = ch.start_ts if ch else 0.0
    end = ch.end_ts if ch else 0.0
    phase = ch.phase() if ch else "finished"
    elapsed = max(0.0, min(now, end) - start) if start else 0.0
    remain = max(0.0, start - now) if now < start else max(0.0, end - now)

    return {
        "generated_at": now,
        "challenge": {
            "id": getattr(ch, "id", ""),
            "title": getattr(ch, "title", ""),
            "start_ts": start,
            "end_ts": end,
            "phase": phase,
            "phase_fa": {
                "upcoming": "پیش از شروع",
                "active": "در جریان",
                "finished": "پایان‌یافته",
            }.get(phase, "—"),
            "elapsed": elapsed,
            "remain": remain,
            "progress": (100.0 * (now - start) / (end - start)) if end > start else 0.0,
        },
        "stats": stats,
        "cycles": cycles,
        "alerts": alerts,
        "candidates": cands[:limit],
        "codes": groups[:limit],
        "top_score": max([float(a.get("score") or 0) for a in alerts], default=0.0),
        "updated_fa": utils.jalali_pretty(now),
    }


# نگاشت به «شکل متعارف»: نویسه‌هایی که در شبکهٔ کوچک یکسان دیده می‌شوند
# به یک نماینده تبدیل می‌شوند تا همهٔ خوانش‌های یک رمز در یک خوشه بمانند.
# فقط جفت‌هایی که واقعاً یکسان دیده می‌شوند (نه هر چیز گرد):
# O↔0 و l↔1 و I↔1 — بقیه (Q/D/G) عمداً جدا می‌مانند تا خانواده‌ها قاطی نشوند.
_CANON = str.maketrans({
    "O": "0", "D": "0",
    "l": "1", "I": "1", "|": "1",
})


def canonical_code(code: str) -> str:
    """شکل متعارف برای مقایسهٔ خوانش‌ها (فقط برای خوشه‌بندی، نه نمایش)."""
    return code.translate(_CANON).lower()


def _code_rank(code: str) -> tuple:
    """کیفیت شکل کد: طول ۱۰، شروع دو حرف کوچک، دنبالهٔ l/I مشکوک."""
    score = 0
    if len(code) == 10:
        score += 3
    elif 9 <= len(code) <= 11:
        score += 1
    # الگوی واقعی دیده‌شده: «fvKA4jd9DP» — دو حرف کوچک، سپس مخلوط بزرگ/رقم
    if len(code) >= 2 and code[:2].islower() and code[2:3].isupper():
        score += 3
    elif len(code) >= 2 and code[:2].islower():
        score += 1
    # سه نویسهٔ کوچک پیاپی در ابتدا الگوی رمز ۱۴۰۰ نیست (اغلب خطای OCR)
    if len(code) >= 3 and code[:3].islower():
        score -= 3
    if any(ch.isupper() for ch in code[2:]) and any(ch.isdigit() for ch in code[2:]):
        score += 2
    if code.endswith(("l", "I", "1")):
        score -= 2
    if code.count("O") + code.count("0") > 1:
        score -= 1
    return (-score, -len(code), code)


def _cluster_readings(codes_raw: list[dict], limit: int) -> list[dict[str, Any]]:
    """خوشه‌بندی خوانش‌ها + شمارش آرا برای هر گونه."""
    groups: list[dict[str, Any]] = []
    for c in sorted(codes_raw, key=lambda x: _rank_key(x)):
        canon = canonical_code(c["code"])
        for g in groups:
            if (g["canon"] == canon
                    or _close(g["code"], c["code"])
                    or _prefix_overlap(g["code"], c["code"])):
                g["variants"] += 1
                if c.get("source") == "ocr":
                    g["raw"] += 1
                g["votes"][c["code"]] = g["votes"].get(c["code"], 0) + 1
                if c["ts"] > g["ts"]:
                    g["ts"] = c["ts"]
                break
        else:
            groups.append({
                "code": c["code"], "canon": canon, "product_id": c["product_id"],
                "ts": c["ts"], "variants": 1, "source": c.get("source", ""),
                "raw": 1 if c.get("source") == "ocr" else 0,
                "votes": {c["code"]: 1},
            })
    # سرِ خوشه: خوانش خام موتور (نه گونهٔ حدسی)، سپس کیفیت شکل
    raw_codes = {
        c["code"] for c in codes_raw
        if c.get("source") == "ocr" and canonical_code(c["code"]) in {g["canon"] for g in groups}
    }
    for g in groups:
        ranked = sorted(
            g["votes"].items(),
            key=lambda kv: (0 if kv[0] in raw_codes else 1,) + _code_rank(kv[0]),
        )
        g["code"] = ranked[0][0]
        g["votes_total"] = ranked[0][1]
        g["readings"] = [{"code": k, "votes": v} for k, v in ranked[:6]]
    groups.sort(key=lambda g: (0 if any(r["code"] in raw_codes for r in g["readings"]) else 1,)
                + _code_rank(g["code"]))
    return groups[:limit]


def _rank_key(c: dict) -> tuple:
    src_rank = 0 if c.get("source") == "ocr" else 1
    return (src_rank,) + _code_rank(c["code"])


def code_families(codes: list[dict]) -> list[dict]:
    """خوشه‌بندی خوانش‌ها در «خانوادهٔ یک رمز» و شمارش آرا.

    هر خانواده یک رمز است با چند گونهٔ خوانده‌شده؛ رأی هر گونه تعداد
    خوانش‌های مستقل آن است. مرتب‌سازی: بیشترین رأی، سپس کیفیت شکل.
    این کار جای «یک رمز قطعی» را می‌گیرد: ابزار **همهٔ گونه‌ها را با
    شمارش** نشان می‌دهد و کاربر ابتدا پرتکرارترین را می‌آزماید.
    """
    canon_map: dict[str, dict[str, Any]] = {}
    for c in codes:
        code = str(c.get("code") or "")
        if not code:
            continue
        key = canonical_code(code)
        slot = canon_map.setdefault(key, {"readings": {}, "raw": set()})
        slot["readings"][code] = slot["readings"].get(code, 0) + 1
        if c.get("source") == "ocr":
            slot["raw"].add(code)

    families: list[dict[str, Any]] = []
    for key, slot in canon_map.items():
        ranked = sorted(
            slot["readings"].items(),
            key=lambda kv: (-kv[1],) + _code_rank(kv[0]),
        )
        best, votes = ranked[0]
        families.append({
            "code": best,
            "canon": key,
            "votes": votes,
            "raw": bool(slot["raw"]),
            "readings": [{"code": c, "votes": v} for c, v in ranked],
            "variants": sum(slot["readings"].values()),
            "product_id": "",
            "source": "ocr" if "ocr" in slot["sources"] else "variant",
        })
    # محصول و زمان را از نخستین عضو پیدا کن
    for fam in families:
        for c in codes:
            if canonical_code(str(c.get("code") or "")) == fam["canon"]:
                fam["product_id"] = c.get("product_id", "")
                fam["ts"] = c.get("ts", 0)
                break
    # خوانش خام موتور بر گونهٔ حدسی مقدم است، سپس کیفیت شکل
    families.sort(key=lambda f: (0 if f["raw"] else 1,) + _code_rank(f["code"]))
    return families


def _prefix_overlap(a: str, b: str) -> bool:
    """آیا یکی پیشوند/پسوند دیگری است (با تحمل یک نویسهٔ لبهٔ اضافه)؟"""
    if not a or not b:
        return False
    short, long = (a, b) if len(a) <= len(b) else (b, a)
    if len(long) - len(short) > 2:
        return False
    return long.startswith(short) or long.endswith(short)


def _close(a: str, b: str) -> bool:
    if a == b:
        return True
    if abs(len(a) - len(b)) > 2:
        return False
    # فاصلهٔ لِوِنشتاین ساده برای واریانت‌های OCR
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1] <= 2


# --------------------------------------------------------------------------- #
# قالب
# --------------------------------------------------------------------------- #

_TEMPLATE = """<!doctype html>
<html lang="fa" dir="rtl"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>داشبورد شکار گنج</title>
<style>
:root {
  --bg:#0b0d12; --panel:#12151d; --panel2:#161a24; --line:#232936; --txt:#e8ecf4;
  --dim:#8b95a8; --accent:#ffd400; --good:#35d07f; --bad:#ff5c70; --warn:#ffab40;
  --blue:#4fa8ff;
}
* { box-sizing: border-box; }
body {
  margin:0; padding:22px 18px 60px; background:var(--bg); color:var(--txt);
  font-family:"Vazirmatn","IRANSans","Segoe UI",Tahoma,sans-serif;
}
.wrap { max-width:1180px; margin:0 auto; }
header { display:flex; align-items:center; justify-content:space-between; gap:14px;
         flex-wrap:wrap; margin-bottom:18px; }
h1 { font-size:23px; margin:0; }
h1 small { font-size:13px; color:var(--dim); font-weight:400; margin-right:8px; }
.pill { border:1px solid var(--line); background:var(--panel); border-radius:999px;
        padding:6px 13px; font-size:13px; color:var(--dim); }
.pill b { color:var(--txt); font-weight:600; }
.pill.up { color:var(--blue); border-color:#1e3a5c; }
.pill.on { color:var(--good); border-color:#17452f; }
.pill.off { color:var(--bad); border-color:#4a1f27; }
.grid4 { display:grid; grid-template-columns:repeat(auto-fit,minmax(155px,1fr)); gap:11px; }
.card { background:var(--panel); border:1px solid var(--line); border-radius:14px; padding:14px 16px; }
.card .v { font-size:26px; font-weight:700; letter-spacing:.5px; }
.card .l { font-size:12.5px; color:var(--dim); margin-top:3px; }
.card.hot { border-color:#5a4a10; background:linear-gradient(180deg,#1b1708,#12151d); }
.card.hot .v { color:var(--accent); }
.bar { height:8px; border-radius:6px; background:#1c2130; overflow:hidden; margin-top:12px; }
.bar > i { display:block; height:100%; background:linear-gradient(90deg,#ffd400,#ff9f1c); }
h2 { font-size:17px; margin:26px 0 12px; color:var(--accent); }
h2 span { color:var(--dim); font-size:13px; font-weight:400; }
table { width:100%; border-collapse:collapse; font-size:13.5px; }
th { text-align:right; color:var(--dim); font-weight:500; padding:8px 6px;
     border-bottom:1px solid var(--line); white-space:nowrap; }
td { padding:9px 6px; border-bottom:1px solid #1a1f2b; vertical-align:top; }
code { background:#1b2130; padding:2px 7px; border-radius:6px; font-size:12.5px;
       font-family:ui-monospace,Menlo,Consolas,monospace; direction:ltr; display:inline-block; }
.code-main { color:var(--accent); font-weight:700; font-size:14px; }
.tag { font-size:11.5px; padding:2px 8px; border-radius:999px; border:1px solid var(--line);
       color:var(--dim); }
.tag.strong { color:var(--good); border-color:#17452f; }
.tag.finished { color:var(--dim); }
.empty { color:var(--dim); font-size:13.5px; padding:14px 0; }
.thumb { width:56px; height:56px; border-radius:9px; object-fit:cover; background:#1b2130; }
.flash { animation:fl 1.1s ease-out; }
@keyframes fl { from { background:#1b2a1f; } to { background:transparent; } }
footer { margin-top:30px; color:var(--dim); font-size:12px; line-height:2; }
.dot { display:inline-block; width:8px; height:8px; border-radius:50%; background:var(--dim);
       margin-left:6px; }
.dot.live { background:var(--good); box-shadow:0 0 0 0 rgba(53,208,127,.6);
            animation:pulse 1.8s infinite; }
@keyframes pulse { 70% { box-shadow:0 0 0 9px rgba(53,208,127,0); } 100% { box-shadow:0 0 0 0 rgba(53,208,127,0); } }
</style>
</head><body><div class="wrap">

<header>
  <h1>🎁 داشبورد شکار گنج <small id="chtitle">—</small></h1>
  <div style="display:flex;gap:8px;flex-wrap:wrap">
    <span class="pill" id="phase">—</span>
    <span class="pill" id="clock">—</span>
    <span class="pill" id="live" hidden><span class="dot live"></span>زنده</span>
  </div>
</header>

<div class="card" style="padding:16px 18px">
  <div style="display:flex;justify-content:space-between;gap:10px;flex-wrap:wrap;font-size:13px;color:var(--dim)">
    <span id="range">—</span>
    <span id="progress-txt">—</span>
  </div>
  <div class="bar"><i id="progress" style="width:0%"></i></div>
</div>

<h2 style="margin-top:20px">شمارنده‌ها</h2>
<div class="grid4" id="kpis"></div>

<h2>نامزدهای نشانه‌دار <span id="cand-count"></span></h2>
<div id="cands"></div>

<h2>رمزهای استخراج‌شده <span id="code-count"></span></h2>
<div id="codes"></div>

<h2>هشدارها</h2>
<div id="alerts"></div>

<h2>چرخه‌های اسکن</h2>
<div id="cycles"></div>

<footer>
  digiganj — داشبورد فقط «مشاهده» است؛ هیچ رمزی را روی حساب شما ثبت نمی‌کند.
  زمان فایل تصویر روی CDN قابل‌اتکاترین نشانهٔ انتشار است؛ «اولین مشاهده» به
  شبکهٔ شما بستگی دارد.<br>
  آخرین به‌روزرسانی: <span id="updated">—</span>
</footer>
</div>

<script>
const LIVE = __LIVE__;
const REFRESH_MS = __REFRESH__;

function fa(n) {
  if (n === null || n === undefined || n === "") return "—";
  const s = String(n);
  const map = ["۰","۱","۲","۳","۴","۵","۶","۷","۸","۹"];
  return s.replace(/[0-9]/g, d => map[+d]).replace(/,/g, "٬");
}
function dur(sec) {
  sec = Math.max(0, Math.floor(sec || 0));
  if (sec < 60) return fa(sec) + " ثانیه";
  const m = Math.floor(sec / 60), h = Math.floor(m / 60);
  if (h) return fa(h) + " ساعت و " + fa(m % 60) + " دقیقه";
  return fa(m) + " دقیقه و " + fa(sec % 60) + " ثانیه";
}
function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g, c => (
    {"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
}
// همهٔ زمان‌ها به وقت تهران (UTC+3:30) نمایش داده می‌شوند، نه منطقهٔ مرورگر
const TEHRAN_OFFSET = 3.5 * 3600;
function tehran(ts) {
  const d = new Date((ts + TEHRAN_OFFSET) * 1000);
  // از getUTC* استفاده می‌کنیم تا جابه‌جایی محلی دوباره اعمال نشود
  return {
    getFullYear: () => d.getUTCFullYear(),
    getMonth: () => d.getUTCMonth(),
    getDate: () => d.getUTCDate(),
    getHours: () => d.getUTCHours(),
    getMinutes: () => d.getUTCMinutes(),
    getSeconds: () => d.getUTCSeconds(),
  };
}

const JMONTHS = ["فروردین","اردیبهشت","خرداد","تیر","مرداد","شهریور",
                 "مهر","آبان","آذر","دی","بهمن","اسفند"];

// تبدیل میلادی → جلالی (الگوریتم استاندارد بدون کتابخانه)
function g2j(gy, gm, gd) {
  const g_d_m = [0,31,59,90,120,151,181,212,243,273,304,334];
  let jy = (gy <= 1600) ? 0 : 979;
  gy -= (gy <= 1600) ? 621 : 1600;
  const gy2 = (gm > 2) ? (gy + 1) : gy;
  let days = (365 * gy) + Math.floor((gy2 + 3) / 4) - Math.floor((gy2 + 99) / 100)
           + Math.floor((gy2 + 399) / 400) - 80 + gd + g_d_m[gm - 1];
  jy += 33 * Math.floor(days / 12053); days %= 12053;
  jy += 4 * Math.floor(days / 1461);   days %= 1461;
  if (days > 365) { jy += Math.floor((days - 1) / 365); days = (days - 1) % 365; }
  const jm = (days < 186) ? 1 + Math.floor(days / 31) : 7 + Math.floor((days - 186) / 30);
  const jd = 1 + ((days < 186) ? (days % 31) : ((days - 186) % 30));
  return [jy, jm, jd];
}

function jdate(ts, withTime) {
  if (!ts) return "—";
  try {
    const d = tehran(ts);
    const [jy, jm, jd] = g2j(d.getFullYear(), d.getMonth() + 1, d.getDate());
    const p = n => String(n).padStart(2, "0");
    let out = fa(p(jy) + "/" + p(jm) + "/" + p(jd));
    if (withTime) out += " " + fa(p(d.getHours()) + ":" + p(d.getMinutes()));
    return out;
  } catch (e) { return "—"; }
}

function jpretty(ts) {
  if (!ts) return "—";
  try {
    const d = tehran(ts);
    const [jy, jm, jd] = g2j(d.getFullYear(), d.getMonth() + 1, d.getDate());
    const p = n => String(n).padStart(2, "0");
    return fa(jd) + " " + JMONTHS[jm - 1] + " " + fa(jy) + "، ساعت " +
           fa(p(d.getHours()) + ":" + p(d.getMinutes()) + ":" + p(d.getSeconds()));
  } catch (e) { return "—"; }
}
function table(headers, rows, emptyMsg) {
  if (!rows.length) return '<div class="empty">' + emptyMsg + "</div>";
  return "<table><tr>" + headers.map(h => "<th>" + h + "</th>").join("") + "</tr>" +
    rows.map(r => "<tr>" + r.map(c => "<td>" + c + "</td>").join("") + "</tr>").join("") +
    "</table>";
}

function render(s) {
  const ch = s.challenge || {};
  document.getElementById("chtitle").textContent = ch.title || "";
  const ph = document.getElementById("phase");
  ph.textContent = ch.phase_fa || "—";
  ph.className = "pill " + (ch.phase === "active" ? "on" : ch.phase === "upcoming" ? "up" : "off");
  document.getElementById("clock").textContent = jpretty(s.generated_at);
  document.getElementById("range").textContent =
    ch.start_ts ? ("شروع: " + jdate(ch.start_ts, true) + " ← پایان: " + jdate(ch.end_ts, true)) : "—";

  const prog = Math.max(0, Math.min(100, ch.progress || 0));
  document.getElementById("progress").style.width = prog.toFixed(1) + "%";
  let ptxt;
  if (ch.phase === "upcoming") ptxt = "شروع تا " + dur(ch.remain);
  else if (ch.phase === "active") ptxt = "سپری‌شده " + dur(ch.elapsed) + " از " + dur(ch.end_ts - ch.start_ts);
  else ptxt = "پایان‌یافته";
  document.getElementById("progress-txt").textContent = ptxt;

  const st = s.stats || {};
  const kpis = [
    ["products", "محصول بررسی‌شده"],
    ["images", "تصویر ثبت‌شده"],
    ["images_downloaded", "تصویر دانلودشده"],
    ["images_ocred", "تصویر OCR‌شده"],
    ["alerts", "هشدار"],
    ["codes", "رمز"],
  ].map(([k, l]) =>
    '<div class="card"><div class="v">' + fa(st[k] || 0) + '</div><div class="l">' + l + "</div></div>"
  ).join("");
  const top = '<div class="card hot"><div class="v">' + fa(Math.round(s.top_score || 0)) +
    '</div><div class="l">بالاترین امتیاز</div></div>';
  document.getElementById("kpis").innerHTML = top + kpis;

  const cands = s.candidates || [];
  document.getElementById("cand-count").textContent = "(" + fa(cands.length) + ")";
  document.getElementById("cands").innerHTML = table(
    ["تصویر", "محصول", "امتیاز", "زمان فایل", "فاصله", "رمزها", "نوع"],
    cands.map(c => {
      const img = c.file_path ? '<img class="thumb" loading="lazy" src="/local/' +
        encodeURIComponent(c.file_path) + '">' : '<div class="thumb"></div>';
      const d = c.stamp_ts && ch.start_ts ? (c.stamp_ts - ch.start_ts) : null;
      const dist = d === null ? "—" : (d >= 0 ? dur(d) + " پس از شروع" : dur(-d) + " پیش از شروع");
      return [
        img, "dkp-" + esc(c.product_id), fa(Math.round(c.score || 0)),
        jdate(c.stamp_ts, true), dist,
        c.ocr_codes ? "<code>" + esc(c.ocr_codes) + "</code>" : "—",
        '<span class="tag ' + (c.kind || "") + '">' + esc(c.kind || "—") + "</span>",
      ];
    }),
    "هنوز نامزدی نیست."
  );

  const codes = s.codes || [];
  document.getElementById("code-count").textContent = "(" + fa(codes.length) + " خانواده)";
  document.getElementById("codes").innerHTML = table(
    ["رمز پیشنهادی", "خوانش‌های دیگر", "محصول", "گونه", "نوع", "زمان"],
    codes.map(c => {
      const alts = (c.readings || []).slice(1, 4).map(r =>
        "<code>" + esc(r.code) + "</code>").join(" ") || "—";
      return [
        '<code class="code-main">' + esc(c.code) + "</code>",
        alts,
        "dkp-" + esc(c.product_id),
        fa(c.variants),
        c.source === "ocr" ? "خام" : "حدسی",
        jdate(c.ts, true),
      ];
    }),
    "هنوز رمزی استخراج نشده."
  );

  const alerts = s.alerts || [];
  document.getElementById("alerts").innerHTML = table(
    ["محصول", "امتیاز", "دلیل", "زمان"],
    alerts.map(a => [
      "dkp-" + esc(a.product_id), fa(Math.round(a.score || 0)),
      esc(a.reason || "—"), jdate(a.ts, true),
    ]),
    "هشداری ثبت نشده."
  );

  const cycles = (s.cycles || []).slice().reverse();
  document.getElementById("cycles").innerHTML = table(
    ["#", "گالری", "تازه", "هشدار", "مدت", "زمان"],
    cycles.map(c => [
      fa(c.id), fa(c.galleries || 0), fa(c.new_images || 0),
      fa(c.alerts || 0), dur(c.elapsed || 0), jdate(c.started_ts, true),
    ]),
    "چرخه‌ای اجرا نشده."
  );

  document.getElementById("updated").textContent = jpretty(s.generated_at);
}

async function tick() {
  try {
    const r = await fetch("state.json?t=" + Date.now(), { cache: "no-store" });
    if (r.ok) render(await r.json());
  } catch (e) { /* شبکه قطع شد؛ بار بعدی */ }
}

const initial = __STATE__;
render(initial);
if (LIVE) {
  document.getElementById("live").hidden = false;
  setInterval(tick, REFRESH_MS);
}
</script>
</body></html>
"""


def render_dashboard(state: dict[str, Any], live: bool = False, refresh_ms: int = 4000) -> str:
    state_json = json.dumps(state, ensure_ascii=False, default=str)
    # جلوگیری از بسته‌شدن اسکریپت با </script> داخل داده
    state_json = state_json.replace("</", "<\\/")
    html = _TEMPLATE
    html = html.replace("__LIVE__", "true" if live else "false")
    html = html.replace("__REFRESH__", str(max(1000, int(refresh_ms))))
    html = html.replace("__STATE__", state_json)
    return html


def write_dashboard(path: str | Path, store: db.Store, cfg=None, challenge=None,
                    live: bool = False, refresh_ms: int = 4000,
                    limit: int = 12) -> Path:
    state = collect_state(store, cfg, challenge, limit=limit)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(render_dashboard(state, live=live, refresh_ms=refresh_ms), encoding="utf-8")
    return p


# --------------------------------------------------------------------------- #
# سرور زنده
# --------------------------------------------------------------------------- #


def serve(store: db.Store, cfg=None, challenge=None, host: str = "127.0.0.1",
          port: int = 8811, interval: float = 3.0, quiet: bool = False) -> None:
    """سرور داشبورد زنده؛ ``/`` صفحه و ``/state.json`` وضعیت تازه.

    تصاویر محلی از ``/local/<path>`` سرو می‌شوند تا بندانگشتی‌ها بدون
    وابستگی به CDN دیده شوند.
    """
    con = Console()
    cache = {"state": collect_state(store, cfg, challenge), "ts": time.time()}
    lock = threading.Lock()

    def state_now() -> dict[str, Any]:
        with lock:
            if time.time() - cache["ts"] >= interval:
                try:
                    cache["state"] = collect_state(store, cfg, challenge)
                    cache["ts"] = time.time()
                except Exception:
                    pass
            return cache["state"]

    class Handler(BaseHTTPRequestHandler):
        # HTTP/1.1 لازم است تا اتصال برای درخواست‌های تکراری باز بماند؛
        # داشبورد هر چند ثانیه state.json را می‌خواند و با HTTP/1.0 هر بار
        # یک اتصال تازه برقرار می‌شد (کندی محسوس و بار اضافی روی پروکسی).
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):  # کم‌صدا
            return

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_GET(self):  # noqa: N802
            path = self.path.split("?", 1)[0]
            if path in ("/", "/index.html"):
                body = render_dashboard(state_now(), live=True).encode("utf-8")
                self._send(200, body, "text/html; charset=utf-8")
                return
            if path == "/state.json":
                body = json.dumps(state_now(), ensure_ascii=False, default=str).encode("utf-8")
                self._send(200, body, "application/json; charset=utf-8")
                return
            if path.startswith("/local/"):
                from urllib.parse import unquote

                raw = unquote(path[len("/local/"):])
                p = Path(raw)
                # محافظت واقعی: فقط فایل‌های تصویری که داخل پوشهٔ کاری (یا
                # مسیرهای مجاز داده) هستند سرو می‌شوند. بدون این بررسی،
                # سروری که روی 0.0.0.0 باز است می‌توانست هر .jpg/.png روی
                # دیسک را برای هر کسی که نشانی پیش‌نمایش را دارد بخواند.
                target = self._allowed_local(p)
                if target is None:
                    self._send(403, b"forbidden", "text/plain")
                    return
                ctype = {
                    ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                    ".png": "image/png", ".webp": "image/webp",
                }.get(target.suffix.lower(), "application/octet-stream")
                try:
                    self._send(200, target.read_bytes(), ctype)
                except OSError:
                    self._send(404, b"not found", "text/plain")
                return
            self._send(404, b"not found", "text/plain")

        @staticmethod
        def _allowed_local(p: Path) -> Path | None:
            """فایل تصویری مجاز برای سرو، یا None.

            مجاز = پسوند تصویر + بودن داخل یکی از ریشه‌های مجاز.
            """
            if p.suffix.lower() not in (".jpg", ".jpeg", ".png", ".webp"):
                return None
            try:
                resolved = p.resolve(strict=True)
            except (OSError, RuntimeError):
                return None
            if not resolved.is_file():
                return None
            roots = [Path.cwd().resolve()]
            if cfg is not None:
                for attr in ("data_dir", "reports_dir", "scans_dir"):
                    raw_root = getattr(cfg, attr, "") or ""
                    if raw_root:
                        try:
                            roots.append(Path(raw_root).resolve())
                        except (OSError, RuntimeError):
                            continue
            for root in roots:
                try:
                    resolved.relative_to(root)
                    return resolved
                except ValueError:
                    continue
            return None

    httpd = ThreadingHTTPServer((host, port), Handler)
    actual = httpd.server_address[1]
    if not quiet:
        con.banner("digiganj — داشبورد زنده", f"http://127.0.0.1:{actual}/")
        con.hint("برای توقف: Ctrl+C")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
