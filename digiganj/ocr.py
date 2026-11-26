"""تشخیص رمز در تصویر.

سه موتور، به‌ترتیب کیفیت:

۱) **tesseract** — اگر باینری ``tesseract`` روی سیستم نصب باشد (دقیق‌ترین برای
   متن‌های تیز). با ``--psm 7`` اجرا می‌شود مگر تنظیمات چیز دیگری بگویند.
۲) **موتور داخلی** — یک طبقه‌بند نویسهٔ سبک که با فونت‌های مونو/هندسی متداول
   آموزش دیده و **بدون هیچ وابستگی خارجی** کار می‌کند. مناسب رمزهایی است که
   با فونت تمیز و بدون چرخش روی تصویر چاپ شده‌اند (رایج‌ترین حالت ایونت).
۳) **برش + تطبیق الگو** — برای حالتی که بخشی از متن از قبل معلوم است.

اصل کار موتور داخلی
-------------------
* تصویر خاکستری می‌شود؛ دو حالت (متن روشن روی زمینهٔ تیره، و برعکس) بررسی
  می‌شود و کنتراست با کشش هیستوگرام نرمال می‌شود.
* خطوط متن با پروفایل افقی روشنایی پیدا و به نوارهایی بریده می‌شوند
  (اتصال فاصله‌ها تا همیشه «خط» بمانند؛ ارتفاع‌های بی‌ربط مثل لوگو/حاشیه رد
  می‌شوند).
* در هر نوار، ستون‌های تهی نویسه‌ها را جدا می‌کنند؛ نویسه‌های چسبیده با
  کمینهٔ پروفایل عمودی شکسته می‌شوند.
* هر نویسه به شبکهٔ ۱۲×۱۶ نرمال می‌شود (بدون حساسیت به اندازه/ضخامت) و با
  میانگین‌های آموزش‌دیدهٔ فونت‌های رایج مقایسه می‌شود.
* در پایان، رشته‌های شبیه رمز با الگوی قابل‌تنظیم استخراج می‌شوند.

این موتور «متن آزاد» را نمی‌فهمد؛ هدفش فقط یافتن رمز است و همین آن را
سبک و قابل اتکا می‌کند.
"""

from __future__ import annotations

import math
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from . import utils

try:
    import numpy as np
except Exception:  # pragma: no cover - numpy همیشه لازم نیست
    np = None  # type: ignore

try:
    from PIL import Image, ImageOps, ImageFilter
except Exception:  # pragma: no cover
    Image = None  # type: ignore

GLYPHS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
GLYPH_INDEX = {ch: i for i, ch in enumerate(GLYPHS)}
GRID_W, GRID_H = 12, 16


@dataclass
class TextLine:
    """یک نوار متن با نویسه‌های جداشده."""

    text: str
    score: float
    y: int
    height: int
    charset: str = ""

    def as_dict(self) -> dict:
        return {
            "text": self.text,
            "score": round(self.score, 2),
            "y": self.y,
            "height": self.height,
        }


@dataclass
class OcrResult:
    engine: str
    texts: list[str]
    codes: list[str]
    lines: list[TextLine]
    elapsed: float
    error: str = ""

    @property
    def best_text(self) -> str:
        return self.texts[0] if self.texts else ""

    @property
    def score(self) -> float:
        return max((l.score for l in self.lines), default=0.0)

    def as_dict(self) -> dict:
        return {
            "engine": self.engine,
            "codes": self.codes,
            "lines": [l.as_dict() for l in self.lines],
            "elapsed_ms": round(self.elapsed * 1000, 1),
            "error": self.error,
        }


# --------------------------------------------------------------------------- #
# وابستگی‌ها
# --------------------------------------------------------------------------- #


def pillow_available() -> bool:
    return Image is not None and np is not None


def tesseract_available() -> bool:
    return shutil.which("tesseract") is not None


# --------------------------------------------------------------------------- #
# قواعد رمز
# --------------------------------------------------------------------------- #

CONFUSABLE_MAP = str.maketrans({
    "0": "O", "O": "0",
    "1": "I", "I": "1", "l": "1", "|": "1", "!": "1",
    "5": "S", "S": "5",
    "8": "B", "B": "8",
    "2": "Z", "Z": "2",
    "6": "G", "G": "6",
    "9": "g", "q": "9",
})

# اشتباهات واقعی مشاهده‌شده در تصاویر کارت رمز (فونت مونو و وزن ضخیم):
# حروف گرد در شبکهٔ کوچک به هم می‌چسبند، دنبالهٔ حروف پایین‌رو گم می‌شود.
# هر جفت، دو طرف تبدیل‌پذیر است تا هر خوانشی قابل بازبینی باشد.
CONFUSABLE_PAIRS: dict[str, tuple[str, ...]] = {
    "p": ("o", "q", "b"),
    "b": ("p", "6", "h"),
    "q": ("g", "p", "9"),
    "d": ("a", "o", "q"),
    "o": ("p", "0", "e"),
    "a": ("d", "o"),
    "e": ("o", "c"),
    "c": ("e", "o"),
    "n": ("h", "m", "u"),
    "h": ("n", "b", "k"),
    "u": ("n", "v"),
    "m": ("n", "w"),
    "w": ("m", "v"),
    "v": ("u", "y"),
    "y": ("v", "j"),
    "j": ("y", "i"),
    "i": ("j", "l", "1"),
    "l": ("i", "1", "I"),
    "I": ("l", "1", "i"),
    "Q": ("O", "0", "G"),
    "O": ("Q", "0", "D"),
    "D": ("O", "0"),
    "G": ("6", "C", "Q"),
    "C": ("G", "O"),
    "P": ("R", "F"),
    "R": ("P", "B"),
    "F": ("P", "E"),
    "E": ("F", "B"),
    "B": ("8", "R", "E"),
    "S": ("5", "8"),
    "Z": ("2", "7"),
    "2": ("Z", "7"),
    "7": ("2", "T", "1"),
    "T": ("7", "Y"),
    "Y": ("T", "V"),
    "N": ("M", "H"),
    "M": ("N", "W"),
    "H": ("N", "M"),
    "K": ("X", "R"),
    "U": ("V", "0"),
    "V": ("U", "Y"),
    "A": ("4", "R"),
    "6": ("G", "b", "8"),
    "8": ("B", "6", "S"),
    "9": ("g", "q"),
    "g": ("9", "q", "y"),
}.copy()

# جلوی اشتباه رایج: خود-تبدیل حذف شود
for _k, _vs in list(CONFUSABLE_PAIRS.items()):
    CONFUSABLE_PAIRS[_k] = tuple(v for v in _vs if v != _k)

# ریسک اشتباه: کدام نویسه‌ها در عمل بیشتر اشتباه خوانده می‌شوند؟
# ۳ = تقریباً همیشه محتمل (شکل یکسان در شبکهٔ کوچک)
# ۲ = محتمل (فقط یک بخش شکل متفاوت است)
# ۱ = کم‌احتمال (شکل کاملاً متفاوت)
CONFUSABLE_RISK: dict[str, int] = {
    "0": 3, "O": 3, "Q": 3, "o": 3, "p": 3, "q": 3, "g": 3,
    "l": 3, "1": 3, "I": 3, "i": 3, "|": 3,
    "5": 2, "S": 2, "8": 2, "B": 2, "6": 2, "G": 2, "9": 2,
    "2": 2, "Z": 2, "c": 2, "e": 2, "a": 2, "d": 2, "b": 2,
    "u": 2, "v": 2, "n": 2, "h": 2, "m": 2, "w": 2, "y": 2, "j": 2,
    "C": 1, "D": 1, "E": 1, "F": 1, "H": 1, "K": 1, "M": 1, "N": 1,
    "P": 1, "R": 1, "T": 1, "U": 1, "V": 1, "W": 1, "X": 1, "Y": 1,
    "7": 1, "A": 1, "4": 1,
}


def normalize_code_token(token: str) -> str:
    return token.strip()


def confusables(ch: str) -> tuple[str, ...]:
    """نویسه‌های هم‌شکل که در شبکهٔ کوچک با ``ch`` اشتباه می‌شوند."""
    out: list[str] = []
    for v in CONFUSABLE_PAIRS.get(ch, ()):
        if v not in out:
            out.append(v)
    # نگاشت قدیمی تک‌حرفی هم پشتیبانی شود
    legacy = ch.translate(CONFUSABLE_MAP)
    if legacy != ch and legacy not in out:
        out.append(legacy)
    return tuple(out)


def _fmt_for(code: str) -> dict[str, Any]:
    fid = detect_format(code)
    return next((f for f in CODE_FORMATS if f["id"] == fid), {}) or {}


def case_variants(code: str) -> list[str]:
    """گونه‌های حرف بزرگ/کوچک — رمز به بزرگی و کوچکی حروف حساس است.

    OCR اغلب «x» و «X» را یکی می‌خواند. سه نسخهٔ ارزان و پرارزش ساخته
    می‌شود: همهٔ حروف کوچک، همهٔ حروف بزرگ، و «متعارف» (حروف کوچک جز
    آغاز هر بخشی که با رقم جدا شده — الگوی رایج رمزها).
    """
    if not code:
        return []
    out: list[str] = []
    lower = code.lower()
    upper = code.upper()
    for v in (lower, upper, code):
        if v not in out and (not detect_format(code) or match_format(v, _fmt_for(code))):
            out.append(v)
    # الگوی رایج: حرف اولِ هر «کلمه» بزرگ، بقیه کوچک
    import re as _re

    parts = _re.split(r"(\d+)", code)
    normal = ""
    for part in parts:
        if part.isdigit() or not part:
            normal += part
        else:
            normal += part[0].upper() + part[1:].lower()
    if normal and normal not in out and (not detect_format(code) or match_format(normal, _fmt_for(code))):
        out.append(normal)
    return [v for v in out if v != code]


def code_variants(code: str, max_variants: int = 8, positions: int = 3,
                  per_position: int = 2, min_risk: int = 1) -> list[str]:
    """نسخه‌های محتمل رمز برای وقتی نویسه‌ای اشتباه خوانده شده باشد.

    دو نوع واریانت ساخته می‌شود:

    * **تک‌موقعیتی**: یک نویسهٔ مشتبه با هم‌شکلش عوض می‌شود (``p``↔``o``).
    * **چندموقعیتی**: اگر دو نویسهٔ مجاور هم مشتبه باشند، نسخهٔ جایگزینی هر
      دو با هم هم ساخته می‌شود (مثل ``20``→``2O``).

    تعداد محدود است تا فهرست نامزدها از کنترل خارج نشود؛ کاربر خودش
    نسخه‌ها را به‌ترتیب در فرم رسمی می‌آزماید.
    """
    if not code:
        return []
    # قالب کد اصلی: گونه‌ها نباید از قالب بیرون بزنند. مثال واقعی: در قالب
    # «DK + فقط حروف»، تبدیل K→X یا O→0 کد را بی‌اعتبار می‌کند و فقط
    # نامزدهای ساختگی تولید می‌کند که کاربر را از جواب درست دور می‌کند.
    fmt_id = detect_format(code)
    fmt = next((f for f in CODE_FORMATS if f["id"] == fmt_id), None)

    def _valid(cand: str) -> bool:
        if fmt is None:
            return True
        return match_format(cand, fmt)

    idx = [
        i for i, ch in enumerate(code)
        if confusables(ch) and CONFUSABLE_RISK.get(ch, 1) >= min_risk
    ]
    if not idx:
        return []
    # موقعیت‌های پرریسک اول: اگر «p» ممکن است «o» باشد، پیش از «H→N» بیاید
    idx.sort(key=lambda i: (-CONFUSABLE_RISK.get(code[i], 1), i))
    variants: list[str] = []
    seen = {code}
    # ۱) تک‌موقعیتی (فقط نویسه‌های هم‌ریسک با همان سطح)
    for i in idx[: max(positions, 1)]:
        risk = CONFUSABLE_RISK.get(code[i], 1)
        for repl in [r for r in confusables(code[i]) if CONFUSABLE_RISK.get(r, 1) >= risk][:per_position]:
            cand = code[:i] + repl + code[i + 1:]
            if not _valid(cand):
                continue
            if cand not in seen:
                seen.add(cand)
                variants.append(cand)
                if len(variants) >= max_variants:
                    return variants
    # ۲) دو‌موقعیتی مجاور (اگر جا بود)
    for i in idx:
        if i + 1 not in idx:
            continue
        for r1 in confusables(code[i])[:2]:
            for r2 in confusables(code[i + 1])[:2]:
                cand = code[:i] + r1 + r2 + code[i + 2:]
                if not _valid(cand):
                    continue
                if cand not in seen:
                    seen.add(cand)
                    variants.append(cand)
                    if len(variants) >= max_variants:
                        return variants
    return variants


def _entropy(s: str) -> float:
    if not s:
        return 0.0
    counts: dict[str, int] = {}
    for ch in s:
        counts[ch] = counts.get(ch, 0) + 1
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


# --------------------------------------------------------------------------- #
# قالب‌های رمز در دوره‌های مختلف ایونت
# --------------------------------------------------------------------------- #
#
# ۱۴۰۰ (شکار گنج قدیم): ۱۰ کاراکتر مخلوط حرف بزرگ/کوچک و رقم — مثل «fvKA4jd9DP»
# ۱۴۰۵ (جاکدی + شکار گنج): با «DK» شروع می‌شود، ۵ یا ۶ کاراکتر، فقط حروف —
#                          مثل «DKABC» یا «DKABCD»
#
# قالب‌ها عمداً «داده» هستند، نه کد ثابت: اگر سال بعد باز عوض شد، فقط این
# فهرست به‌روز می‌شود.
CODE_FORMATS: tuple[dict[str, Any], ...] = (
    {
        "id": "dk-hunt",
        "title": "شکار گنج (DK + حروف)",
        "prefix": "DK",
        "lengths": (5, 6),
        "letters_only": True,
        "regex": r"DK[A-Za-z]{3,4}",
    },
    {
        "id": "legacy-1400",
        "title": "شکار گنج ۱۴۰۰ (۱۰ کاراکتر مخلوط)",
        "prefix": "",
        "lengths": (10,),
        "letters_only": False,
        "regex": r"(?=[A-Za-z0-9]*[A-Za-z])(?=[A-Za-z0-9]*\d)[A-Za-z0-9]{10}",
    },
)


def match_format(code: str, fmt: dict[str, Any]) -> bool:
    """آیا ``code`` با این قالب می‌خواند؟"""
    if not code:
        return False
    lengths = tuple(fmt.get("lengths") or ())
    if lengths and len(code) not in lengths:
        return False
    prefix = str(fmt.get("prefix") or "")
    if prefix and not code.upper().startswith(prefix.upper()):
        return False
    if fmt.get("letters_only") and not code.isalpha():
        return False
    return True


def detect_format(code: str) -> str:
    """شناسهٔ قالبی که کد با آن می‌خواند (یا رشتهٔ خالی)."""
    for fmt in CODE_FORMATS:
        if match_format(code, fmt):
            return str(fmt["id"])
    return ""


def hunt_code_pattern(formats: Iterable[dict[str, Any]] | None = None) -> str:
    """الگوی جستجوی ترکیبی برای همهٔ قالب‌ها (برای یک بار پیمایش متن)."""
    parts: list[str] = []
    for fmt in (formats or CODE_FORMATS):
        rx = str(fmt.get("regex") or "")
        if rx:
            parts.append(f"(?:{rx})")
    return "|".join(parts) if parts else r"(?!)"


def is_hunt_code(code: str) -> bool:
    """آیا این رشته کد گنج است (به‌هر قالبی)؟"""
    return bool(detect_format(code))


def plausible_code(code: str) -> bool:
    """فیلتر رمزهای بی‌معنی (تاریخ، شمارهٔ تلفن، تکرار، مسیر فایل و…).

    کدی که با یکی از قالب‌های رسمی ایونت بخواند، پیش از هر فیلتر دیگری
    پذیرفته می‌شود: قالب رسمی خودش گویاترین شاهد است.
    """
    if not code:
        return False
    if is_hunt_code(code):
        return True
    if len(set(code)) < 4:
        return False
    if re.fullmatch(r"0+", code) or re.fullmatch(r"(\d)\1+", code):
        return False
    if _entropy(code) < 2.2:
        return False
    # تاریخ‌های ۸رقمی مثل 20211126
    if re.fullmatch(r"20\d{6}", code):
        return False
    # عددهای گرد
    if re.fullmatch(r"\d0{4,}", code):
        return False
    return True


def find_codes(text: str, pattern: str | None = None,
               min_len: int = 5, max_len: int = 24) -> list[str]:
    """استخراج توکن‌های شبیه رمز از متن؛ بدون تکرار و با حفظ ترتیب.

    الگوی پیش‌فرض ترکیب همهٔ قالب‌های شناخته‌شده است، پس هم کد قدیمی
    ۱۰ کاراکتری و هم کد جدید «DK…» در یک پیمایش پیدا می‌شوند.
    """
    pattern = pattern or hunt_code_pattern()
    found: list[str] = []
    seen: set[str] = set()
    for m in re.finditer(pattern, text or ""):
        tok = normalize_code_token(m.group(0))
        if not (min_len <= len(tok) <= max_len):
            continue
        tok = tok.strip(".,:;-_/\\'\"`|")
        if tok in seen:
            continue
        if not plausible_code(tok):
            continue
        seen.add(tok)
        found.append(tok)
    return found


def assemble_codes(tokens: Iterable[str], gap_chars: str = " -_|.,:") -> list[str]:
    """چسباندن توکن‌های کوتاهِ پیاپی که با فاصله/خط جدا شده‌اند.

    OCR ممکن است رمز را تکه‌تکه ببیند («fvKA4jd» + «9DP»). اگر طول کل در
    بازهٔ معقول باشد، ترکیب‌ها را هم به کاندیدها اضافه می‌کنیم.
    """
    toks = [t for t in tokens if t]
    out: list[str] = []
    for i in range(len(toks)):
        acc = toks[i]
        for j in range(i + 1, min(len(toks), i + 3)):
            acc += toks[j]
            if 8 <= len(acc) <= 24:
                out.append(acc)
    return out


# --------------------------------------------------------------------------- #
# موتور داخلی
# --------------------------------------------------------------------------- #

GRID_W, GRID_H = 20, 24       # شبکهٔ نرمال‌سازی نویسه
TOPK_PER_GLYPH = 6            # نویسه‌های نزدیک برای جستجوی پرتوی
BEAM_WIDTH = 10               # تعداد مسیرهای نگه‌داشته‌شده
BEAM_EXPAND = 5               # انشعاب هر نویسه
BEAM_GAP = 0.25               # هرس گزینه‌های دورتر از بهترین
FEATURE_WEIGHT = 0.6          # وزن جریمهٔ اختلاف ویژگی‌های شکل
KNN_K = 2                     # بهترین k نمونه در امتیازدهی


class GlyphMatcher:
    """طبقه‌بند نویسه با چند نمونه از فونت‌های رایج.

    هر نویسه با شبکهٔ ۲۰×۲۴ و یک بردار ویژگی شکل (نسبت ابعاد، چگالی، مرکز
    جرم) نمایش داده می‌شود. جدول با ``digiganj ocr-train`` ساخته و به‌صورت
    فشرده (npz) ذخیره می‌شود؛ جدا از کد است چون فونت‌ها مجوزهای متفاوت دارند.
    """

    def __init__(self, table_path: str | Path | None = None):
        self.table: dict[str, list[tuple["np.ndarray", "np.ndarray"]]] = {}
        self.meta: dict[str, str] = {}
        self._index: "np.ndarray" | None = None
        self._chars: list[str] = []
        if table_path:
            self.load(table_path)

    # ---- ذخیره/بازخوانی ----
    def load(self, path: str | Path) -> None:
        p = Path(path)
        if not p.exists() or np is None:
            return
        with np.load(p, allow_pickle=False) as data:
            for key in data.files:
                if key.startswith("g_"):
                    ch = key[2:]
                    arr = data[key]
                    feats = data.get(f"f_{ch}")
                    samples = []
                    for i in range(arr.shape[0]):
                        f = feats[i] if feats is not None and i < feats.shape[0] else np.zeros(4)
                        samples.append((arr[i], f))
                    if samples:
                        self.table[ch] = samples
                elif key.startswith("m_"):
                    try:
                        value = data[key]
                        self.meta[key[2:]] = (
                            str(value.item()) if hasattr(value, "item") else str(value)
                        )
                    except Exception:
                        pass
        self._build_index()

    def save(self, path: str | Path) -> None:
        p = Path(path)
        utils.ensure_dir(p.parent)
        payload: dict[str, "np.ndarray"] = {}
        for ch, samples in self.table.items():
            if not samples:
                continue
            payload[f"g_{ch}"] = np.stack([s[0] for s in samples])
            payload[f"f_{ch}"] = np.stack([s[1] for s in samples])
        for k, v in self.meta.items():
            payload[f"m_{k}"] = np.array(v)
        np.savez_compressed(p, **payload)

    def add(self, ch: str, grid: "np.ndarray", feat: "np.ndarray") -> None:
        self.table.setdefault(ch, []).append(
            (grid.astype(np.float32), feat.astype(np.float32))
        )
        self._index = None

    @property
    def size(self) -> int:
        return sum(len(v) for v in self.table.values())

    def _build_index(self) -> None:
        """همهٔ نمونه‌ها را در یک ماتریس مسطح می‌چیند (برای ضرب ماتریسی سریع)."""
        if not self.table or np is None:
            self._index = None
            return
        rows: list["np.ndarray"] = []
        chars: list[str] = []
        for ch, samples in self.table.items():
            for grid, _feat in samples:
                v = grid.reshape(-1)
                n = np.linalg.norm(v) or 1.0
                rows.append(v / n)
                chars.append(ch)
        self._index = np.stack(rows)
        self._chars = chars

    # ---- طبقه‌بندی ----
    def classify_top(self, grid: "np.ndarray", feat: "np.ndarray",
                     topn: int = TOPK_PER_GLYPH, k: int = KNN_K) -> list[tuple[float, str]]:
        """نویسه‌های نزدیک با امتیاز (شباهت شبکه − جریمهٔ فاصلهٔ ویژگی).

        برای هر نویسه، میانگین بهترین k نمونه استفاده می‌شود تا یک نمونهٔ
        استثنایی نتیجه را به هم نزند (مقاوم‌تر از کمینهٔ فاصله).
        """
        if self._index is None or np is None:
            return []
        v = grid.reshape(-1)
        norm = np.linalg.norm(v) or 1.0
        sims = (self._index @ (v / norm)).tolist()
        # بهترین k شباهت برای هر نویسه
        per_char: dict[str, list[float]] = {}
        for sim, ch in zip(sims, self._chars):
            per_char.setdefault(ch, []).append(sim)
        # k نویسهٔ نزدیک از نظر ویژگی‌های شکل
        feat_d: dict[str, list[float]] = {}
        for ch, samples in self.table.items():
            if not samples:
                continue
            feat_d[ch] = sorted(float(np.abs(feat - f).mean()) for _g, f in samples)
        scored: list[tuple[float, str]] = []
        for ch, sim_list in per_char.items():
            sim_list.sort(reverse=True)
            top_sims = sim_list[:k]
            grid_score = sum(top_sims) / len(top_sims)
            dists = feat_d.get(ch) or [9.0]
            feat_score = sum(dists[:k]) / len(dists[:k])
            scored.append((grid_score - FEATURE_WEIGHT * feat_score, ch))
        scored.sort(reverse=True)
        return scored[: max(1, topn)]


class GlyphSegmenter:
    """نرمال‌سازی تصویر و برش نویسه."""

    @staticmethod
    def to_gray(img: "Image.Image") -> "np.ndarray":
        return np.asarray(img.convert("L"), dtype=np.float32)

    @staticmethod
    def stretch(arr: "np.ndarray", low: float = 2.0, high: float = 98.0) -> "np.ndarray":
        lo = np.percentile(arr, low)
        hi = np.percentile(arr, high)
        if hi - lo < 1e-3:
            return arr
        return np.clip((arr - lo) * (255.0 / (hi - lo)), 0, 255)

    @staticmethod
    def ink(arr: "np.ndarray", dark_text: bool) -> "np.ndarray":
        return arr < 128 if dark_text else arr > 128

    @staticmethod
    def contours(mask: "np.ndarray") -> list[tuple[int, int]]:
        """بازه‌های ستونیِ دارای جوهر (جداسازی نویسه‌ها)."""
        cols = mask.sum(axis=0)
        segs: list[tuple[int, int]] = []
        start = None
        for x, v in enumerate(cols):
            if v > 0 and start is None:
                start = x
            elif v == 0 and start is not None:
                if x - start >= 2:
                    segs.append((start, x))
                start = None
        if start is not None and len(cols) - start >= 2:
            segs.append((start, len(cols)))
        return GlyphSegmenter._resplit_wide(mask, segs)

    @staticmethod
    def _resplit_wide(mask, segs):
        if not segs:
            return segs
        widths = sorted(b - a for a, b in segs)
        typical = widths[len(widths) // 2]
        out: list[tuple[int, int]] = []
        for x0, x1 in segs:
            w = x1 - x0
            if w <= max(6, typical * 2.1):
                out.append((x0, x1))
                continue
            prof = mask[:, x0:x1].sum(axis=0)
            n = max(2, round(w / max(1, typical)))
            cuts = [x0]
            for k in range(1, n):
                target = int(k * w / n)
                lo = max(1, target - max(2, typical // 2))
                hi = min(w - 1, target + max(2, typical // 2))
                idx = lo + int(np.argmin(prof[lo:hi])) if hi > lo else target
                cuts.append(x0 + idx)
            cuts.append(x1)
            for a, b in zip(cuts, cuts[1:]):
                if b - a >= 2:
                    out.append((a, b))
        return out

    @staticmethod
    def normalize(mask: "np.ndarray") -> "np.ndarray | None":
        """جعبهٔ نویسه را به قاب مربع مرکزی در شبکهٔ GRID_H×GRID_W نگاشت می‌کند."""
        rows = np.where(mask.any(axis=1))[0]
        cols = np.where(mask.any(axis=0))[0]
        if len(rows) == 0 or len(cols) == 0:
            return None
        crop = mask[rows[0]: rows[-1] + 1, cols[0]: cols[-1] + 1]
        h, w = crop.shape
        if h < 3 or w < 1:
            return None
        side = max(h, w)
        canvas = np.zeros((side, side), dtype=np.float32)
        y0 = (side - h) // 2
        x0 = (side - w) // 2
        canvas[y0: y0 + h, x0: x0 + w] = crop
        ys = np.clip((np.arange(GRID_H) * side / GRID_H).astype(int), 0, side - 1)
        xs = np.clip((np.arange(GRID_W) * side / GRID_W).astype(int), 0, side - 1)
        return canvas[np.ix_(ys, xs)]

    @staticmethod
    def features(mask: "np.ndarray", ref_h: float | None = None) -> "np.ndarray":
        """ویژگی‌های شکل یک نویسه.

        ``ref_h`` ارتفاع مرجع خط است (میانهٔ ارتفاع نویسه‌های همان خط). نسبت
        ارتفاع نویسه به این مرجع، ویژگی کلیدی جداکردن حروف دارای دنباله
        (p، q، g، y، j) از حروف کوچک معمولی (o، c، e) است.
        """
        rows = np.where(mask.any(axis=1))[0]
        cols = np.where(mask.any(axis=0))[0]
        if len(rows) == 0 or len(cols) == 0:
            return np.zeros(5, dtype=np.float32)
        crop = mask[rows[0]: rows[-1] + 1, cols[0]: cols[-1] + 1].astype(np.float32)
        h, w = crop.shape
        total = crop.sum() or 1.0
        cy = float((crop * np.arange(h).reshape(-1, 1)).sum() / total / h)
        cx = float((crop * np.arange(w).reshape(1, -1)).sum() / total / w)
        rel = h / ref_h if ref_h else 1.0
        return np.array(
            [min(h / max(1, w), 4.0) / 4.0, crop.mean(), cy, cx, min(rel, 3.0) / 3.0],
            dtype=np.float32,
        )

    @staticmethod
    def zone(mask: "np.ndarray") -> "np.ndarray":
        """توزیع عمودی جوهر داخل جعبهٔ نویسه (۴ نوار افقی).

        بدون نیاز به خط مبنا: تصویر داخل جعبه به چهار نوار هم‌ارتفاع تقسیم
        می‌شود و سهم جوهر هر نوار ثبت می‌گردد. دنبالهٔ q/p/g/y/j باعث
        می‌شود سهم نوار پایینی بالا برود؛ e/c/o جوهر بیشتری در میانه دارند.
        """
        rows = np.where(mask.any(axis=1))[0]
        cols = np.where(mask.any(axis=0))[0]
        if len(rows) == 0 or len(cols) == 0:
            return np.zeros(4, dtype=np.float32)
        crop = mask[rows[0]: rows[-1] + 1, cols[0]: cols[-1] + 1].astype(np.float32)
        h = crop.shape[0]
        total = crop.sum() or 1.0
        shares = []
        for k in range(4):
            y0 = int(k * h / 4)
            y1 = max(y0 + 1, int((k + 1) * h / 4))
            shares.append(float(crop[y0:y1].sum() / total))
        return np.array(shares, dtype=np.float32)


def line_geometry(masks: list["np.ndarray"]) -> tuple[float, float, float]:
    """هندسهٔ افقی خط: (بالای خط، خط مبنا، پایین دنباله‌ها).

    برای جداسازی حروف دارای دنباله (p/q/g/y/j) از حروف کوچک معمولی
    (o/c/e/a) حیاتی است. مرزها با صدک گرفته می‌شوند تا نویسه‌های استثنایی
    (مثل «L» بلند یا «-» کوتاه) خط مبنا را جابه‌جا نکنند.
    """
    boxes = []
    for m in masks:
        rows = np.where(m.any(axis=1))[0]
        if len(rows):
            boxes.append((int(rows[0]), int(rows[-1]) + 1))
    if not boxes:
        return 0.0, 0.0, 0.0
    tops = sorted(b[0] for b in boxes)
    bots = sorted(b[1] for b in boxes)
    top = float(tops[max(0, int(len(tops) * 0.25))])
    base = float(bots[max(0, int(len(bots) * 0.75))])
    bottom = float(bots[-1])
    return top, base, bottom


@dataclass
class LineCandidate:
    text: str
    score: float


class InternalOcrEngine:
    """موتور داخلی: خطوط را پیدا، نویسه‌ها را جدا و رتبه‌بندی می‌کند و
    با جستجوی پرتوی چند کاندید برای کل رشته می‌سازد."""

    name = "internal"

    def __init__(self, matcher: GlyphMatcher | None = None):
        self.matcher = matcher or GlyphMatcher()

    # ---- یک نوار (خط) ----
    def read_band(self, band: "np.ndarray") -> list[list[tuple[float, str]]]:
        segs = GlyphSegmenter.contours(band)
        h = band.shape[0]
        glyphs: list[tuple["np.ndarray", int, int]] = []
        for x0, x1 in segs:
            glyph = band[:, x0:x1]
            rows = np.where(glyph.any(axis=1))[0]
            if len(rows) == 0:
                continue
            th = rows[-1] - rows[0] + 1
            tw = x1 - x0
            # نویسه‌های ریز (نقطه/خط‌تیره) یا پهنای نامعقول، جداکننده تلقی می‌شوند.
            if th < max(4, h * 0.28) or tw < 1 or tw > h * 2.5:
                continue
            glyphs.append((glyph, th, tw))
        if not glyphs:
            return []
        out: list[list[tuple[float, str]]] = []
        for glyph, _th, _tw in glyphs:
            grid = GlyphSegmenter.normalize(glyph)
            if grid is None:
                continue
            feat = np.concatenate(
                [GlyphSegmenter.features(glyph), GlyphSegmenter.zone(glyph)]
            )
            options = self.matcher.classify_top(grid, feat)
            if options:
                out.append(options)
        return out

    @staticmethod
    def beam(cands: list[list[tuple[float, str]]], width: int = BEAM_WIDTH,
             expand: int = BEAM_EXPAND, gap: float = BEAM_GAP) -> list[LineCandidate]:
        """جستجوی پرتوی روی گزینه‌های هر نویسه (محدود، بدون انفجار ترکیبی)."""
        beams: list[tuple[str, float]] = [("", 0.0)]
        for options in cands:
            if not options:
                continue
            best_score = options[0][0]
            new: list[tuple[str, float]] = []
            for prefix, score in beams:
                for rank, (sc, ch) in enumerate(options[:expand]):
                    if rank > 0 and sc < best_score - gap:
                        break
                    new.append((prefix + ch, score + sc))
            new.sort(key=lambda t: -t[1])
            beams = new[:width]
        return [LineCandidate(t, s) for t, s in beams]

    # ---- خواندن مستندات ----
    @staticmethod
    def dark_regions(arr: "np.ndarray", min_frac: float = 0.02,
                     max_frac: float = 0.75, max_regions: int = 4) -> list[tuple[int, int, int, int]]:
        """مستطیل‌های تیرهٔ بزرگ را پیدا می‌کند (کادر/بنر رمز).

        روش: ماسک تیره → تقلیل تفکیک‌پذیری (downsample) → مؤلفه‌های پیوستهٔ
        بزرگ → بازگرداندن کادر به ابعاد اصلی. پایین‌آوردن تفکیک‌پذیری، هم
        سرعت را زیاد می‌کند و هم شکاف‌های متن داخل کادر را می‌بندد.
        """
        h, w = arr.shape
        if h < 32 or w < 48:
            return []
        # مقیاس: حداکثر ۱۶۰ پیکسل در بُعد بزرگ‌تر
        factor = max(1, int(max(h, w) / 160))
        small = arr[::factor, ::factor]
        dark = small < 110
        sh, sw = dark.shape
        visited = np.zeros_like(dark, dtype=bool)
        regions: list[tuple[float, int, int, int, int]] = []
        # جستجوی مؤلفه‌های پیوسته با پیمایش ساده (بدون scipy)
        for y in range(sh):
            row = dark[y]
            for x in np.nonzero(row & ~visited[y])[0]:
                if visited[y, x]:
                    continue
                # BFS
                stack = [(int(y), int(x))]
                visited[y, x] = True
                xs: list[int] = []
                ys: list[int] = []
                count = 0
                while stack:
                    cy, cx = stack.pop()
                    xs.append(cx)
                    ys.append(cy)
                    count += 1
                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        ny, nx = cy + dy, cx + dx
                        if 0 <= ny < sh and 0 <= nx < sw and dark[ny, nx] and not visited[ny, nx]:
                            visited[ny, nx] = True
                            stack.append((ny, nx))
                if count < (sh * sw) * min_frac:
                    continue
                x0, x1 = min(xs), max(xs) + 1
                y0, y1 = min(ys), max(ys) + 1
                bw = x1 - x0
                bh = y1 - y0
                if bw <= 0 or bh <= 0:
                    continue
                # ناحیه باید پرتراکم باشد (کادر، نه خط‌های پراکنده)
                fill = count / float(bw * bh)
                if fill < 0.55:
                    continue
                # نسبت ابعاد معقول برای نوار رمز
                aspect = bw / bh
                if aspect < 1.2 or aspect > 20:
                    continue
                if bh / sh > 0.95:  # کل ارتفاع → احتمالاً پس‌زمینه
                    continue
                regions.append(
                    (
                        fill * aspect,
                        x0 * factor,
                        y0 * factor,
                        min(w, x1 * factor),
                        min(h, y1 * factor),
                    )
                )
        regions.sort(reverse=True)
        return [(x0, y0, x1, y1) for _s, x0, y0, x1, y1 in regions[:max_regions]]

    def _read_region(self, arr: "np.ndarray",
                     box: tuple[int, int, int, int]) -> list[list[list[tuple[float, str]]]]:
        """خواندن متن روشن داخل یک مستطیل تیره (همهٔ خطوط).

        خروجی: فهرست خطوط؛ هر خط فهرست گزینه‌های هر نویسه. (سه‌سطحی، شبیه
        خروجی ``_lines``.)
        """
        x0, y0, x1, y1 = box
        region = arr[y0:y1, x0:x1]
        if region.size == 0:
            return []
        # حاشیهٔ کادر را حذف نکنید: در تست، حذف ۳ پیکسل باعث چسبیدن نویسه‌ها
        # و از دست رفتن کل خط می‌شود. کشش کنتراست (5,95) خودش نویز لبه را
        # کم می‌کند.
        region = GlyphSegmenter.stretch(region, 5, 95)
        return self._lines(region, dark_text=False)

    def read_image_all(self, img: "Image.Image") -> list[list[list[tuple[float, str]]]]:
        """همهٔ خطوط ممکن (روی متن تیره، متن روشن، و داخل کادرهای تیره)."""
        arr = GlyphSegmenter.stretch(GlyphSegmenter.to_gray(img))
        out: list[list[list[tuple[float, str]]]] = []
        # ۱) متن تیره روی زمینهٔ روشن
        out.extend(self._lines(arr, dark_text=True))
        # ۲) متن روشن روی زمینهٔ تیره
        out.extend(self._lines(arr, dark_text=False))
        # ۳) داخل کادرهای تیره (رمز معمولاً داخل نوار/بنر تیره است)
        for box in self.dark_regions(arr):
            lines = self._read_region(arr, box)
            out.extend(lines)
        return out

    def _lines(self, arr: "np.ndarray", dark_text: bool) -> list[list[list[tuple[float, str]]]]:
        ink = GlyphSegmenter.ink(arr, dark_text)
        row_ink = ink.sum(axis=1)
        if row_ink.max() <= 0:
            return []
        thresh = max(1.0, float(row_ink.max()) * 0.06)
        lines: list[list[list[tuple[float, str]]]] = []
        start = None
        gap = 0
        for y, v in enumerate(row_ink):
            if v > thresh:
                if start is None:
                    start = y
                gap = 0
            elif start is not None:
                gap += 1
                if gap > 3:
                    band = ink[start: y - gap + 1]
                    start = None
                    if band.shape[0] >= 6:
                        cand = self.read_band(band)
                        if cand:
                            lines.append(cand)
        if start is not None:
            band = ink[start:]
            if band.shape[0] >= 6:
                cand = self.read_band(band)
                if cand:
                    lines.append(cand)
        return lines

    def read(self, img_path: str | Path, code_pattern: str | None = None,
             max_variants: int = 24) -> OcrResult:
        started = utils.now_ts()
        if not pillow_available():
            return OcrResult(self.name, [], [], [], utils.now_ts() - started,
                             "Pillow/numpy نصب نیست.")
        if self.matcher.size == 0:
            return OcrResult(self.name, [], [], [], utils.now_ts() - started,
                             "جدول آموزش موتور داخلی یافت نشد.")
        try:
            img = Image.open(img_path)
            img = ImageOps.exif_transpose(img).convert("RGB")
        except Exception as exc:
            return OcrResult(self.name, [], [], [], utils.now_ts() - started,
                             f"تصویر باز نشد: {exc}")
        if img.width < 24 or img.height < 8:
            return OcrResult(self.name, [], [], [], utils.now_ts() - started,
                             "تصویر خیلی کوچک است.")
        # همهٔ خطوط کاندید از همهٔ حالت‌ها/مقیاس‌ها جمع می‌شوند؛ سپس خطوطی که
        # رمز معتبر دارند اولویت می‌گیرند (متن‌های دیگر مثل نام محصول نه).
        candidates: list[list[LineCandidate]] = []
        scales = _scale_plan(img)
        for scale in scales:
            work = img if scale == 1.0 else img.resize(
                (max(1, int(img.width * scale)), max(1, int(img.height * scale))),
                Image.LANCZOS,
            )
            if GlyphSegmenter.to_gray(work).size > 6_000_000:
                continue
            for line_opts in self.read_image_all(work):
                variants = self.beam(line_opts)
                if variants:
                    candidates.append(variants)

        def rank(group: list[LineCandidate]) -> tuple[int, float]:
            # اولویت: وجود رمز معتبر، سپس امتیاز نرمال‌شده با طول
            has = 1 if find_codes(group[0].text, code_pattern) else 0
            norm = group[0].score / max(1, len(group[0].text))
            return (has, norm)

        candidates.sort(key=rank, reverse=True)
        best: list[LineCandidate] = candidates[0] if candidates else []
        # خطوط دیگر به‌عنوان متن مکمل برای استخراج رمز نگه داشته می‌شوند
        extra_texts = [g[0].text for g in candidates[1:12]]
        texts = [c.text for c in best] + extra_texts
        # نسخه‌های بریده: OCR ممکن است یک نویسهٔ اضافی (مثل «l» یا «I») در لبه
        # بیندازد (خط حاشیه/سایهٔ کادر). هم خودِ متن و هم نسخه‌های کوتاه‌شده
        # را بررسی می‌کنیم.
        expanded: list[str] = []
        for text in texts:
            expanded.append(text)
            if len(text) >= 9:
                expanded.append(text[1:])
                expanded.append(text[:-1])
                if len(text) >= 10:
                    expanded.append(text[1:-1])
        codes: list[str] = []
        for text in expanded:
            for c in find_codes(text, code_pattern):
                if c not in codes:
                    codes.append(c)
        lines = [TextLine(v.text, min(1.0, (v.score / max(1, len(v.text)) + 1) / 2),
                          0, 0) for v in best]
        return OcrResult(self.name, texts[:max_variants], codes[:max_variants],
                         lines, utils.now_ts() - started)


def _scale_plan(img) -> list[float]:
    base = img.height
    plan = [1.0]
    if base < 60:
        plan.insert(0, 2.0)
    if base < 32:
        plan.insert(0, 3.0)
    if img.width > 2400:
        plan = [0.75, 1.0]
    return plan


# --------------------------------------------------------------------------- #
# موتور Tesseract
# --------------------------------------------------------------------------- #


def tesseract_read(img_path: str | Path, lang: str = "eng", psm: int = 7) -> OcrResult:
    started = utils.now_ts()
    exe = shutil.which("tesseract")
    if not exe:
        return OcrResult("tesseract", [], [], [], utils.now_ts() - started,
                         "tesseract نصب نیست.")
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
        out_base = tmp.name[:-4]
    try:
        cmd = [exe, str(img_path), out_base, "-l", lang, "--psm", str(psm)]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if proc.returncode != 0:
            return OcrResult("tesseract", [], [], [], utils.now_ts() - started,
                             utils.shorten(proc.stderr, 200))
        text = Path(out_base + ".txt").read_text(encoding="utf-8", errors="replace")
        lines = []
        for i, raw in enumerate(text.splitlines()):
            raw = raw.strip()
            if raw:
                lines.append(TextLine(text=raw, score=0.8, y=i, height=0))
        codes = find_codes(text)
        return OcrResult("tesseract", [text.strip()], codes, lines,
                         utils.now_ts() - started)
    except subprocess.TimeoutExpired:
        return OcrResult("tesseract", [], [], [], utils.now_ts() - started, "زمان‌بر شد.")
    finally:
        for suffix in (".txt", ".tsv", ".hocr"):
            try:
                Path(out_base + suffix).unlink()
            except OSError:
                pass


# --------------------------------------------------------------------------- #
# رابط یکپارچه
# --------------------------------------------------------------------------- #


class OcrRunner:
    """موتور OCR مناسب را انتخاب و اجرا می‌کند؛ نتیجه را کش می‌کند."""

    def __init__(self, cfg, glyphs_path: str | Path | None = None):
        self.cfg = cfg
        self._cache: dict[str, OcrResult] = {}
        self._tesseract = bool(cfg.ocr_enabled) and tesseract_available()
        self._internal: InternalOcrEngine | None = None
        if cfg.ocr_enabled:
            matcher = GlyphMatcher(glyphs_path) if glyphs_path else GlyphMatcher()
            self._internal = InternalOcrEngine(matcher)
        self.warnings: list[str] = []

    @property
    def available(self) -> bool:
        return self._tesseract or bool(self._internal and self._internal.matcher.size)

    @property
    def engines(self) -> list[str]:
        out = []
        if self._tesseract:
            out.append("tesseract")
        if self._internal and self._internal.matcher.size:
            out.append("internal")
        return out

    def note(self, message: str) -> None:
        if message not in self.warnings:
            self.warnings.append(message)

    # ---- اجرا روی یک تصویر ----
    def run(self, img_path: str | Path, code_pattern: str | None = None,
            prefer: str = "auto", extra_lines: Iterable[str] = ()) -> OcrResult:
        key = f"{img_path}|{prefer}"
        if key in self._cache:
            return self._cache[key]
        pattern = code_pattern or self.cfg.code_regex
        results: list[OcrResult] = []

        if self._tesseract and prefer in ("auto", "tesseract"):
            pre = _preprocess_for_tesseract(img_path, self.cfg.ocr_max_pixels)
            res = tesseract_read(pre, self.cfg.ocr_lang, self.cfg.ocr_psm)
            _unlink_quiet(pre)
            results.append(res)

        if self._internal and self._internal.matcher.size and prefer in (
            "auto", "internal"
        ):
            res = self._internal.read(img_path, pattern)
            results.append(res)

        if self._internal and not self._internal.matcher.size and prefer == "auto":
            self.note("جدول آموزش موتور داخلی یافت نشد؛ فقط tesseract استفاده می‌شود.")

        merged_texts: list[str] = []
        merged_codes: list[str] = []
        merged_lines: list[TextLine] = []
        elapsed = 0.0
        engine_names: list[str] = []
        errors: list[str] = []
        for res in sorted(results, key=lambda r: -r.score):
            engine_names.append(res.engine)
            elapsed += res.elapsed
            merged_texts.extend(t for t in res.texts if t)
            merged_lines.extend(res.lines)
            for code in res.codes:
                if code not in merged_codes:
                    merged_codes.append(code)
            if res.error:
                errors.append(f"{res.engine}: {res.error}")
        for line in extra_lines:
            merged_texts.append(line)
            for code in find_codes(line, pattern):
                if code not in merged_codes:
                    merged_codes.append(code)

        # کدهای ترکیبی از تکه‌های پیاپی
        for text in list(merged_texts):
            toks = re.findall(r"[A-Za-z0-9]{3,24}", text)
            for cand in assemble_codes(toks):
                if cand not in merged_codes and plausible_code(cand):
                    merged_codes.append(cand)

        result = OcrResult(
            engine="+".join(engine_names) or "none",
            texts=merged_texts,
            codes=merged_codes[: self.cfg.ocr_max_candidates],
            lines=merged_lines,
            elapsed=elapsed,
            error="؛ ".join(errors),
        )
        self._cache[key] = result
        return result

    # ---- آماده‌سازی تصویر برای tesseract ----
    def ready(self) -> bool:
        if not self.cfg.ocr_enabled:
            return False
        return self.available


def _preprocess_for_tesseract(img_path: str | Path, max_pixels: int) -> str:
    """تصویر را برای tesseract بهینه می‌کند؛ مسیر فایل موقت برمی‌گرداند."""
    if not pillow_available():
        return str(img_path)
    try:
        img = Image.open(img_path)
        img = ImageOps.exif_transpose(img).convert("L")
        if img.width * img.height > max_pixels:
            ratio = (max_pixels / (img.width * img.height)) ** 0.5
            img = img.resize((max(1, int(img.width * ratio)), max(1, int(img.height * ratio))))
        if min(img.width, img.height) < 400:
            f = min(4, max(2, 800 // min(img.width, img.height)))
            img = img.resize((img.width * f, img.height * f), Image.LANCZOS)
        img = ImageOps.autocontrast(img)
        img = img.filter(ImageFilter.SHARPEN)
        fd = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        fd.close()
        img.save(fd.name)
        return fd.name
    except Exception:
        return str(img_path)


def _unlink_quiet(path: str | Path) -> None:
    try:
        Path(path).unlink()
    except OSError:
        pass


# --------------------------------------------------------------------------- #
# رابط عمومی
# --------------------------------------------------------------------------- #


def detect_text(img_path: str | Path, cfg, runner: OcrRunner | None = None) -> OcrResult:
    runner = runner or OcrRunner(cfg)
    return runner.run(img_path)


def engine_status(cfg) -> dict:
    """وضعیت موتورهای OCR برای `digiganj doctor`."""
    runner = OcrRunner(cfg)
    return {
        "enabled": cfg.ocr_enabled,
        "tesseract_binary": tesseract_available(),
        "pillow": pillow_available(),
        "engines": runner.engines,
        "ready": runner.available,
        "warnings": runner.warnings,
    }


def _render_ref_geometry(font, size: int, stroke: int = 0,
                         ref_text: str = "xoenam") -> tuple[float, float, float]:
    """هندسهٔ خط مرجع (بالا، خط مبنا، پایین) از رشته‌ای بدون نویسهٔ دنباله‌دار."""
    from PIL import ImageDraw

    canvas = Image.new("L", (max(20, size * (len(ref_text) + 2)), size * 3), 0)
    ImageDraw.Draw(canvas).text((size // 2, size // 2), ref_text, fill=255, font=font,
                                stroke_width=stroke, stroke_fill=255)
    mask = np.asarray(canvas, dtype=np.float32) > 96
    cols = mask.sum(axis=0)
    masks: list[np.ndarray] = []
    start = None
    for x, v in enumerate(cols):
        if v > 0 and start is None:
            start = x
        elif v == 0 and start is not None:
            rows = np.where(mask[:, start:x].any(axis=1))[0]
            if len(rows):
                masks.append(mask[rows[0]: rows[-1] + 1, start:x])
            start = None
    if start is not None:
        rows = np.where(mask[:, start:].any(axis=1))[0]
        if len(rows):
            masks.append(mask[rows[0]: rows[-1] + 1, start:])
    if not masks:
        return 0.0, float(size), float(size)
    return line_geometry(masks)


def extract_code_card(path: str | "Path", out_path: str | "Path | None" = None,
                      padding: int = 8) -> str:
    """برش ناحیهٔ کادر رمز از تصویر محصول و ذخیرهٔ بزرگ‌شده.

    چرا؟ موتور OCR گاهی یک نویسه را اشتباه می‌خواند (مثل `p`→`o`). کادر رمز
    معمولاً کوچک است و در تصویر ۶۴۰×۶۴۰ گم می‌شود. این تابع همان ناحیه را
    جدا و ۳ برابر بزرگ می‌کند تا **شما با چشم** بخوانید — سریع‌تر و قطعی‌تر
    از هر OCR‌ای.

    خروجی: مسیر فایل ذخیره‌شده (یا رشتهٔ خالی اگر ناحیه‌ای پیدا نشد).
    """
    try:
        from PIL import Image
    except ImportError:
        return ""
    src = Path(path)
    if not src.exists():
        return ""
    try:
        img = Image.open(src)
        img.load()
    except Exception:
        return ""
    gray = np.asarray(img.convert("L"), dtype=np.float32)
    regions = InternalOcrEngine.dark_regions(gray, min_frac=0.02, max_frac=0.75)
    if not regions:
        # شاید کادر روشن با متن تیره است؛ کل تصویر را نگه می‌داریم
        box = (0, 0, img.width, img.height)
    else:
        # بزرگ‌ترین ناحیهٔ با نسبت ابعاد افقی
        def area(r):
            return (r[2] - r[0]) * (r[3] - r[1])

        box = max(regions, key=area)
    x0 = max(0, box[0] - padding)
    y0 = max(0, box[1] - padding)
    x1 = min(img.width, box[2] + padding)
    y1 = min(img.height, box[3] + padding)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return ""
    crop = img.crop((x0, y0, x1, y1))
    scale = 3
    crop = crop.resize((crop.width * scale, crop.height * scale), Image.LANCZOS)
    target = Path(out_path) if out_path else src.with_name(src.stem + "-code-card.png")
    target.parent.mkdir(parents=True, exist_ok=True)
    crop.save(target)
    return str(target)


def train_glyph_table(out_path: str | Path, fonts_dir: str | Path | None = None,
                      scales: Iterable[int] = (16, 22, 30, 44),
                      thickness: Iterable[bool] = (False, True)) -> dict:
    """ساخت جدول آموزش برای موتور داخلی.

    برای هر فونت TrueType (اگر مسیر داده شود) و هر اندازه/ضخامت، هر نویسه از
    ``GLYPHS`` را رندر، نرمال و ذخیره می‌کند. فونت‌های مربی: ``fonts_dir/**/*.ttf``.
    اگر فونتی داده نشود، فونت پیش‌فرض PIL (bitmap) استفاده می‌شود.
    """
    if not pillow_available():
        raise RuntimeError("برای آموزش، Pillow و numpy لازم است.")
    from PIL import ImageDraw, ImageFont

    matcher = GlyphMatcher()
    fonts: list[tuple[str, str]] = []
    if fonts_dir:
        for pattern in ("**/*.ttf", "**/*.otf"):
            for fp in sorted(Path(fonts_dir).glob(pattern)):
                fonts.append((fp.name, str(fp)))
    if not fonts:
        fonts.append(("default", ""))
    count = 0
    for name, path in fonts:
        for size in scales:
            for bold in thickness:
                try:
                    font = ImageFont.truetype(path, size) if path else ImageFont.load_default()
                except Exception:
                    continue
                for ch in GLYPHS:
                    # بوم بزرگ با حاشیهٔ کافی تا دنبالهٔ حروف (j، g، y) هرگز بریده نشود
                    canvas = Image.new("L", (size * 3, size * 4), 0)
                    draw = ImageDraw.Draw(canvas)
                    stroke = 1 if bold else 0
                    draw.text((size // 2, size // 2), ch, fill=255, font=font,
                              stroke_width=stroke, stroke_fill=255)
                    mask = np.asarray(canvas, dtype=np.float32) > 96
                    grid = GlyphSegmenter.normalize(mask)
                    if grid is None:
                        continue
                    feat = np.concatenate(
                        [GlyphSegmenter.features(mask), GlyphSegmenter.zone(mask)]
                    )
                    matcher.add(ch, grid, feat)
                    count += 1
    matcher.meta["version"] = "1"
    matcher.meta["fonts"] = ",".join(n for n, _ in fonts)[:500]
    matcher.save(out_path)
    return {"glyphs": count, "classes": len(matcher.table), "path": str(out_path)}
