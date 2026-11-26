"""خروجی کنسول: رنگ، جدول و متن راست‌به‌چپ.

برای اینکه لاگ‌ها در ترمینال‌های مختلف و در فایل هم خوانا بمانند:
- اگر خروجی به ترمینال وصل باشد و NO_COLOR تنظیم نشده باشد، رنگ اضافه می‌شود.
- همهٔ پیام‌ها فارسی نوشته می‌شوند؛ اعداد با ارقام فارسی نمایش داده می‌شوند.
"""

from __future__ import annotations

import os
import shutil
import sys
import unicodedata
from typing import Iterable, Sequence

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
MAGENTA = "\033[35m"
CYAN = "\033[36m"
GRAY = "\033[90m"


def _supports_color(stream) -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    try:
        return bool(stream.isatty())
    except Exception:
        return False


def _width(text: str) -> int:
    """عرض نمایشی متن (حروف فارسی/عربی و CJK دو ستونی حساب نمی‌شوند)."""
    total = 0
    for ch in text:
        if unicodedata.combining(ch):
            continue
        if unicodedata.east_asian_width(ch) in ("W", "F"):
            total += 2
        elif ch in "\u200c\u200d\u200e\u200f\u2066\u2067\u2068\u2069":
            continue
        else:
            total += 1
    return total


def pad(text: str, width: int, align: str = "right") -> str:
    gap = max(0, width - _width(text))
    if align == "left":
        return text + " " * gap
    if align == "center":
        left = gap // 2
        return " " * left + text + " " * (gap - left)
    return " " * gap + text


def truncate_display(text: str, width: int) -> str:
    if _width(text) <= width:
        return text
    out = ""
    w = 0
    for ch in text:
        cw = _width(ch)
        if w + cw > width - 1:
            break
        out += ch
        w += cw
    return out + "…"


class Console:
    def __init__(self, stream=None, color: bool | None = None, verbose: bool = True):
        self.stream = stream or sys.stdout
        self.color = _supports_color(self.stream) if color is None else color
        self.verbose = verbose

    # ---- رنگ ----
    def c(self, text: str, color: str, bold: bool = False) -> str:
        if not self.color:
            return text
        prefix = (BOLD if bold else "") + color
        return f"{prefix}{text}{RESET}"

    def write(self, text: str = "") -> None:
        self.stream.write(text + "\n")
        try:
            self.stream.flush()
        except Exception:
            pass

    # ---- پیام‌ها ----
    def say(self, text: str) -> None:
        if self.verbose:
            self.write(text)

    def good(self, text: str) -> None:
        self.write(self.c("✔ ", GREEN) + text)

    def warn(self, text: str) -> None:
        self.write(self.c("⚠ ", YELLOW) + text)

    def err(self, text: str) -> None:
        self.write(self.c("✖ ", RED) + text)

    def info(self, text: str) -> None:
        self.write(self.c("• ", CYAN) + text)

    def hint(self, text: str) -> None:
        self.write(self.c("… ", GRAY) + self.c(text, DIM))

    def title(self, text: str) -> None:
        self.rule()
        self.write(self.c(text, BOLD + CYAN))
        self.rule()

    def rule(self, char: str = "─") -> None:
        width = shutil.get_terminal_size((80, 20)).columns
        self.write(self.c(char * max(20, min(width, 100)), GRAY))

    def banner(self, title: str, subtitle: str = "") -> None:
        self.rule("━")
        self.write(self.c("  " + title, BOLD))
        if subtitle:
            self.write(self.c("  " + subtitle, GRAY))
        self.rule("━")

    # ---- جدول ----
    def table(
        self,
        headers: Sequence[str],
        rows: Iterable[Sequence[str]],
        aligns: Sequence[str] | None = None,
    ) -> None:
        rows = [[("" if cell is None else str(cell)) for cell in row] for row in rows]
        if not rows:
            self.hint("(ردیفی برای نمایش نیست)")
            return
        cols = len(headers)
        aligns = list(aligns or ["right"] * cols)
        widths = [_width(h) for h in headers]
        for row in rows:
            for i in range(cols):
                widths[i] = max(widths[i], min(_width(row[i]), 46))
        term = shutil.get_terminal_size((100, 20)).columns
        while sum(widths) + 2 * (cols - 1) > term and max(widths) > 8:
            widths[widths.index(max(widths))] -= 1
        head = "  ".join(
            self.c(pad(h, widths[i], aligns[i]), BOLD) for i, h in enumerate(headers)
        )
        self.write(head)
        self.write(self.c("  ".join("─" * w for w in widths), GRAY))
        for row in rows:
            cells = [
                pad(truncate_display(row[i], widths[i]), widths[i], aligns[i])
                for i in range(cols)
            ]
            self.write("  ".join(cells))

    def kv(self, pairs: Sequence[tuple[str, str]], sep: str = "  ") -> None:
        """جفت‌های کلید/مقدار.

        اگر مجموع طول ردیف از عرض ترمینال بیشتر شود، هر جفت در یک خط می‌آید
        (چون مقادیر بلند مثل مسیر فایل یا فهرست رمزها در یک خط جا نمی‌شوند
        و جدول را می‌شکنند).
        """
        width = max((_width(k) for k, _ in pairs), default=0)
        one_line = sep.join(f"{pad(k, width)} = {v}" for k, v in pairs)
        term_w = shutil.get_terminal_size((100, 20)).columns
        if _width(one_line) <= term_w - 2:
            self.write(
                sep.join(
                    f"{self.c(pad(k, width), GRAY)} {self.c('=', DIM)} {v}" for k, v in pairs
                )
            )
            return
        for k, v in pairs:
            self.write(f"  {self.c(pad(k, width), GRAY)} {self.c('=', DIM)} {v}")


class Logger:
    """لاگر فایل + کنسول با مُهر زمان جلالی.

    سطح‌ها: DEBUG < INFO < WARN < ERROR. سطح کنسول جدا از سطح فایل است تا
    فایل کامل بماند ولی کنسول شلوغ نشود.
    """

    def __init__(
        self,
        path: str | None = None,
        level: str = "INFO",
        console: Console | None = None,
        console_level: str | None = None,
    ):
        self.path = path
        self.console = console or Console()
        self.level = self._norm(level)
        self.console_level = self._norm(console_level or level)
        self._file = None
        if path:
            from . import utils

            utils.ensure_dir(os.path.dirname(path) or ".")
            self._file = open(path, "a", encoding="utf-8")

    @staticmethod
    def _norm(level: str) -> int:
        return {"DEBUG": 10, "INFO": 20, "WARN": 30, "WARNING": 30, "ERROR": 40}.get(
            str(level).upper(), 20
        )

    def close(self) -> None:
        if self._file:
            try:
                self._file.close()
            except Exception:
                pass
            self._file = None

    def _write(self, lvl: str, msg: str, to_console: bool = True) -> None:
        from . import utils

        ts = utils.now_ts()
        stamp = utils.jalali_datetime(ts, fa=False)
        if self._file:
            try:
                self._file.write(f"{stamp} [{lvl}] {msg}\n")
                self._file.flush()
            except Exception:
                pass
        if not to_console:
            return
        value = self._norm(lvl)
        if value < self.console_level:
            return
        prefix = {
            "DEBUG": self.console.c("·", GRAY),
            "INFO": self.console.c("•", CYAN),
            "WARN": self.console.c("⚠", YELLOW),
            "ERROR": self.console.c("✖", RED),
            "DONE": self.console.c("✔", GREEN),
        }.get(lvl, "•")
        line = f"{self.console.c(stamp, GRAY)} {prefix} {msg}"
        self.console.write(line)

    # ---- API ----
    def debug(self, msg: str) -> None:
        if self.level <= 10:
            self._write("DEBUG", msg)

    def info(self, msg: str) -> None:
        if self.level <= 20:
            self._write("INFO", msg)

    def good(self, msg: str) -> None:
        if self.level <= 20:
            self._write("DONE", msg)

    def warn(self, msg: str) -> None:
        if self.level <= 30:
            self._write("WARN", msg)

    def error(self, msg: str) -> None:
        self._write("ERROR", msg)

    def event(self, name: str, msg: str) -> None:
        """رویداد مهم: هم در فایل، هم در کنسول با برچسب."""
        self._write("DONE", f"[{name}] {msg}")


def make_logger(cfg, level: str | None = None, console: Console | None = None) -> Logger:
    return Logger(
        path=cfg.log_file,
        level=level or cfg.log_level,
        console=console,
        console_level=level or "INFO",
    )
