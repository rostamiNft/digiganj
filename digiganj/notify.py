"""هشدار: تلگرام، وبهوک و صدای ترمینال.

هدف: وقتی رمز پیدا شد، بدانید — هر جا که هستید. همهٔ پیام‌ها فارسی‌اند.
هیچ داده‌ای بی‌اجازه به سرور خارجی نمی‌رود؛ فقط اگر خودتان توکن/آدرس بدهید.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from typing import Any

from . import utils
from .console import Console


@dataclass
class Notification:
    title: str
    body: str
    image_path: str = ""
    image_url: str = ""
    product_url: str = ""


def _bell() -> None:
    try:
        sys.stdout.write("\a")
        sys.stdout.flush()
    except Exception:
        pass


def build_alert_text(alert: dict[str, Any], challenge=None, product_url: str = "") -> str:
    """متن فارسی هشدار."""
    lines = ["🎁 «شکار گنج» — نشانهٔ احتمالی رمز پیدا شد!", ""]
    name = alert.get("product_name") or ""
    pid = alert.get("product_id", "")
    lines.append(f"• محصول: {name or '—'} (dkp-{pid})")
    if product_url:
        lines.append(f"• پیوند: {product_url}")
    score = alert.get("score", 0)
    lines.append(f"• امتیاز اطمینان: {utils.fa_num(f'{float(score):.0f}')} از ۱۰۰")
    stamp = alert.get("stamp_ts")
    if stamp:
        lines.append(f"• زمان فایل: {utils.jalali_pretty(float(stamp))}")
        if challenge is not None:
            delta = max(0.0, float(stamp) - challenge.start_ts)
            lines.append(f"• فاصله از شروع چالش: {utils.human_duration(delta)}")
    codes = alert.get("codes") or alert.get("ocr_codes") or ""
    if codes:
        lines.append(f"• رمزهای خوانده‌شده: {codes}")
    lines.append(f"• زمان هشدار: {utils.jalali_pretty(utils.now_ts())}")
    if alert.get("url"):
        lines.append(f"• تصویر: {alert['url']}")
    lines.append("")
    lines.append("یادآوری: این فقط «نشانه» است؛ پیش از هر اقدامی، تصویر را با چشم بررسی کنید.")
    return "\n".join(lines)


class Notifier:
    """ارسال هشدار به کانال‌های تنظیم‌شده."""

    def __init__(self, cfg, logger=None, console: Console | None = None):
        self.cfg = cfg
        self.logger = logger
        self.console = console or Console()
        self._tg_ok = bool(cfg.telegram_bot_token and cfg.telegram_chat_id)

    # ------------------------------------------------------------------ #
    # کانال‌ها
    # ------------------------------------------------------------------ #
    def _send_telegram(self, note: Notification) -> bool:
        if not self._tg_ok:
            return False
        try:
            import httpx
        except Exception:
            if self.logger:
                self.logger.debug("httpx برای ارسال تلگرام در دسترس نیست.")
            return False
        token = self.cfg.telegram_bot_token
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        text = f"*{note.title}*\n\n{note.body}"
        try:
            with httpx.Client(timeout=10.0) as client:
                r = client.post(
                    url,
                    data={
                        "chat_id": self.cfg.telegram_chat_id,
                        "text": text,
                        "parse_mode": "Markdown",
                        "disable_web_page_preview": "true",
                    },
                )
                return bool(r.status_code == 200)
        except Exception as exc:
            if self.logger:
                self.logger.warn(f"ارسال تلگرام ناموفق: {utils.shorten(exc, 120)}")
            return False

    def _send_webhook(self, note: Notification) -> bool:
        if not self.cfg.webhook_url:
            return False
        try:
            import httpx
        except Exception:
            return False
        payload = {
            "title": note.title,
            "body": note.body,
            "image_url": note.image_url,
            "product_url": note.product_url,
            "sent_at": utils.iso_utc(utils.now_ts()),
        }
        try:
            with httpx.Client(timeout=10.0) as client:
                r = client.post(self.cfg.webhook_url, content=json.dumps(payload).encode(), headers={
                    "Content-Type": "application/json; charset=utf-8",
                })
                return 200 <= r.status_code < 300
        except Exception as exc:
            if self.logger:
                self.logger.warn(f"وبهوک ناموفق: {utils.shorten(exc, 120)}")
            return False

    def _desktop(self, note: Notification) -> None:
        """اعلان میزکار اگر ابزاری نصب باشد (notify-send / osascript)."""
        import shutil
        import subprocess

        try:
            if shutil.which("notify-send"):
                subprocess.run(
                    ["notify-send", note.title, note.body[:500]],
                    capture_output=True,
                    timeout=3,
                )
            elif shutil.which("osascript"):
                script = f'display notification "{note.body[:200]}" with title "{note.title}"'
                subprocess.run(["osascript", "-e", script], capture_output=True, timeout=3)
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    # رابط
    # ------------------------------------------------------------------ #
    def channels(self) -> list[str]:
        out = ["کنسول"]
        if self._tg_ok:
            out.append("تلگرام")
        if self.cfg.webhook_url:
            out.append("وبهوک")
        return out

    def send(self, note: Notification, bell: bool = True) -> dict[str, bool]:
        result = {"console": True, "telegram": False, "webhook": False}
        self.console.write("")
        self.console.write(self.console.c("━" * 60, "\033[33m"))
        self.console.write(self.console.c(note.title, "\033[1;33m"))
        self.console.write("")
        for line in note.body.splitlines():
            self.console.write("  " + line)
        self.console.write(self.console.c("━" * 60, "\033[33m"))
        if bell:
            _bell()
        if self._tg_ok:
            result["telegram"] = self._send_telegram(note)
        if self.cfg.webhook_url:
            result["webhook"] = self._send_webhook(note)
        if self._tg_ok or self.cfg.webhook_url:
            self._desktop(note)
        return result

    def alert(self, alert: dict[str, Any], challenge=None, product_url: str = "") -> dict[str, bool]:
        note = Notification(
            title="🎁 نشانهٔ رمز شکار گنج",
            body=build_alert_text(alert, challenge, product_url),
            image_url=alert.get("url", ""),
            product_url=product_url,
        )
        return self.send(note)

    def test(self) -> dict[str, bool]:
        note = Notification(
            title="پیام آزمایشی digiganj",
            body=(
                "این یک پیام آزمایشی است.\n"
                f"زمان: {utils.jalali_pretty(utils.now_ts())}\n"
                f"کانال‌ها: {', '.join(self.channels())}"
            ),
        )
        return self.send(note, bell=False)
