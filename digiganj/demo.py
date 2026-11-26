"""سرور دموی محلی: شبیه‌سازی کامل دیجی‌کالا برای آزمودن بدون تماس واقعی.

چرا؟ برای اینکه بتوانید کل جریان (فهرست، گالری، تحلیل زمان، OCR، هشدار،
گزارش) را روی یک «دیجی‌کالای کوچک» تمرین کنید — بدون فشار به سرویس واقعی،
و به‌گونه‌ای که رفتار «انتشار تصویر رمز در لحظهٔ شروع» بازتولید شود.

سرور استاندارد کتابخانه‌ای پایتون است؛ فقط ``http.server`` و ``PIL`` لازم
دارد. با ``digiganj demo`` اجرا می‌شود.

پاسخ‌های شبیه‌سازی‌شده
---------------------
* ``GET /ajax/treasure-hunt/products/?pageno=N`` → JSON با click_impression و
  trackerData.pages (ساختار ۱۴۰۰).
* ``GET /v1/product/dkp-{id}/`` → JSON با ``data.product.images.image_list``.
* ``GET /product/dkp-{id}/`` → HTML با گالری (برای آزمودن مسیر HTML).
* ``GET /digikala-products/{file}?x-oss-process=...`` → تصویر JPEG/PNG
  (با پارامترهای پردازش، شبیه CDN واقعی).
* زمان‌بندی: تصویر «گنج» فقط پس از ``--reveal-after`` ثانیه در گالری یک
  محصول خاص ظاهر می‌شود و نام فایلش زمان همان لحظه را دارد.
"""

from __future__ import annotations

import argparse
import io
import json
import threading
import time
import urllib.parse
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_OK = True
except Exception:
    PIL_OK = False


# --------------------------------------------------------------------------- #
# دادهٔ مصنوعی
# --------------------------------------------------------------------------- #

FAKE_NAMES = [
    "گوشی موبایل سامسونگ مدل Galaxy S24 Ultra",
    "کنسول بازی سونی مدل PlayStation 5 Slim",
    "هدفون بی‌سیم اپل مدل AirPods 3",
    "ساعت هوشمند اپل مدل Watch Series 9",
    "تبلت سامسونگ مدل Galaxy Tab S9",
    "هارد اکسترنال ای‌دیتا مدل HD770G",
    "پاوربانک شیائومی مدل 20000mAh",
    "لپ‌تاپ ایسوس مدل VivoBook 15",
    "مانیتور ال‌جی مدل 27MP400",
    "کیبورد مکانیکی ریزر مدل BlackWidow",
    "ماوس لاجیتک مدل MX Master 3S",
    "اسپیکر جی‌بی‌ال مدل Flip 6",
    "دوربین کانن مدل EOS R50",
    "چاپگر اچ‌پی مدل LaserJet M141a",
    "میکروفون هایپرایکس مدل QuadCast",
]


def _gallery_images(pid: int, n: int = 5, base_ts: int = 1680000000) -> list[str]:
    out = []
    for i in range(n):
        # هش شبه‌SHA1 + زمان (الگوی واقعی CDN دیجی‌کالا)
        h = (f"{pid:08x}{i:04x}" * 4)[:40]
        out.append(
            f"/digikala-products/{h}_{base_ts + i * 37}.jpg"
            "?x-oss-process=image/resize,m_lfit,h_800,w_800/quality,q_60"
        )
    return out


@dataclass
class DemoState:
    products: int = 60
    pages: int = 3
    per_page: int = 25
    treasure_product: int = 1007
    treasure_code: str = "fvKA4jd9DP"
    reveal_after: float = 12.0
    started_at: float = field(default_factory=time.time)
    revealed: bool = False
    treasure_ts: int = 0

    @property
    def page_count(self) -> int:
        return max(1, (self.products + self.per_page - 1) // self.per_page)

    def elapsed(self) -> float:
        return time.time() - self.started_at

    def refresh(self) -> None:
        if not self.revealed and self.elapsed() >= self.reveal_after:
            self.revealed = True
            self.treasure_ts = int(time.time())
            print(f"[demo] تصویر گنج در ثانیهٔ {self.elapsed():.1f} منتشر شد "
                  f"(timestamp={self.treasure_ts}, رمز={self.treasure_code})")

    def product(self, pid: int) -> dict[str, Any]:
        idx = max(0, (pid - 1001) % self.products) if pid >= 1001 else 0
        return {
            "id": pid,
            "name": FAKE_NAMES[idx % len(FAKE_NAMES)],
            "uri": f"/product/dkp-{pid}/demo-product-{pid}/",
            "price": {"selling_price": 1_000_000 + idx * 150_000},
            "position": idx + 1,
        }

    def gallery(self, pid: int) -> list[str]:
        imgs = _gallery_images(pid, 5)
        self.refresh()
        if self.revealed and pid == self.treasure_product:
            # تصویر رمز با زمان انتشار واقعی
            imgs.insert(
                0,
                f"/digikala-products/treasure{pid}_{self.treasure_ts}.jpg"
                "?x-oss-process=image/resize,m_lfit,h_1600,w_1600/quality,q_80"
                "/watermark,image_ZGstdy8xLnBuZw==,t_90,g_nw,x_15,y_15",
            )
        return imgs


# --------------------------------------------------------------------------- #
# رندر تصویر رمز
# --------------------------------------------------------------------------- #


def render_code_image(code: str, width: int = 640, height: int = 640) -> bytes:
    if not PIL_OK:
        return b"\x89PNG\r\n\x1a\n"  # هدر خالی PNG
    img = Image.new("RGB", (width, height), (250, 248, 243))
    d = ImageDraw.Draw(img)
    # پس‌زمینهٔ ساده
    for i in range(0, height, 40):
        d.line([(0, i), (width, i)], fill=(238, 235, 228), width=1)
    # کادر و متن رمز
    box = (width // 8, height // 3, width * 7 // 8, height * 2 // 3)
    d.rounded_rectangle(box, radius=18, fill=(18, 18, 24))
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
            size=width // 14,
        )
    except Exception:
        font = ImageFont.load_default()
    text = code
    bbox = d.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    d.text(((width - tw) / 2 - bbox[0], (height - th) / 2 - bbox[1]),
           text, fill=(250, 250, 250), font=font)
    d.text((width // 8, height // 3 - 40), "شکار گنج", fill=(30, 30, 40))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=88)
    return buf.getvalue()


def render_plain_image(pid: int, width: int = 480, height: int = 480) -> bytes:
    if not PIL_OK:
        return b"\x89PNG\r\n\x1a\n"
    import random

    rnd = random.Random(pid)
    base = (rnd.randint(20, 240), rnd.randint(20, 240), rnd.randint(20, 240))
    img = Image.new("RGB", (width, height), base)
    d = ImageDraw.Draw(img)
    for _ in range(40):
        x0, y0 = rnd.randint(0, width), rnd.randint(0, height)
        x1, y1 = x0 + rnd.randint(-200, 200), y0 + rnd.randint(-200, 200)
        d.line([(x0, y0), (x1, y1)], fill=(
            rnd.randint(0, 255), rnd.randint(0, 255), rnd.randint(0, 255)), width=rnd.randint(1, 6))
    d.text((20, 20), f"dkp-{pid}", fill=(255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    return buf.getvalue()


# --------------------------------------------------------------------------- #
# Handler
# --------------------------------------------------------------------------- #


def make_handler(state: DemoState):
    class Handler(BaseHTTPRequestHandler):
        server_version = "DigiganjDemo/1.0"

        def log_message(self, fmt, *args):  # کم‌صدا
            if self.path.startswith("/ajax") or self.path.startswith("/v1"):
                return
            super().log_message(fmt, *args)

        def _send(self, code: int, body: bytes, ctype: str = "application/json"):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            try:
                self.wfile.write(body)
            except BrokenPipeError:
                pass

        def _json(self, obj: Any, code: int = 200):
            data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self._send(code, data, "application/json; charset=utf-8")

        def do_GET(self):  # noqa: N802
            parsed = urllib.parse.urlparse(self.path)
            path = parsed.path
            qs = urllib.parse.parse_qs(parsed.query)

            # فهرست محصولات چالش
            if path.rstrip("/") == "/ajax/treasure-hunt/products":
                page = int((qs.get("pageno") or ["1"])[0])
                start = (page - 1) * state.per_page + 1001
                end = min(start + state.per_page, 1001 + state.products)
                items = [state.product(pid) for pid in range(start, end)]
                self._json({
                    "status": True,
                    "data": {
                        "click_impression": items,
                        "trackerData": {"pages": state.page_count},
                        "data_layer": "",
                        "products": "{}",
                        "filters": "[]",
                    },
                })
                return

            # API محصول (گالری)
            if path.startswith("/v1/product/dkp-"):
                pid = int(path.split("/v1/product/dkp-")[1].split("/")[0])
                gallery = state.gallery(pid)
                self._json({
                    "status": 200,
                    "data": {
                        "product": {
                            "id": pid,
                            "title_fa": state.product(pid)["name"],
                            "uri": state.product(pid)["uri"],
                            "images": {"main": gallery[0], "image_list": gallery},
                        }
                    },
                })
                return

            # صفحهٔ HTML محصول
            if path.startswith("/product/dkp-"):
                pid = int(path.split("/product/dkp-")[1].split("/")[0])
                gallery = state.gallery(pid)
                payload = json.dumps({"product": {"id": pid, "images": {
                    "main": gallery[0], "image_list": gallery}}}, ensure_ascii=False)
                html = (
                    "<!doctype html><html><head><title>demo</title></head><body>"
                    f"<h1>{state.product(pid)['name']}</h1>"
                    f'<script id="__NEXT_DATA__" type="application/json">{payload}</script>'
                    "</body></html>"
                )
                self._send(200, html.encode("utf-8"), "text/html; charset=utf-8")
                return

            # تصاویر CDN
            if path.startswith("/digikala-products/"):
                name = path.rsplit("/", 1)[-1]
                if name.startswith("treasure"):
                    data = render_code_image(state.treasure_code)
                else:
                    try:
                        pid = int(name.split("_")[0][:8], 16) or 1
                    except ValueError:
                        pid = 1
                    data = render_plain_image(pid)
                self._send(200, data, "image/jpeg")
                return

            if path in ("/", "/index.html"):
                self._send(200, b"<h1>digiganj demo</h1>", "text/html; charset=utf-8")
                return
            self._send(404, b'{"error":"not found"}')

    return Handler


class DemoServer:
    def __init__(self, port: int = 8099, **kwargs):
        self.state = DemoState(**kwargs)
        self.httpd = ThreadingHTTPServer(("0.0.0.0", port),
                                         make_handler(self.state))
        self.thread: threading.Thread | None = None

    @property
    def base_url(self) -> str:
        port = self.httpd.server_address[1]
        # 0.0.0.0 فقط برای bind است؛ آدرس قابل استفاده برای کلاینت 127.0.0.1 است.
        return f"http://127.0.0.1:{port}"

    def start(self) -> "DemoServer":
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        return self

    def stop(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="سرور دموی شبیه‌ساز دیجی‌کالا")
    ap.add_argument("--port", type=int, default=8099)
    ap.add_argument("--products", type=int, default=60)
    ap.add_argument("--treasure-product", type=int, default=1007,
                    help="شناسهٔ محصولی که تصویر رمز در آن منتشر می‌شود (پیش‌فرض: 1007)")
    ap.add_argument("--reveal-after", type=float, default=12.0)
    ap.add_argument("--code", default="fvKA4jd9DP")
    args = ap.parse_args(argv)
    srv = DemoServer(
        port=args.port,
        products=args.products,
        treasure_product=args.treasure_product,
        reveal_after=args.reveal_after,
        treasure_code=args.code,
    )
    print(f"سرور دمو روی {srv.base_url} بالا آمد.")
    print(f"محصول گنج: dkp-{args.treasure_product} — پس از {args.reveal_after:.0f} ثانیه منتشر می‌شود.")
    try:
        srv.httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nخداحافظ.")
    finally:
        srv.httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
