"""تعامل با دیجی‌کالا: فهرست محصولات و گالری تصاویر.

منابع داده (به‌ترتیب تلاش)
-------------------------
۱) فهرست:  ``/ajax/treasure-hunt/products/?pageno=N`` (همان نقطۀ دسترسی‌ای که
   ایونت استفاده می‌کند؛ پاسخ JSON شامل ``data.click_impression`` و
   ``data.trackerData.pages`` است). اگر در دسترس نبود، صفحهٔ HTML فهرست
   به‌عنوان جایگزین تجزیه می‌شود.
۲) گالری:  ``/v1/product/dkp-{id}/`` روی دامنهٔ API (پاسخ JSON کامل)؛ در صورت
   شکست، صفحهٔ HTML محصول و استخراج ``image_list`` از دادهٔ تزریق‌شده.

همهٔ فراخوانی‌ها از طریق :class:`~digiganj.net.HttpClient` انجام می‌شود تا
تلاش مجدد/محدودسازی نرخ/کش یک‌جا اعمال شود.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Iterable
from urllib.parse import quote

from . import models, utils
from .console import Logger
from .net import BlockedError, HttpClient, NetError, NotFoundError, Response


# --------------------------------------------------------------------------- #
# تجزیهٔ پاسخ فهرست
# --------------------------------------------------------------------------- #


@dataclass
class ListingPage:
    page: int
    pages: int
    products: list[models.Product] = field(default_factory=list)
    total: int = 0
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def is_last(self) -> bool:
        return self.page >= self.pages


def parse_listing(payload: Any, page: int) -> ListingPage:
    """تجزیهٔ JSON فهرست شکار گنج.

    ساختار شناخته‌شده::

        {"status": true,
         "data": {"click_impression": [{"id": …, "name": …, "product_url": …,
                                       "price": …, "position": …}],
                  "trackerData": {"pages": 46, …}}}
    """
    if not isinstance(payload, dict):
        raise ValueError("پاسخ فهرست JSON معتبر نیست.")
    data = payload.get("data") or {}
    if not isinstance(data, dict):
        raise ValueError("کلید data در پاسخ فهرست معتبر نیست.")
    items = (
        data.get("click_impression")
        or data.get("products")
        or data.get("items")
        or payload.get("products")
        or []
    )
    if isinstance(items, str):
        try:
            items = json.loads(items)
        except json.JSONDecodeError:
            items = []
    pages = 0
    tracker = data.get("trackerData") or data.get("tracker_data") or {}
    if isinstance(tracker, dict):
        try:
            pages = int(tracker.get("pages") or tracker.get("total_pages") or 0)
        except (TypeError, ValueError):
            pages = 0
    for key in ("pages", "page_count", "total_pages"):
        if not pages and isinstance(data.get(key), (int, str)):
            try:
                pages = int(data[key])
            except (TypeError, ValueError):
                pages = 0
    products: list[models.Product] = []
    for rank, obj in enumerate(items, start=1):
        if not isinstance(obj, dict):
            continue
        prod = models.Product.from_api(obj, page=page, rank=rank)
        if prod.id:
            products.append(prod)
    if not pages:
        pages = page if products else 0
    return ListingPage(page=page, pages=pages, products=products,
                       total=len(products), raw=data if isinstance(data, dict) else {})


# --------------------------------------------------------------------------- #
# تجزیهٔ گالری
# --------------------------------------------------------------------------- #

_IMG_URL_RE = re.compile(
    r"https?://[^\s\"'<>\\]+?\.(?:jpe?g|png|webp|avif|gif)(?:\?[^\s\"'<>\\]*)?",
    re.IGNORECASE,
)
_IMAGE_LIST_RE = re.compile(r'"image_list"\s*:\s*\[(.*?)\]', re.S)
_NEXT_DATA_RE = re.compile(
    r'<script[^>]+id="__NEXT_DATA__"[^>]*>(.*?)</script>', re.S | re.IGNORECASE
)


def _balanced_json_at(text: str, start: int) -> str | None:
    """از موقعیت ``start`` (که باید ``{`` یا ``[`` باشد) بلوک JSON متوازن را برمی‌گرداند.

    شمارش آکولاد با احترام به رشته‌های درون‌گزاره‌ای انجام می‌شود تا مقادیر
    حاوی ``{``/``}`` شمارش را خراب نکنند.
    """
    if start >= len(text) or text[start] not in "{[":
        return None
    open_ch = text[start]
    close_ch = "}" if open_ch == "{" else "]"
    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _images_blocks(html: str) -> list[dict[str, Any]]:
    """همهٔ بلوک‌های JSON که کلید ``images`` دارند را پیدا و تجزیه می‌کند."""
    found: list[dict[str, Any]] = []
    for m in re.finditer(r'"images"\s*:\s*', html):
        idx = m.end()
        if idx >= len(html) or html[idx] not in "{[":
            continue
        block = _balanced_json_at(html, idx)
        if not block:
            continue
        try:
            data = json.loads(block)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict) and (
            "image_list" in data or "main" in data or "images" in data
        ):
            found.append(data)
        elif isinstance(data, list):
            found.append({"image_list": data})
    return found


def _clean_urls(urls: Iterable[Any], base_url: str = "") -> list[str]:
    """پاک‌سازی و یکدست‌سازی آدرس تصاویر.

    مسیرهای نسبی (که در HTML/CSS دیده می‌شوند) در صورت داشتن ``base_url``
    به آدرس مطلق تبدیل می‌شوند؛ در غیر این صورت کنار گذاشته می‌شوند.
    """
    out: list[str] = []
    seen: set[str] = set()
    root = (base_url or "").rstrip("/")
    for u in urls:
        if not isinstance(u, str):
            continue
        u = u.strip().replace("\\/", "/")
        if not u:
            continue
        if u.startswith("//"):
            u = "https:" + u
        elif not u.startswith("http"):
            if u.startswith("/") and root:
                u = root + u
            else:
                continue
        if not re.search(r"\.(jpe?g|png|webp|avif|gif)(\?|$)", u, re.IGNORECASE):
            continue
        base = models.normalize_image_url(u)
        if base in seen:
            continue
        seen.add(base)
        out.append(u)
    return out


def _dig(d: Any, *keys: str) -> Any:
    for k in keys:
        if isinstance(d, dict) and k in d:
            return d[k]
    return None


def parse_listing_html(html: str, page: int = 1, base_url: str = "") -> list[models.Product]:
    """تجزیهٔ HTML فهرست (مسیر جایگزین اگر API جواب نداد).

    الگوی واقعی: ``div.c-listing a.c-product-box__img`` با ``href`` شامل
    ``/product/dkp-<id>/``. تصویر ممکن است در ``src`` یا ``data-src`` باشد
    و آدرس پروتکل‌نسبی (``//cdn…``) داشته باشد.
    """
    return DigikalaClient._parse_listing_html(html, page)


def extract_gallery_from_json(payload: Any, base_url: str = "") -> list[str]:
    """استخراج لیست تصاویر گالری از ساختار JSON محصول."""
    if not isinstance(payload, dict):
        return []
    cand = payload.get("data") or payload
    product = _dig(cand, "product") or {}
    images = None
    for holder in (product, cand, payload):
        if not isinstance(holder, dict):
            continue
        images = holder.get("images") or holder.get("image") or holder.get("gallery")
        if images:
            break
    urls: list[Any] = []
    if isinstance(images, dict):
        urls.extend(list(images.get("image_list") or []))
        if images.get("main"):
            urls.append(images["main"])
    elif isinstance(images, list):
        urls = list(images)
    if not urls:
        # بعضی نسخه‌های API: data.product.media.images
        media = _dig(product, "media") or {}
        if isinstance(media, dict):
            urls = list(media.get("images") or media.get("image_list") or [])
    if not urls:
        # جستجوی بازگشتی برای image_list
        stack = [payload]
        while stack:
            node = stack.pop()
            if isinstance(node, dict):
                if isinstance(node.get("image_list"), list):
                    urls = list(node["image_list"])
                    if isinstance(node.get("main"), str):
                        urls.append(node["main"])
                    break
                stack.extend(node.values())
            elif isinstance(node, list):
                stack.extend(node)
    return _clean_urls(urls, base_url)


def extract_gallery_from_html(html: str, base_url: str = "") -> list[str]:
    """استخراج گالری از HTML صفحهٔ محصول.

    ترتیب تلاش: JSON تزریق‌شده (__NEXT_DATA__ یا window.state) → آرایهٔ
    ``image_list`` → همهٔ آدرس‌های تصویری CDN.
    """
    urls: list[str] = []
    for m in _NEXT_DATA_RE.finditer(html):
        try:
            data = json.loads(m.group(1))
        except json.JSONDecodeError:
            continue
        urls = extract_gallery_from_json(data, base_url)
        if urls:
            break
    if not urls:
        blocks = _images_blocks(html)
        # بلوکی که image_list دارد اولویت دارد؛ سپس بلوک با بیشترین تصویر
        blocks.sort(
            key=lambda b: (
                "image_list" in b,
                len(b.get("image_list") or []) + (1 if b.get("main") else 0),
            ),
            reverse=True,
        )
        if blocks:
            urls = extract_gallery_from_json({"images": blocks[0]}, base_url)
    if not urls:
        for m2 in _IMAGE_LIST_RE.finditer(html):
            raw = "[" + m2.group(1) + "]"
            try:
                chunk = json.loads(raw.replace("\\/", "/"))
            except json.JSONDecodeError:
                chunk = _IMG_URL_RE.findall(m2.group(1))
            urls.extend(chunk)
    if not urls:
        # آخرین راه: همهٔ آدرس‌های تصویری که از CDN محصول آمده‌اند
        urls = [
            u
            for u in _IMG_URL_RE.findall(html)
            if "digikala-products" in u or "dkstatics" in u
        ]
    return _clean_urls(urls, base_url)


# --------------------------------------------------------------------------- #
# کلاینت دیجی‌کالا
# --------------------------------------------------------------------------- #


class DigikalaClient:
    def __init__(self, cfg, http: HttpClient,
                 logger: Logger | None = None):
        self.cfg = cfg
        self.http = http
        self.logger = logger
        self._product_url_cache: dict[str, str] = {}

    # ------------------------------------------------------------------ #
    # فهرست محصولات
    # ------------------------------------------------------------------ #
    def listing_url(self, page: int, challenge) -> str:
        path = challenge.list_path or self.cfg.products_path
        sep = "&" if "?" in path else "?"
        return self.cfg.url(f"{path}{sep}pageno={page}&sortby=4")

    def fetch_listing_page(self, page: int, challenge) -> ListingPage:
        url = self.listing_url(page, challenge)
        headers = self.http.identity.xhr_headers(self.cfg.challenge_url(challenge))
        resp = self.http.request(
            "GET",
            url,
            headers=headers,
            purpose="listing",
            expect_json=True,
            use_cache=True,
            max_bytes=6 * 1024 * 1024,
        )
        try:
            payload = resp.json()
        except Exception:
            # شاید HTML فهرست برگشته باشد
            html = resp.text
            products = self._parse_listing_html(html, page)
            if products:
                pages = self._pages_from_html(html, page, len(products))
                return ListingPage(page=page, pages=pages, products=products,
                                   total=len(products))
            raise NetError("پاسخ فهرست قابل تجزیه نبود.", url, resp.status)
        listing = parse_listing(payload, page)
        if not listing.products:
            html_fallback = self._listing_html_fallback(page, challenge)
            if html_fallback and html_fallback.products:
                return html_fallback
        return listing

    def _listing_html_fallback(self, page: int, challenge) -> ListingPage | None:
        try:
            resp = self.http.get(
                self.cfg.challenge_url(challenge) + f"?pageno={page}",
                purpose="listing-html",
                headers=self.http.identity.headers(self.cfg.base_url + "/"),
                use_cache=False,
            )
        except NetError:
            return None
        products = self._parse_listing_html(resp.text, page)
        if not products:
            return None
        return ListingPage(
            page=page,
            pages=self._pages_from_html(resp.text, page, len(products)),
            products=products,
            total=len(products),
        )

    @staticmethod
    def _parse_listing_html(html: str, page: int) -> list[models.Product]:
        ids = re.findall(r"dkp-(\d+)", html)
        seen: dict[str, models.Product] = {}
        for pid in ids:
            if pid in seen:
                continue
            seen[pid] = models.Product(id=pid, page=page, rank=len(seen) + 1)
        return list(seen.values())

    @staticmethod
    def _pages_from_html(html: str, page: int, count: int) -> int:
        m = re.search(r'"pages"\s*:\s*(\d+)', html)
        if m:
            return max(int(m.group(1)), page)
        m = re.search(r"صفحه\s*[^0-9]{0,10}(\d{1,4})", html)
        if m:
            return max(int(m.group(1)), page)
        return page if count else 0

    def fetch_listing(self, challenge, max_products: int = 0,
                      progress=None) -> tuple[list[models.Product], int]:
        """خروجی: (محصولات یکتا، تعداد صفحه‌ها). صفحه‌ها به‌صورت موجی خوانده
        می‌شوند تا با کمترین تأخیر، کل فهرست به دست بیاید."""
        first = self.fetch_listing_page(1, challenge)
        products: dict[str, models.Product] = {p.id: p for p in first.products}
        pages = first.pages or 1
        page = 2
        while page <= pages:
            batch_end = min(pages, page + self.cfg.list_page_batch - 1)
            batch = list(range(page, batch_end + 1))

            def one(pg: int):
                try:
                    return self.fetch_listing_page(pg, challenge)
                except NotFoundError:
                    return None
                except (BlockedError, NetError) as exc:
                    if self.logger:
                        self.logger.warn(f"صفحهٔ فهرست {utils.fa_num(pg)} ناموفق: {exc}")
                    return None

            for res in _parallel(one, batch, self.cfg.concurrency):
                if not res:
                    continue
                for prod in res.products:
                    if prod.id and prod.id not in products:
                        products[prod.id] = prod
                pages = max(pages, res.pages)
                if progress:
                    try:
                        progress(len(products), pages)
                    except Exception:
                        pass
                if max_products and len(products) >= max_products:
                    break
            page = batch_end + 1
            if max_products and len(products) >= max_products:
                break
        ordered = sorted(products.values(), key=lambda p: (p.page, p.rank))
        if max_products:
            ordered = ordered[:max_products]
        return ordered, pages

    # ------------------------------------------------------------------ #
    # گالری محصول
    # ------------------------------------------------------------------ #
    def _gallery_urls(self, pid: str) -> list[tuple[str, str, dict[str, str]]]:
        """فهرست (url، منبع، هدرها) به‌ترتیب اولویت."""
        api = self.cfg.api_url(self.cfg.gallery_api_path.format(id=pid))
        html = self.cfg.url(
            (self._product_url_cache.get(pid)
             or self.cfg.gallery_html_path.format(id=pid))
        )
        ref = self.cfg.base_url + "/"
        out: list[tuple[str, str, dict[str, str]]] = []
        source = (self.cfg.gallery_source or "auto").lower()
        if source in ("auto", "api"):
            out.append((api, "api", self.http.identity.api_headers(ref)))
        if source in ("auto", "html"):
            out.append((html, "html", self.http.identity.headers(self.cfg.base_url + "/")))
        return out

    def fetch_gallery(self, product: models.Product) -> tuple[list[models.GalleryItem], str]:
        """گالری یک محصول: ابتدا API، در صورت شکست HTML.

        خروجی: (آیتم‌ها، منبع). اگر هیچ‌کدام نشد، لیست خالی و منبع رشتهٔ خطا.
        """
        last_error = ""
        for url, source, headers in self._gallery_urls(str(product.id)):
            try:
                resp: Response = self.http.request(
                    "GET",
                    url,
                    headers=headers,
                    purpose=f"gallery-{source}",
                    use_cache=False,
                    max_bytes=6 * 1024 * 1024,
                )
            except NotFoundError:
                last_error = "۴۰۴"
                continue
            except (BlockedError, NetError) as exc:
                last_error = str(exc)
                continue
            if source == "api":
                try:
                    payload = resp.json()
                    urls = extract_gallery_from_json(payload, self.cfg.base_url)
                    product.uri = _dig(payload.get("data") or {}, "product", "uri") or product.uri
                except Exception:
                    urls = []
            else:
                urls = extract_gallery_from_html(resp.text, self.cfg.base_url)
                m = re.search(r'<link[^>]+rel="canonical"[^>]+href="([^"]+)"', resp.text)
                if m:
                    self._product_url_cache[str(product.id)] = m.group(1)
            if urls:
                items = [
                    models.GalleryItem.build(str(product.id), u, self.cfg, source)
                    for u in urls[: self.cfg.max_images_per_product]
                ]
                return items, source
            last_error = "بدون تصویر"
        return [], last_error or "ناموفق"

    # ------------------------------------------------------------------ #
    # دانلود تصویر
    # ------------------------------------------------------------------ #
    def download_image(self, url: str, dest, max_bytes: int | None = None) -> tuple[bytes, str]:
        """دانلود تصویر با کیفیت معقول؛ خروجی (داده، sha1)."""
        suffix = self.cfg.full_image_suffix
        target = models.with_suffix(url, suffix)
        resp = self.http.get(
            target,
            headers={
                "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
                "Referer": self.cfg.base_url + "/",
                "Sec-Fetch-Dest": "image",
                "Sec-Fetch-Mode": "no-cors",
                "Sec-Fetch-Site": "cross-site",
            },
            purpose="image",
            use_cache=False,
            max_bytes=max_bytes or self.cfg.max_download_bytes,
        )
        data = resp.content
        if not data:
            raise NetError("تصویر خالی بود.", target, resp.status)
        sha1 = utils.sha1_hex(data)
        return data, sha1


# --------------------------------------------------------------------------- #
# کمکی موازی‌سازی
# --------------------------------------------------------------------------- #


def _parallel(fn, items: list[Any], workers: int):
    """اجرای موازی با حفظ ترتیب نتایج."""
    from concurrent.futures import ThreadPoolExecutor

    if not items:
        return []
    workers = max(1, min(workers, len(items)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(fn, item) for item in items]
        results = []
        for fut in futures:
            try:
                results.append(fut.result())
            except Exception:
                results.append(None)
        return results


def check_availability(http: HttpClient, cfg) -> dict[str, Any]:
    """بررسی سریع دسترسی به نقاط انتهایی اصلی (برای `digiganj doctor`)."""
    out: dict[str, Any] = {}
    tests = {
        "صفحهٔ اصلی": (cfg.base_url + "/", "html"),
        "فهرست شکار گنج": (cfg.url(cfg.products_path + "?pageno=1&sortby=4"), "json"),
        "API محصول": (cfg.api_url(cfg.gallery_api_path.format(id="1774519")), "json"),
        "تصویر CDN": (
            cfg.base_url + "/favicon.ico",
            "image",
        ),
    }
    for name, (url, kind) in tests.items():
        item: dict[str, Any] = {"url": url, "ok": False, "status": 0, "detail": ""}
        try:
            resp = http.request(
                "GET",
                url,
                headers=http.identity.headers(cfg.base_url + "/"),
                purpose="doctor",
                retries=1,
                max_bytes=2 * 1024 * 1024,
            )
            item["status"] = resp.status
            item["ok"] = resp.ok
            if kind == "json":
                try:
                    resp.json()
                    item["detail"] = "JSON معتبر"
                except Exception:
                    item["detail"] = "پاسخ JSON نبود"
                    item["ok"] = False
            elif kind == "html":
                item["detail"] = "HTML دریافت شد"
            else:
                item["detail"] = f"{utils.human_bytes(len(resp.content))}"
        except Exception as exc:
            item["detail"] = utils.shorten(str(exc), 120)
        out[name] = item
    return out


def build_product_url(cfg, pid: str, uri: str = "") -> str:
    if uri.startswith("http"):
        return uri
    if uri.startswith("/"):
        return cfg.base_url + uri
    return f"{cfg.base_url}/product/dkp-{pid}/"


def quote_path(path: str) -> str:
    """کدگذاری امن مسیر فارسی برای درخواست."""
    return "/".join(quote(part) for part in path.split("/"))
