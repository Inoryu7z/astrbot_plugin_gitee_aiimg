from __future__ import annotations

import asyncio
import base64
import json
import re
import time
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlsplit

import httpx

from astrbot.api import logger

from .gitee_sizes import normalize_size_text
from .image_format import guess_image_mime_and_ext
from .openai_compat_backend import _build_collage, resolution_to_size


def _origin(url: str) -> str:
    try:
        u = urlsplit(url)
        if u.scheme and u.netloc:
            return f"{u.scheme}://{u.netloc}"
    except Exception:
        pass
    return ""


def _is_http_url(url: str) -> bool:
    s = str(url or "").strip().lower()
    return s.startswith("http://") or s.startswith("https://")


def _looks_like_ark_multi_image_generate_endpoint(url: str) -> bool:
    s = str(url or "").strip().lower()
    if not s:
        return False
    return (
        "volces.com" in s
        and "/images/generations" in s
        and ("/ark/" in s or "ark." in s or "/api/v3/" in s)
    )


def _extract_image_ref(data: Any) -> str | None:
    """Extract first image ref from OpenAI-like response body."""
    if isinstance(data, dict):
        items = data.get("data")
        if isinstance(items, list):
            for item in items:
                ref = _extract_image_ref(item)
                if ref:
                    return ref

        output = data.get("output")
        if isinstance(output, (list, dict)):
            ref = _extract_image_ref(output)
            if ref:
                return ref

        b64 = data.get("b64_json")
        if isinstance(b64, str) and b64.strip():
            return f"data:image/png;base64,{b64.strip()}"

        for key in ("url", "image_url", "image"):
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
            ref = _extract_image_ref(value)
            if ref:
                return ref
        return None

    if isinstance(data, list):
        for item in data:
            ref = _extract_image_ref(item)
            if ref:
                return ref
        return None

    if isinstance(data, str):
        s = data.strip()
        if s.startswith("data:image/"):
            return s
        if s.startswith(("http://", "https://", "/")):
            return s
        return None

    return None


class OpenAIFullURLBackend:
    """OpenAI-compatible image backend using user-provided full endpoint URLs."""

    def __init__(
            self,
            *,
            imgr,
            full_generate_url: str,
            api_keys: list[str],
            default_model: str = "",
            full_edit_url: str = "",
            timeout: int = 120,
            max_retries: int = 2,
            default_size: str = "4096x4096",
            supports_edit: bool = True,
            extra_body: dict | None = None,
    ):
        self.imgr = imgr
        self.full_generate_url = str(full_generate_url or "").strip()
        self.full_edit_url = str(full_edit_url or "").strip()
        self.api_keys = [str(k).strip() for k in (api_keys or []) if str(k).strip()]
        self.default_model = str(default_model or "").strip()
        self.timeout = max(1, int(timeout or 120))
        self.max_retries = max(0, int(max_retries or 0))
        self.default_size = normalize_size_text(
            str(default_size or "4096x4096").strip()
        )
        self.supports_edit = bool(supports_edit)
        self.extra_body = extra_body or {}

        self._key_index = 0
        self._client: httpx.AsyncClient | None = None

    async def close(self) -> None:
        if self._client is not None:
            try:
                await self._client.aclose()
            except Exception:
                pass
            self._client = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=float(self.timeout), follow_redirects=True
            )
        return self._client

    def _next_key(self) -> str:
        if not self.api_keys:
            raise RuntimeError("未配置 API Key")
        key = self.api_keys[self._key_index]
        self._key_index = (self._key_index + 1) % len(self.api_keys)
        return key

    @staticmethod
    def _headers(api_key: str, *, is_json: bool) -> dict[str, str]:
        headers = {"Authorization": f"Bearer {api_key}"}
        if is_json:
            headers["Content-Type"] = "application/json"
        return headers

    @staticmethod
    def _is_retryable_status(status_code: int) -> bool:
        return status_code == 429 or status_code >= 500

    @staticmethod
    def _as_bool(value: Any, *, default: bool = False) -> bool:
        if isinstance(value, bool):
            return value
        if value is None:
            return default
        if isinstance(value, (int, float)):
            return value != 0
        if isinstance(value, str):
            v = value.strip().lower()
            if v in {"1", "true", "yes", "y", "on", "enable", "enabled"}:
                return True
            if v in {"0", "false", "no", "n", "off", "disable", "disabled", ""}:
                return False
        return default

    def _resolve_size(self, size: str | None, resolution: str | None) -> str:
        final_size = normalize_size_text(size)
        if not final_size:
            final_size = normalize_size_text(resolution_to_size(str(resolution or "")))
        if not final_size:
            final_size = self.default_size
        return final_size

    @staticmethod
    def _collect_local_options(*sources: dict | None) -> dict[str, Any]:
        opts: dict[str, Any] = {}
        for src in sources:
            if not isinstance(src, dict):
                continue
            for key, val in src.items():
                k = str(key or "").strip()
                if not k.startswith("__"):
                    continue
                opts[k] = val
        return opts

    @staticmethod
    def _strip_local_keys(payload: dict[str, Any]) -> dict[str, Any]:
        return {
            k: v for k, v in payload.items() if not str(k or "").strip().startswith("__")
        }

    def _merge_payload(
            self, payload: dict[str, Any], extra_body: dict | None = None
    ) -> dict[str, Any]:
        out = dict(payload)
        if isinstance(self.extra_body, dict):
            out.update(self.extra_body)
        if isinstance(extra_body, dict):
            out.update(extra_body)
        out["watermark"] = False
        return self._strip_local_keys(out)

    async def _post_json(
        self,
        url: str,
        api_key: str,
        payload: dict[str, Any],
    ) -> httpx.Response:
        client = self._get_client()
        attempts = self.max_retries + 1
        last_exc: Exception | None = None

        for attempt in range(attempts):
            try:
                resp = await client.post(
                    url,
                    headers=self._headers(api_key, is_json=True),
                    json=payload,
                )
                if (
                    self._is_retryable_status(resp.status_code)
                    and attempt + 1 < attempts
                ):
                    await asyncio.sleep(min(2.0, 0.4 * (2**attempt)))
                    continue
                return resp
            except Exception as exc:
                last_exc = exc
                if attempt + 1 < attempts:
                    await asyncio.sleep(min(2.0, 0.4 * (2**attempt)))
                    continue

        raise RuntimeError(
            f"请求失败（已重试 {self.max_retries} 次）: {last_exc}"
        ) from last_exc

    async def _post_multipart(
        self,
        url: str,
        api_key: str,
        data: dict[str, str],
        files: dict[str, tuple[str, bytes, str]],
    ) -> httpx.Response:
        client = self._get_client()
        attempts = self.max_retries + 1
        last_exc: Exception | None = None

        for attempt in range(attempts):
            try:
                resp = await client.post(
                    url,
                    headers=self._headers(api_key, is_json=False),
                    data=data,
                    files=files,
                )
                if (
                    self._is_retryable_status(resp.status_code)
                    and attempt + 1 < attempts
                ):
                    await asyncio.sleep(min(2.0, 0.4 * (2**attempt)))
                    continue
                return resp
            except Exception as exc:
                last_exc = exc
                if attempt + 1 < attempts:
                    await asyncio.sleep(min(2.0, 0.4 * (2**attempt)))
                    continue

        raise RuntimeError(
            f"请求失败（已重试 {self.max_retries} 次）: {last_exc}"
        ) from last_exc

    @staticmethod
    def _coerce_form_value(v: Any) -> str:
        if v is None:
            return ""
        if isinstance(v, (str, int, float, bool)):
            return str(v)
        try:
            return json.dumps(v, ensure_ascii=False)
        except Exception:
            return str(v)

    async def _save_ref(self, ref: str, *, endpoint_url: str) -> Path:
        ref = (ref or "").strip()
        if not ref:
            raise RuntimeError("空图片引用")

        if ref.startswith("data:image/"):
            try:
                _header, b64_data = ref.split(",", 1)
            except ValueError:
                raise RuntimeError("data:image 缺少 base64 数据") from None
            image_bytes = base64.b64decode((b64_data or "").strip())
            return await self.imgr.save_image(image_bytes)

        if _is_http_url(ref):
            return await self.imgr.download_image(ref)

        origin = _origin(endpoint_url)
        if origin and ref.startswith("/"):
            return await self.imgr.download_image(
                urljoin(origin + "/", ref.lstrip("/"))
            )
        if origin:
            return await self.imgr.download_image(urljoin(origin + "/", ref))
        raise RuntimeError(f"不支持的图片引用: {ref}")

    async def _save_response(self, resp: httpx.Response, *, endpoint_url: str) -> Path:
        if resp.status_code != 200:
            raise RuntimeError(
                f"请求失败 HTTP {resp.status_code}: {(resp.text or '')[:300]}"
            )

        content_type = (resp.headers.get("content-type") or "").lower()
        if content_type.startswith("image/"):
            return await self.imgr.save_image(resp.content)

        try:
            data = resp.json()
        except Exception as exc:
            raise RuntimeError(f"返回内容不是有效 JSON: {(resp.text or '')[:200]}") from exc

        ref = _extract_image_ref(data)
        if not ref:
            raise RuntimeError(f"未在响应中找到图片地址/数据: {str(data)[:200]}")

        return await self._save_ref(ref, endpoint_url=endpoint_url)

    async def generate(
        self,
        prompt: str,
        *,
        model: str | None = None,
        size: str | None = None,
        resolution: str | None = None,
        extra_body: dict | None = None,
    ) -> Path:
        if not self.full_generate_url:
            raise RuntimeError("未配置 full_generate_url（完整文生图 URL）")

        final_model = str(model or self.default_model or "").strip()
        if not final_model:
            raise RuntimeError("未配置 model")

        final_size = self._resolve_size(size, resolution)
        payload: dict[str, Any] = {
            "model": final_model,
            "prompt": (prompt or "").strip() or "a high quality image",
        }
        if final_size:
            payload["size"] = final_size
        payload = self._merge_payload(payload, extra_body)

        key = self._next_key()
        t0 = time.perf_counter()
        resp = await self._post_json(self.full_generate_url, key, payload)
        out = await self._save_response(resp, endpoint_url=self.full_generate_url)
        logger.info("[OpenAIFullURL][generate] 耗时: %.2fs", time.perf_counter() - t0)
        return out

    async def edit(
        self,
        prompt: str,
        images: list[bytes],
        *,
        model: str | None = None,
        size: str | None = None,
        resolution: str | None = None,
        extra_body: dict | None = None,
    ) -> Path:
        if not self.supports_edit:
            raise RuntimeError("该后端不支持改图/图生图")
        if not images:
            raise ValueError("至少需要一张图片")

        endpoint = self.full_edit_url or self.full_generate_url
        if not endpoint:
            raise RuntimeError("未配置 full_edit_url 或 full_generate_url")

        final_model = str(model or self.default_model or "").strip()
        if not final_model:
            raise RuntimeError("未配置 model")

        final_size = self._resolve_size(size, resolution)
        local_opts = self._collect_local_options(self.extra_body, extra_body)
        edit_mode = str(local_opts.get("__edit_mode") or "auto").strip().lower()
        if not edit_mode or edit_mode == "auto":
            if _looks_like_ark_multi_image_generate_endpoint(endpoint):
                edit_mode = "json_image_array"
            else:
                edit_mode = "collage"

        base_payload: dict[str, Any] = {
            "model": final_model,
            "prompt": (prompt or "").strip() or "Edit this image",
        }
        if final_size:
            base_payload["size"] = final_size
        base_payload = self._merge_payload(base_payload, extra_body)

        key = self._next_key()
        t0 = time.perf_counter()

        if edit_mode == "json_image_array":
            image_field = str(local_opts.get("__edit_image_field") or "image").strip() or "image"
            single_as_array = self._as_bool(
                local_opts.get("__edit_single_image_as_array"), default=False
            )
            force_single_output = self._as_bool(
                local_opts.get("__edit_force_single_output"),
                default=_looks_like_ark_multi_image_generate_endpoint(endpoint),
            )

            refs: list[str] = []
            for img in images:
                mime, _ext = guess_image_mime_and_ext(img)
                b64 = base64.b64encode(img).decode("utf-8")
                refs.append(f"data:{mime};base64,{b64}")

            payload = dict(base_payload)
            payload[image_field] = refs if (len(refs) > 1 or single_as_array) else refs[0]
            if force_single_output and "sequential_image_generation" not in payload:
                payload["sequential_image_generation"] = "disabled"

            logger.info(
                "[OpenAIFullURL][edit] mode=json_image_array refs=%s field=%s endpoint=%s",
                len(refs),
                image_field,
                endpoint,
            )
            response = await self._post_json(endpoint, key, payload)
            out = await self._save_response(response, endpoint_url=endpoint)
            logger.info("[OpenAIFullURL][edit] 耗时: %.2fs", time.perf_counter() - t0)
            return out

        merged_img = _build_collage(images)
        mime, ext = guess_image_mime_and_ext(merged_img)
        b64 = base64.b64encode(merged_img).decode("utf-8")
        data_url = f"data:{mime};base64,{b64}"

        logger.info(
            "[OpenAIFullURL][edit] mode=collage refs=%s endpoint=%s",
            len(images),
            endpoint,
        )

        response: httpx.Response | None = None
        for json_payload in (
            dict(base_payload, image=data_url),
            dict(base_payload, images=[data_url]),
            dict(base_payload, image_url=data_url),
        ):
            resp = await self._post_json(endpoint, key, json_payload)
            response = resp
            if resp.status_code == 200:
                break
            if resp.status_code not in {400, 415, 422}:
                break

        if response is None:
            raise RuntimeError("改图请求未得到有效响应")

        if response.status_code != 200:
            form_data = {
                str(k): self._coerce_form_value(v) for k, v in base_payload.items()
            }
            response = await self._post_multipart(
                endpoint,
                key,
                data=form_data,
                files={"image": (f"image.{ext}", merged_img, mime)},
            )

        out = await self._save_response(response, endpoint_url=endpoint)
        logger.info("[OpenAIFullURL][edit] 耗时: %.2fs", time.perf_counter() - t0)
        return out
