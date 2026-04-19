from __future__ import annotations

import asyncio
import base64
import json
import random
import re
import time
from collections import deque
from typing import Any
from urllib.parse import urljoin, urlsplit

import httpx

from astrbot.api import logger


def _clamp_int(value: Any, *, default: int, min_value: int, max_value: int) -> int:
    try:
        value_int = int(value)
    except (TypeError, ValueError):
        return default
    return max(min_value, min(max_value, value_int))


def _guess_image_mime(data: bytes) -> str:
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return "image/webp"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    return "image/jpeg"


def _build_data_url(image_bytes: bytes) -> str:
    mime = _guess_image_mime(image_bytes)
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def _looks_like_proxy_video_url(url: str) -> bool:
    lowered = (url or "").strip().lower()
    if "generated_video" in lowered:
        return True
    try:
        path = urlsplit(url).path or ""
    except Exception:
        path = ""
    match = re.search(r"/images/p_([A-Za-z0-9+/_=-]+)", path)
    if not match:
        return False
    token = match.group(1)
    padded = token + ("=" * (-len(token) % 4))
    for decoder in (base64.urlsafe_b64decode, base64.b64decode):
        try:
            decoded = decoder(padded.encode("ascii")).decode("utf-8", errors="ignore")
        except Exception:
            continue
        decoded_l = decoded.lower()
        if "generated_video" in decoded_l:
            return True
        if any(ext in decoded_l for ext in (".mp4", ".webm", ".mov")):
            return True
    return False


def _is_valid_video_url(url: str) -> bool:
    if not isinstance(url, str):
        return False
    url = url.strip()
    if len(url) < 10:
        return False
    if not url.startswith(("http://", "https://")):
        return False
    lowered = url.lower()
    if any(c in url for c in ["<", ">", '"', "'", "\n", "\r", "\t"]):
        return False
    if any(ext in lowered for ext in (".mp4", ".webm", ".mov")):
        return True
    if _looks_like_proxy_video_url(url):
        return True
    return False


_VIDEO_URL_RE = re.compile(
    r"(https?://[^\s<>\"')\]\}]+?\.(?:mp4|webm|mov)(?:\?[^\s<>\"')\]\}]*)?)",
    re.IGNORECASE,
)
_GENERIC_URL_RE = re.compile(
    r"(https?://[^\s<>\"')\]\}]+)",
    re.IGNORECASE,
)


def _extract_video_url_from_content(content: str) -> str | None:
    if not content:
        return None
    if "<video" in content and "src=" in content:
        html_patterns = [
            r'<video[^>]*src=["\']([^"\'>]+)["\'][^>]*>',
            r'src=["\']([^"\'>]+\.mp4[^"\'>]*)["\']',
        ]
        for pattern in html_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                url = match.group(1).strip()
                if _is_valid_video_url(url):
                    return url
    match = _VIDEO_URL_RE.search(content)
    if match:
        url = match.group(1).strip()
        if _is_valid_video_url(url):
            return url
    md_patterns = [
        r"!?\[[^\]]*\]\(([^\)]+\.(?:mp4|webm|mov)[^\)]*)\)",
        r"!?\[[^\]]*\]:\s*([^\s]+\.(?:mp4|webm|mov)[^\s]*)",
    ]
    for pattern in md_patterns:
        match = re.search(pattern, content, re.IGNORECASE)
        if match:
            url = match.group(1).strip()
            if _is_valid_video_url(url):
                return url
    for match in _GENERIC_URL_RE.finditer(content):
        url = match.group(1).strip().rstrip(".,;")
        if _is_valid_video_url(url):
            return url
    return None


def _deep_find_video_url(
    data: Any, *, max_depth: int = 6, max_nodes: int = 2000
) -> str | None:
    queue: deque[tuple[Any, int]] = deque([(data, 0)])
    seen = 0
    while queue:
        obj, depth = queue.popleft()
        seen += 1
        if seen > max_nodes:
            return None
        if depth > max_depth:
            continue
        if isinstance(obj, str):
            url = _extract_video_url_from_content(obj) or (
                obj.strip() if _is_valid_video_url(obj) else None
            )
            if url:
                return url
            continue
        if isinstance(obj, dict):
            for key in ("video_url", "file_url", "url", "href", "download_url"):
                val = obj.get(key)
                if isinstance(val, str) and _is_valid_video_url(val):
                    return val.strip()
                if isinstance(val, dict):
                    nested_url = val.get("url") or val.get("file_url")
                    if isinstance(nested_url, str) and _is_valid_video_url(nested_url):
                        return nested_url.strip()
            for val in obj.values():
                queue.append((val, depth + 1))
            continue
        if isinstance(obj, list):
            for item in obj:
                queue.append((item, depth + 1))
            continue
    return None


def _extract_video_url_from_response(
    response_data: Any,
) -> tuple[str | None, str | None]:
    try:
        if not isinstance(response_data, dict):
            return None, f"无效的响应格式: {type(response_data).__name__}"
        direct = response_data.get("video_url")
        if isinstance(direct, str) and _is_valid_video_url(direct):
            return direct, None
        choices = response_data.get("choices")
        if not isinstance(choices, list) or not choices:
            return None, "API 响应缺少 choices"
        choice0 = choices[0]
        if not isinstance(choice0, dict):
            return None, "choices[0] 格式错误"
        message = choice0.get("message")
        if not isinstance(message, dict):
            return None, "choices[0] 缺少 message"
        content = message.get("content")
        if isinstance(content, str):
            url = _extract_video_url_from_content(content)
            if url:
                return url, None
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, str):
                    url = _extract_video_url_from_content(part)
                    if url:
                        return url, None
                if isinstance(part, dict):
                    part_url = (
                        part.get("url")
                        or part.get("video_url")
                        or (
                            part.get("video_url", {})
                            if isinstance(part.get("video_url"), dict)
                            else None
                        )
                    )
                    if isinstance(part_url, str) and _is_valid_video_url(part_url):
                        return part_url, None
                    if isinstance(part_url, dict):
                        nested = part_url.get("url")
                        if isinstance(nested, str) and _is_valid_video_url(nested):
                            return nested, None
                    text = part.get("text")
                    if isinstance(text, str):
                        url = _extract_video_url_from_content(text)
                        if url:
                            return url, None
        for field in ("attachments", "media", "files"):
            items = message.get(field)
            if isinstance(items, list):
                for item in items:
                    if isinstance(item, dict):
                        url = (
                            item.get("url")
                            or item.get("file_url")
                            or item.get("video_url")
                        )
                        if isinstance(url, str) and _is_valid_video_url(url):
                            return url, None
        deep = _deep_find_video_url(response_data)
        if deep:
            return deep, None
        content_preview = ""
        if isinstance(content, str):
            content_preview = content[:200]
        logger.warning(
            f"[GrokVideo] 未能提取视频 URL，content 片段: {content_preview}..."
        )
        return None, "未能从 API 响应中提取到有效的视频 URL"
    except Exception as e:
        logger.warning(f"[GrokVideo] URL 提取异常: {e}")
        return None, f"URL 提取失败: {e}"


# ==============================================================================
# 原有 Grok 服务 (保留，防止报错)
# ==============================================================================
class GrokVideoService:
    def __init__(self, *, settings: dict):
        self.settings = settings if isinstance(settings, dict) else {}

        self.server_url: str = str(
            self.settings.get("server_url", "https://api.x.ai")
        ).rstrip("/")
        self.api_key: str = str(self.settings.get("api_key", "")).strip()
        self.model: str = (
            str(self.settings.get("model", "grok-imagine-0.9")).strip()
            or "grok-imagine-0.9"
        )

        self.timeout_seconds: int = _clamp_int(
            self.settings.get("timeout_seconds", 180),
            default=180,
            min_value=1,
            max_value=3600,
        )
        self.max_retries: int = _clamp_int(
            self.settings.get("max_retries", 2),
            default=2,
            min_value=0,
            max_value=10,
        )
        self.empty_response_retry: int = _clamp_int(
            self.settings.get("empty_response_retry", 2),
            default=2,
            min_value=0,
            max_value=10,
        )
        self.retry_delay: int = _clamp_int(
            self.settings.get("retry_delay", 2),
            default=2,
            min_value=0,
            max_value=60,
        )

        self.presets: dict[str, str] = self._load_presets()
        self.api_url = urljoin(self.server_url + "/", "v1/chat/completions")

        logger.info(
            "[GrokVideo] Initialized: model=%s, timeout=%ss, retries=%s, empty_retry=%s, presets=%s",
            self.model,
            self.timeout_seconds,
            self.max_retries,
            self.empty_response_retry,
            len(self.presets),
        )

    def _load_presets(self) -> dict[str, str]:
        presets: dict[str, str] = {}
        items = self.settings.get("presets", [])
        for item in items:
            if isinstance(item, str) and ":" in item:
                key, val = item.split(":", 1)
                key = key.strip()
                val = val.strip()
                if key and val:
                    presets[key] = val
        return presets

    def get_preset_names(self) -> list[str]:
        return list(self.presets.keys())

    def build_prompt(self, prompt: str, preset: str | None = None) -> str:
        prompt = (prompt or "").strip()
        if preset and preset in self.presets:
            preset_prompt = self.presets[preset]
            if prompt:
                return f"{preset_prompt}, {prompt}"
            return preset_prompt
        return prompt

    async def generate_video_url(
        self,
        prompt: str,
        image_bytes: bytes | None = None,
        *,
        preset: str | None = None,
    ) -> str:
        if not self.api_key:
            raise RuntimeError("Missing API key for video provider (api_key)")
        if not image_bytes:
            raise ValueError("缺少参考图")

        final_prompt = self.build_prompt(prompt, preset=preset)
        if not final_prompt:
            raise ValueError("缺少提示词")

        image_url = _build_data_url(image_bytes)
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": final_prompt},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                }
            ],
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        timeout = httpx.Timeout(
            connect=10.0,
            read=float(self.timeout_seconds),
            write=30.0,
            pool=float(self.timeout_seconds) + 10.0,
        )

        async def _request_once() -> Any:
            async with httpx.AsyncClient(
                timeout=timeout, follow_redirects=True
            ) as client:
                resp = await client.post(self.api_url, json=payload, headers=headers)

            if resp.status_code != 200:
                detail = resp.text[:500]
                if resp.status_code == 401:
                    raise RuntimeError("Grok API Key 无效或已过期 (401)")
                if resp.status_code == 403:
                    raise RuntimeError("Grok API 访问被拒绝 (403)")
                raise RuntimeError(
                    f"Grok API 请求失败 HTTP {resp.status_code}: {detail}"
                )

            try:
                return resp.json()
            except Exception as e:
                text = (resp.text or "").strip()
                if text.startswith("data:"):
                    lines = [
                        ln.strip()
                        for ln in text.splitlines()
                        if ln.strip().startswith("data:")
                    ]
                    chunks: list[dict[str, Any]] = []
                    for ln in lines:
                        data_str = ln[5:].strip()
                        if not data_str or data_str == "[DONE]":
                            continue
                        try:
                            chunks.append(json.loads(data_str))
                        except Exception:
                            continue
                    if chunks:
                        if all(
                            isinstance(c, dict)
                            and str(c.get("object", "")).endswith(".chunk")
                            for c in chunks
                        ):
                            content_parts: list[str] = []
                            for c in chunks:
                                for ch in c.get("choices", []) or []:
                                    delta = ch.get("delta") or {}
                                    part = delta.get("content")
                                    if isinstance(part, str) and part:
                                        content_parts.append(part)
                            content = "".join(content_parts)
                            return {
                                "choices": [
                                    {"message": {"content": content}}
                                ]
                            }
                        return chunks[-1]
                raise RuntimeError(
                    f"API 响应 JSON 解析失败: {e}, body={resp.text[:200]}"
                ) from e

        async def _request_with_retries() -> Any:
            last_exc: Exception | None = None
            for attempt in range(self.max_retries + 1):
                try:
                    logger.info(
                        f"[GrokVideo] 调用 API attempt={attempt + 1}/{self.max_retries + 1}, "
                        f"prompt={final_prompt[:60]}..."
                    )
                    return await _request_once()
                except Exception as e:
                    last_exc = e
                    if attempt >= self.max_retries:
                        break
                    delay = max(0, self.retry_delay) + random.uniform(0, 0.5)
                    logger.warning(f"[GrokVideo] 请求失败: {e}，{delay:.1f}s 后重试...")
                    await asyncio.sleep(delay)
            raise last_exc or RuntimeError("请求失败")

        t_start = time.perf_counter()
        last_parse_error: str | None = None

        for attempt in range(self.empty_response_retry + 1):
            data = await _request_with_retries()
            video_url, parse_error = _extract_video_url_from_response(data)
            if video_url:
                t_end = time.perf_counter()
                logger.info(
                    f"[GrokVideo] 成功: 耗时={t_end - t_start:.2f}s, url={video_url[:80]}..."
                )
                return video_url

            last_parse_error = parse_error or "API 响应未包含视频 URL"
            if attempt >= self.empty_response_retry:
                break

            delay = max(0, self.retry_delay) + random.uniform(0, 0.5)
            logger.warning(
                f"[GrokVideo] 响应无视频URL: {last_parse_error}，{delay:.1f}s 后重试..."
            )
            await asyncio.sleep(delay)

        raise RuntimeError(f"Grok 视频生成失败: {last_parse_error}")


# ==============================================================================
# 新增 豆包 Seedance 服务 (兼容异步任务模式)
# ==============================================================================
class DoubaoSeedanceService:
    def __init__(self, *, settings: dict):
        self.settings = settings if isinstance(settings, dict) else {}

        self.server_url: str = str(
            self.settings.get("server_url", "https://ark.cn-beijing.volces.com")
        ).rstrip("/")
        self.api_key: str = str(self.settings.get("api_key", "")).strip()
        self.model: str = (
            str(self.settings.get("model", "doubao-seedance-1-5-pro-251215")).strip()
            or "doubao-seedance-1-5-pro-251215"
        )

        self.timeout_seconds: int = _clamp_int(
            self.settings.get("timeout_seconds", 300), default=300, min_value=60, max_value=3600
        )
        self.max_retries: int = _clamp_int(
            self.settings.get("max_retries", 1), default=1, min_value=0, max_value=5
        )
        self.polling_interval: int = _clamp_int(
            self.settings.get("polling_interval", 10), default=10, min_value=2, max_value=30
        )
        self.retry_delay: int = _clamp_int(
            self.settings.get("retry_delay", 2), default=2, min_value=0, max_value=60
        )

        self.default_ratio: str = str(self.settings.get("ratio", "9:16"))
        self.default_duration: int = _clamp_int(
            self.settings.get("duration", 6), default=6, min_value=2, max_value=12
        )
        self.default_resolution: str = str(self.settings.get("resolution", "1080p"))
        self.watermark: bool = bool(self.settings.get("watermark", False))
        self.generate_audio: bool = bool(self.settings.get("generate_audio", False))

        self.presets: dict[str, str] = self._load_presets()
        self.create_task_url = urljoin(self.server_url + "/", "api/v3/contents/generations/tasks")
        
        logger.info(
            "[DoubaoVideo] Initialized: model=%s", self.model
        )

    def _load_presets(self) -> dict[str, str]:
        presets: dict[str, str] = {}
        items = self.settings.get("presets", [])
        for item in items:
            if isinstance(item, str) and ":" in item:
                key, val = item.split(":", 1)
                key = key.strip()
                val = val.strip()
                if key and val:
                    presets[key] = val
        return presets

    def get_preset_names(self) -> list[str]:
        return list(self.presets.keys())

    def build_prompt(self, prompt: str, preset: str | None = None) -> str:
        prompt = (prompt or "").strip()
        if preset and preset in self.presets:
            preset_prompt = self.presets[preset]
            if prompt:
                return f"{preset_prompt}, {prompt}"
            return preset_prompt
        return prompt

    async def generate_video_url(
        self,
        prompt: str,
        image_bytes: bytes | None = None,
        *,
        preset: str | None = None,
        **kwargs
    ) -> str:
        if not self.api_key:
            raise RuntimeError("Missing API key (api_key)")

        final_prompt = self.build_prompt(prompt, preset=preset)
        if not final_prompt:
            raise ValueError("缺少提示词")

        content = [{"type": "text", "text": final_prompt}]
        if image_bytes:
            image_url = _build_data_url(image_bytes)
            content.append({
                "type": "image_url",
                "image_url": {"url": image_url}
            })

        payload = {
            "model": self.model,
            "content": content,
            "ratio": kwargs.get("ratio", self.default_ratio),
            "duration": kwargs.get("duration", self.default_duration),
            "resolution": kwargs.get("resolution", self.default_resolution),
            "watermark": kwargs.get("watermark", self.watermark),
            "generate_audio": kwargs.get("generate_audio", self.generate_audio),
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        timeout = httpx.Timeout(connect=10.0, read=60.0, write=30.0, pool=70.0)
        
        task_id = await self._create_task(payload, headers, timeout)
        return await self._poll_task_result(task_id, headers, timeout)

    async def _create_task(self, payload: dict, headers: dict, timeout: httpx.Timeout) -> str:
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                logger.info(f"[DoubaoVideo] 创建任务 attempt={attempt + 1}")
                async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
                    resp = await client.post(self.create_task_url, json=payload, headers=headers)
                if resp.status_code != 200:
                    raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:500]}")
                task_id = resp.json().get("id")
                if not task_id: raise RuntimeError("响应中无 task id")
                logger.info(f"[DoubaoVideo] 任务创建: {task_id}")
                return task_id
            except Exception as e:
                last_exc = e
                if attempt >= self.max_retries: break
                await asyncio.sleep(self.retry_delay)
        raise last_exc or RuntimeError("创建任务失败")

    async def _poll_task_result(self, task_id: str, headers: dict, timeout: httpx.Timeout) -> str:
        t_start = time.perf_counter()
        get_url = f"{self.create_task_url}/{task_id}"
        
        while True:
            if time.perf_counter() - t_start > self.timeout_seconds:
                raise TimeoutError(f"任务超时 ({self.timeout_seconds}s)")

            try:
                async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
                    resp = await client.get(get_url, headers=headers)
                data = resp.json()
                status = data.get("status")
                
                if status == "succeeded":
                    video_url = data.get("content", {}).get("video_url")
                    if video_url: return video_url
                    raise RuntimeError("任务成功但无 video_url")
                elif status == "failed":
                    err = data.get("error", {})
                    raise RuntimeError(f"任务失败: {err.get('message', 'Unknown')}")
                elif status in ["queued", "running"]:
                    await asyncio.sleep(self.polling_interval)
                else:
                    await asyncio.sleep(self.polling_interval)
            except Exception as e:
                if isinstance(e, (RuntimeError, TimeoutError)): raise
                await asyncio.sleep(5)
GrokVideoService = DoubaoSeedanceService