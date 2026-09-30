# -*- coding: utf-8 -*-
"""
智谱 GLM 视觉模型 MCP Server

提供一个 `vision` 工具: 传入本地图片路径(或 base64) + 问题, 调用智谱视觉模型返回文字描述。
用 stdio 方式运行, 供 WorkBuddy 作为自定义 MCP 接入。

依赖: mcp(2.x), httpx
环境变量:
  ZHIPU_API_KEY      (必填) 智谱 API Key
  ZHIPU_VISION_MODEL (可选) 默认 glm-5v-turbo
"""
import base64
import os

import httpx
from mcp.server.mcpserver import MCPServer

API_URL = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
DEFAULT_MODEL = "glm-5v-turbo"

MIME = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
    ".gif": "image/gif",
}

mcp = MCPServer("zhipu-vision", version="1.0.0")


def _to_data_url(image_source: str) -> str:
    """image_source 可以是本地路径, 也可以直接是 base64 或 data URL"""
    s = image_source.strip()
    if s.startswith("data:"):
        return s
    if os.path.isfile(s):
        ext = os.path.splitext(s)[1].lower()
        mime = MIME.get(ext, "image/jpeg")
        with open(s, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        return f"data:{mime};base64,{b64}"
    return s  # 裸 base64 或 URL 原样交给模型


@mcp.tool()
async def vision(
    image: str,
    prompt: str = "详细描述这张图片的内容",
    model: str = DEFAULT_MODEL,
) -> str:
    """调用智谱 GLM 视觉模型看图并回答。

    Args:
        image: 本地图片绝对路径, 或图片 base64 字符串 / data URL
        prompt: 要对图片问的问题或要求
        model: 视觉模型名, 默认 glm-5v-turbo
    """
    api_key = os.environ.get("ZHIPU_API_KEY", "")
    if not api_key:
        return "错误: 未配置 ZHIPU_API_KEY 环境变量"
    data_url = _to_data_url(image)

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ],
        "temperature": 0.2,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    try:
        async with httpx.AsyncClient(timeout=120) as client:
            r = await client.post(API_URL, json=payload, headers=headers)
    except Exception as e:
        return f"错误: 请求失败 {e}"
    if r.status_code != 200:
        return f"错误: HTTP {r.status_code}\n{r.text[:1000]}"
    try:
        data = r.json()
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        return f"错误: 解析响应失败 {e}\n{r.text[:1000]}"


if __name__ == "__main__":
    import asyncio
    asyncio.run(mcp.run_stdio_async())
