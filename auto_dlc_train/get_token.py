"""独立获取平台 token（脱离 TarsAI 平台，仅依赖 httpx + PyYAML）。

取 token 优先级（由高到低）：
  1. 环境变量 TARS_TOKEN        —— 浏览器 DevTools 里拿到的 access token
  2. 本地缓存 refresh_token      —— 调认证中心换新 access token（refresh 自动轮换）
  3. credentials.yaml 账号密码   —— 登录拿 (access, refresh)，并持久化 refresh

缓存文件：.token_cache.json（gitignored），只存 refresh_token。
登录凭证：credentials.yaml（gitignored），结构见 README「登录凭证」。

用法：
  python get_token.py             # 打印可用的 access token（自动走优先级链）
  python -c "import asyncio, get_token; print(asyncio.run(get_token.resolve_token()))"
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import pathlib
from typing import Any

import httpx
import yaml

AUTH_BASE_URL = os.environ.get("TARS_AUTH_BASE_URL", "https://open.tars-ai.com")
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
TOKEN_CACHE_FILE = PROJECT_ROOT / ".token_cache.json"
CREDENTIALS_FILE = PROJECT_ROOT / "credentials.yaml"


class TokenError(RuntimeError):
    """无法自动获取 token。"""


def _read_cache() -> dict[str, Any]:
    if not TOKEN_CACHE_FILE.exists():
        return {}
    try:
        return json.loads(TOKEN_CACHE_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _write_cache(data: dict[str, Any]) -> None:
    TOKEN_CACHE_FILE.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def get_cached_refresh_token() -> str | None:
    return _read_cache().get("refresh_token") or None


def _load_credentials() -> dict[str, str]:
    """读 credentials.yaml 的 data_catalog 段（username+password）。"""
    if not CREDENTIALS_FILE.exists():
        return {}
    try:
        cfg = yaml.safe_load(CREDENTIALS_FILE.read_text(encoding="utf-8")) or {}
        section = cfg.get("data_catalog") or {}
        if isinstance(section, dict):
            return {k: str(v) for k, v in section.items() if v is not None}
    except (yaml.YAMLError, OSError):
        pass
    return {}


async def _post(path: str, payload: dict[str, Any]) -> dict[str, Any] | None:
    """调认证中心，返回规范化后的 data（已剥 userToken 包装）或 None。"""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.post(f"{AUTH_BASE_URL}{path}", json=payload)
            r.raise_for_status()
            resp = r.json()
        # 认证中心成功 code 为 0 或 200；非此即被拒。
        if resp.get("code") not in {0, 200}:
            return None
        data = resp.get("data")
        if isinstance(data, dict) and isinstance(data.get("userToken"), dict):
            data = data["userToken"]
        if not isinstance(data, dict):
            return None
        return data
    except (httpx.HTTPError, OSError, ValueError, TypeError):
        return None


async def mint_access_token(refresh_token: str | None) -> tuple[str, str] | None:
    """用 refresh_token 换新 access_token，返回 (access, rotated_refresh) 或 None。"""
    if not refresh_token:
        return None
    data = await _post(
        "/api/v1/auth/token/refresh", {"refreshToken": refresh_token}
    )
    if not data:
        return None
    access = data.get("accessToken")
    new_refresh = data.get("refreshToken")
    if not access or not new_refresh:
        return None
    return access, new_refresh


async def login_password() -> tuple[str, str] | None:
    """用 credentials.yaml 账号密码登录，返回 (access, refresh) 或 None。

    认证中心要求密码 SHA-256 哈希后传输（明文被拒）。
    """
    creds = _load_credentials()
    username = creds.get("username")
    password = creds.get("password")
    if not username or not password:
        return None
    pwd_hash = hashlib.sha256(password.encode()).hexdigest()
    data = await _post(
        "/api/v1/auth/login",
        {
            "loginType": "PASSWORD",
            "clientId": "tarsai",
            "username": username,
            "password": pwd_hash,
        },
    )
    if not data:
        return None
    access = data.get("accessToken")
    refresh = data.get("refreshToken")
    if not access or not refresh:
        return None
    return access, refresh


async def resolve_token() -> str:
    """取可用的 access token：环境变量 → 缓存 refresh 换新 → 账号密码登录。"""
    env_token = os.environ.get("TARS_TOKEN")
    if env_token:
        return env_token

    cached_refresh = get_cached_refresh_token()
    if cached_refresh:
        result = await mint_access_token(cached_refresh)
        if result:
            access, new_refresh = result
            _write_cache({"refresh_token": new_refresh})
            return access

    result = await login_password()
    if result:
        access, refresh = result
        _write_cache({"refresh_token": refresh})
        return access

    raise TokenError(
        "无法自动获取 token。任选其一：\n"
        f"  1) export TARS_TOKEN=<浏览器 DevTools 里的 access token>\n"
        f"  2) 手动把 refresh_token 写入 {TOKEN_CACHE_FILE.name}"
        f"（{TOKEN_CACHE_FILE}）\n"
        f"  3) 创建 {CREDENTIALS_FILE.name} 填账号密码（见 README「登录凭证」）"
    )


async def _main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--refresh-cache",
        metavar="REFRESH_TOKEN",
        help="把 refresh_token 写入缓存后退出（后续自动换新）",
    )
    ap.add_argument("--clear-cache", action="store_true", help="删除 token 缓存文件")
    args = ap.parse_args()

    if args.clear_cache:
        if TOKEN_CACHE_FILE.exists():
            TOKEN_CACHE_FILE.unlink()
            print(f"已删除缓存 {TOKEN_CACHE_FILE}")
        else:
            print("没有缓存文件")
        return
    if args.refresh_cache:
        _write_cache({"refresh_token": args.refresh_cache})
        print(f"已写入缓存 {TOKEN_CACHE_FILE}")
        return

    try:
        print(await resolve_token())
    except TokenError as e:
        raise SystemExit(str(e))


if __name__ == "__main__":
    asyncio.run(_main())
