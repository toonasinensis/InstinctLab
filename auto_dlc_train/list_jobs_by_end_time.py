"""查询 PAI 训练任务列表，按结束时间过滤，输出任务名称到文本文件。

复用 get_token.resolve_token() 取 token（环境变量 TARS_TOKEN → 缓存 refresh → 账号密码）。
调 GET /api/v1/pai/train/page，自动翻页取全量，按 endTimestamp 过滤后写文件。

用法：
  # 查 meiguogan 训练结束时间晚于 2026-08-28 18:43:33 的任务名
  python list_jobs_by_end_time.py \
      --end-time "2026-08-28 18:43:33" \
      --user meiguogan \
      --out ../tmp/meiguogan_jobs_end_after_20260828_184333.txt

  # 不写文件，只打印到终端
  python list_jobs_by_end_time.py --end-time "2026-08-28 18:43:33" --user meiguogan --out -
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import httpx

from get_token import resolve_token

_PAI_BASE = "https://open.tars-ai.com"
_PAGE_PATH = "/api/v1/pai/train/page"

# 列表接口返回的原始字段（供参考）：
#   name / userName / userId / status / endTimestamp(ms) / startTimestamp / submitTimestamp
_STATUS_UNFINISHED = {"Running", "Pending", "Queued", "Submitted", "Unknown"}


def parse_time(s: str, tz_hours: int) -> int:
    """解析 'YYYY-MM-DD HH:MM:SS'（按 tz_hours 时区），返回 epoch 毫秒。"""
    dt = datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
    dt = dt.replace(tzinfo=timezone(timedelta(hours=tz_hours)))
    return int(dt.timestamp() * 1000)


def fmt_ms(ms: object, tz_hours: int) -> str:
    if not isinstance(ms, (int, float)) or ms in (None, 0):
        return "-"
    dt = datetime.fromtimestamp(ms / 1000, tz=timezone(timedelta(hours=tz_hours)))
    return dt.strftime("%Y-%m-%d %H:%M:%S")


async def fetch_all_jobs(token: str) -> list[dict]:
    """翻页拉取当前用户（onlyMe=true）的全量训练任务，返回原始 list。"""
    headers = {"Authorization": f"Bearer {token}"}
    all_jobs: list[dict] = []
    page = 1
    page_size = 100
    async with httpx.AsyncClient(timeout=60) as client:
        while True:
            params = {"page": page, "pageSize": page_size, "onlyMe": True}
            resp = await client.get(f"{_PAI_BASE}{_PAGE_PATH}", headers=headers, params=params)
            rj = resp.json()
            if rj.get("code") not in (0, 200):
                raise RuntimeError(f"接口报错 code={rj.get('code')} msg={rj.get('msg')} traceId={rj.get('traceId')}")
            data = rj.get("data") or {}
            lst = data.get("list") or []
            all_jobs.extend(lst)
            total = data.get("total")
            if not lst or (total is not None and len(all_jobs) >= total) or len(lst) < page_size:
                break
            page += 1
    return all_jobs


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--end-time", required=True, help="过滤阈值：'YYYY-MM-DD HH:MM:SS'，只保留结束时间晚于它的任务")
    ap.add_argument("--tz", type=int, default=8, help="--end-time 的时区小时数（默认 8，即北京时间）")
    ap.add_argument("--user", default=None, help="可选：按 userName 过滤（默认不过滤，只查当前登录用户自己的任务）")
    ap.add_argument("--out", default=None, help="输出文本文件路径（默认 ../tmp/jobs_end_after_<时间戳>.txt）；'--out -' 表示只打印不写文件")
    ap.add_argument("--include-running", action="store_true", help="把仍在运行（endTimestamp 为空）的任务也视为'会晚于阈值结束'纳入结果")
    args = ap.parse_args()

    threshold_ms = parse_time(args.end_time, args.tz)

    token = await resolve_token()
    jobs = await fetch_all_jobs(token)

    # 按 userName 过滤（可选）
    if args.user:
        jobs = [j for j in jobs if j.get("userName") == args.user]

    # 按结束时间过滤
    matched: list[dict] = []
    running: list[dict] = []
    for j in jobs:
        end = j.get("endTimestamp")
        if isinstance(end, (int, float)) and end > threshold_ms:
            matched.append(j)
        elif end in (None, 0) and args.include_running:
            running.append(j)

    matched.sort(key=lambda j: j.get("endTimestamp") or 0, reverse=True)
    names = [j.get("name", "") for j in matched]
    if args.include_running:
        names += [j.get("name", "") for j in running]

    # 输出文件
    out = args.out
    if out is None:
        stamp = args.end_time.replace(" ", "").replace(":", "").replace("-", "")
        out = str(Path(__file__).resolve().parent.parent / "tmp" / f"jobs_end_after_{stamp}.txt")
    if out == "-":
        sys.stdout.write("\n".join(names) + ("\n" if names else ""))
    else:
        out_path = Path(out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("\n".join(names) + ("\n" if names else ""), encoding="utf-8")
        print(f"已写入 {len(names)} 个任务名 → {out_path}")

    # 终端汇总
    print(f"\n过滤条件：结束时间晚于 {args.end_time} (UTC{'+' if args.tz >= 0 else ''}{args.tz})")
    print(f"任务总数 {len(jobs)}，命中 {len(matched)} 个（已结束且 endTimestamp > 阈值）")
    if not args.include_running:
        still_running = [j for j in jobs if j.get("endTimestamp") in (None, 0)]
        if still_running:
            print(f"  另有 {len(still_running)} 个仍在运行（endTimestamp 为空，未计入；加 --include-running 可纳入）")
    print()
    for j in matched:
        print(f"{fmt_ms(j.get('endTimestamp'), args.tz)}  {j.get('status', ''):<12}  {j.get('name', '')}")
    if args.include_running and running:
        print("--- 仍在运行的任务 ---")
        for j in running:
            print(f"{fmt_ms(j.get('startTimestamp'), args.tz)}  {j.get('status', ''):<12}  {j.get('name', '')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))