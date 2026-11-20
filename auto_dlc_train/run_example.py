"""独立调用 PAI 训练工具示例（脱离 TarsAI 平台）。

token 自动获取（get_token.resolve_token）：
  环境变量 TARS_TOKEN → 本地缓存 refresh 换新 → credentials.yaml 登录
详见 README「token 获取」与 get_token.py。

用法：
  python run_example.py --list-clusters
  python run_example.py --from-yaml ./my_train_job.yaml --dry-run
  python run_example.py --from-yaml ./my_train_job.yaml
"""
from __future__ import annotations

import argparse
import asyncio
from types import SimpleNamespace

from get_token import TokenError, resolve_token
from pai_tool.agent_tools import (
    create_train_job_from_yaml,
    list_clusters,
    list_train_job_resources,
)


async def _ctx() -> SimpleNamespace:
    token = await resolve_token()
    return SimpleNamespace(access_token=token)


async def _main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--list-clusters", action="store_true", help="列集群")
    ap.add_argument("--list-resources", action="store_true", help="聚合查资源（集群/资源池/规格/项目）")
    ap.add_argument("--from-yaml", metavar="PATH", help="按 YAML 创建训练任务")
    ap.add_argument("--dry-run", action="store_true", help="仅解析+转 id+预览，不真正创建")
    args = ap.parse_args()

    try:
        ctx = await _ctx()
    except TokenError as e:
        raise SystemExit(str(e))

    if args.list_clusters:
        print(await list_clusters(_ctx=ctx))
    elif args.list_resources:
        print(await list_train_job_resources(_ctx=ctx))
    elif args.from_yaml:
        print(await create_train_job_from_yaml(yaml_path=args.from_yaml, dry_run=args.dry_run, _ctx=ctx))
    else:
        ap.print_help()


if __name__ == "__main__":
    asyncio.run(_main())
