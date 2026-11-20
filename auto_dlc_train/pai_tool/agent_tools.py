"""PAI 创建分布式训练任务 agent tool —— @tool 形式(独立查询型)。

调 TARS 平台 PAI 接口创建分布式训练任务,复用平台登录态鉴权(_ctx.access_token)。
接口: POST https://open.tars-ai.com/api/v1/pai/train/create-job

接入方式:独立脚本直接调用(async 函数,传 _ctx.access_token 即平台登录 token)。
必填字段填中文语义,函数内部调查询接口转 id 后调 create_train_job。
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Optional
from urllib.parse import quote

import httpx
import yaml as _yaml

from .registry import tool

_PAI_BASE = "https://open.tars-ai.com"
_CREATE_JOB_PATH = "/api/v1/pai/train/create-job"


def _json(data: dict) -> str:
    return json.dumps(data, ensure_ascii=False, default=str)


def _coerce_json(value: Any, expect: str = "dict") -> Any:
    """把模型可能传成字符串的 JSON 归一化为 Python 对象。

    即便 schema 已是 object/array,LLM 仍可能传 JSON 字符串(历史习惯/description 误导)。
    后端 Java Jackson 期望嵌套对象(InstanceTypeInfo 无 String-argument 构造器),
    收到字符串会反序列化失败,故在此防御性归一化。

    expect:
      - "dict": 归一化为 dict(str→json.loads;dict→原样;其它→原样交后端报错定位)
      - "list": 归一化为 list(每项再按 "dict" 归一化)
    返回 None 表示空,调用方据此跳过不写入 body。
    """
    if value is None:
        return None
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        try:
            value = json.loads(s)
        except json.JSONDecodeError:
            return value  # 非合法 JSON,原样返回让后端/日志定位
    if expect == "list" and isinstance(value, list):
        return [_coerce_json(it, "dict") for it in value]
    return value


def _coerce_config(value: Any) -> Optional[str]:
    """dataSet/storageMountList[].config 后端要求 JSON 字符串(文档明确,非对象)。

    dict→json.dumps;str→原样保留;None→None。
    """
    if value is None:
        return None
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


async def _fetch_user_id(token: str) -> Optional[int]:
    """调认证中心(/api/v1/auth/user)拿当前用户数字 id,用于构造任务列表 URL。失败返回 None。"""
    try:
        async with httpx.AsyncClient(timeout=5) as cli:
            r = await cli.get(
                f"{_PAI_BASE}/api/v1/auth/user",
                headers={"Authorization": f"Bearer {token}"},
            )
        data = (r.json() or {}).get("data") or {}
        if isinstance(data, dict):
            uid = data.get("id")
            if isinstance(uid, int):
                return uid
            if isinstance(uid, str) and uid.isdigit():
                return int(uid)
    except Exception:  # noqa: BLE001
        return None
    return None


def _build_job_url(job_name: str, user_id: Optional[int]) -> str:
    """构造 PAI 任务列表 URL(按 jobName 过滤),方便用户点击查看刚创建的任务。

    有 userId → onlyMe=false&userId=<id>;拿不到 → onlyMe=true 兜底(靠登录态识别)。
    """
    enc = quote(job_name)
    if user_id is not None:
        return f"{_PAI_BASE}/pai/distributed-training?onlyMe=false&userId={user_id}&jobName={enc}&page=1&pageSize=20"
    return f"{_PAI_BASE}/pai/distributed-training?onlyMe=true&jobName={enc}&page=1&pageSize=20"


@tool(
    "create_train_job",
    "创建分布式训练任务(TARS PAI 平台)。必填:clusterId(集群id)/jobName(任务名)/specId(规格id)/imageId(镜像id)/podCount(节点数)/priority(优先级)/command(启动命令)。可选:dataSet(数据集挂载)/storageMountList(存储挂载)/instanceTypeInfo(实例规格)/maxRunTime(最大运行小时,默认120)等。复用平台登录态鉴权,用户须已登录。",
    params={
        "clusterId": "集群 id(数字)",
        "jobName": "任务名称",
        "specId": "规格 id(数字)",
        "imageId": "镜像 id",
        "podCount": "节点数量(数字,分布式训练的节点数)",
        "priority": "优先级(数字)",
        "command": "启动命令(训练入口命令)",
        "projectId": "可选,研发项目 id",
        "gpuCount": "可选,单节点 GPU 卡数",
        "cpuCount": "可选,单节点 CPU 核数(仅阿里云)",
        "dataSet": "可选,数据集挂载列表(传 JSON 数组对象,非字符串),每项 {datasetLocalId(必填,number),mountPath(必填),version?,readOnly?,config?};config 是 JSON 字符串非对象",
        "storageMountList": "可选,存储挂载列表(传 JSON 数组对象,非字符串),每项 {mountPath(必填),fileSystemPath(必填),fileStorageId(必填,number),readOnly?,workspaceSourceFlag?,config?};config 是 JSON 字符串非对象",
        "instanceTypeInfo": "可选,实例规格对象(传 JSON 对象非字符串!),形如 {\"id\":\"ml.xni3c.22xlarge\",\"family\":\"ml.xni3c\",\"cpu\":22,\"memoryMiB\":...,\"gpu\":1,\"gpuMemoryMiB\":...,\"gpuType\":\"...\"};id 必填;火山引擎集群用此替代 specId",
        "enableTensorboard": "可选,火山云集群是否开 tensorBoard",
        "errorMonitor": "可选,阿里云集群是否开自动容错",
        "resourcePoolId": "可选,资源池 id",
        "publicJobGroup": "可选,公共任务关联的用户组 id",
        "maxRunTime": "可选,最大运行时长(小时),默认 120",
    },
)
async def create_train_job(
    *,
    clusterId: int,
    jobName: str,
    specId: Optional[int] = None,
    imageId: str,
    podCount: int,
    priority: int,
    command: str,
    projectId: Optional[int] = None,
    gpuCount: Optional[int] = None,
    cpuCount: Optional[int] = None,
    dataSet: Optional[list[dict]] = None,
    storageMountList: Optional[list[dict]] = None,
    instanceTypeInfo: Optional[dict] = None,
    enableTensorboard: Optional[bool] = None,
    errorMonitor: Optional[bool] = None,
    resourcePoolId: Optional[int] = None,
    publicJobGroup: Optional[int] = None,
    maxRunTime: Optional[int] = None,
    _ctx=None,
) -> str:
    """创建分布式训练任务(PAI)。复用平台登录态(_ctx.access_token)鉴权。"""
    # 1) 拿用户登录态 token(复用平台共享鉴权)
    token = getattr(_ctx, "access_token", None) if _ctx else None
    if not token:
        return _json({"status": "error", "message": "未登录,无法获取 access_token(请先登录 TARS 平台)"})

    # 2) 归一化嵌套参数 + 构造请求体
    # LLM 即便 schema 是 object/array 也可能传 JSON 字符串;后端 Jackson 期望嵌套对象
    # (InstanceTypeInfo 无 String 构造器),需防御性 str→object 转换。
    instanceTypeInfo = _coerce_json(instanceTypeInfo, "dict")
    dataSet = _coerce_json(dataSet, "list")
    storageMountList = _coerce_json(storageMountList, "list")
    # dataSet/storageMountList 每项的 config 后端要求 JSON 字符串(文档明确),dict 需 dumps
    for _ml in (dataSet, storageMountList):
        if isinstance(_ml, list):
            for _item in _ml:
                if isinstance(_item, dict) and _item.get("config") is not None:
                    _item["config"] = _coerce_config(_item["config"])

    body: dict = {
        "clusterId": clusterId,
        "jobName": jobName,
        "imageId": imageId,
        "podCount": podCount,
        "priority": priority,
        "command": command,
    }
    optional_fields = {
        "specId": specId,
        "projectId": projectId,
        "gpuCount": gpuCount,
        "cpuCount": cpuCount,
        "dataSet": dataSet,
        "storageMountList": storageMountList,
        "instanceTypeInfo": instanceTypeInfo,
        "enableTensorboard": enableTensorboard,
        "errorMonitor": errorMonitor,
        "resourcePoolId": resourcePoolId,
        "publicJobGroup": publicJobGroup,
        "maxRunTime": maxRunTime,
    }
    for k, v in optional_fields.items():
        if v is not None:
            body[k] = v

    # 3) 调 PAI 接口(Bearer 鉴权)
    url = f"{_PAI_BASE}{_CREATE_JOB_PATH}"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(url, json=body, headers=headers)
        try:
            rj = resp.json()
        except ValueError:
            return _json({
                "status": "error",
                "message": f"PAI 返回非 JSON: HTTP {resp.status_code} {resp.text[:200]}",
                "request_body": body,
            })
    except Exception as e:  # noqa: BLE001
        return _json({"status": "error", "message": f"调用 PAI 接口失败: {e}", "request_body": body})

    # 4) 解析响应 {code, data, msg, traceId};code=0 成功
    code = rj.get("code")
    if code in (0, 200):  # 实测 cluster-option 成功 code=200(create-job 文档写 0,兼容两者)
        uid = await _fetch_user_id(token)
        return _json({
            "status": "ok",
            "data": rj.get("data"),
            "msg": rj.get("msg", ""),
            "traceId": rj.get("traceId", ""),
            "jobName": jobName,
            "job_url": _build_job_url(jobName, uid),  # 平台任务列表链接(按 jobName 过滤),给用户点击查看
        })
    return _json({
        "status": "error",
        "code": code,
        "msg": rj.get("msg", ""),
        "traceId": rj.get("traceId", ""),
        "request_body": body,
    })


# ---- 查询 tool(名字→id 转换) ----
@tool(
    "list_clusters",
    "查询 PAI 平台可用集群列表,返回 [{id, name, company}]。用于把用户说的集群名(如'集群2')转成 clusterId。复用平台登录态鉴权。",
    params={},
)
async def list_clusters(*, _ctx=None) -> str:
    """查询集群列表(GET /api/v1/pai/cluster/cluster-option)。"""
    token = getattr(_ctx, "access_token", None) if _ctx else None
    if not token:
        return _json({"status": "error", "message": "未登录,无法获取 access_token(请先登录 TARS 平台)"})
    url = f"{_PAI_BASE}/api/v1/pai/cluster/cluster-option"
    headers = {"Authorization": f"Bearer {token}"}
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(url, headers=headers)
        rj = resp.json()
    except Exception as e:  # noqa: BLE001
        return _json({"status": "error", "message": f"调用集群列表接口失败: {e}"})
    code = rj.get("code")
    if code in (0, 200):  # 实测 cluster-option 成功 code=200(create-job 文档写 0,兼容两者)
        data = rj.get("data") or []
        return _json({"status": "ok", "count": len(data), "clusters": data})
    return _json({"status": "error", "code": code, "msg": rj.get("msg", ""), "traceId": rj.get("traceId", "")})


@tool(
    "list_gpu_types",
    "查询集群下可用 GPU 类型(机器规格)枚举,返回 [{specId, name}]。用于把用户说的规格名(如'NVIDIA-A800')转成 specId(即 gpu-type-enum 的 key)。复用平台登录态鉴权。",
    params={"clusterId": "集群 id(数字,从 list_clusters 拿)"},
)
async def list_gpu_types(*, clusterId: int, _ctx=None) -> str:
    """查询集群下 GPU 类型枚举(GET /api/v1/pai/spec/gpu-type-enum/{clusterId})。key 即 specId。"""
    token = getattr(_ctx, "access_token", None) if _ctx else None
    if not token:
        return _json({"status": "error", "message": "未登录,无法获取 access_token(请先登录 TARS 平台)"})
    url = f"{_PAI_BASE}/api/v1/pai/spec/gpu-type-enum/{clusterId}"
    headers = {"Authorization": f"Bearer {token}"}
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(url, headers=headers)
        rj = resp.json()
    except Exception as e:  # noqa: BLE001
        return _json({"status": "error", "message": f"调用 GPU 类型枚举接口失败: {e}"})
    code = rj.get("code")
    if code in (0, 200):
        data = rj.get("data") or []
        # 原始 [{key, value}] → [{specId, name}],key 即 specId
        gpu_types = [{"specId": d.get("key"), "name": d.get("value")} for d in data]
        return _json({"status": "ok", "count": len(gpu_types), "gpu_types": gpu_types})
    return _json({"status": "error", "code": code, "msg": rj.get("msg", ""), "traceId": rj.get("traceId", "")})


@tool(
    "list_instance_specs",
    "查询火山引擎集群的计算规格信息(仅火山引擎集群3/4),返回 [{id, family, cpu, memoryMiB, gpu, gpuMemoryMiB, gpuType}]。火山引擎集群创建任务时填 instanceTypeInfo(替代 specId)。复用平台登录态鉴权。",
    params={
        "resourceId": "资源 id(数字,火山引擎集群从 list_resource_pools 拿)",
        "kind": "规格类型:GPU(GPU型)/General(通用CPU机型)/Rdma(高性能GPU型)",
    },
)
async def list_instance_specs(*, resourceId: int, kind: str, _ctx=None) -> str:
    """查询火山引擎计算规格(GET /api/v1/pai/spec/list-instance-specs/{resourceId}/{kind})。"""
    token = getattr(_ctx, "access_token", None) if _ctx else None
    if not token:
        return _json({"status": "error", "message": "未登录,无法获取 access_token(请先登录 TARS 平台)"})
    url = f"{_PAI_BASE}/api/v1/pai/spec/list-instance-specs/{resourceId}/{kind}"
    headers = {"Authorization": f"Bearer {token}"}
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(url, headers=headers)
        rj = resp.json()
    except Exception as e:  # noqa: BLE001
        return _json({"status": "error", "message": f"调用计算规格接口失败: {e}"})
    code = rj.get("code")
    if code in (0, 200):
        data = rj.get("data") or []
        return _json({"status": "ok", "count": len(data), "instance_specs": data})
    return _json({"status": "error", "code": code, "msg": rj.get("msg", ""), "traceId": rj.get("traceId", "")})


@tool(
    "list_projects",
    "查询可选研发项目列表,返回 [{id(projectId), name, description, canSelect}]。用于把用户说的项目名(如'AWE')转成 projectId。复用平台登录态鉴权。",
    params={},
)
async def list_projects(*, _ctx=None) -> str:
    """查询可选项目列表(GET /api/v1/pai/project/list/optional)。"""
    token = getattr(_ctx, "access_token", None) if _ctx else None
    if not token:
        return _json({"status": "error", "message": "未登录,无法获取 access_token(请先登录 TARS 平台)"})
    url = f"{_PAI_BASE}/api/v1/pai/project/list/optional"
    headers = {"Authorization": f"Bearer {token}"}
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(url, headers=headers)
        rj = resp.json()
    except Exception as e:  # noqa: BLE001
        return _json({"status": "error", "message": f"调用项目列表接口失败: {e}"})
    code = rj.get("code")
    if code in (0, 200):
        data = rj.get("data") or []
        return _json({"status": "ok", "count": len(data), "projects": data})
    return _json({"status": "error", "code": code, "msg": rj.get("msg", ""), "traceId": rj.get("traceId", "")})


@tool(
    "list_images",
    "查询集群下镜像列表,返回 [{id, imageId, imageName, imageUri, tag, status, ...}]。用于把用户说的镜像名(如'jay-dsw-a100-wlcb-v1:0127')转成 imageId。复用平台登录态鉴权。",
    params={
        "clusterId": "集群 id(数字,从 list_clusters 拿)",
        "imageName": "可选,镜像名关键词过滤",
    },
)
async def list_images(*, clusterId: int, imageName: Optional[str] = None, _ctx=None) -> str:
    """查询镜像列表(POST /api/v1/pai/image/list,body: {page,pageSize,clusterId,imageName?})。"""
    token = getattr(_ctx, "access_token", None) if _ctx else None
    if not token:
        return _json({"status": "error", "message": "未登录,无法获取 access_token(请先登录 TARS 平台)"})
    url = f"{_PAI_BASE}/api/v1/pai/image/list"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    body: dict = {"page": 1, "pageSize": 100, "clusterId": clusterId}
    if imageName:
        body["imageName"] = imageName
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(url, json=body, headers=headers)
        rj = resp.json()
    except Exception as e:  # noqa: BLE001
        return _json({"status": "error", "message": f"调用镜像列表接口失败: {e}"})
    code = rj.get("code")
    if code in (0, 200):
        data = rj.get("data") or []
        return _json({"status": "ok", "count": len(data), "images": data})
    return _json({"status": "error", "code": code, "msg": rj.get("msg", ""), "traceId": rj.get("traceId", "")})


@tool(
    "list_datasets",
    "查询数据集列表(不分页),返回 [{id(datasetLocalId), datasetName, storageType, fileSystemPath, mountPath, ...}]。用于把用户说的数据集名(如'VLTA-CPFS-taplab')转成 datasetLocalId(create_train_job 的 dataSet[].datasetLocalId)。复用平台登录态鉴权。",
    params={
        "clusterId": "可选,集群 id(数字,按集群过滤)",
        "datasetName": "可选,数据集名关键词过滤",
    },
)
async def list_datasets(*, clusterId: Optional[int] = None, datasetName: Optional[str] = None, _ctx=None) -> str:
    """查询数据集列表(POST /api/v1/pai/dataset/list,body 全可选)。"""
    token = getattr(_ctx, "access_token", None) if _ctx else None
    if not token:
        return _json({"status": "error", "message": "未登录,无法获取 access_token(请先登录 TARS 平台)"})
    url = f"{_PAI_BASE}/api/v1/pai/dataset/list"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    body: dict = {}
    if clusterId is not None:
        body["clusterId"] = clusterId
    if datasetName:
        body["datasetName"] = datasetName
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(url, json=body, headers=headers)
        rj = resp.json()
    except Exception as e:  # noqa: BLE001
        return _json({"status": "error", "message": f"调用数据集列表接口失败: {e}"})
    code = rj.get("code")
    if code in (0, 200):
        data = rj.get("data") or []
        return _json({"status": "ok", "count": len(data), "datasets": data})
    return _json({"status": "error", "code": code, "msg": rj.get("msg", ""), "traceId": rj.get("traceId", "")})


@tool(
    "list_storages",
    "查询可选文件存储列表,返回 [{id(fileStorageId), storageName, storageType, storageTypeName, clusterId, zone}]。用于把用户说的存储名(如'tars-datasets-wlcb')转成 fileStorageId(create_train_job 的 storageMountList[].fileStorageId)。复用平台登录态鉴权。",
    params={
        "clusterId": "可选,集群 id(数字,按集群过滤)",
        "resourceId": "可选,资源池 id(数字,按资源池可用区过滤)",
    },
)
async def list_storages(*, clusterId: Optional[int] = None, resourceId: Optional[int] = None, _ctx=None) -> str:
    """查询可选文件存储列表(GET /api/v1/pai/file-storage/list/optional)。"""
    token = getattr(_ctx, "access_token", None) if _ctx else None
    if not token:
        return _json({"status": "error", "message": "未登录,无法获取 access_token(请先登录 TARS 平台)"})
    url = f"{_PAI_BASE}/api/v1/pai/file-storage/list/optional"
    headers = {"Authorization": f"Bearer {token}"}
    params: dict = {}
    if clusterId is not None:
        params["clusterId"] = clusterId
    if resourceId is not None:
        params["resourceId"] = resourceId
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(url, headers=headers, params=params)
        rj = resp.json()
    except Exception as e:  # noqa: BLE001
        return _json({"status": "error", "message": f"调用文件存储列表接口失败: {e}"})
    code = rj.get("code")
    if code in (0, 200):
        data = rj.get("data") or []
        return _json({"status": "ok", "count": len(data), "storages": data})
    return _json({"status": "error", "code": code, "msg": rj.get("msg", ""), "traceId": rj.get("traceId", "")})


@tool(
    "list_resource_pools",
    "查询资源池列表(可按集群过滤),返回 [{id(resourcePoolId), name, gpuType, gpuCount, availableCount, clusterName, ...}]。用于把用户说的资源池名(如'A800-Group1')转成 resourcePoolId。复用平台登录态鉴权。",
    params={"clusterId": "可选,按集群过滤(数字,从 list_clusters 拿)"},
)
async def list_resource_pools(*, clusterId: Optional[int] = None, _ctx=None) -> str:
    """查询资源池列表(GET /api/v1/pai/resource/allocate-static?clusterId=...)。"""
    token = getattr(_ctx, "access_token", None) if _ctx else None
    if not token:
        return _json({"status": "error", "message": "未登录,无法获取 access_token(请先登录 TARS 平台)"})
    url = f"{_PAI_BASE}/api/v1/pai/resource/allocate-static"
    headers = {"Authorization": f"Bearer {token}"}
    params: dict = {}
    if clusterId is not None:
        params["clusterId"] = clusterId
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(url, headers=headers, params=params)
        rj = resp.json()
    except Exception as e:  # noqa: BLE001
        return _json({"status": "error", "message": f"调用资源池列表接口失败: {e}"})
    code = rj.get("code")
    if code in (0, 200):
        data = rj.get("data") or []
        return _json({"status": "ok", "count": len(data), "resource_pools": data})
    return _json({"status": "error", "code": code, "msg": rj.get("msg", ""), "traceId": rj.get("traceId", "")})


@tool(
    "list_running_jobs",
    "查询当前用户正在运行(Running)的分布式训练任务列表,自动翻页取全量,返回 [{id, name, status, podCount, gpuCount, clusterName, ...}]。"
    "复用平台登录态鉴权。",
    params={
        "pageSize": "可选,单页条数(默认100)",
        "jobName": "可选,按任务名关键词过滤",
    },
)
async def list_running_jobs(*, pageSize: Optional[int] = 100, jobName: Optional[str] = None, _ctx=None) -> str:
    """查询 Running 状态的任务(GET /api/v1/pai/train/page,status=Running,onlyMe=true,自动翻页)。"""
    token = getattr(_ctx, "access_token", None) if _ctx else None
    if not token:
        return _json({"status": "error", "message": "未登录,无法获取 access_token(请先登录 TARS 平台)"})
    url = f"{_PAI_BASE}/api/v1/pai/train/page"
    headers = {"Authorization": f"Bearer {token}"}
    running: list[dict] = []
    page = 1
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            while True:
                params: dict = {"page": page, "pageSize": pageSize, "onlyMe": True, "status": "Running"}
                if jobName:
                    params["jobName"] = jobName
                resp = await client.get(url, headers=headers, params=params)
                rj = resp.json()
                code = rj.get("code")
                if code not in (0, 200):
                    return _json({"status": "error", "code": code, "msg": rj.get("msg", ""), "traceId": rj.get("traceId", "")})
                data = rj.get("data") or {}
                lst = data.get("list") or []
                running.extend(lst)
                total = data.get("total")
                if not lst or (total is not None and len(running) >= total) or len(lst) < pageSize:
                    break
                page += 1
    except Exception as e:  # noqa: BLE001
        return _json({"status": "error", "message": f"调用训练任务列表接口失败: {e}"})
    jobs = [
        {
            "id": j.get("id"),
            "name": j.get("name"),
            "status": j.get("status"),
            "clusterName": j.get("clusterName"),
            "resourceName": j.get("resourceName"),
            "podCount": j.get("podCount"),
            "gpuCount": j.get("gpuCount"),
            "priority": j.get("priority"),
            "submitTimestamp": j.get("submitTimestamp"),
            "startTimestamp": j.get("startTimestamp"),
            "queueReason": j.get("queueReason"),
        }
        for j in running
    ]
    return _json({"status": "ok", "count": len(jobs), "running_jobs": jobs})


# ---- YAML 模板复用:从标准 YAML 创建分布式训练任务 ----
def _match_item(items: list[dict], value: Any, fields: list[str]) -> Optional[dict]:
    """在 items 里按 fields 顺序模糊匹配 value(相等或包含),返回首个匹配项。

    镜像等场景用户给的可能是 imageName / imageUri / tag 片段,故用多字段+包含匹配。
    """
    if not value:
        return None
    v = str(value).strip()
    for it in items or []:
        image_name = str(it.get("imageName") or "").strip()
        tag = str(it.get("tag") or "").strip()
        if image_name and tag and f"{image_name}:{tag}" == v:
            return it
        for f in fields:
            fv = it.get(f)
            if fv is None:
                continue
            if str(fv).strip() == v:
                return it
    for it in items or []:
        for f in fields:
            fv = it.get(f)
            if fv is None:
                continue
            s = str(fv).strip()
            if v in s or s in v:
                return it
    return None


def _cloud_type(cluster_id: Any, company: Any) -> str:
    """判断云厂商:阿里云(集群1/2)/ 火山引擎(集群3/4)。company 字段优先,fallback 用 id。"""
    c = str(company or "")
    if "火山" in c or "volc" in c.lower():
        return "volc"
    if "阿里" in c or "aliyun" in c.lower():
        return "aliyun"
    return "volc" if cluster_id in (3, 4) else "aliyun"


@tool(
    "create_train_job_from_yaml",
    "从标准 YAML 文件创建分布式训练任务(快速复用模式)。YAML key 对齐后端字段,Name 后缀字段"
    "(clusterName/specName/imageName/projectName/resourcePoolName/dataSet[].datasetName/"
    "storageMountList[].storageName)填中文名,tool 自动转 id 后调 create_train_job。"
    "第一次逐步创建后导出 YAML,下次直接用它创建,免去逐项问参。复用平台登录态鉴权。",
    params={
        "yaml_path": "标准 train_job YAML 文件路径(模板见 README.md);"
                     "相对路径相对当前工作目录(cwd)",
        "dry_run": "可选,只解析+转 id+展示将提交的参数,不真正创建(默认 false;建议先 dry_run 确认再正式创建)",
    },
)
async def create_train_job_from_yaml(*, yaml_path: str, dry_run: bool = False, _ctx=None) -> str:
    """读标准 YAML → Name 字段转 id → create_train_job。

    YAML schema 见 README.md(key 对齐后端字段,Name 后缀填名字):
    - 必填: jobName/clusterName/priority/podCount/command/imageName
    - 阿里云(集群1/2): specName+gpuCount[+cpuCount]+errorMonitor
    - 火山引擎(集群3/4): instanceType[+enableTensorboard]
    - 选填: resourcePoolName/projectName/maxRunTime/publicJobGroup/dataSet/storageMountList
    Name 后缀字段由本 tool 调查询接口转 id,用户只需写中文名。
    """
    token = getattr(_ctx, "access_token", None) if _ctx else None
    if not token:
        return _json({"status": "error", "message": "未登录,无法获取 access_token(请先登录 TARS 平台)"})

    # 1) 读 YAML
    p = Path(yaml_path).expanduser()
    if not p.is_absolute():
        p = Path.cwd() / p
    if not p.exists():
        return _json({"status": "error", "message": f"YAML 文件不存在: {p}"})
    try:
        cfg = _yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except _yaml.YAMLError as e:  # noqa: BLE001
        return _json({"status": "error", "message": f"YAML 解析失败: {e}"})
    if not isinstance(cfg, dict):
        return _json({"status": "error", "message": f"YAML 顶层应为 dict,实际 {type(cfg).__name__}"})

    # 2) 校验必填
    required = ["jobName", "clusterName", "priority", "podCount", "command", "imageName"]
    missing = [f for f in required if cfg.get(f) in (None, "")]
    if missing:
        return _json({"status": "error", "message": f"YAML 缺必填字段: {missing}", "yaml_path": str(p)})

    steps: list[str] = []  # 记录 Name→id 转换过程,便于定位

    # 3) 集群 clusterName → clusterId + 云厂商
    clusters_r = json.loads(await list_clusters(_ctx=_ctx))
    if clusters_r.get("status") != "ok":
        return _json({"status": "error", "message": "查集群列表失败", "detail": clusters_r})
    cinfo = _match_item(clusters_r["clusters"], cfg["clusterName"], ["name"])
    if not cinfo:
        return _json({"status": "error", "message": f"未找到集群 '{cfg['clusterName']}'",
                      "available": [c.get("name") for c in clusters_r["clusters"]]})
    cluster_id = cinfo.get("id")
    cloud = _cloud_type(cluster_id, cinfo.get("company"))
    steps.append(f"集群 '{cfg['clusterName']}' → clusterId={cluster_id} [{cloud}]")

    # 4) 镜像 imageName → imageId(多字段模糊匹配)
    images_r = json.loads(await list_images(clusterId=cluster_id, _ctx=_ctx))
    if images_r.get("status") != "ok":
        return _json({"status": "error", "message": "查镜像列表失败", "detail": images_r})
    iinfo = _match_item(images_r["images"], cfg["imageName"], ["imageName", "imageUri", "imageId", "tag"])
    if not iinfo:
        return _json({"status": "error", "message": f"未找到镜像 '{cfg['imageName']}'",
                      "available": [i.get("imageName") for i in images_r["images"][:20]]})
    image_id = iinfo.get("imageId") or iinfo.get("id")
    steps.append(f"镜像 '{cfg['imageName']}' → imageId={image_id}")

    # 5) 资源池 resourcePoolName → resourcePoolId(选填)
    resource_pool_id = None
    if cfg.get("resourcePoolName"):
        rp_r = json.loads(await list_resource_pools(clusterId=cluster_id, _ctx=_ctx))
        if rp_r.get("status") != "ok":
            return _json({"status": "error", "message": "查资源池列表失败", "detail": rp_r})
        rp = _match_item(rp_r["resource_pools"], cfg["resourcePoolName"], ["name"])
        if not rp:
            return _json({"status": "error", "message": f"未找到资源池 '{cfg['resourcePoolName']}'",
                          "available": [r.get("name") for r in rp_r["resource_pools"]]})
        resource_pool_id = rp.get("id")
        steps.append(f"资源池 '{cfg['resourcePoolName']}' → resourcePoolId={resource_pool_id}")

    # 6) 研发项目 projectName → projectId(选填)
    project_id = None
    if cfg.get("projectName"):
        pj_r = json.loads(await list_projects(_ctx=_ctx))
        if pj_r.get("status") != "ok":
            return _json({"status": "error", "message": "查项目列表失败", "detail": pj_r})
        pj = _match_item(pj_r["projects"], cfg["projectName"], ["name"])
        if not pj:
            return _json({"status": "error", "message": f"未找到项目 '{cfg['projectName']}'",
                          "available": [x.get("name") for x in pj_r["projects"]]})
        project_id = pj.get("id")
        steps.append(f"项目 '{cfg['projectName']}' → projectId={project_id}")

    # 7) 机器规格(按云厂商分支)
    spec_id = None
    instance_type_info = None
    if cloud == "aliyun":
        if not cfg.get("specName"):
            return _json({"status": "error", "message": "阿里云集群 YAML 必须填 specName(GPU 类型名,如 NVIDIA-RTX4090)"})
        if not cfg.get("gpuCount"):
            return _json({"status": "error", "message": "阿里云集群 YAML 必须填 gpuCount(单节点卡数)"})
        gt_r = json.loads(await list_gpu_types(clusterId=cluster_id, _ctx=_ctx))
        if gt_r.get("status") != "ok":
            return _json({"status": "error", "message": "查 GPU 类型失败", "detail": gt_r})
        gt = _match_item(gt_r["gpu_types"], cfg["specName"], ["name"])
        if not gt:
            return _json({"status": "error", "message": f"未找到 GPU 类型 '{cfg['specName']}'",
                          "available": [g.get("name") for g in gt_r["gpu_types"]]})
        spec_id = gt.get("specId")
        steps.append(f"规格 '{cfg['specName']}' → specId={spec_id}")
    else:  # 火山引擎:instanceType 即 instance_specs 的 id,直接构造 instanceTypeInfo
        if not cfg.get("instanceType"):
            return _json({"status": "error", "message": "火山引擎集群 YAML 必须填 instanceType(实例规格 id,如 ml.xni3c.22xlarge)"})
        instance_type_info = {"id": cfg["instanceType"]}
        steps.append(f"实例规格 '{cfg['instanceType']}' → instanceTypeInfo.id")

    # 8) 数据集挂载 dataSet[].datasetName → datasetLocalId(选填,可多条)
    data_set = None
    if cfg.get("dataSet"):
        ds_r = json.loads(await list_datasets(clusterId=cluster_id, _ctx=_ctx))
        if ds_r.get("status") != "ok":
            return _json({"status": "error", "message": "查数据集列表失败", "detail": ds_r})
        data_set = []
        for d in cfg["dataSet"]:
            di = _match_item(ds_r["datasets"], d.get("datasetName"), ["datasetName", "name"])
            if not di:
                return _json({"status": "error", "message": f"未找到数据集 '{d.get('datasetName')}'",
                              "available": [x.get("datasetName") for x in ds_r["datasets"][:20]]})
            item = {"datasetLocalId": di.get("id") or di.get("datasetLocalId"), "mountPath": d["mountPath"]}
            if d.get("version") is not None:
                item["version"] = d["version"]
            if d.get("readOnly") is not None:
                item["readOnly"] = d["readOnly"]
            if d.get("config") is not None:
                item["config"] = d["config"]  # create_train_job 内部 _coerce_config 转 JSON 字符串
            data_set.append(item)
            steps.append(f"数据集 '{d.get('datasetName')}' → datasetLocalId={item['datasetLocalId']}")

    # 9) 存储挂载 storageMountList[].storageName → fileStorageId(选填,可多条)
    storage_mount_list = None
    if cfg.get("storageMountList"):
        st_r = json.loads(await list_storages(clusterId=cluster_id, resourceId=resource_pool_id, _ctx=_ctx))
        if st_r.get("status") != "ok":
            return _json({"status": "error", "message": "查存储列表失败", "detail": st_r})
        storage_mount_list = []
        for s in cfg["storageMountList"]:
            si = _match_item(st_r["storages"], s.get("storageName"), ["storageName", "name"])
            if not si:
                return _json({"status": "error", "message": f"未找到存储 '{s.get('storageName')}'",
                              "available": [x.get("storageName") for x in st_r["storages"][:20]]})
            item = {"fileStorageId": si.get("id") or si.get("fileStorageId"),
                    "fileSystemPath": s["fileSystemPath"], "mountPath": s["mountPath"]}
            if s.get("readOnly") is not None:
                item["readOnly"] = s["readOnly"]
            if s.get("workspaceSourceFlag") is not None:
                item["workspaceSourceFlag"] = s["workspaceSourceFlag"]
            if s.get("config") is not None:
                item["config"] = s["config"]
            storage_mount_list.append(item)
            steps.append(f"存储 '{s.get('storageName')}' → fileStorageId={item['fileStorageId']}")

    # 10) 组装入参,调 create_train_job
    kwargs: dict = dict(
        clusterId=cluster_id,
        jobName=cfg["jobName"],
        imageId=image_id,
        podCount=cfg["podCount"],
        priority=cfg["priority"],
        command=cfg["command"],
    )
    if spec_id is not None:
        kwargs["specId"] = spec_id
    if instance_type_info is not None:
        kwargs["instanceTypeInfo"] = instance_type_info
    if resource_pool_id is not None:
        kwargs["resourcePoolId"] = resource_pool_id
    if project_id is not None:
        kwargs["projectId"] = project_id
    # 以下字段 YAML key 与后端一致,直接透传
    for k in ("gpuCount", "cpuCount", "errorMonitor", "enableTensorboard",
              "maxRunTime", "publicJobGroup"):
        if cfg.get(k) is not None:
            kwargs[k] = cfg[k]
    if data_set is not None:
        kwargs["dataSet"] = data_set
    if storage_mount_list is not None:
        kwargs["storageMountList"] = storage_mount_list

    if dry_run:
        return _json({"status": "dry_run", "yaml_path": str(p), "cloud": cloud,
                      "resolved": steps, "will_submit": kwargs})

    result = json.loads(await create_train_job(_ctx=_ctx, **kwargs))
    result["yaml_path"] = str(p)
    result["resolved"] = steps
    return _json(result)


# ---- 全量聚合:一次查清集群/资源池/机器规格/项目,免去逐层询问 ----
async def _query_cluster_resources(c: dict, _ctx=None) -> dict:
    """查单个集群的资源池 + 机器规格(阿里云 gpu_types 集群级 / 火山 instance_specs 按资源池)。"""
    cid = c.get("id")
    cloud = _cloud_type(cid, c.get("company"))
    rp_r = json.loads(await list_resource_pools(clusterId=cid, _ctx=_ctx))
    rps = rp_r.get("resource_pools", []) if rp_r.get("status") == "ok" else []
    entry: dict = {
        "clusterId": cid,
        "clusterName": c.get("name"),
        "company": c.get("company"),
        "cloud": cloud,  # aliyun | volc
        "resource_pools": [],
    }
    if cloud == "aliyun":
        # 阿里云:GPU 类型是集群级枚举(specId)
        gt_r = json.loads(await list_gpu_types(clusterId=cid, _ctx=_ctx))
        entry["machine_specs"] = [
            {"specId": g.get("specId"), "name": g.get("name")}
            for g in (gt_r.get("gpu_types") or [])
        ]
        entry["machine_spec_type"] = "specId"  # 提示 agent:阿里云填 specName→specId
        entry["resource_pools"] = [
            {"id": r.get("id"), "name": r.get("name"),
             "gpuCount": r.get("gpuCount"), "availableCount": r.get("availableCount")}
            for r in rps
        ]
    else:
        # 火山:计算规格按资源池查(instanceTypeInfo),并发拉每个资源池的 instance_specs
        async def _q_rp(r: dict) -> dict:
            is_r = json.loads(await list_instance_specs(resourceId=r.get("id"), kind="GPU", _ctx=_ctx))
            specs = is_r.get("instance_specs", []) if is_r.get("status") == "ok" else []
            return {"id": r.get("id"), "name": r.get("name"),
                    "gpuCount": r.get("gpuCount"), "availableCount": r.get("availableCount"),
                    "instance_specs": specs}
        entry["resource_pools"] = await asyncio.gather(*[_q_rp(r) for r in rps])
        entry["machine_specs"] = None
        entry["machine_spec_type"] = "instanceTypeInfo"  # 火山:填 instanceType→instanceTypeInfo.id
    return entry


@tool(
    "list_train_job_resources",
    "一次性并发聚合查询创建分布式训练任务所需的核心选项:集群 → 资源池 → 机器规格(阿里云 GPU 类型 specId / "
    "火山引擎计算规格 instanceTypeInfo)+ 研发项目,按集群分组结构化返回。用于开场把全量选项渲染成一张表,"
    "让用户一轮选定集群/资源池/规格/项目,免去'选集群→选资源池→选规格'多轮询问。"
    "镜像/数据集/存储数量大且按集群过滤,不在此聚合,选定集群后用 list_images/list_datasets/list_storages 查。"
    "复用平台登录态鉴权。",
    params={
        "cluster_id": "可选,只查指定集群(数字 clusterId);不传则查全部集群(并发,耗时不随集群数线性增长)",
    },
)
async def list_train_job_resources(*, cluster_id: Optional[int] = None, _ctx=None) -> str:
    """聚合查询集群/资源池/机器规格/项目(并发)。返回 {clusters:[...], projects:[...]}。

    每个集群 entry:
      - 阿里云: {cloud:'aliyun', machine_spec_type:'specId', machine_specs:[{specId,name}],
                resource_pools:[{id,name,gpuCount,availableCount}]}
      - 火山:   {cloud:'volc', machine_spec_type:'instanceTypeInfo', machine_specs:null,
                resource_pools:[{id,name,gpuCount,availableCount,instance_specs:[{id,family,cpu,gpu,...}]}]}
    """
    token = getattr(_ctx, "access_token", None) if _ctx else None
    if not token:
        return _json({"status": "error", "message": "未登录,无法获取 access_token(请先登录 TARS 平台)"})

    # 1) 集群(可过滤)
    clusters_r = json.loads(await list_clusters(_ctx=_ctx))
    if clusters_r.get("status") != "ok":
        return _json({"status": "error", "message": "查集群列表失败", "detail": clusters_r})
    clusters = clusters_r.get("clusters") or []
    if cluster_id is not None:
        clusters = [c for c in clusters if c.get("id") == cluster_id]
    if not clusters:
        return _json({"status": "error", "message": "无可用集群"})

    # 2) 研发项目(全局,与集群查询并发)
    # 3) 每个集群的资源池+规格(集群间并发,火山集群内资源池并发)
    projects_task = asyncio.create_task(list_projects(_ctx=_ctx))
    cluster_entries = await asyncio.gather(
        *[_query_cluster_resources(c, _ctx) for c in clusters]
    )
    projects_r = json.loads(await projects_task)
    projects = [
        {"id": p.get("id"), "name": p.get("name")}
        for p in (projects_r.get("projects") or [])
    ] if projects_r.get("status") == "ok" else []

    return _json({
        "status": "ok",
        "count": len(cluster_entries),
        "clusters": cluster_entries,
        "projects": projects,
        "note": "阿里云填 specName(→specId)+gpuCount;火山引擎填 instanceType(→instanceTypeInfo.id),不填 specId/gpuCount。"
                "镜像/数据集/存储选定集群后用 list_images/list_datasets/list_storages 查。",
    })
