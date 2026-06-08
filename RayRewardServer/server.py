from __future__ import annotations
from handlers import *
from pathlib import Path

import argparse
from typing import Any, Dict, List

import ray
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from commons.config_loader import load_service_config
from core.scheduler import init_handler_groups


class InferenceRequest(BaseModel):
    items: List[Any]


def parse_args():
    parser = argparse.ArgumentParser(description="Ray multi-handler GPU service")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--ray-address", type=str, default=None, help="Ray cluster address, e.g. ray://host:10001")

    # 其余参数交给 OmegaConf dotlist 解析（兼容：--a.b 1 / a.b=1 / --flag）
    args, unknown = parser.parse_known_args()
    return args, unknown


def build_app(handler_groups: Dict[str, Any]) -> FastAPI:
    app = FastAPI(title="Multi-Handler Ray GPU Service")

    @app.get("/handlers")
    def list_handlers() -> Dict[str, Any]:
        return {
            "handlers": [
                {
                    "name": name,
                    "num_workers": len(getattr(group, "workers", [])),
                }
                for name, group in handler_groups.items()
            ]
        }

    @app.get("/health")
    def global_health() -> Dict[str, Any]:
        """判断是否所有 handler 都已可用（readiness 聚合）。

        返回:
          - ok: bool，所有 handler overall_status 都为 ok 才为 True
          - handlers: 各 handler 的 health() 详情
        """
        handler_reports: Dict[str, Any] = {}
        all_ok = True

        for name, group in handler_groups.items():
            try:
                report = group.health()
            except Exception as e:
                report = {
                    "handler": name,
                    "overall_status": "error",
                    "ok": False,
                    "error": f"group.health() failed: {type(e).__name__}: {e}",
                }
            handler_reports[name] = report
            if report.get("overall_status") != "ok":
                all_ok = False

        return {
            "ok": all_ok,
            "overall_status": "ok" if all_ok else "error",
            "num_handlers": len(handler_groups),
            "handlers": handler_reports,
        }

    @app.get("/{handler_name}/health")
    def handler_health(handler_name: str) -> Dict[str, Any]:
        group = handler_groups.get(handler_name)
        if group is None:
            raise HTTPException(status_code=404, detail=f"Unknown handler '{handler_name}'")
        return group.health()

    @app.post("/{handler_name}")
    def handler_infer(handler_name: str, req: InferenceRequest) -> Dict[str, Any]:
        group = handler_groups.get(handler_name)
        if group is None:
            raise HTTPException(status_code=404, detail=f"Unknown handler '{handler_name}'")

        results = group.infer(req.items)
        return {
            "handler": handler_name,
            "num_items": len(req.items),
            "results": results,
        }

    return app


def main():
    args, unknown = parse_args()

    # 仿 EasyR1：structured defaults + file config + CLI overrides
    cfg_obj = load_service_config(
        config_path=args.config,
        cli_argv=unknown,
        ray_address=args.ray_address,
    )
    cfg = cfg_obj.to_dict()

    host = cfg_obj.server.host
    port = cfg_obj.server.port

    # 初始化 Ray（address=None 表示本地启动）
    ray.init(address=cfg_obj.ray_address, ignore_reinit_error=True)

    ROOT_DIR = Path(__file__).resolve().parent  # RayRewardServer 根目录
    ray.init(
        address=cfg_obj.ray_address,
        runtime_env={"working_dir": str(ROOT_DIR)},
        ignore_reinit_error=True,
    )

    # 初始化各个 handler 对应的 HandlerGroup
    handler_groups = init_handler_groups(cfg)

    app = build_app(handler_groups)

    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
