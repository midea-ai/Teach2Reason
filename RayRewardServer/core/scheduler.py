# scheduler.py
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import ray
import yaml
from ray.util.placement_group import placement_group

from .workers import GPUWorker


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class HandlerGroup:
    """管理某一个 handler 的一组 GPU workers。

    - 按 config 创建 placement group + 多个 GPUWorker actor；
    - infer(): 把一个 batch 拆块，发给多个 worker 并并行执行；
    - health(): 聚合所有 worker.health()。
    """

    def __init__(self, name: str, handler_cfg: Dict[str, Any], cluster_cfg: Dict[str, Any], worker_start_id: int):
        self.name = name
        self.handler_cfg = handler_cfg
        self.cluster_cfg = cluster_cfg
        self.workers: List[ray.actor.ActorHandle] = []
        self.worker_start_id = worker_start_id
        self._init_workers()

    def _init_workers(self) -> None:
        num_workers = int(self.handler_cfg.get("num_workers", 1))
        gpu_per_worker = float(self.handler_cfg.get("gpu_per_worker", 1.0))
        cpu_per_worker = float(self.cluster_cfg.get("cpus_per_worker", 1.0))

        class_name = self.handler_cfg["class_name"]
        init_kwargs = self.handler_cfg.get("init_kwargs", {})

        # vLLM will initialize torch.distributed even when dp/tp=1.
        # If this service is launched inside a torchrun environment, all Ray
        # actors may inherit the same MASTER_PORT and crash with EADDRINUSE.
        # We therefore assign a per-actor port starting from this base.
        torch_dist_port_base = int(self.cluster_cfg.get("torch_dist_port_base", 61000))

        # 为当前 handler 创建一个 placement group，把每个 worker 放在一个 bundle 里
        bundles = [{"GPU": gpu_per_worker, "CPU": cpu_per_worker} for _ in range(num_workers)]
        strategy = str(self.handler_cfg.get("placement_strategy", "PACK")).upper()
        if strategy not in ("PACK", "SPREAD", "STRICT_PACK", "STRICT_SPREAD"):
            raise ValueError(f"Invalid placement_strategy for handler {self.name}: {strategy}")
        self.pg = placement_group(bundles, strategy=strategy)
        ray.get(self.pg.ready())

        for i in range(num_workers):
            worker = GPUWorker.options(
                num_gpus=gpu_per_worker,
                num_cpus=cpu_per_worker,
                placement_group=self.pg,
                placement_group_bundle_index=i,
                name=f"{self.name}_worker_{i}",
            ).remote(
                class_name=class_name,
                handler_init_kwargs=init_kwargs,
                worker_index=i + self.worker_start_id,
                torch_dist_port_base=torch_dist_port_base,
            )
            self.workers.append(worker)

    def infer(self, items: List[Any]) -> List[Any]:
        """把一个 batch 拆到多个 GPU worker 上，并保持原始顺序。"""
        if not items:
            return []

        n = len(items)
        num_workers = min(len(self.workers), n)

        indices = np.arange(n)
        index_chunks = np.array_split(indices, num_workers)

        futures = []
        for worker, idx_chunk in zip(self.workers[:num_workers], index_chunks):
            idx_list = idx_chunk.tolist()
            if not idx_list:
                continue
            chunk_items = [items[i] for i in idx_list]
            items_with_idx: List[Tuple[int, Any]] = list(zip(idx_list, chunk_items))
            futures.append(worker.infer.remote(items_with_idx))

        results_per_worker = ray.get(futures)
        flat: List[Tuple[int, Any]] = [pair for sub in results_per_worker for pair in sub]
        flat.sort(key=lambda x: x[0])
        return [val for _, val in flat]

    def health(self) -> Dict[str, Any]:
        """聚合所有 worker.health() 的健康状态。"""
        worker_reports: List[Dict[str, Any]] = []
        # 先把所有 health 调用发出去，让 Ray 并行执行，再逐个收集结果（避免串行等待）。
        futures: List[Tuple[int, Any]] = []
        for idx, w in enumerate(self.workers):
            futures.append((idx, w.health.remote()))

        for idx, ref in futures:
            try:
                report = ray.get(ref)
            except Exception as e:
                report = {
                    "ok": False,
                    "error": f"worker call failed: {type(e).__name__}: {e}",
                }
            report = dict(report) if isinstance(report, dict) else {"ok": False, "raw": report}
            report.setdefault("ok", False)
            report.setdefault("worker_index", idx)
            worker_reports.append(report)

        overall_ok = all(r.get("ok", False) for r in worker_reports)

        return {
            "handler": self.name,
            "overall_status": "ok" if overall_ok else "error",
            "num_workers": len(self.workers),
            "workers": worker_reports,
        }


def init_handler_groups(config: Dict[str, Any]) -> Dict[str, HandlerGroup]:
    cluster_cfg = config.get("cluster", {})
    handlers_cfg: Dict[str, Any] = config.get("handlers", {})

    # 1) 简单的 GPU 资源检查：所有 handler 的 GPU 需求和不能超过 Ray 报告的总 GPU
    total_available_gpus = float(ray.available_resources().get("GPU", 0.0))
    total_needed = 0.0
    for name, hcfg in handlers_cfg.items():
        num_workers = int(hcfg.get("num_workers", 1))
        gpu_per_worker = float(hcfg.get("gpu_per_worker", 1.0))
        total_needed += num_workers * gpu_per_worker

    if total_needed > total_available_gpus + 1e-6:
        raise RuntimeError(
            f"Handlers need {total_needed} GPUs in total, "
            f"but Ray only reports {total_available_gpus}."
        )

    # 2) 为每个 handler 创建一个 HandlerGroup
    groups: Dict[str, HandlerGroup] = {}
    worker_start_id = 0
    for name, hcfg in handlers_cfg.items():
        groups[name] = HandlerGroup(name=name, handler_cfg=hcfg, cluster_cfg=cluster_cfg,
                                    worker_start_id=worker_start_id)
        worker_start_id += hcfg.get("num_workers", 1)

    return groups
