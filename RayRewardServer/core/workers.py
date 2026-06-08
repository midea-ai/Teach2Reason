# workers.py
from __future__ import annotations
import os
import socket
import importlib
from typing import Any, Dict, List, Tuple

import ray


@ray.remote
class GPUWorker:
    """通用 GPU worker：包装一个用户自定义 Handler 类。

    - __init__ 里按 class_path 加载 handler 类并实例化；
    - infer() 负责处理一个 chunk 的 batch；
    - health() 调用 handler.health()（如果存在）并做规范化。
    """

    def __init__(
            self,
            class_name: str,
            handler_init_kwargs: dict | None = None,
            worker_index: int = 0,
            torch_dist_port_base: int = 61000,
    ):
        # NOTE:
        # This service is often launched inside a PyTorch distributed job
        # (torchrun / kubeflow pytorchjob). In that environment, env vars like
        # MASTER_ADDR/MASTER_PORT/RANK/WORLD_SIZE are commonly set.
        # vLLM internally calls torch.distributed.init_process_group() even
        # when data_parallel_size=1, and will try to create a TCPStore server
        # on MASTER_PORT. If multiple Ray actors inherit the same MASTER_PORT,
        # they will fight for the same port and crash with EADDRINUSE.
        #
        # Fix: sanitize/override torch.distributed env vars *per actor* BEFORE
        # importing the handler module (because handler modules import vllm at
        # import time).
        self._setup_local_torch_dist_env(worker_index, torch_dist_port_base)

        module_path, class_name = class_name.split(":")
        module = importlib.import_module(module_path)
        handler_cls = getattr(module, class_name)
        self.handler = handler_cls(**(handler_init_kwargs or {}))

    @staticmethod
    def _port_is_free(port: int, host: str = "0.0.0.0") -> bool:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((host, port))
            return True
        except OSError:
            return False
        finally:
            try:
                s.close()
            except Exception:
                pass

    @classmethod
    def _pick_dist_port(cls, worker_index: int, base: int) -> int:
        # Deterministic scan: base+idx, base+idx+N, ... to avoid collisions
        # when many actors start concurrently.
        # Keep the window reasonably small.
        start = base + int(worker_index)
        for k in range(0, 1000):
            port = start + 16 * k  # stride reduces collision between adjacent idx
            if 1024 <= port <= 65535 and cls._port_is_free(port):
                return port
        # Fall back: let OS choose a free port (rare path)
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("0.0.0.0", 0))
        port = s.getsockname()[1]
        s.close()
        return int(port)

    @classmethod
    def _setup_local_torch_dist_env(cls, worker_index: int, torch_dist_port_base: int) -> None:
        # Clear potentially dangerous vars from outer torchrun/elastic context.
        # Keep this list conservative; it's fine to remove more.
        for k in [
            "RANK",
            "WORLD_SIZE",
            "LOCAL_RANK",
            "LOCAL_WORLD_SIZE",
            "NODE_RANK",
            "GROUP_RANK",
            "ROLE_RANK",
            "ROLE_WORLD_SIZE",
            "TORCHELASTIC_RUN_ID",
            "TORCHELASTIC_RESTART_COUNT",
            "TORCHELASTIC_MAX_RESTARTS",
            "TORCHELASTIC_ERROR_FILE",
            "VLLM_HOST_IP",
            "VLLM_RPC_BASE_PORT",
        ]:
            os.environ.pop(k, None)

        # os.environ["VLLM_USE_V1"] = "1"  # 使用VLLM v1
        # Per-actor unique port.
        port = cls._pick_dist_port(worker_index, int(torch_dist_port_base))
        base_vllm_port = int(os.environ.get("VLLM_PORT_BASE", 50000))
        worker_torch_port = torch_dist_port_base + (worker_index * 10)
        worker_vllm_port = base_vllm_port + (worker_index * 50)
        os.environ["RAY_PYTORCH_MASTER_PORT"] = str(70000 + worker_index * 40)
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(worker_torch_port)
        # vLLM 相关端口设置
        os.environ["VLLM_HOST_IP"] = "127.0.0.1"
        # os.environ["VLLM_RPC_BASE_PORT"] = str(worker_vllm_port)
        os.environ["VLLM_PORT"] = str(worker_vllm_port)

    def infer(self, items_with_idx: List[Tuple[int, Any]]) -> List[Tuple[int, Any]]:
        """对一小块 (chunk) 数据做推理。

        items_with_idx: [(原始索引, item), ...]
        返回:        [(原始索引, output), ...]
        """
        if not items_with_idx:
            return []

        indices, items = zip(*items_with_idx)

        # 优先用批处理接口
        if hasattr(self.handler, "process_batch"):
            outputs = self.handler.process_batch(list(items))
        else:
            outputs = [self.handler(item) for item in items]

        # 规范化/校验 outputs，避免 zip() 静默截断导致“丢数据但不报错”
        if outputs is None:
            raise ValueError(
                f"[{type(self.handler).__name__}] process_batch/__call__ returned None; "
                f"expected a sequence with length {len(items)}."
            )

        # 允许 list/tuple 等序列；明确拒绝 str/bytes/dict（这些用 list() 会产生很诡异的结果）
        if isinstance(outputs, (str, bytes, dict)):
            raise TypeError(
                f"[{type(self.handler).__name__}] returned {type(outputs).__name__}; "
                "expected a list-like sequence of outputs."
            )

        if not isinstance(outputs, list):
            try:
                outputs = list(outputs)
            except Exception as e:
                raise TypeError(
                    f"[{type(self.handler).__name__}] returned non-iterable outputs: {type(outputs).__name__}"
                ) from e

        if len(outputs) != len(items):
            raise ValueError(
                f"[{type(self.handler).__name__}] output length mismatch: "
                f"got {len(outputs)} outputs for {len(items)} inputs. "
                "Your handler must return exactly one output per input item."
            )

        return list(zip(indices, outputs))

    def health(self) -> Dict[str, Any]:
        """真正意义上的健康检查。

        - 如果 handler 实现了 health()，就用它的结果；
        - 支持 bool / dict / 其他类型（会包一层）。
        """
        info: Dict[str, Any]

        if hasattr(self.handler, "health"):
            try:
                res = self.handler.health()
            except Exception as e:
                return {
                    "ok": False,
                    "error": f"handler.health() raised: {type(e).__name__}: {e}",
                }

            if isinstance(res, bool):
                info = {"ok": res}
            elif isinstance(res, dict):
                # 确保有 ok 字段
                ok = res.get("ok", True)
                info = {"ok": bool(ok), **res}
            else:
                info = {"ok": True, "detail": res}
        else:
            # 没实现 health，就只能认为“看起来正常”
            info = {"ok": True, "detail": "no custom health() defined in handler"}

        return info
