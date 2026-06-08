from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from typing import Any, Dict, Optional


def recursive_post_init(dataclass_obj: Any) -> None:
    """递归调用 dataclass_obj 及其子 dataclass 的 post_init()（如果存在）。"""
    if hasattr(dataclass_obj, "post_init"):
        dataclass_obj.post_init()  # type: ignore[attr-defined]

    for attr in fields(dataclass_obj):
        value = getattr(dataclass_obj, attr.name)

        if is_dataclass(value):
            recursive_post_init(value)
            continue

        # 处理 Dict[str, dataclass] 的情况（例如 handlers: Dict[str, HandlerConfig]）
        if isinstance(value, dict):
            for v in value.values():
                if is_dataclass(v):
                    recursive_post_init(v)


@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 8000

    def post_init(self) -> None:
        self.port = int(self.port)
        if not (1 <= self.port <= 65535):
            raise ValueError(f"server.port must be in [1, 65535], got {self.port}")


@dataclass
class ClusterConfig:
    nnodes: int = 1
    gpus_per_node: float = 1.0
    cpus_per_worker: float = 1.0
    # Base port for per-actor torch.distributed TCPStore (MASTER_PORT).
    # When this service is launched inside a torchrun / kubeflow pytorchjob,
    # env vars like MASTER_PORT may be inherited by all Ray actors and cause
    # port collisions (EADDRINUSE). We therefore set a unique port per actor
    # starting from this base.
    torch_dist_port_base: int = 61000

    def post_init(self) -> None:
        self.nnodes = int(self.nnodes)
        self.gpus_per_node = float(self.gpus_per_node)
        self.cpus_per_worker = float(self.cpus_per_worker)
        if self.nnodes <= 0:
            raise ValueError(f"cluster.nnodes must be > 0, got {self.nnodes}")
        if self.gpus_per_node <= 0:
            raise ValueError(f"cluster.gpus_per_node must be > 0, got {self.gpus_per_node}")
        if self.cpus_per_worker <= 0:
            raise ValueError(f"cluster.cpus_per_worker must be > 0, got {self.cpus_per_worker}")

        self.torch_dist_port_base = int(self.torch_dist_port_base)
        if not (1024 <= self.torch_dist_port_base <= 65535):
            raise ValueError(
                f"cluster.torch_dist_port_base must be in [1024, 65535], got {self.torch_dist_port_base}"
            )


@dataclass
class HandlerConfig:
    """单个 handler 的配置。

    注意：本项目通过 handlers.registry + class_name 来定位 Handler 类。
    """
    class_name: str = ""
    num_workers: int = 1
    gpu_per_worker: float = 1.0
    init_kwargs: Dict[str, Any] = field(default_factory=dict)

    # 预留：未来如果你想让不同 handler 使用不同的 placement group 策略
    placement_strategy: str = "PACK"

    def post_init(self) -> None:
        if not self.class_name:
            raise ValueError("handlers.<name>.class_name must be set (e.g. 'DummyHandler').")

        self.num_workers = int(self.num_workers)
        if self.num_workers <= 0:
            raise ValueError(f"handlers.<name>.num_workers must be > 0, got {self.num_workers}")

        self.gpu_per_worker = float(self.gpu_per_worker)
        if self.gpu_per_worker <= 0:
            raise ValueError(f"handlers.<name>.gpu_per_worker must be > 0, got {self.gpu_per_worker}")

        if self.init_kwargs is None:
            self.init_kwargs = {}
        if not isinstance(self.init_kwargs, dict):
            raise TypeError(
                f"handlers.<name>.init_kwargs must be a dict, got {type(self.init_kwargs).__name__}"
            )

        self.placement_strategy = str(self.placement_strategy).upper()
        if self.placement_strategy not in ("PACK", "SPREAD", "STRICT_PACK", "STRICT_SPREAD"):
            raise ValueError(
                f"handlers.<name>.placement_strategy must be one of PACK/SPREAD/STRICT_PACK/STRICT_SPREAD, got {self.placement_strategy}"
            )


@dataclass
class ServiceConfig:
    server: ServerConfig = field(default_factory=ServerConfig)
    cluster: ClusterConfig = field(default_factory=ClusterConfig)
    handlers: Dict[str, HandlerConfig] = field(default_factory=dict)

    # 允许从 CLI 覆盖（--ray-address / ray_address=...）
    ray_address: Optional[str] = None

    def post_init(self) -> None:
        # OmegaConf merge 后，handlers 的 value 可能是 dict，统一转成 HandlerConfig
        fixed: Dict[str, HandlerConfig] = {}
        for name, h in (self.handlers or {}).items():
            if isinstance(h, HandlerConfig):
                fixed[name] = h
            elif isinstance(h, dict):
                fixed[name] = HandlerConfig(**h)
            else:
                raise TypeError(
                    f"handlers.{name} must be a mapping or HandlerConfig, got {type(h).__name__}"
                )
        self.handlers = fixed

        if not self.handlers:
            raise ValueError("At least one handler must be configured under `handlers`.")

    def deep_post_init(self) -> None:
        recursive_post_init(self)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
