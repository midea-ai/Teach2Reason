from __future__ import annotations

from typing import List, Optional

from omegaconf import OmegaConf

from .config_schema import ServiceConfig


def _unknown_to_dotlist(unknown: List[str]) -> List[str]:
    """把 argparse.parse_known_args() 的 unknown tokens 规范化为 OmegaConf dotlist。

    支持：
      - key=value
      - --key=value
      - --key value
      - --flag        (等价于 flag=true)

    其中 key 可以是 dotted key（例如 handlers.dummy.num_workers）。
    """
    dotlist: List[str] = []
    i = 0
    while i < len(unknown):
        tok = unknown[i]

        # 允许直接传 key=value（OmegaConf/Hydra 风格）
        if not tok.startswith("--"):
            if "=" in tok:
                dotlist.append(tok)
            i += 1
            continue

        # --key 或 --key=value
        key = tok[2:]
        if not key:
            i += 1
            continue

        if "=" in key:
            # --a.b=1 -> a.b=1
            dotlist.append(key)
            i += 1
            continue

        # --key value 或 --flag
        if i + 1 < len(unknown) and not unknown[i + 1].startswith("--"):
            dotlist.append(f"{key}={unknown[i + 1]}")
            i += 2
        else:
            dotlist.append(f"{key}=true")
            i += 1

    return dotlist


def load_service_config(
    config_path: str = "config.yaml",
    cli_argv: Optional[List[str]] = None,
    ray_address: Optional[str] = None,
) -> ServiceConfig:
    """仿 EasyR1 的方式加载配置：structured defaults + file + cli overrides。

    - structured 默认值来自 ServiceConfig dataclass
    - file 来自 config_path（若 cli 里带 config=... 会覆盖）
    - cli overrides 来自 unknown tokens（支持 --a.b 1 / a.b=1 等写法）
    """
    cli_argv = cli_argv or []
    dotlist = _unknown_to_dotlist(cli_argv)
    cli_cfg = OmegaConf.from_dotlist(dotlist) if dotlist else OmegaConf.create()

    # 兼容 EasyR1 风格：允许直接在 CLI 里传 config=xxx
    if "config" in cli_cfg:
        config_path = str(cli_cfg["config"])
        del cli_cfg["config"]

    if ray_address is not None:
        cli_cfg = OmegaConf.merge(cli_cfg, OmegaConf.create({"ray_address": ray_address}))

    default_cfg = OmegaConf.structured(ServiceConfig())

    if config_path:
        file_cfg = OmegaConf.load(config_path)
        merged = OmegaConf.merge(default_cfg, file_cfg)
    else:
        merged = default_cfg

    merged = OmegaConf.merge(merged, cli_cfg)

    cfg_obj: ServiceConfig = OmegaConf.to_object(merged)
    cfg_obj.deep_post_init()
    return cfg_obj
