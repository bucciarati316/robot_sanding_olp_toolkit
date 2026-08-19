"""元数据驱动的 CAM 插件基类。

跨模块数据契约统一定义在 :mod:`schemas`。本模块只保留插件行为协议，
并重导出旧调用方使用的 schema 名称，避免形成第二套平行类型。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List

from .schemas import (
    DataType,
    DebugItem,
    ParamDef,
    ParamType,
    ToolpathResult,
)

__all__ = [
    "BaseAlgorithm",
    "DataType",
    "DebugItem",
    "ParamDef",
    "ParamType",
    "ToolpathResult",
]


class BaseAlgorithm(ABC):
    """所有元数据驱动刀路算法必须实现的唯一行为协议。"""

    NAME: str = "Unnamed Algorithm"
    SUPPORTED_EXTS: List[str] = []

    def __init__(self) -> None:
        self.is_loaded: bool = False
        self.logs: List[str] = []

    def log(self, message: str) -> None:
        """记录算法消息，并保留现有控制台输出行为。"""
        self.logs.append(message)
        print(f"[{self.NAME}] {message}")

    @abstractmethod
    def load_geometry(self, filepath: str) -> bool:
        """加载几何体并在成功时更新 ``is_loaded``。"""

    @abstractmethod
    def get_parameters(self) -> List[ParamDef]:
        """返回由 GUI 消费的参数定义。"""

    @abstractmethod
    def generate(self, **kwargs: Any) -> ToolpathResult:
        """生成符合 :class:`schemas.ToolpathResult` 的刀路结果。"""
