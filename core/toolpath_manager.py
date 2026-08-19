"""旧刀路管理器 API 的兼容适配器。

生产插件注册表只有 :mod:`auto_manager` 中的一份。本模块保留历史 API，
但所有读写均委托给同一个 ``AlgorithmManager`` 实例。
"""

from __future__ import annotations

from typing import List, Type

from .auto_manager import AlgorithmManager, get_manager as get_algorithm_manager
from .core_algorithm import BaseAlgorithm


class PluginError(Exception):
    """插件兼容 API 的基础异常。"""


class PluginNotFoundError(PluginError):
    """请求的插件不存在。"""


class PluginLoadError(PluginError):
    """插件发现或实例化失败。"""


class InvalidPluginError(PluginError):
    """插件未实现规范 BaseAlgorithm 协议。"""


class ToolpathEngineManager:
    """委托给规范 ``AlgorithmManager`` 的无状态兼容门面。"""

    def __init__(self, manager: AlgorithmManager | None = None) -> None:
        self._manager = manager or get_algorithm_manager()

    def discover(self, algorithms_pkg: str = "algorithms") -> int:
        if algorithms_pkg != "algorithms":
            raise PluginLoadError("兼容入口仅支持规范 algorithms 包")
        self._manager.discover_algorithms(prefix="algo")
        return len(self.list_algorithms())

    def register(self, cls: Type[BaseAlgorithm]) -> None:
        try:
            self._manager.register_algorithm(cls, prefix="algo")
        except TypeError as exc:
            raise InvalidPluginError(str(exc)) from exc
        except ValueError as exc:
            raise PluginLoadError(str(exc)) from exc

    def unregister(self, name: str) -> bool:
        return self._manager.unregister_algorithm(name, prefix="algo")

    def get(self, name: str) -> BaseAlgorithm:
        try:
            return self._manager.create_algorithm(name, prefix="algo")
        except ValueError as exc:
            raise PluginNotFoundError(str(exc)) from exc

    def list_algorithms(self) -> List[str]:
        return self._manager.get_all_algorithm_names(prefix="algo")

    def get_algorithms_for(self, file_ext: str) -> List[str]:
        normalized = file_ext.lower()
        return [
            name
            for name in self.list_algorithms()
            if normalized
            in {
                ext.lower()
                for ext in self._manager.get_algorithm_class(
                    name,
                    prefix="algo",
                ).SUPPORTED_EXTS
            }
        ]


_compat_manager: ToolpathEngineManager | None = None


def get_manager() -> ToolpathEngineManager:
    """返回共享规范注册表之上的兼容门面。"""
    global _compat_manager
    if _compat_manager is None:
        _compat_manager = ToolpathEngineManager()
    return _compat_manager


def list_algorithms() -> List[str]:
    return get_manager().list_algorithms()


def get_algorithms_for(file_ext: str) -> List[str]:
    return get_manager().get_algorithms_for(file_ext)
