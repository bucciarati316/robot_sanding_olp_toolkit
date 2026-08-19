"""auto_manager - 第二层：热插拔算法管理器

本模块实现了CAM框架的插件发现与注册系统。
通过动态导入实现零配置的插件加载。

核心功能：
    - 启动时自动发现 `algorithms/` 目录中的算法
    - 支持热重载，无需重启应用程序
    - 零外部配置（无需 JSON/XML/插件清单文件）
    - 线程安全的单例初始化

用法：
    >>> from core.auto_manager import get_manager
    >>> manager = get_manager()
    >>> print(manager.get_all_algorithm_names())
    ['Zigzag Contour', 'Parametric Spiral', ...]
    >>> ZigzagClass = manager.get_algorithm_class('Zigzag Contour')
"""

from __future__ import annotations

import os
import sys
import importlib.util
import inspect
import traceback
from pathlib import Path
from typing import Dict, List, Type

# Import Layer 1 base class
from .core_algorithm import BaseAlgorithm


class AlgorithmManager:
    """
    用于动态算法发现与注册的单例管理器。

    此类在初始化时自动扫描 `algorithms/` 目录，
    发现所有有效的 BaseAlgorithm 子类，并按其 NAME 属性进行注册，以便后续检索。

    支持按文件名前缀过滤扫描结果（prefix 参数），
    从而将刀路算法（algo_*.py）和 IK 求解器（ik_*.py）注册到不同的注册表，
    供 Tab4 和 Tab5 分别使用。

    该管理器是一个真正的单例——调用 `AlgorithmManager()` 或 `get_manager()`
    始终返回同一实例，即使跨模块导入也不例外。

    类属性：
        _instance: 单例实例（首次实例化前为 None）。

    实例属性：
        _all_registries: Dict[str, Dict[str, Type[BaseAlgorithm]]]，按前缀分组的注册表。
        _scan_paths: Dict[str, Path]，按前缀存储的扫描目录路径。

    线程安全：
        本实现使用简单的 None 检查来创建单例，
        这在单线程环境中是安全的（大多数 GUI 应用程序）。
        如需多线程使用，请考虑添加锁。

    示例：
        >>> manager = get_manager()
        >>> manager.discover_algorithms(prefix="algo")   # Tab4 刀路算法
        >>> manager.discover_algorithms(prefix="ik")    # Tab5 IK 求解器
        >>> names = manager.get_all_algorithm_names(prefix="algo")
        >>> ZigzagAlgorithm = manager.get_algorithm_class("Zigzag Contour", prefix="algo")
    """

    _instance: "AlgorithmManager | None" = None

    def __new__(cls) -> "AlgorithmManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._all_registries: Dict[str, Dict[str, Type[BaseAlgorithm]]] = {}
            cls._instance._scan_paths: Dict[str, Path] = {}
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if not getattr(self, '_initialized', False):
            self._initialized = True
            self._discover_algorithms(prefix=None)

    def discover_algorithms(self, plugins_dir: str = None, prefix: str = None) -> None:
        """
        公开的插件发现入口。

        推荐用法：
            manager.discover_algorithms(prefix="algo")   # Tab4 刀路算法
            manager.discover_algorithms(prefix="ik")    # Tab5 IK 求解器

        参数：
            plugins_dir: 要扫描的目录路径（默认为 "algorithms"）。
            prefix: 文件名前缀过滤（如 "algo" 只扫描 algo_*.py，"ik" 只扫描 ik_*.py）。
                    为 None 时扫描所有 .py 文件。

        行为：
            1. 确定算法目录路径
            2. 如果目录不存在则创建（并给出通知）
            3. 遍历所有 .py 文件（排除双下划线文件）
            4. 按 prefix 过滤文件名
            5. 使用 importlib.util 动态导入每个模块
            6. 检查模块中是否有 BaseAlgorithm 子类
            7. 将有效子类注册到对应 prefix 的注册表中

        注意：
            调用此方法会清除对应 prefix 的注册表并重新扫描。
        """
        self._discover_algorithms(plugins_dir=plugins_dir, prefix=prefix)

    def _discover_algorithms(self, plugins_dir: str = None, prefix: str = None) -> None:
        """
        扫描并注册所有有效的算法插件（内部方法）。

        参数：
            plugins_dir: 要扫描的目录路径（默认为 "algorithms"）。
            prefix: 文件名前缀过滤（如 "algo" 只扫描 algo_*.py，"ik" 只扫描 ik_*.py）。
                    为 None 时扫描所有 .py 文件。
        """
        key = prefix if prefix else "__all__"
        self._all_registries[key] = {}

        # ``auto_manager.py`` now lives in ``core/``; plugins remain a
        # project-level package so both installed and source checkouts work.
        base_path = Path(__file__).resolve().parents[1]
        algo_dir = Path(plugins_dir) if plugins_dir else (base_path / "algorithms")

        if not algo_dir.exists():
            algo_dir.mkdir(parents=True, exist_ok=True)
            print(f"[AlgorithmManager] 已创建 algorithms 目录: {algo_dir}")
            print(f"[AlgorithmManager] 提示: 在此添加算法 .py 文件以实现自动注册")
            return

        self._scan_paths[key] = algo_dir
        print(f"[AlgorithmManager] 正在扫描目录: {algo_dir}" +
              (f" (prefix='{prefix}')" if prefix else ""))

        for py_file in algo_dir.glob("*.py"):
            if py_file.name.startswith("__"):
                continue
            if prefix and not py_file.name.startswith(prefix):
                continue

            self._load_module(py_file, prefix)

        count = len(self._all_registries[key])
        print(f"[AlgorithmManager] 注册完成 [{key}]: 已加载 {count} 个算法")

    def _load_module(self, py_file: Path, prefix: str = None) -> None:
        """
        动态加载单个 Python 模块并注册其中的算法类。

        参数：
            py_file: 要加载的 .py 文件路径。
            prefix: 注册到哪个 prefix 注册表。
        """
        key = prefix if prefix else "__all__"
        module_name = py_file.stem
        package_name = "algorithms"
        full_name = f"{package_name}.{module_name}"

        try:
            # 使用完整包名创建 spec。dataclasses 等运行时会通过 cls.__module__
            # 查询 sys.modules；短名称 spec + 仅注册完整名称会导致动态模块加载崩溃。
            spec = importlib.util.spec_from_file_location(full_name, py_file)
            if spec is None or spec.loader is None:
                print(f"[AlgorithmManager] 无法为 {py_file.name} 创建 spec")
                return

            module = importlib.util.module_from_spec(spec)
            # Set __package__ so relative imports (e.g. from ._utils) resolve correctly
            module.__package__ = package_name
            # Ensure parent package is in sys.modules for relative import resolution
            if package_name not in sys.modules:
                import types
                pkg_module = types.ModuleType(package_name)
                pkg_module.__path__ = [str(py_file.parent)]
                sys.modules[package_name] = pkg_module
            # Use full "algorithms.ik_PILM" key so sibling relative imports resolve
            sys.modules[full_name] = module
            spec.loader.exec_module(module)

            self._register_algorithms_from_module(module, py_file.name, prefix)

        except Exception as e:
            print(f"[AlgorithmManager] 加载失败 {py_file.name}: {e}")
            traceback.print_exc()

    def _register_algorithms_from_module(self, module, source_file: str, prefix: str = None) -> None:
        """
        检查模块并注册找到的所有 BaseAlgorithm 子类。

        参数：
            module: 要检查的已加载 Python 模块。
            source_file: 源文件名（用于日志记录）。
            prefix: 注册到哪个 prefix 注册表。
        """
        key = prefix if prefix else "__all__"
        registry = self._all_registries.setdefault(key, {})
        classes_found = 0
        for name, obj in inspect.getmembers(module, inspect.isclass):
            if (
                issubclass(obj, BaseAlgorithm)
                and obj is not BaseAlgorithm
                and obj.__module__ == module.__name__
            ):
                # 统一用文件名（去掉 ik_ 前缀）作为注册键，显示名用 NAME
                reg_key = obj.NAME
                registry[reg_key] = obj
                print(f"[AlgorithmManager] 已注册: {reg_key} (NAME={obj.NAME}, 来自 {source_file})")
                classes_found += 1

        if classes_found == 0:
            print(f"[AlgorithmManager] 在 {source_file} 中未找到算法类")

    def get_all_algorithm_names(self, prefix: str = None) -> List[str]:
        """
        获取所有已注册算法的名称。

        参数：
            prefix: 指定前缀注册表（如 "algo"、"ik"）。为 None 时返回所有。

        返回：
            算法 NAME 值的列表，适用于填充 UI 菜单。

        示例：
            >>> manager.get_all_algorithm_names(prefix="algo")
            ['Paper: Mesh Constant Arc Length Slicing']
            >>> manager.get_all_algorithm_names(prefix="ik")
            ['伪逆直接求解法', 'SLSQP非线性优化']
        """
        key = prefix if prefix else "__all__"
        registry = self._all_registries.get(key, {})
        return list(registry.keys())

    def get_algorithm_class(self, name: str, prefix: str = None) -> Type[BaseAlgorithm]:
        """
        通过 NAME 检索算法类。

        参数：
            name: 要检索的算法类的 NAME 属性。
            prefix: 指定前缀注册表。

        返回：
            算法类（不是实例）。

        抛出：
            ValueError: 如果没有注册该名称的算法。

        示例：
            >>> SLSQPClass = manager.get_algorithm_class('SLSQP非线性优化', prefix='ik')
            >>> solver = SLSQPClass(kinematics_engine=kin_engine, ...)
        """
        key = prefix if prefix else "__all__"
        registry = self._all_registries.get(key, {})
        if name not in registry:
            available = ", ".join(registry.keys()) or "(none)"
            raise ValueError(
                f"未找到算法 '{name}'。可用算法 [{key}]: {available}"
            )
        return registry[name]

    def register_algorithm(
        self,
        cls: Type[BaseAlgorithm],
        *,
        prefix: str = "algo",
    ) -> None:
        """把插件注册到规范注册表，供兼容适配器和测试使用。"""
        if not isinstance(cls, type) or not issubclass(cls, BaseAlgorithm):
            raise TypeError("插件必须继承 BaseAlgorithm")
        name = getattr(cls, "NAME", None)
        if not name:
            raise ValueError("插件类缺少 NAME")
        self._all_registries.setdefault(prefix, {})[name] = cls

    def unregister_algorithm(self, name: str, *, prefix: str = "algo") -> bool:
        """从规范注册表移除插件。"""
        return self._all_registries.setdefault(prefix, {}).pop(name, None) is not None

    def create_algorithm(self, name: str, *, prefix: str = "algo") -> BaseAlgorithm:
        """从规范注册表创建新的插件实例。"""
        return self.get_algorithm_class(name, prefix=prefix)()

    def reload_plugins(self, prefix: str = None) -> None:
        """
        通过重新扫描算法目录来热重载所有插件。

        参数：
            prefix: 指定前缀注册表（如 "algo"、"ik"）。为 None 时重载所有。

        示例：
            >>> manager.reload_plugins(prefix="ik")  # 只重载 IK 求解器
        """
        print("[AlgorithmManager] 正在热重载插件...")
        self._discover_algorithms(prefix=prefix)


def get_manager() -> AlgorithmManager:
    """
    获取单例 AlgorithmManager 实例的全局辅助函数。

    这是从应用程序任何位置访问管理器的推荐方式。

    返回：
        单例 AlgorithmManager 实例。

    示例：
        >>> from core.auto_manager import get_manager
        >>> manager = get_manager()
        >>> manager.discover_algorithms(prefix="algo")   # Tab4 刀路算法
        >>> manager.discover_algorithms(prefix="ik")    # Tab5 IK 求解器
        >>> manager.get_all_algorithm_names(prefix="algo")
        ['Paper: Mesh Constant Arc Length Slicing']
    """
    return AlgorithmManager()
