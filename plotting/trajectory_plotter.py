"""
JointTrajectoryPlotter — 关节轨迹曲线绑图（matplotlib + PySide6 集成）。
"""

from __future__ import annotations

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from PySide6.QtWidgets import QWidget


# 默认配色表（与 MATLAB 默认 6 色一致）
_DEFAULT_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c",
    "#d62728", "#9467bd", "#8c564b",
]


class JointTrajectoryPlotter:
    """
    在 PySide6 界面中嵌入 matplotlib 图表，绘制关节轨迹及其 URDF 限位。

    Parameters
    ----------
    n_joints : int
        关节数量（默认 6）。
    figsize : tuple[float, float]
        图表尺寸（英寸）。
    dark_bg : bool
        是否使用深色背景（默认 True，与项目深色 UI 风格一致）。
    """

    def __init__(
        self,
        n_joints: int = 6,
        figsize: tuple[float, float] = (9, 4.5),
        dark_bg: bool = True,
    ):
        self._n_joints = n_joints

        # ── matplotlib Figure & Axes ──────────────────────────────────────
        self._fig = Figure(figsize=figsize)
        self._ax: plt.Axes = self._fig.add_subplot(111)

        if dark_bg:
            self._fig.patch.set_facecolor("#1e1e1e")
            self._ax.set_facecolor("#2d2d2d")
            self._ax.tick_params(colors="#cccccc")
            self._ax.xaxis.label.set_color("#cccccc")
            self._ax.yaxis.label.set_color("#cccccc")
            self._ax.title.set_color("#eeeeee")
            self._ax.spines["top"].set_color("#555555")
            self._ax.spines["bottom"].set_color("#555555")
            self._ax.spines["left"].set_color("#555555")
            self._ax.spines["right"].set_color("#555555")
            self._ax.grid(True, color="#404040", alpha=0.5)
            for spine in self._ax.spines.values():
                spine.set_edgecolor("#555555")
        else:
            self._fig.patch.set_facecolor("white")
            self._ax.set_facecolor("#f8f8f8")
            self._ax.grid(True, alpha=0.3)

        self._ax.set_xlabel("Step")
        self._ax.set_ylabel("Joint Position (rad)")
        self._ax.set_title("Joint Trajectory")

        self._fig.tight_layout()

        # ── 轨迹线 ─────────────────────────────────────────────────────────
        # self._traj_lines[i] → 第 i 个关节的轨迹折线
        self._traj_lines: list[plt.Line2D] = []
        for i in range(n_joints):
            color = _DEFAULT_COLORS[i % len(_DEFAULT_COLORS)]
            line, = self._ax.plot([], [], color=color, lw=1.5, label=f"J{i + 1}")
            self._traj_lines.append(line)

        self._ax.legend(
            loc="upper left",
            ncol=min(n_joints, 3),
            fontsize="small",
            framealpha=0.7,
        )

        # ── 限位虚线 ───────────────────────────────────────────────────────
        # 占位：初始化时隐藏，等 set_limits() 调用时再显示
        self._limit_lower: list[plt.Line2D] = []
        self._limit_upper: list[plt.Line2D] = []
        for i in range(n_joints):
            color = _DEFAULT_COLORS[i % len(_DEFAULT_COLORS)]
            ln_low, = self._ax.plot([], [], "--", color=color, lw=1.0, alpha=0.6)
            ln_upp, = self._ax.plot([], [], "--", color=color, lw=1.0, alpha=0.6)
            self._limit_lower.append(ln_low)
            self._limit_upper.append(ln_upp)

        self._limits_set = False
        self._lower_limits: np.ndarray | None = None
        self._upper_limits: np.ndarray | None = None
        self._trajectory = np.empty((0, n_joints), dtype=float)

        # Current-frame locator: one dashed vertical cursor plus one colored
        # cross marker on every joint curve.
        self._frame_cursor = self._ax.axvline(
            0.0, color="#00E5FF", lw=1.2, ls="--", alpha=0.9, visible=False
        )
        self._frame_markers: list[plt.Line2D] = []
        for i in range(n_joints):
            marker, = self._ax.plot(
                [], [], marker="+", ms=11, mew=1.8, ls="None",
                color=_DEFAULT_COLORS[i % len(_DEFAULT_COLORS)],
                visible=False,
            )
            self._frame_markers.append(marker)

        # ── Qt 集成 ────────────────────────────────────────────────────────
        self._canvas = FigureCanvasQTAgg(self._fig)
        self._canvas.setMinimumHeight(200)

        self._fig.tight_layout()

    # ── Public API ───────────────────────────────────────────────────────────

    def set_limits(self, lower: np.ndarray, upper: np.ndarray) -> None:
        """
        设置并绘制关节限位水平虚线。

        Parameters
        ----------
        lower : np.ndarray
            各关节下限，长度 >= n_joints。
        upper : np.ndarray
            各关节上限，长度 >= n_joints。
        """
        lower = np.asarray(lower)
        upper = np.asarray(upper)

        n = min(len(lower), len(upper), self._n_joints)

        self._lower_limits = lower[:n].astype(float, copy=True)
        self._upper_limits = upper[:n].astype(float, copy=True)
        y_min, y_max = float("inf"), float("-inf")
        x_max = max(len(self._trajectory) - 1, 1)
        for i in range(n):
            self._limit_lower[i].set_data([0, x_max], [lower[i], lower[i]])
            self._limit_upper[i].set_data([0, x_max], [upper[i], upper[i]])
            y_min = min(y_min, lower[i])
            y_max = max(y_max, upper[i])

        # 设置固定显示范围（不限位被设置前使用 autoscaling）
        self._ax.set_xlim(0, x_max)
        margin = (y_max - y_min) * 0.1 if y_max > y_min else 0.1
        self._ax.set_ylim(y_min - margin, y_max + margin)

        self._limits_set = True
        self._canvas.draw()

    def set_trajectory(self, trajectory: np.ndarray) -> None:
        """
        设置完整轨迹。播放时不得用已播放前缀替换此数据。

        Parameters
        ----------
        trajectory : np.ndarray
            轨迹数组，shape (K, n_joints) 或 (n_joints,)（单帧）。
        """
        traj = np.asarray(trajectory, dtype=float)
        if traj.ndim == 1:
            traj = traj[None, :]
        if traj.ndim != 2 or len(traj) == 0:
            raise ValueError("trajectory 必须为非空 (K,n_joints) 数组")
        if not np.all(np.isfinite(traj)):
            raise ValueError("trajectory 包含非有限值")
        K, nj = traj.shape
        if nj != self._n_joints:
            raise ValueError(
                f"trajectory has {nj} joints but plotter requires {self._n_joints}"
            )
        self._trajectory = traj.copy()

        x = np.arange(K)

        for i in range(nj):
            self._traj_lines[i].set_data(x, traj[:, i])

        x_max = max(K - 1, 1)
        self._ax.set_xlim(0, x_max)
        if self._limits_set:
            for i in range(min(len(self._lower_limits), self._n_joints)):
                self._limit_lower[i].set_data(
                    [0, x_max], [self._lower_limits[i], self._lower_limits[i]]
                )
                self._limit_upper[i].set_data(
                    [0, x_max], [self._upper_limits[i], self._upper_limits[i]]
                )
        else:
            self._ax.relim()
            self._ax.autoscale_view()
        self.set_current_frame(0)

    def set_current_frame(self, frame_index: int) -> None:
        """Move the current-frame locator without altering the full curves."""
        if len(self._trajectory) == 0:
            self._frame_cursor.set_visible(False)
            for marker in self._frame_markers:
                marker.set_visible(False)
            self._canvas.draw_idle()
            return
        index = int(np.clip(frame_index, 0, len(self._trajectory) - 1))
        self._frame_cursor.set_xdata([index, index])
        self._frame_cursor.set_visible(True)
        for joint, marker in enumerate(self._frame_markers):
            marker.set_data([index], [self._trajectory[index, joint]])
            marker.set_visible(True)
        self._canvas.draw_idle()

    def clear(self) -> None:
        """清空轨迹线，保留限位。"""
        for line in self._traj_lines:
            line.set_data([], [])
        self._trajectory = np.empty((0, self._n_joints), dtype=float)
        self._frame_cursor.set_visible(False)
        for marker in self._frame_markers:
            marker.set_visible(False)
        self._ax.relim()
        self._ax.autoscale_view()
        self._canvas.draw()

    def to_widget(self) -> QWidget:
        """
        返回可直接嵌入 QLayout 的 QWidget。

        Returns
        -------
        QWidget
            承载 matplotlib FigureCanvas 的 Qt 控件。
        """
        return self._canvas

    def fig(self) -> Figure:
        """返回底层的 matplotlib Figure 对象。"""
        return self._fig

    def close(self) -> None:
        """关闭 Figure，释放资源。"""
        plt.close(self._fig)
