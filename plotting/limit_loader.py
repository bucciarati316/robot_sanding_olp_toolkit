"""
URDFLimitLoader — 从 URDF（或 Pinocchio 模型）中提取关节限位。
"""

import numpy as np
import pinocchio as pin


class URDFLimitLoader:
    """
    从 URDF 文件或 Pinocchio 模型对象中提取关节位置上下限。

    Parameters
    ----------
    urdf_path : str
        URDF 文件路径。
    """

    def __init__(self, urdf_path: str):
        self.urdf_path = urdf_path
        self._model: pin.Model = pin.buildModelFromUrdf(urdf_path)
        self.lower: np.ndarray = self._model.lowerPositionLimit
        self.upper: np.ndarray = self._model.upperPositionLimit

    def get_limits(self, nq: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        """
        返回关节限位的 (lower, upper) 元组。

        Parameters
        ----------
        nq : int, optional
            期望的关节数量。若为 None，则返回模型全部限位。

        Returns
        -------
        lower : np.ndarray
            下限数组。
        upper : np.ndarray
            上限数组。
        """
        lower = self.lower.copy()
        upper = self.upper.copy()

        if nq is not None:
            lower = lower[:nq]
            upper = upper[:nq]

        # 将 NaN / Inf 钳制到 [-pi, pi]
        lower = np.nan_to_num(lower, nan=-np.pi, posinf=np.pi, neginf=-np.pi)
        upper = np.nan_to_num(upper, nan=np.pi, posinf=np.pi, neginf=-np.pi)

        return lower, upper
