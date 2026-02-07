import math

import numpy as np


def quat2axisangle(quat: np.ndarray) -> np.ndarray:
    """
    Convert quaternion to axis-angle representation.
    Based on robosuite implementation.
    Args:
        quat: (x, y, z, w) or (w, x, y, z) depending on convention.
              Robosuite/MuJoCo typically use (w, x, y, z).
    """
    # Create a copy to avoid modifying the original array
    q = quat.copy()

    # Clip quaternion to valid range
    if q[3] > 1.0:
        q[3] = 1.0
    elif q[3] < -1.0:
        q[3] = -1.0

    den = np.sqrt(1.0 - q[3] * q[3])

    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (q[:3] * 2.0 * math.acos(q[3])) / den
