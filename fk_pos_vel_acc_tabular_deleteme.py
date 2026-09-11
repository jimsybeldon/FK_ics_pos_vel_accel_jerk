import json
from pathlib import Path
import numpy as np


def analyze_coupler_kinematics(
    json_path: str | Path, omega: float = 1.0, is_periodic: bool = True
) -> dict[str, np.ndarray | dict]:
    """Computes tabular position, velocity, and acceleration vectors for a coupler path

    given a constant input crank angular velocity (omega). Identifies extremum
    and zero-crossing points for velocity magnitude, acceleration magnitude,
    and tangential acceleration (acceleration vs. deceleration).

    Parameters:
        json_path: Path to the JSON packet containing 'theta_list_rad' and 'coupler_path'.
        omega: Constant input crank angular velocity in rad/s.
        is_periodic: Flag indicating whether the crank trajectory wraps around 2*pi.

    Returns:
        Dictionary containing arrays for position, velocity, acceleration,
        velocity/acceleration magnitudes, tangential/normal components,
        and critical points.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    theta = np.array(data["theta_list_rad"], dtype=np.float64)
    pos = np.array(data["coupler_path"], dtype=np.float64)
    n_pts = len(pos)

    if n_pts < 3:
        raise ValueError("At least 3 points are required for central difference.")

    if is_periodic:
        dtheta_start = theta[1] - theta[0]
        dtheta_end = theta[-1] - theta[-2]

        pos_padded = np.vstack([pos[-1:], pos, pos[:1]])
        theta_padded = np.concatenate(
            [[theta[0] - dtheta_end], theta, [theta[-1] + dtheta_start]]
        )

        dtheta_forward = theta_padded[2:] - theta_padded[1:-1]
        dtheta_backward = theta_padded[1:-1] - theta_padded[:-2]
        dtheta_central = theta_padded[2:] - theta_padded[:-2]

        dr_dtheta = (pos_padded[2:] - pos_padded[:-2]) / dtheta_central[:, None]
        d2r_dtheta2 = (
            2.0
            * (
                (pos_padded[2:] - pos_padded[1:-1]) / dtheta_forward[:, None]
                - (pos_padded[1:-1] - pos_padded[:-2]) / dtheta_backward[:, None]
            )
            / dtheta_central[:, None]
        )
    else:
        dr_dtheta = np.gradient(pos, theta, axis=0)
        d2r_dtheta2 = np.gradient(dr_dtheta, theta, axis=0)

    vel = dr_dtheta * omega
    acc = d2r_dtheta2 * (omega**2)

    vel_mag = np.linalg.norm(vel, axis=1)
    acc_mag = np.linalg.norm(acc, axis=1)

    v_unit = np.divide(
        vel,
        vel_mag[:, None],
        out=np.zeros_like(vel),
        where=vel_mag[:, None] != 0,
    )

    # Tangential acceleration (a_t > 0 is tangential acceleration, a_t < 0 is tangential deceleration)
    a_tangential = np.sum(acc * v_unit, axis=1)
    a_normal = np.sqrt(np.maximum(0.0, acc_mag**2 - a_tangential**2))

    # Extracted critical points method supporting signed tangential arrays
    def extract_critical_points(
        arr: np.ndarray, tol: float = 1e-4, is_signed: bool = False
    ) -> dict:
        max_idx = int(np.argmax(arr))
        min_idx = int(np.argmin(arr))

        if is_signed:
            zero_indices = np.where(np.abs(arr) <= tol)[0].tolist()
        else:
            zero_indices = np.where(arr <= tol)[0].tolist()

        return {
            "max": {
                "index": max_idx,
                "theta_rad": float(theta[max_idx]),
                "position": pos[max_idx].tolist(),
                "value": float(arr[max_idx]),
            },
            "min": {
                "index": min_idx,
                "theta_rad": float(theta[min_idx]),
                "position": pos[min_idx].tolist(),
                "value": float(arr[min_idx]),
            },
            "zeros": [
                {
                    "index": int(idx),
                    "theta_rad": float(theta[idx]),
                    "position": pos[idx].tolist(),
                    "value": float(arr[idx]),
                }
                for idx in zero_indices
            ],
        }

    critical_points = {
        "velocity": extract_critical_points(vel_mag, is_signed=False),
        "acceleration": extract_critical_points(acc_mag, is_signed=False),
        "tangential_acceleration": extract_critical_points(
            a_tangential, is_signed=True
        ),
    }

    return {
        "theta_rad": theta,
        "position": pos,
        "velocity": vel,
        "acceleration": acc,
        "velocity_magnitude": vel_mag,
        "acceleration_magnitude": acc_mag,
        "a_tangential": a_tangential,
        "a_normal": a_normal,
        "critical_points": critical_points,
    }


if __name__ == "__main__":
    file_path = "rank1_cad_packet.json"
    results = analyze_coupler_kinematics(file_path, omega=10.0, is_periodic=True)

    v_crit = results["critical_points"]["velocity"]
    a_crit = results["critical_points"]["acceleration"]
    at_crit = results["critical_points"]["tangential_acceleration"]

    print(
        f"Max Velocity: {v_crit['max']['value']:.4f} at Position {v_crit['max']['position']}"
    )
    print(
        f"Min Velocity: {v_crit['min']['value']:.4f} at Position {v_crit['min']['position']}"
    )
    print(f"Zero Velocity Points Count: {len(v_crit['zeros'])}\n")

    print(
        f"Max Accel Mag: {a_crit['max']['value']:.4f} at Position {a_crit['max']['position']}"
    )
    print(
        f"Min Accel Mag: {a_crit['min']['value']:.4f} at Position {a_crit['min']['position']}"
    )
    print(f"Zero Accel Mag Points Count: {len(a_crit['zeros'])}\n")

    print(
        f"Max Tangential Accel (Max Accel along Path): {at_crit['max']['value']:.4f} at Position {at_crit['max']['position']}"
    )
    print(
        f"Min Tangential Accel (Max Decel along Path): {at_crit['min']['value']:.4f} at Position {at_crit['min']['position']}"
    )
    print(f"Tangential Zero-Crossing Points Count: {len(at_crit['zeros'])}")