
from __future__ import annotations
import json
from dataclasses import dataclass
import numpy as np


# --------------------------------------------------------------------------- #
# Data structures
# --------------------------------------------------------------------------- #

@dataclass

# checkout this explanation about what @dataclass is about:   https://www.youtube.com/watch?v=5mMpM8zK4pY

class MotionResults:
    theta2: np.ndarray              # crank angle list (radians), from JSON
    CP_pos: np.ndarray              # complex, global frame, from JSON coupler_path
    CP_vel: np.ndarray              # complex, numerical derivative of CP_pos
    CP_acc: np.ndarray              # complex, numerical derivative of CP_vel
    CP_jerk: np.ndarray             # complex, numerical derivative of CP_acc
    CP_acc_tangential: np.ndarray   # real, signed: d|v|/dt (>0 speeding up, <0 decelerating)
    CP_acc_normal: np.ndarray       # real, signed: centripetal/path-curvature component
    valid: np.ndarray               # bool mask, here all True (no assembly logic)


# --------------------------------------------------------------------------- #
# Numerical differentiation (Option A: treat coupler_path as exact truth)
# --------------------------------------------------------------------------- #

def _central_difference(values: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """
    Simple central-difference derivative with respect to theta.
    values: array of complex (or real) samples at theta[k].
    theta:  array of angles (radians), same length.
    Returns: derivative d(values)/d(theta), same length, with forward/backward
             difference at the endpoints.
    """
    n = len(values)
    deriv = np.full_like(values, np.nan, dtype=complex)

    # interior points: central difference
    for k in range(1, n - 1):
        dt = theta[k + 1] - theta[k - 1]
        if dt == 0:
            deriv[k] = 0.0 + 0.0j
        else:
            deriv[k] = (values[k + 1] - values[k - 1]) / dt

    # endpoints: one-sided difference
    if n >= 2:
        dt0 = theta[1] - theta[0]
        dtN = theta[-1] - theta[-2]
        deriv[0] = (values[1] - values[0]) / dt0 if dt0 != 0 else 0.0 + 0.0j
        deriv[-1] = (values[-1] - values[-2]) / dtN if dtN != 0 else 0.0 + 0.0j

    return deriv


# --------------------------------------------------------------------------- #
# Extrema and sign-crossing utilities (unchanged in spirit)
# --------------------------------------------------------------------------- #

def find_local_extrema(theta2: np.ndarray, valid: np.ndarray, mag: np.ndarray):
    """
    Return (minima_idx, maxima_idx): indices of interior local minima/maxima
    of `mag` over the valid portion of the sweep. Simple sign-change-of-slope
    detector. Endpoints are excluded.
    """
    minima, maxima = [], []
    n = len(mag)
    for k in range(1, n - 1):
        if not (valid[k - 1] and valid[k] and valid[k + 1]):
            continue
        left = mag[k] - mag[k - 1]
        right = mag[k + 1] - mag[k]
        if left < 0 and right > 0:
            minima.append(k)
        elif left > 0 and right < 0:
            maxima.append(k)
    return minima, maxima


def find_sign_crossings(theta2: np.ndarray, valid: np.ndarray, values: np.ndarray, CP_pos: np.ndarray):
    """
    Locate zero-crossings of a SIGNED scalar (e.g. tangential acceleration a_t).
    Returns a list of dicts: theta2 (interpolated, radians), CP position
    (interpolated), and the crossing direction ("accel -> decel" or
    "decel -> accel").
    """
    crossings = []
    n = len(values)
    for k in range(n - 1):
        if not (valid[k] and valid[k + 1]):
            continue
        v0, v1 = values[k], values[k + 1]
        if np.isnan(v0) or np.isnan(v1):
            continue
        if v0 == 0.0:
            frac = 0.0
        elif v0 * v1 < 0.0:
            frac = -v0 / (v1 - v0)
        else:
            continue
        th2_interp = theta2[k] + frac * (theta2[k + 1] - theta2[k])
        pos_interp = CP_pos[k] + frac * (CP_pos[k + 1] - CP_pos[k])
        direction = "accel -> decel" if v0 > v1 else "decel -> accel"
        crossings.append({"theta2": th2_interp, "CP": pos_interp, "direction": direction})
    return crossings


# --------------------------------------------------------------------------- #
# Reporting (reused, but now for JSON-driven motion)
# --------------------------------------------------------------------------- #

def report(res: MotionResults, label: str, values: np.ndarray, unit: str, signed: bool = False):
    """
    signed=False (default): `values` is a complex vector quantity; report
        extrema of its MAGNITUDE (always >= 0 -- e.g. speed, |accel|, |jerk|).
    signed=True: `values` is already a real, signed scalar (e.g. tangential
        acceleration) -- report extrema of the signed value itself.
    """
    mag = values if signed else np.abs(values)
    minima, maxima = find_local_extrema(res.theta2, res.valid, mag)

    n_valid = int(np.sum(res.valid))
    print(f"\n--- Coupler point {label} ({unit}) ---")
    print(f"  valid crank-angle samples: {n_valid} / {len(res.theta2)}")

    def _line(tag, k):
        th2_deg = np.degrees(res.theta2[k])
        x, y = res.CP_pos[k].real, res.CP_pos[k].imag
        print(f"  {tag:>18s}: {mag[k]:12.5g} {unit}  at theta2 = {th2_deg:8.3f} deg   "
              f"CP = ({x:9.4f}, {y:9.4f})")

    if not minima and not maxima:
        print("  (no interior local extrema found on the valid portion of the sweep)")
        return

    if minima:
        k_glob_min = min(minima, key=lambda k: mag[k])
        for k in minima:
            _line("local minimum" if k != k_glob_min else "GLOBAL minimum", k)
    if maxima:
        k_glob_max = max(maxima, key=lambda k: mag[k])
        for k in maxima:
            _line("local maximum" if k != k_glob_max else "GLOBAL maximum", k)


def report_crossings(res: MotionResults, label: str, values: np.ndarray, unit: str):
    crossings = find_sign_crossings(res.theta2, res.valid, values, res.CP_pos)
    print(f"\n--- {label}: zero crossings (accel <-> decel transitions) ---")
    if not crossings:
        print("  (no sign change found -- tangential accel stays one sign over this sweep)")
        return
    for c in crossings:
        th2_deg = np.degrees(c["theta2"])
        x, y = c["CP"].real, c["CP"].imag
        print(f"  {c['direction']:>15s}  at theta2 = {th2_deg:8.3f} deg   "
              f"CP = ({x:9.4f}, {y:9.4f})")


# --------------------------------------------------------------------------- #
# Core: use JSON as master coordinate system and theta list
# --------------------------------------------------------------------------- #

def analyze_motion_from_json(json_path: str) -> MotionResults:
    """
    Load rank1_cad_packet.json (or equivalent) and treat its data as the
    authoritative motion:
      - theta_list_rad: crank angle samples (radians)
      - coupler_path:   CP positions [x, y] in GLOBAL frame
    No linkage solving is performed; we only analyze the given motion.
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    theta2 = np.array(data["theta_list_rad"], dtype=float)

    # coupler_path: list of [x, y]; convert to complex
    cp_xy = np.array(data["coupler_path"], dtype=float)  # shape (N, 2)
    CP_pos = cp_xy[:, 0] + 1j * cp_xy[:, 1]

    # numerical derivatives w.r.t. theta2
    CP_vel = _central_difference(CP_pos, theta2)
    CP_acc = _central_difference(CP_vel, theta2)
    CP_jerk = _central_difference(CP_acc, theta2)

    # tangential / normal acceleration decomposition
    speed = np.abs(CP_vel)
    with np.errstate(invalid="ignore", divide="ignore"):
        cross_term = np.conj(CP_vel) * CP_acc
        CP_acc_tangential = np.where(speed > 0, cross_term.real / speed, np.nan)
        CP_acc_normal = np.where(speed > 0, cross_term.imag / speed, np.nan)

    valid = np.ones_like(theta2, dtype=bool)  # all samples considered valid

    return MotionResults(
        theta2=theta2,
        CP_pos=CP_pos,
        CP_vel=CP_vel,
        CP_acc=CP_acc,
        CP_jerk=CP_jerk,
        CP_acc_tangential=CP_acc_tangential,
        CP_acc_normal=CP_acc_normal,
        valid=valid,
    )


# --------------------------------------------------------------------------- #
# Main: print the same style of report, but driven purely by JSON motion
# --------------------------------------------------------------------------- #

def main():
    # Path to your rank1_cad_packet.json (or equivalent)
    json_path = "rank1_cad_packet.json"

    res = analyze_motion_from_json(json_path)

    print("=" * 70)
    print("Four-bar coupler-point motion analysis (JSON-driven, no linkage solving)")
    print("=" * 70)
    print(f"theta2 sweep: {np.degrees(res.theta2[0]):.3f} to "
          f"{np.degrees(res.theta2[-1]):.3f} deg, {len(res.theta2)} samples")

    report(res, "velocity (speed)", res.CP_vel, "length/rad")  # units depend on your length units
    report(res, "acceleration (|a|, unsigned -- see tangential below for decel)",
           res.CP_acc, "length/rad^2")
    report(res, "acceleration, TANGENTIAL (signed: + speeding up, - decelerating)",
           res.CP_acc_tangential, "length/rad^2", signed=True)
    report_crossings(res, "Tangential acceleration", res.CP_acc_tangential, "length/rad^2")
    report(res, "jerk", res.CP_jerk, "length/rad^3")


if __name__ == "__main__":
    main()
