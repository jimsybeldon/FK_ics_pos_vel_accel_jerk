"""
Four-bar linkage coupler-point kinematics via the polar (Euler-notation) vector-loop
dyad equations.

Vector loop (per the supplied diagram / notation):

    Z2*e^(i*th2) + Z3*e^(i*th3) - Z4*e^(i*th4) = Z1*e^(i*th1)

    A  = ground pivot, global origin (0,0)
    D  = ground pivot, global position (X, Y)
    B  = A + Z2*e^(i*th2)                      (crank pivot)
    C  = D + Z4*e^(i*th4) = B + Z3*e^(i*th3)   (coupler/rocker pivot)
    CP = B + Z_cp*e^(i*(th3 + phi_cp))         (coupler point, offset U,V from B)

All th_i in this module are tracked in a frame where th1 = 0 (Z1 lies along the
local +x axis), exactly as specified: "Track THETA angles relative to Z1, with
Z1 providing the direction defined as THETA = 0."  Everything is then rotated
back into the true global frame (A at the global origin, global x horizontal)
by the fixed angle phi1 = angle of the vector A->D, at the very end -- rotation
is just multiplication by the constant unit complex number e^(i*phi1), so it
applies identically to position, velocity, acceleration and jerk vectors.

Governing dyad equations (Euler / polar notation, omega_i = dtheta_i/dt,
alpha_i = domega_i/dt, jerk_i = dalpha_i/dt):

    Position:      Z2*e^(i th2) + Z3*e^(i th3) - Z4*e^(i th4) = Z1
    Velocity:      Z2*om2*i*e^(i th2) + Z3*om3*i*e^(i th3) - Z4*om4*i*e^(i th4) = 0
    Acceleration:  Z2*(i*al2 - om2^2)*e^(i th2)
                 + Z3*(i*al3 - om3^2)*e^(i th3)
                 - Z4*(i*al4 - om4^2)*e^(i th4) = 0
    Jerk (d/dt of the acceleration equation; derived below):
        d/dt[ Zk*(i*alk - omk^2)*e^(i thk) ]
            = Zk*e^(i thk) * [ i*(jerkk - omk^3) - 3*omk*alk ]
        =>  Z2*e^(i th2)*[i*(j2-om2^3) - 3*om2*al2]
          + Z3*e^(i th3)*[i*(j3-om3^3) - 3*om3*al3]
          - Z4*e^(i th4)*[i*(j4-om4^3) - 3*om4*al4] = 0

The crank (link 2) is the driver at constant angular velocity: om2 = OMEGA2,
al2 = 0, j2 = 0.  At each th2 the velocity equation is linear in (om3, om4),
the acceleration equation is linear in (al3, al4), and the jerk equation is
linear in (j3, j4) -- each solved as a 2x2 real linear system from the
real/imag parts of the complex equation.  This is the closed-form analytical
approach; no numerical differentiation of theta3/theta4 is used anywhere.

Position (th3, th4) itself is solved analytically via the standard two-circle
(diagonal) closed form: B is known once th2 is known; C must lie at distance
Z3 from B and distance Z4 from D simultaneously, i.e. at the intersection of
two circles. This is algebraically identical to the classical law-of-cosines
/ Freudenstein diagonal method, just expressed without an arccos branch cut.
Two intersection points exist (open / crossed circuit); the circuit is fixed
at th2 = 0 by the user's choice and then held by continuity (nearest-point
tracking) through the sweep, consistent with the branch-defect discussion --
a jump means the linkage geometry doesn't allow the requested continuous
input range on this circuit.

Author's note: proof-of-concept, printout-only, no plotting. Edit the INPUTS
block at the bottom of the file.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np


# --------------------------------------------------------------------------- #
# Linkage definition
# --------------------------------------------------------------------------- #

@dataclass
class FourBarInputs:
    # Ground pivot D location in the GLOBAL frame; A is the global origin.
    Dx: float
    Dy: float

    # Link magnitudes
    Z2: float          # crank   (A -> B)
    Z3: float           # coupler (B -> C)
    Z4: float           # rocker  (D -> C)

    # Coupler point offset from pivot B, in the LOCAL frame of link 3
    # (U along the B->C direction, V orthogonal to it, CCW positive).
    U: float
    V: float

    # Crank kinematics (driver, constant angular velocity)
    omega2: float       # rad/s, constant (alpha2 = 0, jerk2 = 0)

    # Sweep of the crank angle theta2, RELATIVE TO Z1 (Z1 direction = 0),
    # in radians. For a true crank-rocker this can run a full 0..2*pi.
    # For a branch-tolerant / non-Grashof linkage, restrict this to the
    # actual operating window -- the code will flag angles where no real
    # assembly exists (loop cannot close) rather than silently continuing.
    theta2_start: float = 0.0
    theta2_end: float = 2 * np.pi
    n_steps: int = 3600

    # Which of the two circle-circle intersections to start on at
    # theta2_start.  +1 and -1 select the two circuits (open / crossed);
    # which physical circuit that corresponds to depends on geometry, so
    # just try both if you need a specific one -- the sign is held by
    # continuity for the rest of the sweep.
    circuit_start: int = +1


@dataclass
class FourBarResults:
    theta2: np.ndarray
    valid: np.ndarray          # False where the loop could not close (branch/assembly limit)
    theta3: np.ndarray
    theta4: np.ndarray
    CP_pos: np.ndarray         # complex, global frame
    CP_vel: np.ndarray         # complex, global frame
    CP_acc: np.ndarray         # complex, global frame
    CP_jerk: np.ndarray        # complex, global frame
    CP_acc_tangential: np.ndarray  # real, signed: d|v|/dt  (>0 speeding up, <0 decelerating)
    CP_acc_normal: np.ndarray      # real, signed: centripetal/path-curvature component


# --------------------------------------------------------------------------- #
# Core analytical solution
# --------------------------------------------------------------------------- #

def solve_fourbar(inp: FourBarInputs) -> FourBarResults:
    # --- Ground link Z1 (magnitude, and its orientation phi1 in the true
    #     global frame). All th_i below are computed in the th1 = 0 frame;
    #     phi1 is applied once, at the very end, to rotate into the global
    #     frame the user actually wants (A at origin, global x horizontal).
    Z1 = float(np.hypot(inp.Dx, inp.Dy))
    if Z1 <= 0.0:
        raise ValueError("Ground pivots A and D coincide -- Z1 is degenerate.")
    phi1 = float(np.arctan2(inp.Dy, inp.Dx))

    Z2, Z3, Z4 = inp.Z2, inp.Z3, inp.Z4
    U, V = inp.U, inp.V
    Zcp = float(np.hypot(U, V))
    phi_cp = float(np.arctan2(V, U))   # constant offset angle of CP from the link-3 direction

    om2 = inp.omega2
    al2 = 0.0
    j2 = 0.0

    theta2_arr = np.linspace(inp.theta2_start, inp.theta2_end, inp.n_steps)

    theta3_arr = np.full_like(theta2_arr, np.nan)
    theta4_arr = np.full_like(theta2_arr, np.nan)
    valid = np.zeros_like(theta2_arr, dtype=bool)

    CP_pos = np.full(theta2_arr.shape, np.nan + 1j * np.nan, dtype=complex)
    CP_vel = np.full(theta2_arr.shape, np.nan + 1j * np.nan, dtype=complex)
    CP_acc = np.full(theta2_arr.shape, np.nan + 1j * np.nan, dtype=complex)
    CP_jerk = np.full(theta2_arr.shape, np.nan + 1j * np.nan, dtype=complex)

    # D in the th1 = 0 local frame sits at (Z1, 0)
    D_local = complex(Z1, 0.0)

    prev_C = None  # for branch continuity via nearest-point tracking

    for k, th2 in enumerate(theta2_arr):
        B = Z2 * np.exp(1j * th2)

        # --- Position: intersect circle(center=B, r=Z3) with circle(center=D, r=Z4)
        d_vec = D_local - B
        d = abs(d_vec)

        if d > (Z3 + Z4) or d < abs(Z3 - Z4) or d == 0.0:
            # No real assembly at this theta2 -- outside the range this
            # linkage can physically close (dead point / non-Grashof limit).
            continue

        a = (Z3 ** 2 - Z4 ** 2 + d ** 2) / (2 * d)
        h_sq = Z3 ** 2 - a ** 2
        if h_sq < 0.0:
            continue
        h = np.sqrt(h_sq)

        mid = B + a * d_vec / d
        perp = 1j * d_vec / d  # unit vector rotated +90 deg

        C_plus = mid + h * perp
        C_minus = mid - h * perp

        if prev_C is None:
            C = C_plus if inp.circuit_start >= 0 else C_minus
        else:
            # hold the branch by continuity: pick whichever intersection is
            # closer to the previous step's C. A large jump here signals a
            # true branch/circuit change, not just numerical noise.
            C = C_plus if abs(C_plus - prev_C) <= abs(C_minus - prev_C) else C_minus
        prev_C = C

        th3 = np.angle(C - B)
        th4 = np.angle(C - D_local)

        theta3_arr[k] = th3
        theta4_arr[k] = th4
        valid[k] = True

        # --- Velocity: Z2*om2*i*e^(i th2) + Z3*om3*i*e^(i th3) - Z4*om4*i*e^(i th4) = 0
        # Linear system for (om3, om4). Rearranged:
        #   Z3*i*e^(i th3) * om3  -  Z4*i*e^(i th4) * om4  =  -Z2*om2*i*e^(i th2)
        rhs_v = -Z2 * om2 * 1j * np.exp(1j * th2)
        c3_v = Z3 * 1j * np.exp(1j * th3)
        c4_v = -Z4 * 1j * np.exp(1j * th4)
        om3, om4 = _solve_2x2_complex(c3_v, c4_v, rhs_v)

        # --- Acceleration: Z2*(i*al2-om2^2)e^(i th2) + Z3*(i*al3-om3^2)e^(i th3)
        #                    - Z4*(i*al4-om4^2)e^(i th4) = 0
        # Linear system for (al3, al4).
        term2_a = Z2 * (1j * al2 - om2 ** 2) * np.exp(1j * th2)
        rhs_a = -(term2_a - Z3 * (om3 ** 2) * np.exp(1j * th3) + Z4 * (om4 ** 2) * np.exp(1j * th4))
        c3_a = Z3 * 1j * np.exp(1j * th3)
        c4_a = -Z4 * 1j * np.exp(1j * th4)
        al3, al4 = _solve_2x2_complex(c3_a, c4_a, rhs_a)

        # --- Jerk: d/dt of the acceleration equation.
        #   Zk*e^(i thk)*[ i*(jk - omk^3) - 3*omk*alk ]   summed with the loop's signs.
        # Linear system for (j3, j4).
        term2_j = Z2 * np.exp(1j * th2) * (1j * (j2 - om2 ** 3) - 3 * om2 * al2)
        # move everything except the j3, j4 terms to the RHS
        known_j = (
            term2_j
            + Z3 * np.exp(1j * th3) * (1j * (-om3 ** 3) - 3 * om3 * al3)
            - Z4 * np.exp(1j * th4) * (1j * (-om4 ** 3) - 3 * om4 * al4)
        )
        rhs_j = -known_j
        c3_j = Z3 * 1j * np.exp(1j * th3)
        c4_j = -Z4 * 1j * np.exp(1j * th4)
        j3, j4 = _solve_2x2_complex(c3_j, c4_j, rhs_j)

        # --- Coupler point CP = B + Zcp*e^(i(th3+phi_cp)), and its derivatives.
        # Treat the CP offset as rigidly attached to link 3, so it shares
        # link 3's omega/alpha/jerk; only the phase is shifted by phi_cp.
        th_cp = th3 + phi_cp
        e_cp = np.exp(1j * th_cp)

        pos_cp = B + Zcp * e_cp

        vel_B = Z2 * om2 * 1j * np.exp(1j * th2)
        vel_cp_term = Zcp * om3 * 1j * e_cp
        vel_cp = vel_B + vel_cp_term

        acc_B = Z2 * (1j * al2 - om2 ** 2) * np.exp(1j * th2)
        acc_cp_term = Zcp * (1j * al3 - om3 ** 2) * e_cp
        acc_cp = acc_B + acc_cp_term

        jerk_B = Z2 * np.exp(1j * th2) * (1j * (j2 - om2 ** 3) - 3 * om2 * al2)
        jerk_cp_term = Zcp * e_cp * (1j * (j3 - om3 ** 3) - 3 * om3 * al3)
        jerk_cp = jerk_B + jerk_cp_term

        # --- Rotate local-frame (th1=0) results into the true global frame.
        rot = np.exp(1j * phi1)
        CP_pos[k] = pos_cp * rot          # (A is the global origin already, so no translation needed)
        CP_vel[k] = vel_cp * rot
        CP_acc[k] = acc_cp * rot
        CP_jerk[k] = jerk_cp * rot

    # --- Tangential / normal decomposition of the acceleration vector,
    #     relative to the (instantaneous) velocity direction.
    #
    #     |a_vec| = sqrt(ax^2+ay^2) is always >= 0 -- it cannot show
    #     deceleration, only "how hard, in any direction."  The signed
    #     quantity that actually answers "speeding up or slowing down" is
    #     the component of a_vec ALONG v_vec:
    #
    #         a_t = d|v|/dt = Re( conj(v_vec) * a_vec ) / |v_vec|
    #
    #     which is positive when the coupler point is speeding up along its
    #     path and NEGATIVE when it is decelerating. The remaining,
    #     perpendicular component is purely centripetal (caused by the path
    #     curving) and does not represent speed change at all -- it is
    #     nonzero even at constant speed:
    #
    #         a_n = Im( conj(v_vec) * a_vec ) / |v_vec|
    #
    #     |a_vec|^2 = a_t^2 + a_n^2 exactly, so the two together fully
    #     explain what the magnitude-only report was hiding.

    speed = np.abs(CP_vel)
    with np.errstate(invalid="ignore", divide="ignore"):
        cross_term = np.conj(CP_vel) * CP_acc
        CP_acc_tangential = np.where(speed > 0, cross_term.real / speed, np.nan)
        CP_acc_normal = np.where(speed > 0, cross_term.imag / speed, np.nan)
    CP_acc_tangential = np.where(valid, CP_acc_tangential, np.nan)
    CP_acc_normal = np.where(valid, CP_acc_normal, np.nan)

    return FourBarResults(
        theta2=theta2_arr,
        valid=valid,
        theta3=theta3_arr,
        theta4=theta4_arr,
        CP_pos=CP_pos,
        CP_vel=CP_vel,
        CP_acc=CP_acc,
        CP_jerk=CP_jerk,
        CP_acc_tangential=CP_acc_tangential,
        CP_acc_normal=CP_acc_normal,
    )


def _solve_2x2_complex(c1: complex, c2: complex, rhs: complex) -> tuple[float, float]:
    """
    Solve  c1*x + c2*y = rhs  for REAL x, y, given complex coefficients,
    by splitting into real/imag parts -> a standard 2x2 real linear system.
    (x, y) here are the unknown angular rates (om3, om4), or the unknown
    angular accelerations, or the unknown angular jerks, depending on caller.
    """
    A = np.array([[c1.real, c2.real],
                  [c1.imag, c2.imag]], dtype=float)
    b = np.array([rhs.real, rhs.imag], dtype=float)
    det = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
    if abs(det) < 1e-14:
        return float("nan"), float("nan")
    x = (b[0] * A[1, 1] - A[0, 1] * b[1]) / det
    y = (A[0, 0] * b[1] - b[0] * A[1, 0]) / det
    return x, y


# --------------------------------------------------------------------------- #
# Local extrema of |velocity|, |acceleration|, |jerk| along the sweep
# --------------------------------------------------------------------------- #

def find_local_extrema(theta2: np.ndarray, valid: np.ndarray, mag: np.ndarray):
    """
    Return (minima_idx, maxima_idx): indices of interior local minima/maxima
    of `mag` over the valid, contiguous portion of the sweep. Simple sign-
    change-of-slope detector -- adequate for a proof-of-concept printout.
    Endpoints and points adjacent to an invalid (non-assembling) region are
    excluded, since a "local extremum" there is really just the reachable
    edge of the assembly range, not a stationary point of the physics.
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
    Locate zero-crossings of a SIGNED scalar (e.g. tangential acceleration
    a_t). This is the accel<->decel transition itself -- a root of the
    signed value, not a local min/max of it -- so it needs its own detector
    rather than find_local_extrema.

    Returns a list of dicts: theta2 (interpolated, radians), CP position
    (interpolated), and the crossing direction ("accel -> decel" when a_t
    goes + to -, "decel -> accel" when it goes - to +).

    theta2 and CP position are linearly interpolated between the bracketing
    samples -- adequate for a proof-of-concept printout; for a precise
    location, re-solve the loop equations at the interpolated theta2
    directly instead of interpolating CP_pos.
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
# Printout
# --------------------------------------------------------------------------- #

def report(res: FourBarResults, label: str, values: np.ndarray, unit: str, signed: bool = False):
    """
    signed=False (default): `values` is a complex vector quantity; report
        extrema of its MAGNITUDE (always >= 0 -- e.g. speed, |accel|, |jerk|).
    signed=True: `values` is already a real, signed scalar (e.g. tangential
        acceleration) -- report extrema of the signed value itself, so
        negative numbers (deceleration) show up as such.
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


def report_crossings(res: FourBarResults, label: str, values: np.ndarray, unit: str):
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


def main():
    # ------------------------------------------------------------------- #
    # INPUTS -- edit these. Units: lengths consistent (e.g. mm or in),
    # angles in radians unless noted, omega2 in rad/s.
    # ------------------------------------------------------------------- #
    inp = FourBarInputs(
        Dx=4.0, Dy=0.5,     # ground pivot D, global coords; A is (0,0)
        Z2=1.0,             # crank
        Z3=3.5,             # coupler
        Z4=2.5,             # rocker
        U=1.8, V=0.9,       # coupler point offset from B, in link-3's local frame
        omega2=10.0,        # rad/s, constant
        theta2_start=0.0,
        theta2_end=2 * np.pi,
        n_steps=3600,
        circuit_start=+1,
    )

    res = solve_fourbar(inp)

    n_valid = int(np.sum(res.valid))
    print("=" * 70)
    print("Four-bar coupler-point kinematics (polar/Euler dyad, analytical)")
    print("=" * 70)
    print(f"Z1 = {np.hypot(inp.Dx, inp.Dy):.5g}   "
          f"phi1 (Z1 orientation, global) = {np.degrees(np.arctan2(inp.Dy, inp.Dx)):.3f} deg")
    print(f"Z2={inp.Z2}  Z3={inp.Z3}  Z4={inp.Z4}  U={inp.U}  V={inp.V}  omega2={inp.omega2} rad/s")
    print(f"theta2 sweep: {np.degrees(inp.theta2_start):.1f} to "
          f"{np.degrees(inp.theta2_end):.1f} deg, {inp.n_steps} steps, "
          f"{n_valid} assembled")

    if n_valid == 0:
        print("\nNo assembly found anywhere in the requested theta2 sweep -- "
              "check Z2/Z3/Z4/Z1 (Grashof condition) or narrow the sweep window.")
        return

    report(res, "velocity (speed)", res.CP_vel, "length/s")
    report(res, "acceleration (|a|, unsigned -- see tangential below for decel)",
           res.CP_acc, "length/s^2")
    report(res, "acceleration, TANGENTIAL (signed: + speeding up, - decelerating)",
           res.CP_acc_tangential, "length/s^2", signed=True)
    report_crossings(res, "Tangential acceleration", res.CP_acc_tangential, "length/s^2")
    report(res, "jerk", res.CP_jerk, "length/s^3")


if __name__ == "__main__":
    main()
