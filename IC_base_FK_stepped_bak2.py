import json
import numpy as np

# ============================================================
#  Load JSON geometry (source of truth)
# ============================================================

def load_geometry(seed_file, pivot_file):
    with open(seed_file, "r") as f:
        seed = json.load(f)[0]

    with open(pivot_file, "r") as f:
        pivots = json.load(f)

    a = seed["a"]
    b = seed["b"]
    c = seed["c"]

    A = np.array(pivots["ground_pivots"]["A"])
    D = np.array(pivots["ground_pivots"]["D"])

    B_ref = np.array(pivots["rocker_pivots"]["B"])
    C_ref = np.array(pivots["rocker_pivots"]["C"])

    u_local = pivots["coupler_local_frame"]["u"]
    v_local = pivots["coupler_local_frame"]["v"]

    return A, D, a, b, c, B_ref, C_ref, u_local, v_local


# ============================================================
#  Geometric FK (same as validated version)
# ============================================================

def circle_intersection(B, D, b, c, C_ref):
    BD = D - B
    d = np.linalg.norm(BD)

    x = (b*b - c*c + d*d) / (2*d)
    h_sq = b*b - x*x
    h = np.sqrt(max(h_sq, 0))

    P = B + (x/d) * BD
    perp = np.array([-BD[1], BD[0]]) / d

    C1 = P + h * perp
    C2 = P - h * perp

    return C1 if np.linalg.norm(C1 - C_ref) < np.linalg.norm(C2 - C_ref) else C2


def coupler_point(B, C, u_local, v_local):
    phi = np.arctan2(C[1] - B[1], C[0] - B[0])
    R = np.array([[np.cos(phi), -np.sin(phi)],
                  [np.sin(phi),  np.cos(phi)]])
    return B + R @ np.array([u_local, v_local])


def run_fk(seed_file, pivot_file, steps=720):
    A, D, a, b, c, B_ref, C_ref, u_local, v_local = load_geometry(seed_file, pivot_file)

    thetas = np.linspace(0, 2*np.pi, steps, endpoint=False)

    B_list, C_list, P_list = [], [], []

    for th in thetas:
        B = A + a * np.array([np.cos(th), np.sin(th)])
        C = circle_intersection(B, D, b, c, C_ref)
        P = coupler_point(B, C, u_local, v_local)

        B_list.append(B)
        C_list.append(C)
        P_list.append(P)

    return np.array(B_list), np.array(C_list), np.array(P_list), thetas, A, D, a


# ============================================================
#  Instant centers + kinematic/dynamic ratios
# ============================================================

def compute_ic_and_ratios(B_arr, C_arr, P_arr, thetas, A, D, a):
    steps = len(thetas)
    dtheta = thetas[1] - thetas[0]

    # Angular velocity of crank (constant)
    omega_in = dtheta  # per step, if time step = 1

    # Angular velocity of rocker (finite difference)
    rocker_angles = np.unwrap(np.arctan2(C_arr[:,1] - D[1], C_arr[:,0] - D[0]))
    omega_rocker = np.gradient(rocker_angles)  # per step

    # Coupler point velocity and acceleration (finite differences)
    V_P = np.gradient(P_arr, axis=0)          # velocity per step
    A_P = np.gradient(V_P, axis=0)            # acceleration per step

    # Velocity ratio: |V_P| / |V_B|
    B_vel = np.gradient(B_arr, axis=0)
    V_ratio = np.linalg.norm(V_P, axis=1) / np.linalg.norm(B_vel, axis=1)

    # Acceleration ratio: |A_P| / |A_B|
    A_B = np.gradient(B_vel, axis=0)
    A_ratio = np.linalg.norm(A_P, axis=1) / np.linalg.norm(A_B, axis=1)

    # Torque ratio via virtual work:
    # tau_in * omega_in = tau_out * omega_rocker  =>  tau_out / tau_in = omega_in / omega_rocker
    # (where omega_rocker != 0)
    torque_ratio = np.zeros(steps)
    nonzero = np.abs(omega_rocker) > 1e-9
    torque_ratio[nonzero] = omega_in / omega_rocker[nonzero]

    # Instant centers (for a four-bar):
    # IC12 = A, IC23 = B, IC34 = C, IC41 = D
    IC12 = A
    IC41 = D
    IC23_arr = B_arr
    IC34_arr = C_arr

    # Simple summary print
    print("\n=== IC + Kinematic/Dynamic Ratios Summary ===")
    print("Crank angular velocity (per step):", omega_in)
    print("Rocker angular velocity range:", np.min(omega_rocker), "to", np.max(omega_rocker))
    print("Velocity ratio |V_P|/|V_B|: min", np.min(V_ratio), "max", np.max(V_ratio))
    print("Acceleration ratio |A_P|/|A_B|: min", np.min(A_ratio), "max", np.max(A_ratio))
    print("Torque ratio tau_out/tau_in (virtual work): min", np.min(torque_ratio[nonzero]),
          "max", np.max(torque_ratio[nonzero]))
    print("IC12 (ground-crank):", IC12)
    print("IC41 (ground-rocker):", IC41)
    print("=== IC + Ratios Computation Complete ===\n")

    return {
        "IC12": IC12,
        "IC41": IC41,
        "IC23_traj": IC23_arr,
        "IC34_traj": IC34_arr,
        "omega_in": omega_in,
        "omega_rocker": omega_rocker,
        "V_P": V_P,
        "A_P": A_P,
        "V_ratio": V_ratio,
        "A_ratio": A_ratio,
        "torque_ratio": torque_ratio,
    }


# ============================================================
#  Example usage
# ============================================================

if __name__ == "__main__":
    B_arr, C_arr, P_arr, thetas, A, D, a = run_fk(
        "atlas_seed_Rank1_for_IC.json",
        "atlas_seed_Rank1_Pivots.json"
    )

    ic_data = compute_ic_and_ratios(B_arr, C_arr, P_arr, thetas, A, D, a)
