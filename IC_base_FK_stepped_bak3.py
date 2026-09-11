import json
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
#  Load JSON geometry
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
#  FK stepping
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

    return np.array(B_list), np.array(C_list), np.array(P_list), thetas, A, D


# ============================================================
#  IC + kinematic/dynamic quantities
# ============================================================

def compute_ic_and_ratios(B_arr, C_arr, P_arr, thetas, A, D):
    steps = len(thetas)
    dtheta = thetas[1] - thetas[0]

    omega_in = dtheta

    rocker_angles = np.unwrap(np.arctan2(C_arr[:,1] - D[1], C_arr[:,0] - D[0]))
    omega_rocker = np.gradient(rocker_angles)

    V_P = np.gradient(P_arr, axis=0)
    A_P = np.gradient(V_P, axis=0)

    B_vel = np.gradient(B_arr, axis=0)
    A_B = np.gradient(B_vel, axis=0)

    V_ratio = np.linalg.norm(V_P, axis=1) / np.linalg.norm(B_vel, axis=1)
    A_ratio = np.linalg.norm(A_P, axis=1) / np.linalg.norm(A_B, axis=1)

    torque_ratio = np.zeros(steps)
    nonzero = np.abs(omega_rocker) > 1e-9
    torque_ratio[nonzero] = omega_in / omega_rocker[nonzero]

    IC12 = A
    IC41 = D
    IC23_arr = B_arr
    IC34_arr = C_arr

    return {
        "IC12": IC12,
        "IC41": IC41,
        "IC23_traj": IC23_arr,
        "IC34_traj": IC34_arr,
        "rocker_angles": rocker_angles,
        "omega_in": omega_in,
        "omega_rocker": omega_rocker,
        "V_P": V_P,
        "A_P": A_P,
        "V_ratio": V_ratio,
        "A_ratio": A_ratio,
        "torque_ratio": torque_ratio,
    }


# ============================================================
#  Motion plots
# ============================================================

def plot_motion(B_arr, C_arr, P_arr, thetas, A, D, ic_data):
    crank_deg = np.degrees(thetas)
    V_mag = np.linalg.norm(ic_data["V_P"], axis=1)
    A_mag = np.linalg.norm(ic_data["A_P"], axis=1)

    # Trajectories of B, C, P
    plt.figure(figsize=(6, 6))
    plt.plot(B_arr[:,0], B_arr[:,1], label="B (crank)")
    plt.plot(C_arr[:,0], C_arr[:,1], label="C (rocker)")
    plt.plot(P_arr[:,0], P_arr[:,1], label="P (coupler point)")
    plt.scatter([A[0], D[0]], [A[1], D[1]], c=["k", "k"], marker="x", label="Ground pivots A,D")
    plt.axis("equal")
    plt.legend()
    plt.title("Trajectories of B, C, P")
    plt.xlabel("x")
    plt.ylabel("y")

    # Rocker angle vs crank angle
    plt.figure()
    plt.plot(crank_deg, np.degrees(ic_data["rocker_angles"]))
    plt.title("Rocker angle vs crank angle")
    plt.xlabel("Crank angle (deg)")
    plt.ylabel("Rocker angle (deg)")

    # Velocity and acceleration magnitude of P
    plt.figure()
    plt.plot(crank_deg, V_mag, label="|V_P|")
    plt.plot(crank_deg, A_mag, label="|A_P|")
    plt.title("Coupler point velocity and acceleration magnitudes")
    plt.xlabel("Crank angle (deg)")
    plt.ylabel("Magnitude")
    plt.legend()

    # Torque ratio
    plt.figure()
    plt.plot(crank_deg, ic_data["torque_ratio"])
    plt.title("Torque ratio tau_out / tau_in (virtual work)")
    plt.xlabel("Crank angle (deg)")
    plt.ylabel("Torque ratio")

    # Instant centers (A, D, B, C)
    plt.figure(figsize=(6, 6))
    plt.scatter(A[0], A[1], c="r", label="IC12 (A)")
    plt.scatter(D[0], D[1], c="b", label="IC41 (D)")
    plt.plot(ic_data["IC23_traj"][:,0], ic_data["IC23_traj"][:,1], "g.", alpha=0.5, label="IC23 (B traj)")
    plt.plot(ic_data["IC34_traj"][:,0], ic_data["IC34_traj"][:,1], "m.", alpha=0.5, label="IC34 (C traj)")
    plt.axis("equal")
    plt.title("Instant centers")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()

    plt.show()


# ============================================================
#  Example usage
# ============================================================

if __name__ == "__main__":
    B_arr, C_arr, P_arr, thetas, A, D = run_fk(
        "atlas_seed_Rank1_for_IC.json",
        "atlas_seed_Rank1_Pivots.json"
    )

    ic_data = compute_ic_and_ratios(B_arr, C_arr, P_arr, thetas, A, D)
    plot_motion(B_arr, C_arr, P_arr, thetas, A, D, ic_data)
