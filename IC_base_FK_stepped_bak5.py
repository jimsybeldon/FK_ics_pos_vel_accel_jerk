import json
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Load JSON geometry + coupler path
# ============================================================

def load_cad_packet(json_file):
    with open(json_file, "r") as f:
        data = json.load(f)

    A = np.array(data["ground_pivots"]["A"])
    D = np.array(data["ground_pivots"]["D"])
    B_ref = np.array(data["rocker_pivots"]["B"])
    C_ref = np.array(data["rocker_pivots"]["C"])

    u_local = data["coupler_local_frame"]["u"]
    v_local = data["coupler_local_frame"]["v"]

    theta_list = np.array(data["theta_list_rad"])
    P_arr = np.array(data["coupler_path"])

    return A, D, B_ref, C_ref, u_local, v_local, theta_list, P_arr


# ============================================================
# Compute B and C from geometry (circle-circle intersection)
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


def compute_BC(A, D, theta_list, a, b, c, C_ref):
    B_arr = []
    C_arr = []

    for th in theta_list:
        B = A + a * np.array([np.cos(th), np.sin(th)])
        C = circle_intersection(B, D, b, c, C_ref)
        B_arr.append(B)
        C_arr.append(C)

    return np.array(B_arr), np.array(C_arr)


# ============================================================
# Coupler-point IC computation
# ============================================================

def perpendicular_line(point, velocity):
    # Returns (point, normal_vector)
    vx, vy = velocity
    # perpendicular direction
    nx, ny = -vy, vx
    return point, np.array([nx, ny])


def intersect_lines(P, nP, B, nB):
    # Solve intersection of:
    # (x - P) · nP = 0
    # (x - B) · nB = 0
    A_mat = np.array([nP, nB])
    b_vec = np.array([np.dot(nP, P), np.dot(nB, B)])
    return np.linalg.solve(A_mat, b_vec)


def compute_coupler_IC(P_arr, B_arr):
    V_P = np.gradient(P_arr, axis=0)
    V_B = np.gradient(B_arr, axis=0)

    IC_list = []

    for i in range(len(P_arr)):
        p = P_arr[i]
        b = B_arr[i]
        vp = V_P[i]
        vb = V_B[i]

        p0, nP = perpendicular_line(p, vp)
        b0, nB = perpendicular_line(b, vb)

        IC = intersect_lines(p0, nP, b0, nB)
        IC_list.append(IC)

    return np.array(IC_list), V_P, V_B


# ============================================================
# Plotting
# ============================================================

def plot_all(B_arr, C_arr, P_arr, IC_arr, theta_list, A, D):
    crank_deg = np.degrees(theta_list)

    # ============================================================
    # Existing full-scale plot (unchanged)
    # ============================================================
    plt.figure(figsize=(6,6))
    plt.plot(B_arr[:,0], B_arr[:,1], label="B")
    plt.plot(C_arr[:,0], C_arr[:,1], label="C")
    plt.plot(P_arr[:,0], P_arr[:,1], label="P (coupler)")
    plt.plot(IC_arr[:,0], IC_arr[:,1], ".", alpha=0.5, label="Coupler IC")
    plt.scatter(A[0], A[1], c="r", label="A")
    plt.scatter(D[0], D[1], c="b", label="D")
    plt.axis("equal")
    plt.legend()
    plt.title("Trajectories + Coupler IC locus (full scale)")
    plt.show()

    # ============================================================
    # NEW: Scaled IC plot centered at (0,0), +/- 5 units
    # ============================================================

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(B_arr[:, 0], B_arr[:, 1], label="B")
    ax.plot(C_arr[:, 0], C_arr[:, 1], label="C")
    ax.plot(P_arr[:, 0], P_arr[:, 1], label="P (coupler)")
    ax.plot(IC_arr[:, 0], IC_arr[:, 1], ".", alpha=0.5, label="Coupler IC")
    ax.scatter(A[0], A[1], c="r", label="A")
    ax.scatter(D[0], D[1], c="b", label="D")

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-5.0, 5.0)
    ax.set_ylim(-5.0, 5.0)

    ax.legend()
    ax.set_title("Coupler IC locus (scaled view, center 0,0, range +/-5)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.show()




# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    # Known link lengths from your seed JSON
    a = 1.0
    b = 2.0
    c = 2.0

    A, D, B_ref, C_ref, u_local, v_local, theta_list, P_arr = load_cad_packet(
        "rank1_cad_packet.json"
    )

    B_arr, C_arr = compute_BC(A, D, theta_list, a, b, c, C_ref)

    IC_arr, V_P, V_B = compute_coupler_IC(P_arr, B_arr)

    plot_all(B_arr, C_arr, P_arr, IC_arr, theta_list, A, D)
