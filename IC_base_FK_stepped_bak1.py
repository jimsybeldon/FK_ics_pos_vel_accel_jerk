import json
import numpy as np

# ============================================================
#  Load JSON geometry (authoritative source of truth)
# ============================================================

def load_geometry(seed_file, pivot_file):
    with open(seed_file, "r") as f:
        seed = json.load(f)[0]

    with open(pivot_file, "r") as f:
        pivots = json.load(f)

    # Link lengths
    a = seed["a"]
    b = seed["b"]
    c = seed["c"]

    # Ground pivots
    A = np.array(pivots["ground_pivots"]["A"])
    D = np.array(pivots["ground_pivots"]["D"])

    # Reference moving pivots (used for branch selection)
    B_ref = np.array(pivots["rocker_pivots"]["B"])
    C_ref = np.array(pivots["rocker_pivots"]["C"])

    # Coupler local frame
    u_local = pivots["coupler_local_frame"]["u"]
    v_local = pivots["coupler_local_frame"]["v"]

    print("\n=== JSON Geometry Loaded (Source of Truth) ===")
    print("Ground pivot A:", A)
    print("Ground pivot D:", D)
    print("Link lengths: a =", a, " b =", b, " c =", c)
    print("Reference B:", B_ref)
    print("Reference C:", C_ref)
    print("Coupler local frame (u, v):", (u_local, v_local))
    print("================================================\n")

    return A, D, a, b, c, B_ref, C_ref, u_local, v_local


# ============================================================
#  Circle–circle intersection (pure geometric closure)
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

    # Deterministic branch selection using JSON reference pose
    return C1 if np.linalg.norm(C1 - C_ref) < np.linalg.norm(C2 - C_ref) else C2


# ============================================================
#  Coupler point transform (rigid BC frame)
# ============================================================

def coupler_point(B, C, u_local, v_local):
    phi = np.arctan2(C[1] - B[1], C[0] - B[0])
    R = np.array([[np.cos(phi), -np.sin(phi)],
                  [np.sin(phi),  np.cos(phi)]])
    return B + R @ np.array([u_local, v_local])


# ============================================================
#  FK stepping with validation outputs
# ============================================================

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

    B_arr = np.array(B_list)
    C_arr = np.array(C_list)
    P_arr = np.array(P_list)

    # ============================================================
    #  HUMAN‑VERIFIABLE VALIDATION OUTPUTS
    # ============================================================

    print("=== First 5 FK Positions ===")
    for i in range(5):
        print(f"Step {i}:")
        print("  B =", B_arr[i])
        print("  C =", C_arr[i])
        print("  P =", P_arr[i])

    print("\n=== Last 5 FK Positions ===")
    for i in range(steps-5, steps):
        print(f"Step {i}:")
        print("  B =", B_arr[i])
        print("  C =", C_arr[i])
        print("  P =", P_arr[i])

    # Link length closure errors
    print("\n=== Link Length Closure Errors (should be near zero) ===")
    print("Mean(|B-A| - a):", np.mean(np.linalg.norm(B_arr - A, axis=1) - a))
    print("Mean(|C-B| - b):", np.mean(np.linalg.norm(C_arr - B_arr, axis=1) - b))
    print("Mean(|C-D| - c):", np.mean(np.linalg.norm(C_arr - D, axis=1) - c))

    # Rocker angle continuity (detects branch jumps)
    rocker_angles = np.unwrap(np.arctan2(C_arr[:,1] - D[1], C_arr[:,0] - D[0]))
    max_jump = np.max(np.abs(np.diff(rocker_angles)))
    print("\nMax rocker angle jump (should be small):", max_jump)

    print("\n=== FK Validation Complete ===\n")

    return B_arr, C_arr, P_arr, thetas


# ============================================================
#  Example usage
# ============================================================

if __name__ == "__main__":
    run_fk("atlas_seed_Rank1_for_IC.json", "atlas_seed_Rank1_Pivots.json")
