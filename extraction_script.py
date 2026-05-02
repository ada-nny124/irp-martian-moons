import os
import h5py
import numpy as np
import pandas as pd

# --- Constants for Mars ---
G = 6.67430e-11  
M_MARS = 6.4171e23  
R_MARS = 3.3895e6  
UNBOUND_ID = 2147483647  

OUTPUT_DIR = "extraction_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_masses_exacting(f_phys):
    """Targets the InitialMassTable found in your diagnostic."""
    if "PartType0/Masses" in f_phys:
        return f_phys["PartType0/Masses"][()]
    
    # Use InitialMassTable from Header as revealed by diagnostic
    if "Header" in f_phys and "InitialMassTable" in f_phys["Header"].attrs:
        m_table = f_phys["Header"].attrs["InitialMassTable"]
        m_val = m_table[0] # Particle Type 0
        if m_val > 0:
            num = f_phys["Header"].attrs["NumPart_ThisFile"][0]
            return np.full(num, m_val)
            
    raise KeyError("Could not find non-zero mass in PartType0/Masses or InitialMassTable.")

def calculate_orbital_elements(pos, vel):
    r_mag, v_mag = np.linalg.norm(pos), np.linalg.norm(vel)
    if r_mag == 0: return [0]*5
    energy = (v_mag**2 / 2.0) - (G * M_MARS / r_mag)
    h_vec = np.cross(pos, vel)
    h_mag = np.linalg.norm(h_vec)
    a = -G * M_MARS / (2.0 * energy) if abs(energy) > 1e-10 else np.inf
    e_vec = (np.cross(vel, h_vec) / (G * M_MARS)) - (pos / r_mag)
    e = np.linalg.norm(e_vec)
    i = np.degrees(np.arccos(h_vec[2] / h_mag)) if h_mag > 0 else 0
    return a, e, i, a*(1-e), a*(1+e) if e < 1 else np.inf

def process_simulation(directory):
    summary_list, catalog_list = [], []
    inits = [f for f in os.listdir(directory) if f.startswith("init_")]
    
    for init_name in inits:
        sim_id = init_name.replace("init_", "").replace(".hdf5", "")
        f_phys_path = os.path.join(directory, f"{sim_id}_90000.hdf5")
        f_fof_path = os.path.join(directory, f"{sim_id}_90000_fof_0.0020_0000.hdf5")
        
        if not os.path.exists(f_phys_path) or not os.path.exists(f_fof_path):
            continue

        try:
            # 1. Load Units from Snapshots (as per diagnostic)
            with h5py.File(f_phys_path, "r") as f_p:
                u_l = float(f_p["Units"].attrs["Unit length in cgs (U_L)"][0]) * 1e-2 # cm to m
                u_m = float(f_p["Units"].attrs["Unit mass in cgs (U_M)"][0]) * 1e-3   # g to kg
                u_t = float(f_p["Units"].attrs["Unit time in cgs (U_t)"][0])
                u_v = u_l / u_t
                
                # Load Physics
                box = f_p["Header"].attrs["BoxSize"][0] * u_l
                p_ids = f_p["PartType0/ParticleIDs"][()]
                pos = (f_p["PartType0/Coordinates"][()] * u_l) - (0.5 * box)
                vel = f_p["PartType0/Velocities"][()] * u_v
                mass = get_masses_exacting(f_p) * u_m
                
                df_phys = pd.DataFrame({
                    "p_id": p_ids, "m": mass,
                    "x": pos[:,0], "y": pos[:,1], "z": pos[:,2],
                    "vx": vel[:,0], "vy": vel[:,1], "vz": vel[:,2]
                })

            # 2. Load Clustering (FOF File)
            with h5py.File(f_fof_path, "r") as f_f:
                f_ids = f_f["PartType0/ParticleIDs"][()]
                g_ids = f_f["PartType0/FOFGroupIDs"][()] # Confirmed path
                df_fof = pd.DataFrame({"p_id": f_ids, "g_id": g_ids})

            # 3. Exact Alignment
            df = pd.merge(df_phys, df_fof, on="p_id")
            bound_df = df[(df["g_id"] >= 0) & (df["g_id"] < UNBOUND_ID)]
            
            # Aggregate Fragments
            def calc_com(g):
                tm = g["m"].sum()
                return pd.Series({
                    "m_kg": tm,
                    "cx": (g["x"]*g["m"]).sum()/tm, "cy": (g["y"]*g["m"]).sum()/tm, "cz": (g["z"]*g["m"]).sum()/tm,
                    "cvx": (g["vx"]*g["m"]).sum()/tm, "cvy": (g["vy"]*g["m"]).sum()/tm, "cvz": (g["vz"]*g["m"]).sum()/tm
                })

            frags = bound_df.groupby("g_id").apply(calc_com, include_groups=False).reset_index()

            for _, f in frags.iterrows():
                a, e, i, rp, ra = calculate_orbital_elements(np.array([f.cx, f.cy, f.cz]), np.array([f.cvx, f.cvy, f.cvz]))
                catalog_list.append({
                    "sim_id": sim_id, "fragment_id": int(f.g_id), "mass_kg": f.m_kg, 
                    "a": a, "e": e, "i": i, "rp_Rm": rp/R_MARS, "ra_Rm": ra/R_MARS
                })

            summary_list.append({
                "sim_id": sim_id, "n_particles": len(df_phys), 
                "bound_mass_kg": bound_df["m"].sum(), 
                "max_frag_kg": frags["m_kg"].max() if not frags.empty else 0, 
                "frag_count": len(frags)
            })

        except Exception as e:
            print(f"Error in {sim_id}: {e}")

    pd.DataFrame(summary_list).to_csv(os.path.join(OUTPUT_DIR, "parameter_study_summary.csv"), index=False)
    pd.DataFrame(catalog_list).to_csv(os.path.join(OUTPUT_DIR, "fragment_catalog.csv"), index=False)
    print(f"Extraction complete. Results in {OUTPUT_DIR}/")

if __name__ == "__main__":
    process_simulation("snapshots")