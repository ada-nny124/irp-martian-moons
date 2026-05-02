import os
import h5py
import numpy as np
import pandas as pd

# --- Constants for Mars ---
G = 6.67430e-11  
M_MARS = 6.4171e23  
R_MARS = 3.3895e6  
UNBOUND_ID = 2147483647  # INT_MAX used as the 'no group' flag

OUTPUT_DIR = "extraction_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_group_ids(h5_file):
    """Targets the confirmed FOFGroupIDs path."""
    if "PartType0/FOFGroupIDs" in h5_file:
        return h5_file["PartType0/FOFGroupIDs"][()]
    raise KeyError("FOFGroupIDs not found. Run h5ls on the fof file.")

def get_particle_masses(h5_file):
    """Robustly extracts masses and applies unit conversion."""
    # Attempt to get unit mass from the file attributes
    u_m = 1.0
    if "Units" in h5_file:
        u_m = float(h5_file["Units"].attrs.get("Unit mass in cgs (U_M)", 1.0)) * 1e-3 # to kg
    
    if "PartType0/Masses" in h5_file:
        return h5_file["PartType0/Masses"][()] * u_m
    
    # Fallback to MassTable
    mass_table = h5_file["Header"].attrs["MassTable"]
    num_particles = h5_file["Header"].attrs["NumPart_ThisFile"][0]
    return np.full(num_particles, mass_table[0]) * u_m

def calculate_orbital_elements(pos_vec, vel_vec):
    r_mag = np.linalg.norm(pos_vec)
    v_mag = np.linalg.norm(vel_vec)
    if r_mag == 0: return [0]*5
    
    energy = (v_mag**2 / 2.0) - (G * M_MARS / r_mag)
    h_vec = np.cross(pos_vec, vel_vec)
    h_mag = np.linalg.norm(h_vec)
    
    a = -G * M_MARS / (2.0 * energy) if abs(energy) > 1e-10 else np.inf
    e_vec = (np.cross(vel_vec, h_vec) / (G * M_MARS)) - (pos_vec / r_mag)
    e = np.linalg.norm(e_vec)
    i = np.degrees(np.arccos(h_vec[2] / h_mag)) if h_mag > 0 else 0
    return a, e, i, a*(1-e), a*(1+e) if e < 1 else np.inf

def process_simulation_set(directory):
    summary_list, catalog_list = [], []
    init_files = [f for f in os.listdir(directory) if f.startswith("init_")]
    
    for init_name in init_files:
        sim_id = init_name.replace("init_", "").replace(".hdf5", "")
        final_file = f"{sim_id}_90000.hdf5"
        fof_file = f"{sim_id}_90000_fof_0.0020_0000.hdf5"
        
        if not os.path.exists(os.path.join(directory, final_file)): continue

        try:
            with h5py.File(os.path.join(directory, init_name), "r") as f_init:
                u_l = float(f_init["Units"].attrs["Unit length in cgs (U_L)"]) * 1e-2
                u_t = float(f_init["Units"].attrs["Unit time in cgs (U_t)"])
                u_v = u_l / u_t
                n_particles = f_init["Header"].attrs["NumPart_Total"][0]
                
            with h5py.File(os.path.join(directory, final_file), "r") as f_final, \
                 h5py.File(os.path.join(directory, fof_file), "r") as f_fof:
                
                box = f_final["Header"].attrs["BoxSize"]
                pos = (f_final["PartType0/Coordinates"][()] - 0.5 * box) * u_l
                vel = f_final["PartType0/Velocities"][()] * u_v
                masses = get_particle_masses(f_final)
                group_ids = get_group_ids(f_fof)
                
                df = pd.DataFrame({"id": group_ids, "m": masses, "x": pos[:,0], "y": pos[:,1], "z": pos[:,2], "vx": vel[:,0], "vy": vel[:,1], "vz": vel[:,2]})
                
                # CRITICAL FILTER: Exclude the UNBOUND_ID (2147483647)
                bound_df = df[(df["id"] >= 0) & (df["id"] < UNBOUND_ID)]
                
                def calc_com(g):
                    total_m = g["m"].sum()
                    if total_m <= 0: return None
                    return pd.Series({
                        "f_mass": total_m,
                        "cx": (g["x"]*g["m"]).sum()/total_m, "cy": (g["y"]*g["m"]).sum()/total_m, "cz": (g["z"]*g["m"]).sum()/total_m,
                        "cvx": (g["vx"]*g["m"]).sum()/total_m, "cvy": (g["vy"]*g["m"]).sum()/total_m, "cvz": (g["vz"]*g["m"]).sum()/total_m
                    })

                frags = bound_df.groupby("id").apply(calc_com, include_groups=False).dropna().reset_index()

                for _, f in frags.iterrows():
                    a, e, i, rp, ra = calculate_orbital_elements(np.array([f.cx, f.cy, f.cz]), np.array([f.cvx, f.cvy, f.cvz]))
                    catalog_list.append({
                        "sim_id": sim_id, "fragment_id": int(f.id), "mass_kg": f.f_mass, 
                        "a": a, "e": e, "i": i, "rp_Rm": rp/R_MARS, "ra_Rm": ra/R_MARS
                    })

                summary_list.append({
                    "sim_id": sim_id, "n_particles": int(n_particles), 
                    "bound_mass_kg": bound_df["m"].sum(), 
                    "max_frag_kg": frags["f_mass"].max() if not frags.empty else 0, 
                    "frag_count": len(frags)
                })

        except Exception as e: print(f"Error {sim_id}: {e}")

    pd.DataFrame(summary_list).to_csv(os.path.join(OUTPUT_DIR, "parameter_study_summary.csv"), index=False)
    pd.DataFrame(catalog_list).to_csv(os.path.join(OUTPUT_DIR, "fragment_catalog.csv"), index=False)
    print(f"Extraction complete. Found {len(summary_list)} simulations and {len(catalog_list)} total fragments.")

if __name__ == "__main__":
    process_simulation_set("snapshots")