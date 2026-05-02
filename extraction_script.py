import os
import h5py
import numpy as np
import pandas as pd

# --- Constants for Mars ---
G = 6.67430e-11  
M_MARS = 6.4171e23  
R_MARS = 3.3895e6  

def calculate_orbital_elements(pos_vec, vel_vec):
    """Calculates keplerian elements for a CoM state vector."""
    r_mag = np.linalg.norm(pos_vec)
    v_mag = np.linalg.norm(vel_vec)
    
    # Specific Energy and Angular Momentum
    energy = (v_mag**2 / 2.0) - (G * M_MARS / r_mag)
    h_vec = np.cross(pos_vec, vel_vec)
    h_mag = np.linalg.norm(h_vec)
    
    # Semi-major axis
    if abs(energy) < 1e-10: a = np.inf
    else: a = -G * M_MARS / (2.0 * energy)
    
    # Eccentricity
    e_vec = (np.cross(vel_vec, h_vec) / (G * M_MARS)) - (pos_vec / r_mag)
    e = np.linalg.norm(e_vec)
    
    # Inclination
    i = np.degrees(np.arccos(h_vec[2] / h_mag)) if h_mag > 0 else 0
    
    # Periapsis and Apoapsis
    periapsis = a * (1.0 - e)
    apoapsis = a * (1.0 + e) if e < 1 else np.inf
    
    return a, e, i, periapsis, apoapsis

def extract_metadata_from_filename(filename):
    """
    Parses your naming convention: Ma_xp_A[Mass]_n[Particles]_r[Rp]_v[Vinf]
    Adjust split logic based on your specific file string precision.
    """
    parts = filename.split('_')
    metadata = {
        "sim_id": filename.replace("init_", "").replace(".hdf5", ""),
        "mass_label": parts[2], # e.g., A2000
        "n_label": parts[3],    # e.g., n70
        "rp_label": parts[4],   # e.g., r16
        "v_label": parts[5]     # e.g., v00
    }
    return metadata

def process_simulation_set(directory):
    summary_list = []
    catalog_list = []
    
    # Identify unique simulation IDs by looking for 'init' files
    init_files = [f for f in os.listdir(directory) if f.startswith("init_")]
    
    for init_name in init_files:
        meta = extract_metadata_from_filename(init_name)
        sim_id = meta["sim_id"]
        
        # Define paths for paired files
        final_file = f"{sim_id}_90000.hdf5"
        fof_file = f"{sim_id}_90000_fof_0.0020_0000.hdf5"
        
        if not os.path.exists(os.path.join(directory, final_file)): continue

        try:
            # 1. Extract INPUTS from init file
            with h5py.File(os.path.join(directory, init_name), "r") as f_init:
                u_l = float(f_init["Units"].attrs["Unit length in cgs (U_L)"]) * 1e-2
                n_particles = f_init["Header"].attrs["NumPart_Total"][0]
                # Assuming mass and spin are stored in Header or specific datasets
                total_mass = np.sum(f_init["PartType0/Masses"][()]) * (u_l**3 * 1e-3) # simplified unit conv
                
            # 2. Extract OUTCOMES and Cluster data
            with h5py.File(os.path.join(directory, final_file), "r") as f_final, \
                 h5py.File(os.path.join(directory, fof_file), "r") as f_fof:
                
                # Physics data
                box = f_final["Header"].attrs["BoxSize"]
                pos = (f_final["PartType0/Coordinates"][()] - 0.5 * box) * u_l
                vel = f_final["PartType0/Velocities"][()] * (u_l / 1.0) # check unit time
                masses = f_final["PartType0/Masses"][()] # in internal units
                
                # Clustering IDs
                group_ids = f_fof["PartType0/GroupIDs"][()]
                
                # Build temporary DataFrame for aggregation
                df_temp = pd.DataFrame({
                    "group_id": group_ids,
                    "m": masses,
                    "x": pos[:, 0], "y": pos[:, 1], "z": pos[:, 2],
                    "vx": vel[:, 0], "vy": vel[:, 1], "vz": vel[:, 2]
                })
                
                # Aggregate Fragment Data
                fragments = df_temp[df_temp["group_id"] >= 0].groupby("group_id").apply(lambda g: pd.Series({
                    "f_mass": g["m"].sum(),
                    "com_x": (g["x"] * g["m"]).sum() / g["m"].sum(),
                    "com_y": (g["y"] * g["m"]).sum() / g["m"].sum(),
                    "com_z": (g["z"] * g["m"]).sum() / g["m"].sum(),
                    "com_vx": (g["vx"] * g["m"]).sum() / g["m"].sum(),
                    "com_vy": (g["vy"] * g["m"]).sum() / g["m"].sum(),
                    "com_vz": (g["vz"] * g["m"]).sum() / g["m"].sum(),
                })).reset_index()

                # Calculate Orbital Elements for each fragment
                for _, frag in fragments.iterrows():
                    com_pos = np.array([frag.com_x, frag.com_y, frag.com_z])
                    com_vel = np.array([frag.com_vx, frag.com_vy, frag.com_vz])
                    a, e, i, rp, ra = calculate_orbital_elements(com_pos, com_vel)
                    
                    catalog_list.append({
                        "sim_id": sim_id,
                        "fragment_id": frag.group_id,
                        "fragment_mass_kg": frag.f_mass,
                        "mass_ratio": frag.f_mass / total_mass,
                        "semi_major_axis_a": a,
                        "eccentricity_e": e,
                        "inclination_i": i,
                        "apoapsis_Rm": ra / R_MARS,
                        "periapsis_Rm": rp / R_MARS
                    })

                # Calculate Summary Data
                r_mags = np.linalg.norm(pos, axis=1)
                summary_list.append({
                    "sim_id": sim_id,
                    "n_particles": n_particles,
                    "total_bound_mass_kg": df_temp[df_temp["group_id"] >= 0]["m"].sum(),
                    "largest_remnant_mass_kg": fragments["f_mass"].max() if not fragments.empty else 0,
                    "fragment_count": len(fragments),
                    "debris_spread_km": (np.max(r_mags) - np.min(r_mags)) / 1000.0
                })

        except Exception as e:
            print(f"Error processing {sim_id}: {e}")

    # Export to CSV
    pd.DataFrame(summary_list).to_csv("parameter_study_summary.csv", index=False)
    pd.DataFrame(catalog_list).to_csv("fragment_catalog.csv", index=False)
    print("Extraction complete: 2 files generated.")

if __name__ == "__main__":
    process_simulation_set("snapshots")