import os
import h5py
import numpy as np
import pandas as pd

# --- Constants for Mars ---
G = 6.67430e-11  
M_MARS = 6.4171e23  
R_MARS = 3.3895e6  

# Ensure output directory exists
OUTPUT_DIR = "extraction_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_group_ids(h5_file):
    """Robustly searches for Group IDs in common SPH/FoF output paths."""
    possible_paths = [
        "PartType0/GroupIDs",
        "PartType0/FOFGroupIDs",
        "GroupIDs",
        "FOF/GroupIDs"
    ]
    for path in possible_paths:
        if path in h5_file:
            return h5_file[path][()]
    raise KeyError("Could not find GroupIDs in HDF5 file. Check file structure.")

def get_particle_masses(h5_file):
    """Robustly extracts masses, checking both the dataset and the Header MassTable."""
    if "PartType0/Masses" in h5_file:
        return h5_file["PartType0/Masses"][()]
    else:
        # Fallback to MassTable in Header (Common in Gadget/SWIFT)
        mass_table = h5_file["Header"].attrs["MassTable"]
        # Use the first index for Type0 particles
        mass_value = mass_table[0]
        num_particles = h5_file["Header"].attrs["NumPart_ThisFile"][0]
        return np.full(num_particles, mass_value)

def calculate_orbital_elements(pos_vec, vel_vec):
    """Calculates keplerian elements for a Center of Mass (CoM) state vector."""
    r_mag = np.linalg.norm(pos_vec)
    v_mag = np.linalg.norm(vel_vec)
    
    if r_mag == 0: return 0, 0, 0, 0, 0

    energy = (v_mag**2 / 2.0) - (G * M_MARS / r_mag)
    h_vec = np.cross(pos_vec, vel_vec)
    h_mag = np.linalg.norm(h_vec)
    
    # Semi-major axis
    if abs(energy) < 1e-10: a = np.inf
    else: a = -G * M_MARS / (2.0 * energy)
    
    # Eccentricity vector
    e_vec = (np.cross(vel_vec, h_vec) / (G * M_MARS)) - (pos_vec / r_mag)
    e = np.linalg.norm(e_vec)
    
    # Inclination
    i = np.degrees(np.arccos(h_vec[2] / h_mag)) if h_mag > 0 else 0
    
    # Periapsis and Apoapsis
    periapsis = a * (1.0 - e)
    apoapsis = a * (1.0 + e) if e < 1 else np.inf
    
    return a, e, i, periapsis, apoapsis

def process_simulation_set(directory):
    summary_list = []
    catalog_list = []
    
    # Identify unique simulation IDs by looking for 'init' files
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
                group_ids = get_group_ids(f_fof) # Use the new robust seeker
                
                df_temp = pd.DataFrame({
                    "group_id": group_ids, "m": masses,
                    "x": pos[:, 0], "y": pos[:, 1], "z": pos[:, 2],
                    "vx": vel[:, 0], "vy": vel[:, 1], "vz": vel[:, 2]
                })
                
                # Filter for particles that actually belong to a fragment (Group ID >= 0)
                fragments = df_temp[df_temp["group_id"] >= 0].groupby("group_id").apply(lambda g: pd.Series({
                    "f_mass": g["m"].sum(),
                    "com_x": (g["x"] * g["m"]).sum() / g["m"].sum(),
                    "com_y": (g["y"] * g["m"]).sum() / g["m"].sum(),
                    "com_z": (g["z"] * g["m"]).sum() / g["m"].sum(),
                    "com_vx": (g["vx"] * g["m"]).sum() / g["m"].sum(),
                    "com_vy": (g["vy"] * g["m"]).sum() / g["m"].sum(),
                    "com_vz": (g["vz"] * g["m"]).sum() / g["m"].sum(),
                }), include_groups=False).reset_index()

                for _, frag in fragments.iterrows():
                    com_pos = np.array([frag.com_x, frag.com_y, frag.com_z])
                    com_vel = np.array([frag.com_vx, frag.com_vy, frag.com_vz])
                    a, e, i, rp, ra = calculate_orbital_elements(com_pos, com_vel)
                    
                    catalog_list.append({
                        "sim_id": sim_id, "fragment_id": int(frag.group_id),
                        "fragment_mass_kg": frag.f_mass,
                        "semi_major_axis_a": a, "eccentricity_e": e,
                        "inclination_i": i, "apoapsis_Rm": ra / R_MARS,
                        "periapsis_Rm": rp / R_MARS
                    })

                summary_list.append({
                    "sim_id": sim_id, "n_particles": int(n_particles),
                    "total_bound_mass_kg": df_temp[df_temp["group_id"] >= 0]["m"].sum(),
                    "largest_remnant_mass_kg": fragments["f_mass"].max() if not fragments.empty else 0,
                    "fragment_count": len(fragments)
                })

        except Exception as e:
            print(f"Error processing {sim_id}: {e}")

    # Save to the requested folder
    pd.DataFrame(summary_list).to_csv(os.path.join(OUTPUT_DIR, "parameter_study_summary.csv"), index=False)
    pd.DataFrame(catalog_list).to_csv(os.path.join(OUTPUT_DIR, "fragment_catalog.csv"), index=False)
    print(f"Extraction complete: Files saved in {OUTPUT_DIR}")

if __name__ == "__main__":
    process_simulation_set("snapshots")