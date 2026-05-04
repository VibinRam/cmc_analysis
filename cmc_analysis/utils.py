import numpy as np
import pandas as pd
import glob
import os
import gzip
import re
import pickle
import logging
from matplotlib.collections import LineCollection
import h5py

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.WARNING)
logger.setLevel(logging.WARNING)

SNAP_COLS = [
    "id", "m", "r", "vr", "vt", "E", "J", "binflag",
    "m0", "m1", "id0", "id1", "a", "e", "startype",
    "luminosity", "radius",
    "bin_startype0", "bin_startype1",
    "bin_star_lum0", "bin_star_lum1",
    "bin_star_radius0", "bin_star_radius1",
    "Eb", "eta", "star_phi",
    "rad0", "rad1", "tb",
    "lum0", "lum1",
    "massc0", "massc1",
    "radc0", "radc1",
    "menv0", "menv1",
    "renv0", "renv1",
    "tms0", "tms1",
    "dmdt0", "dmdt1",
    "radrol0", "radrol1",
    "ospin0", "ospin1",
    "B0", "B1",
    "formation0", "formation1",
    "bacc0", "bacc1",
    "tacc0", "tacc1",
    "mass0_0", "mass0_1",
    "epoch0", "epoch1",
    "ospin", "B", "formation"
]

BHFORM_COLS = [
    "time", "r", "binary", "ID", "zams_m",
    "m_progenitor", "bh_mass", "bh_spin", 
    "birth_kick", "vsarray0", "vsarray1",
    "vsarray2", "vsarray3", "vsarray4",
    "vsarray5", "vsarray6", "vsarray7",
    "vsarray8", "vsarray9", "vsarray10",
    "vsarray11", "vsarray12", "vsarray13",
    "vsarray14", "vsarray15"
]

BHMERG_COLS = [
    "time", "type", "r", "id1", "id2", "m1", "m2", "spin1", "spin2",
    "final_id", #--> This column not in old cmc version.
    "m_final", "spin_final", "v_kick", "v_esc", "a_final", "e_final", "a_50M", "e_50M", "a_100M", "e_100M", "a_500M", "e_500M"]

EVENT_TO_COLOR = {
    "formation" : 'c',
    "collision" : 'g',
    "merger" : 'r',
    "accretion" : 'b',
    "escape" : 'm',
    "merged" : 'k',
    "disruption" : 'y'
}

# code unit to Myr conversion factor
CODE_TO_MYR = 516.414

# msun per code unit time to msun per myr conversion
MSUN_PER_CODE_TO_MSUN_PER_MYR = (1/516.414)

# mstar unit msun to msun (esc file seems to use this for bh)
MSTAR_UNIT_MSUN_TO_MSUN = 0.605875

# age of the universe
AGE_OF_UNIV = 13.8e3 #myr

def calc_peters_eqn(a, e, m1, m2):
    # a - AU
    # m - Mo

    # returns  Myr

    const = 3.15e11 # MYr Mo^3/AU^4

    f_e = np.power(1- e**2, 7.0/2.0)

    merg_time = const * f_e * np.power(a, 4) / (m1 * m2 * (m1+m2))

    return myr_to_code_unit(merg_time)

def calc_chirp_mass(m1, m2):

    numer = np.power(m1 * m2, 3.0/5.0)
    dinom = np.power(m1 + m2, 1.0/5.0)

    return numer/dinom

def calc_effective_spin(m1, m2, a1, a2, ct1, ct2):

    numer = m1*a1*ct1 + m2*a2*ct2
    denom = m1 + m2

    return numer/denom

def get_isotropic_tilts():

    s1, s2 = np.random.uniform(-1, 1, 2)

    return s1, s2    

def code_unit_to_myr(time):
    return CODE_TO_MYR * time

def myr_to_code_unit(time):
    return time / CODE_TO_MYR

def dmdt_code_unit_to_per_myr(dmdt):
    return MSUN_PER_CODE_TO_MSUN_PER_MYR * dmdt

def code_unit_esc_to_MSUN(mass):
    return MSTAR_UNIT_MSUN_TO_MSUN * mass

def clean_colnames(list_unproc_colnames):
    clean_colnames = []

    for colname in list_unproc_colnames:

        colname = re.sub(r'^#\d+:', '', colname)

        colname = re.sub(r'\[.*?\]', '', colname)

        clean_colnames.append(colname)

    return clean_colnames

def read_esc(file_name):

    df = pd.read_csv(file_name, sep=r'\s+', index_col=None, na_values='na')

    df.columns = clean_colnames(df.columns)

    return df

def read_snap(out_loc, file_name, prefix):

    if "t=" in file_name:

        h5_file = os.path.join(out_loc, f"{prefix}.snapshots.h5")

        with h5py.File(h5_file, 'r') as f:

            df = pd.DataFrame(f[file_name][:])
            
        time = float(re.search(r't=([0-9.Ee+-]+)', file_name).group(1))

        df.attrs['time'] = time

    elif file_name.endswith(".dat.gz"):

        with gzip.open(file_name, 'rt') as f:
            first_line = f.readline()

        time = float(first_line.split('t=')[1].split()[0])

        df = pd.read_csv(file_name, comment='#', sep=r'\s+', header=None, engine='python', index_col=None)

        df.attrs['time'] = time

    else:

        logger.error(f"{file_name} not a valid file name")
        raise FileNotFoundError

    df.columns = SNAP_COLS
    
    logger.info(f'# Read {file_name}\n')

    return df

def sort_snap(out_loc, prefix):
    
    file_pattern = os.path.join(out_loc, f"{prefix}.snap????.dat.gz")

    all_snap_files = glob.glob(file_pattern)

    if len(all_snap_files) == 0:

        try:

            h5_file = os.path.join(out_loc, f"{prefix}.snapshots.h5")

            with h5py.File(h5_file, 'r') as f:

                all_snap_files = list(f.keys())

            all_snap_files = sorted(all_snap_files, key=lambda s: int(s.split('(')[0]))
        
        except:

            logger.error("No valid snapshots files found")
    
    else:

        all_snap_files.sort()

    return all_snap_files

def parse_bh_tracks(out_loc, prefix):

    bh_tracks = {}

    sorted_snaps = sort_snap(out_loc, prefix)

    for snap_idx in range(len(sorted_snaps)):
        
        df = read_snap(out_loc, sorted_snaps[snap_idx], prefix)
        time = df.attrs['time']
    
        for row in df.itertuples():

            bh_ids = []
            dmdts = []
            masses = []
            companion_ids = []
            companion_types = []

            if row.id != 0 and row.id != -100 and row.startype == 14:
                
                bh_ids.append(int(row.id))
                dmdts.append(0.0)
                masses.append(float(row.m))
                companion_ids.append(None)
                companion_types.append(None)

            if row.binflag == 1 and row.bin_startype0 == 14:

                bh_ids.append(int(row.id0))
                dmdts.append(float(row.dmdt0))
                masses.append(float(row.m0))
                companion_ids.append(int(row.id1))
                companion_types.append(int(row.bin_startype1))

            if row.binflag == 1 and row.bin_startype1 == 14:
             
                bh_ids.append(int(row.id1))
                dmdts.append(float(row.dmdt1))
                masses.append(float(row.m1))
                companion_ids.append(int(row.id0))
                companion_types.append(int(row.bin_startype0))

            for i in range(len(bh_ids)):

                dmdts[i] = dmdt_code_unit_to_per_myr(dmdts[i])

                if bh_ids[i] not in bh_tracks:

                    bh_tracks[bh_ids[i]] = {
                        "snap_idx" : [snap_idx],
                        "time" : [time],
                        "dmdt" : [dmdts[i]],
                        "mass" : [masses[i]],
                        "binflag" : [row.binflag],
                        "companion_id" : [companion_ids[i]],
                        "companion_type" : [companion_types[i]]
                    }

                else:

                    bh_tracks[bh_ids[i]]["snap_idx"].append(snap_idx)
                    bh_tracks[bh_ids[i]]["time"].append(time)
                    bh_tracks[bh_ids[i]]["dmdt"].append(dmdts[i])
                    bh_tracks[bh_ids[i]]["mass"].append(masses[i])
                    bh_tracks[bh_ids[i]]["binflag"].append(row.binflag)
                    bh_tracks[bh_ids[i]]["companion_id"].append(companion_ids[i])
                    bh_tracks[bh_ids[i]]["companion_type"].append(companion_types[i])

    with open(os.path.join(out_loc, "bh_tracks.pkl"), 'wb') as f:
        pickle.dump(bh_tracks, f, protocol=pickle.HIGHEST_PROTOCOL)

    return bh_tracks

def parse_bh_mergers(out_loc, prefix):

    merger_fname = f"{prefix}.semergedisrupt.log"

    merger_file = os.path.join(out_loc, merger_fname)

    bh_mergers = {}

    with open(merger_file, 'r') as f:
    
        for line in f:

            if not line.startswith("t="):
                continue

            if "disruptboth" in line:
                continue

            log_time = float(line.split()[0].split("=")[1])
            id_rem = int(re.search(r'idr=(\d+)', line).group(1))
            mass_rem = float(re.search(r'mr=([0-9.Ee+-]+)', line).group(1))
            rem_type = int(re.search(r'typer=(\d+)', line).group(1))

            if rem_type != 14:
                continue

            parents = []
            parent_masses = []
            parent_types = []

            for match in re.finditer(r'id(\d+)=(\d+)\(m\d+=([0-9.Ee+-]+)\)', line):

                parents.append(int(match.group(2)))
                parent_masses.append(float(match.group(3)))

            for match in re.finditer(r'type(\d+)=([0-9]+)', line):

                parent_types.append(int(match.group(2)))

            if "disrupt1" in line:
           
                id_merged = parents[1]
                mass_merged = parent_masses[1]
                type_merged = parent_types[1]
                id_host = parents[0]
                mass_host = parent_masses[0]
                type_host = parent_types[0]

            elif "disrupt2" in line:

                id_merged = parents[0]
                mass_merged = parent_masses[0]
                type_merged = parent_types[0]
                id_host = parents[1]
                mass_host = parent_masses[1]
                type_host = parent_types[1]

            # Keep only BH+BH mergers; semergedisrupt also logs mixed-type mergers
            # that do not appear in bhmerger.dat.
            if type_host != 14 or type_merged != 14:
                continue
            
            if id_rem not in bh_mergers:

                bh_mergers[id_rem] = {
                    "time" : [log_time],
                    "mass_rem" : [mass_rem],
                    "id_merged" : [id_merged],
                    "mass_merged" : [mass_merged],
                    "type_merged" : [type_merged],
                    "id_host" : [id_host],
                    "mass_host" : [mass_host],
                    "type_host" : [type_host]
                }

            else:

                bh_mergers[id_rem]["time"].append(log_time)
                bh_mergers[id_rem]["mass_rem"].append(mass_rem)
                bh_mergers[id_rem]["id_merged"].append(id_merged)
                bh_mergers[id_rem]["mass_merged"].append(mass_merged)
                bh_mergers[id_rem]["type_merged"].append(type_merged)
                bh_mergers[id_rem]["id_host"].append(id_host)
                bh_mergers[id_rem]["mass_host"].append(mass_host)
                bh_mergers[id_rem]["type_host"].append(type_host)

    mergerdat_fname = f"{prefix}.bhmerger.dat"

    mergerdat_file = os.path.join(out_loc, mergerdat_fname)

    merger_df = pd.read_csv(mergerdat_file, comment='#', sep=r'\s+', engine='python',  header=None,  index_col=None)

    merger_df.columns = BHMERG_COLS

    # Initialize spin arrays once per remnant before filling from bhmerger.dat.
    for id_rem, merger_info in bh_mergers.items():
        n_mergers = len(merger_info["time"])
        merger_info["spin_host"] = np.full(n_mergers, np.nan)
        merger_info["spin_merged"] = np.full(n_mergers, np.nan)
        merger_info["semi_maj"] = np.full(n_mergers, np.nan)
        merger_info["eccentricity"] = np.full(n_mergers, np.nan)

    unmatched_rows = []

    for row in merger_df.itertuples():

        if int(row.id1) in bh_mergers.keys():

            id_rem = int(row.id1)
            id_merged = int(row.id2)
            spin_rem = float(row.spin1)
            spin_merged = float(row.spin2)
            row_time = float(row.time)
            semi_maj = float(row.a_100M)
            eccentricity = float(row.e_100M)

        elif int(row.id2) in bh_mergers.keys():

            id_rem = int(row.id2)
            id_merged = int(row.id1)
            spin_rem = float(row.spin2)
            spin_merged = float(row.spin1)
            row_time = float(row.time)
            semi_maj = float(row.a_100M)
            eccentricity = float(row.e_100M)

        else:
            unmatched_rows.append(row)

            continue

        match_inds = [
            idx
            for idx, candidate in enumerate(bh_mergers[id_rem]["id_merged"])
            if candidate == id_merged
        ]

        if not match_inds:
            unmatched_rows.append(row)
            continue

        # If there are repeated IDs, match by closest event time.
        merg_ind = min(
            match_inds,
            key=lambda idx: abs(bh_mergers[id_rem]["time"][idx] - row_time)
        )

        bh_mergers[id_rem]["spin_host"][merg_ind] = spin_rem

        # Prefer the higher-precision timestamp from bhmerger.dat over the
        # truncated timestamp embedded in semergedisrupt.log.
        bh_mergers[id_rem]["time"][merg_ind] = row_time

        bh_mergers[id_rem]["spin_merged"][merg_ind] = spin_merged

        bh_mergers[id_rem]["semi_maj"][merg_ind] = semi_maj

        bh_mergers[id_rem]["eccentricity"][merg_ind] = eccentricity

    # Add bhmerger-only pairs that are absent from semergedisrupt.
    # For these rows, use the more massive component as host/remnant proxy.
    n_fallback_added = 0

    for row in unmatched_rows:

        if float(row.m1) >= float(row.m2):
            id_rem = int(row.final_id)
            id_host = int(row.id1)
            id_merged = int(row.id2)
            mass_host = float(row.m1)
            mass_merged = float(row.m2)
            spin_host = float(row.spin1)
            spin_merged = float(row.spin2)
            semi_maj = float(row.a_100M)
            eccentricity = float(row.e_100M)
        else:
            id_rem = int(row.final_id)
            id_host = int(row.id2)
            id_merged = int(row.id1)
            mass_host = float(row.m2)
            mass_merged = float(row.m1)
            spin_host = float(row.spin2)
            spin_merged = float(row.spin1)
            semi_maj = float(row.a_100M)
            eccentricity = float(row.e_100M)

        if id_rem not in bh_mergers:

            bh_mergers[id_rem] = {
                "time": [float(row.time)],
                "mass_rem": [float(row.m_final)],
                "id_merged": [id_merged],
                "mass_merged": [mass_merged],
                "type_merged": [14],
                "id_host": [id_host],
                "mass_host": [mass_host],
                "type_host": [14],
                "spin_host": np.array([spin_host]),
                "spin_merged": np.array([spin_merged]),
                "semi_maj": np.array([semi_maj]),
                "eccentricity": np.array([eccentricity])
            }

        else:

            bh_mergers[id_rem]["time"].append(float(row.time))
            bh_mergers[id_rem]["mass_rem"].append(float(row.m_final))
            bh_mergers[id_rem]["id_merged"].append(id_merged)
            bh_mergers[id_rem]["mass_merged"].append(mass_merged)
            bh_mergers[id_rem]["type_merged"].append(14)
            bh_mergers[id_rem]["id_host"].append(id_host)
            bh_mergers[id_rem]["mass_host"].append(mass_host)
            bh_mergers[id_rem]["type_host"].append(14)
            bh_mergers[id_rem]["spin_host"] = np.append(bh_mergers[id_rem]["spin_host"], spin_host)
            bh_mergers[id_rem]["spin_merged"] = np.append(bh_mergers[id_rem]["spin_merged"], spin_merged)
            bh_mergers[id_rem]["semi_maj"] = np.append(bh_mergers[id_rem]["semi_maj"], semi_maj)
            bh_mergers[id_rem]["eccentricity"] = np.append(bh_mergers[id_rem]["eccentricity"], eccentricity)

        n_fallback_added += 1

    if n_fallback_added:
        logger.warning(
            "Added %d bhmerger-only rows that were missing in %s.",
            n_fallback_added,
            merger_fname,
        )

    if unmatched_rows and not n_fallback_added:
        logger.warning(
            "%d rows in %s could not be matched to %s; spins remain NaN for unmatched events.",
            len(unmatched_rows),
            mergerdat_fname,
            merger_fname,
        )

    with open(os.path.join(out_loc, "bh_mergers.pkl"), "wb") as f:
        pickle.dump(bh_mergers, f, protocol=pickle.HIGHEST_PROTOCOL)    

    return bh_mergers  

def parse_bh_collisions(out_loc, prefix):

    collision_fname = f"{prefix}.collision.log"

    coll_file = os.path.join(out_loc, collision_fname)

    bh_collisions = {}

    # 3 body collisions seems to have 0 as the remnant id

    b3_col_ind = 'X3'

    b3_col_count = 0

    with open(coll_file, 'r') as f:
   
        for line in f:
            
            if not line.startswith("t="):
                continue

            time = float(line.split()[0].split("=")[1])
            id_rem = int(re.search(r'idm=(\d+)', line).group(1))
            mass_rem = float(re.search(r'mm=([0-9.Ee+-]+)', line).group(1))
            rem_type = int(re.search(r'typem=(\d+)', line).group(1))

            if rem_type != 14:
                continue

            parents = []
            parent_masses = []
            parent_types = []

            for match in re.finditer(r'id(\d+)=(\d+)\(m\d+=([0-9.Ee+-]+)\)', line):

                parents.append(int(match.group(2)))
                parent_masses.append(float(match.group(3)))

            for match in re.finditer(r'type(\d+)=([0-9]+)', line):

                parent_types.append(int(match.group(2)))

            if id_rem == 0:
                id_rem = b3_col_ind + str(b3_col_count)
                b3_col_count += 1
            if 0 in parents:
                for nth, parent_id in enumerate(parents):
                    if parent_id == 0:
                        parents[nth] = b3_col_ind + str(b3_col_count)
                        b3_col_count += 1

            bh_collisions[id_rem] = {
                "time" : time,
                "mass_rem" : mass_rem,
                "parents" : parents,
                "parent_masses" : parent_masses,
                "parent_types" : parent_types
            }

    with open(os.path.join(out_loc, "bh_collisions.pkl"), 'wb') as f:
        pickle.dump(bh_collisions, f, protocol=pickle.HIGHEST_PROTOCOL)

    return bh_collisions

def parse_bh_escapers(out_loc, prefix):

    esc_filename = f"{prefix}.esc.dat"

    esc_file = os.path.join(out_loc, esc_filename)

    esc_df = read_esc(esc_file)

    bh_escapers = {}

    xe_col_count = 0

    for row in esc_df.itertuples():

        bh_ids = []
        masses = []
        dmdts = []
        binflag = []
        semi_maj = []
        eccentr = []
        companion_ids = []
        companion_types = []
        companion_masses = []
        spins = []
        companion_spins = []
    
        if row.binflag == 0 and int(row.startype) == 14:
            
            if row.id == 0:
                prefixed_id = 'XE' + str(xe_col_count)
                bh_ids.append(prefixed_id)
                xe_col_count += 1
            else:
                bh_ids.append(int(row.id))
            masses.append(float(row.m))     #caution: In the older version of cmc, this mass is in code units.
            dmdts.append(0.0)
            binflag.append(0)
            semi_maj.append(None)
            eccentr.append(None)
            companion_ids.append(None)
            companion_types.append(None)
            companion_masses.append(None)
            spins.append(float(row.bhspin) if pd.notna(row.bhspin) else np.nan)
            companion_spins.append(np.nan)
            
        if row.binflag == 1 and int(row.bin_startype0) == 14:

            if row.id0 == 0:
                prefixed_id = 'XE' + str(xe_col_count)
                bh_ids.append(prefixed_id)
                xe_col_count += 1
            else:
                bh_ids.append(int(row.id0))
            masses.append(float(row.m0))
            dmdts.append(float(row.dmdt0))
            binflag.append(1)
            semi_maj.append(float(row.a))
            eccentr.append(float(row.e))
            companion_ids.append(int(row.id1))
            companion_types.append(int(row.bin_startype1))
            spins.append(float(row.bhspin1) if pd.notna(row.bhspin1) else np.nan)
            companion_spins.append(float(row.bhspin2) if pd.notna(row.bhspin2) else np.nan)

        if row.binflag == 1 and int(row.bin_startype1) == 14:

            if row.id1 == 0:
                prefixed_id = 'XE' + str(xe_col_count)
                bh_ids.append(prefixed_id)
                xe_col_count += 1
            else:
                bh_ids.append(int(row.id1))
            masses.append(float(row.m1))
            dmdts.append(float(row.dmdt1))
            binflag.append(1)
            semi_maj.append(float(row.a))
            eccentr.append(float(row.e))
            companion_ids.append(int(row.id0))
            companion_types.append(int(row.bin_startype0))
            spins.append(float(row.bhspin2) if pd.notna(row.bhspin2) else np.nan)
            companion_spins.append(float(row.bhspin1) if pd.notna(row.bhspin1) else np.nan)
        
        for i in range(len(bh_ids)):
                    
            bh_escapers[bh_ids[i]] = {
                "time" : row.t,
                "mass" : masses[i],
                "dmdt" : dmdts[i],
                "binflag" : binflag[i],
                "a" : semi_maj[i],
                "e" : eccentr[i],
                "companion_id" : companion_ids[i],
                "companion_type" : companion_types[i],
                "spin" : spins[i],
                "companion_spin" : companion_spins[i]
               }
    
    with open(os.path.join(out_loc, "bh_escapers.pkl"), 'wb') as f:
        pickle.dump(bh_escapers, f, protocol=pickle.HIGHEST_PROTOCOL) 
               
    return bh_escapers    

def parse_bh_formations(out_loc, prefix):

    bh_formations_fname = f"{prefix}.bhformation.dat"

    bh_formations_file = os.path.join(out_loc, bh_formations_fname)

    df = pd.read_csv(bh_formations_file, comment='#', sep=r'\s+', engine='python',  header=None,  index_col=None, on_bad_lines='skip')

    df = df.dropna(thresh=10)

    df.columns = BHFORM_COLS

    bh_formations = {}

    for row in df.itertuples():
  
        if row.ID == 0:
            continue

        bh_formations[int(row.ID)] = {
            
            "time" : float(row.time),
            "m_progenitor" : float(row.m_progenitor),
            "m_bh" : float(row.bh_mass),
            "binary" : int(row.binary)
        }

    with open(os.path.join(out_loc, "bh_formations.pkl"), 'wb') as f:
        pickle.dump(bh_formations, f, protocol=pickle.HIGHEST_PROTOCOL)

    return bh_formations

def parse_snap_times(out_loc, prefix):

    sorted_snaps = sort_snap(out_loc, prefix)

    snap_times = {}

    for snap_idx, snap_name in enumerate(sorted_snaps):

        if "t=" in snap_name:

            time = float(re.search(r't=([0-9.Ee+-]+)', snap_name).group(1))

        elif snap_name.endswith(".dat.gz"):

            with gzip.open(snap_name, 'rt') as f:

                first_line = f.readline()

                time = float(first_line.split('t=')[1].split()[0])

        else:

            logger.error("No valid snapshots found.")

        snap_times[snap_idx] = time

    time_to_snap = {v: k for k, v in snap_times.items()}

    return snap_times, time_to_snap

def load_bh_tracks(out_loc):

    bh_tracks_file = os.path.join(out_loc, "bh_tracks.pkl")

    try:

        with open(bh_tracks_file, 'rb') as f:
            bh_tracks = pickle.load(f)

    except FileNotFoundError:

        logger.error("Error: file \"bh_track.pkl\" not found. Try parse_bh_tracks()")
        raise 

    return bh_tracks

def load_bh_collisions(out_loc):

    bh_collisions_file = os.path.join(out_loc, "bh_collisions.pkl")

    try:

        with open(bh_collisions_file, 'rb') as f:
            bh_collisions = pickle.load(f)

    except FileNotFoundError:

        logger.error("Error: file \"bh_collisions.pkl\" not found. Try parse_bh_collisions()")
        raise

    return bh_collisions

def load_bh_mergers(out_loc):

    bh_mergers_file = os.path.join(out_loc, "bh_mergers.pkl")

    try:

        with open(bh_mergers_file, 'rb') as f:
            bh_mergers = pickle.load(f)

    except FileNotFoundError:

        logger.error("Error: file \"bh_mergers.pkl\" not found. Try parse_bh_mergers()")
        raise

    return bh_mergers

def load_bh_escapers(out_loc):

    bh_escapers_file = os.path.join(out_loc, "bh_escapers.pkl")

    try:

        with open(bh_escapers_file, 'rb') as f:
            bh_escapers = pickle.load(f)

    except FileNotFoundError:

        logger.error("Error: file \"bh_escapers.pkl\" not found. Try parse_bh_escapers()")
        raise

    return bh_escapers 

def load_bh_formations(out_loc):

    bh_formations_file = os.path.join(out_loc, "bh_formations.pkl")

    try:

        with open(bh_formations_file, 'rb') as f:
            bh_formations = pickle.load(f)

    except FileNotFoundError:

        logger.error("Error: file \"bh_formations.pkl\" not found. Try parse_bh_formations()")
        raise

    return bh_formations

def augment_formations_with_tracks(bh_formations, bh_tracks):

    # Some CMC runs can miss BH birth events in formation logs.
    # Seed those BHs from their first appearance in the snapshot tracks.
    if bh_formations is None:
        bh_formations = {}

    n_added = 0

    for bh_id, track in bh_tracks.items():

        if not isinstance(bh_id, int):
            continue

        if bh_id in bh_formations:
            continue

        if len(track.get("time", [])) == 0 or len(track.get("mass", [])) == 0:
            continue

        birth_idx = int(np.argmin(track["time"]))

        bh_formations[bh_id] = {
            "time": float(track["time"][birth_idx]),
            "m_progenitor": np.nan,
            "m_bh": float(track["mass"][birth_idx]),
            "binary": -1,
            "source": "track_fallback"
        }
        n_added += 1

    if n_added > 0:
        logger.warning(
            f"Added {n_added} BH formation entries from tracks fallback."
        )

    return bh_formations

class BHWorldLine:

    def __init__(self, wid, birth_time, birth_id, birth_mass):

        self.wid = wid

        self.birth_id = birth_id

        self.birth_time = birth_time

        self.id_history = [
            {
                "time" : birth_time,
                "id" : birth_id
            }
        ]

        self.events = [
            {
                "event" : "formation",
                "time" : birth_time,
                "mass" : birth_mass
            }
        ]

        self.active = True

    def __call__(self, show_id_history=False, 
                 show_events=False,
                 show_events_vertical=False
        ):

        print(f"wid = {self.wid}:")

        if show_id_history:
        
            print(f"t : ", end=' ')

            for ids in self.id_history:

                print(f"{ids['time']:8.5f}", end=' ')
        
            print()

            print(f"id : ", end=' ')

            for ids in self.id_history:

                print(f"{str(ids['id']):8s}", end=' ')

            print() 

        if show_events:

            print(f"t : ", end=' ')

            for event in self.events:

                print(f"{event['time']:10.5f}", end=' ')

            print()

            print(f"event : ", end=' ')

            for event in self.events:

                print(f"{event['event']:10s}", end=' ')

            print()

            print(f"mass : ", end=' ')

            for event in self.events:
                print(f"{event['mass']:10.5f}", end=' ')

            print()

        if show_events_vertical:

            print(f"Ids: ", end='')

            for id_info in self.id_history:

                print(f"{id_info['id']}", end=' ')

            print()

            print(f"{'t':10s} {'event':10s} {'mass':10s} {'partner_id':10s}")

            for event in self.events:

                print(
                    f"{event['time']:10.5f}",
                    f"{event['event']:10s}",
                    f"{event['mass']:10.5f}",
                    f"{event.get('partner_id', 0):10d}"
                )

            print()


    def add_collision(
            self, time, mass, new_id,
            partner_ids, partner_types, 
            partner_masses, disrupt=False
        ):
  
        self.events.append(
            {
                "event" : "collision",
                "time" : time,
                "mass" : mass,
                "partner_ids" : partner_ids,
                "partner_types" : partner_types,
                "partner_masses" : partner_masses,
            }
        )

        if not disrupt:

            self.id_history.append(
                {
                    "time" : time,
                    "id" : new_id
                }
            )

        if disrupt:
   
            self.events.append(
                {
                    "event" : "disruption",
                    "time" : time,
                    "mass" : mass,
                    "disrupted_by" : new_id
                }
            )

            self.active = False

    def add_merger(
            self, time, mass,
            id_rem, mass_host,
            host_type, id_host,
            partner_id,
            partner_type, partner_mass,
            host_spin=np.nan,
            partner_spin=np.nan,
            semi_maj=np.nan,
            eccentricity=np.nan,
            disrupt=False
        ):

        if not disrupt:

            self.events.append(
                {
                    "event" : "merger",
                    "time" : time,
                    "mass" : mass,
                    "id_rem" : id_rem,
                    "mass_host" : mass_host,
                    "host_type" : host_type,
                    "host_spin" : host_spin,
                    "id_host" : id_host,
                    "partner_id" : partner_id,
                    "partner_type" : partner_type,
                    "partner_mass" : partner_mass,
                    "partner_spin" : partner_spin,
                    "semi_maj" : semi_maj,
                    "eccentricity" : eccentricity
                }
            )

            if id_rem != self.id_history[-1]["id"]:

                self.id_history.append(
                    {
                        "time" : time,
                        "id" : id_rem
                    }
                )	
        
        if disrupt:

            self.events.append(
                {
                    "event" : "merged",
                    "time" : time,
                    "mass" : mass,
                    "merged_into" : partner_id
                } 
            )
            self.active = False

    def add_escape(
            self, time, mass, host_id,
            binflag,
            a, e, partner_id, partner_type,
            spin=np.nan, partner_spin=np.nan):

        self.events.append(
            {
                "event" : "escape",
                "time" : time,
                "mass" : mass,
                "host_id" : host_id,
                "binflag" : binflag,
                "a" : a,
                "e" : e,
                "partner_id" : partner_id,
                "partner_type" : partner_type,
                "spin" : spin,
                "partner_spin" : partner_spin,
            }
        )

        self.active = False

    def add_accretion(
            self, time, mass,accr_time,
            snap_idx, dmdt, partner_id, 
            partner_type
        ):

        self.events.append(
            {
                "event" : "accretion",
                "time" : time,
                "mass" : mass,
                "accr_time" : accr_time,
                "snap_idx" : snap_idx,
                "dmdt" : dmdt,
                "partner_id" : partner_id,
                "partner_type" : partner_type
            }
        )

    def get_mergers(self, only_bh=False):

        times = []
        rem_ids = []
        host_masses = []
        host_ids = []
        partner_ids = []
        partner_masses = []
        host_spins = []
        partner_spins = []
        semi_majs = []
        eccentricitys = []

        for event in self.events:

            if event['event'] == "merger":

                if only_bh:

                    if (event['partner_type'] != 14) or (event['host_type'] != 14):

                        continue

                times.append(event['time'])
                rem_ids.append(event['id_rem'])
                host_masses.append(event['mass_host'])
                host_ids.append(event['id_host'])
                partner_ids.append(event['partner_id'])
                partner_masses.append(event['partner_mass'])
                host_spins.append(event.get('host_spin', np.nan))
                partner_spins.append(event.get('partner_spin', np.nan))
                semi_majs.append(event.get('semi_maj', np.nan))
                eccentricitys.append(event.get('eccentricity', np.nan))

        return {
            'times' : times,
            'rem_ids' : rem_ids,
            'host_masses' : host_masses,
            'host_ids' : host_ids,
            'partner_ids' : partner_ids,
            'partner_masses' : partner_masses,
            'host_spins' : host_spins,
            'partner_spins' : partner_spins,
            'semi_majs' : semi_majs,
            'eccentricitys' : eccentricitys,
        }
    
    def get_collisions(self, only_bh=False):

        times = []
        partner_ids = []

        for event in self.events:

            if event['event'] == "collision":

                if only_bh:

                    if event['partner_type'] != 14:

                        continue

                times.append(event['time'])

                partner_ids.append(event['partner_ids'])

        return {
            'times' : times,
            'partner_ids' : partner_ids
        }
    
    def get_accretions(self):

        times = []
        dmdts = []
        partner_ids = []

        for event in self.events:

            if event['event'] == "accretion":

                times.append(event['time'])

                dmdts.append(event['dmdt'])

                partner_ids.append(event['partner_id'])

        return {
            'times' : times,
            'dmdts' : dmdts,
            'partner_ids' : partner_ids
        }
    
    def get_escape(self):

        for event in self.events:

            if event['event'] == "escape":

                return {
                    'time' : event['time'],
                    'mass' : event['mass'],
                    'host_id' : event['host_id'],
                    'binflag' : event['binflag'],
                    'a' : event['a'],
                    'e' : event['e'],
                    'partner_id' : event['partner_id'],
                    'partner_type' : event['partner_type'],
                    'spin' : event.get('spin', np.nan),
                    'partner_spin' : event.get('partner_spin', np.nan)
                }
            
        return None
            
    def get_id_at_time(self, time_query):

        candidates = [x for x in self.id_history if x['time'] <= time_query]

        if not candidates:
            return None

        data_t = max(candidates, key=lambda x: x['time'])

        return data_t['id']

    def sort_history(self):

        self.id_history.sort(key = lambda x: x['time'])

    def sort_events(self):
        
        self.events.sort(key = lambda x: x['time'])

    def get_all_partner_ids(self):

        all_partners = set()

        for event in self.events:

            if event['event'] == "collision":

                for ith, partner in enumerate(event["partner_ids"]):
                    
                    if event["partner_types"][ith] == 14:
                    
                        all_partners.add(partner)

            if event['event'] == "disruption":

                all_partners.add(event["disrupted_by"])

            if event['event'] == "merger":

                if event['partner_type'] == 14:

                    all_partners.add(event['partner_id'])

            if event['event'] == "merged":

                all_partners.add(event['merged_into'])

        return all_partners

    def add_time_mass_graph(self, ax, linestyle='--', write_wid=True):

        text_offset = 0

        time_mass = np.array([
                         [event['time'], event['mass']]
                         for event in self.events
                         ])
        
        time_mass_non_esc = np.array([
                         [event['time'], event['mass']]
                         for event in self.events
                         if event['event'] != "escape"
                         ])
        
        time_mass_esc = np.array([
                         [event['time'], event['mass']]
                         for event in self.events
                         if event['event'] == "escape"
                         ])
        
        color = [EVENT_TO_COLOR[event['event']]
                     for event in self.events]
        
        points = time_mass.reshape(-1, 1, 2)

        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        if linestyle not in [None, '', 'None']:
            lc = LineCollection(
                segments,
                color=color[:-1],
                linewidth=1,
                alpha=0.7,
                linestyle=linestyle
            )

            ax.add_collection(lc)

        if time_mass_esc.size > 0:

            ax.scatter(
                time_mass_esc[:,0], time_mass_esc[:,1],
                facecolors='none', edgecolors=EVENT_TO_COLOR['escape'] 
            )
            color.remove(EVENT_TO_COLOR['escape'])

            if write_wid == True:

                ax.text(
                    time_mass_esc[:,0]+text_offset, time_mass_esc[:,1]+text_offset, 
                    str(self.wid),
                    color='black',
                    fontsize=8,
                    clip_on=True
                    )
                
                write_wid = False

        ax.scatter(
            time_mass_non_esc[:,0], time_mass_non_esc[:,1], 
            marker='*', s=3, 
            color=color
            )
        
        if write_wid == True:

            ax.text(
                time_mass_non_esc[-1,0]+text_offset, time_mass_non_esc[-1,1]+text_offset, 
                str(self.wid),
                color='black',
                fontsize=8,
                clip_on=True
                )
            
            write_wid = False

        return ax

def generate_worldlines(out_loc, verbose=True):

    if not verbose:

        logger.setLevel(logging.ERROR)

    try:
        bh_formations = load_bh_formations(out_loc)
    except FileNotFoundError:
        logger.warning("No bh_formations.pkl found, using tracks fallback only.")
        bh_formations = {}
    bh_formations = load_bh_formations(out_loc)
    bh_collisions = load_bh_collisions(out_loc)
    bh_mergers = load_bh_mergers(out_loc)
    bh_escapers = load_bh_escapers(out_loc)
    bh_tracks = load_bh_tracks(out_loc)
    # bh_formations = augment_formations_with_tracks(bh_formations, bh_tracks)

    bh_worldlines = {}
    wid_counter = 0
    bh_id_to_wid = {}

    for bh_id, bh_info in bh_formations.items():
       
        wid = wid_counter       

        bh_worldlines[wid] = BHWorldLine(
                                 wid = wid,
                                 birth_time = bh_info['time'],
                                 birth_id = bh_id,
                                 birth_mass = bh_info['m_bh']
                             )

        bh_id_to_wid[bh_id] = wid
        
        wid_counter += 1

    for bh_id, bh_info in bh_collisions.items():

        if not isinstance(bh_id, int):
            if isinstance(bh_id, str) and bh_id.startswith("X"):
                logger.info(
                    f"Collision remnant uses synthetic ID {bh_id} with parents {bh_info['parents']}"
                )
            else:
                logger.warning(f"Found {bh_id} with parents: {bh_info['parents']} in collisions")

        non_int_parents = [x for x in bh_info['parents'] if not isinstance(x, int)]
        if non_int_parents:
            if all(isinstance(x, str) and x.startswith("X") for x in non_int_parents):
                logger.info(
                    f"Collision parents include synthetic IDs {non_int_parents} for remnant {bh_id}"
                )
            else:
                logger.warning(f"Found {bh_id} with parents: {bh_info['parents']} in collisions")
         

        ids_collided = []
        types_collided = []
        masses_collided = []

        max_bh_mass = max(
            (parent_mass 
            for parent_mass, parent_type in zip(
                bh_info['parent_masses'],
                bh_info['parent_types']
            )
            if parent_type == 14),
            default = 0.0
        )

        for nth, id_parent in enumerate(bh_info['parents']):

            if bh_info['parent_types'][nth] != 14:
                continue

            if id_parent not in bh_id_to_wid.keys():

                if isinstance(id_parent, int):

                    logger.warning(f"{id_parent} in collisions doesn't have a wid.")

                    id_parent = 'XC' + str(id_parent)

                    logger.info(f"{id_parent} added to formation.")

                bh_info['parents'][nth] = id_parent

                wid = wid_counter

                bh_worldlines[wid] = BHWorldLine(
                                         wid = wid,
                                         birth_id = id_parent,
                                         birth_time = bh_info['time'],
                                         birth_mass = bh_info['parent_masses'][nth]
                                     )

                bh_id_to_wid[id_parent] = wid

                wid_counter += 1
            
            if bh_info['parent_masses'][nth] != max_bh_mass:
                
                wid = bh_id_to_wid[id_parent]
                    
                ids_collided = bh_info['parents'][:nth] + bh_info['parents'][nth+1:]
                types_collided = bh_info['parent_types'][:nth] + bh_info['parent_types'][nth+1:]
                masses_collided = bh_info['parent_masses'][:nth] + bh_info['parent_masses'][nth+1:]

                bh_worldlines[wid].add_collision(
                                       time = bh_info['time'],
                                       mass = bh_info['mass_rem'],
                                       new_id = bh_id,
                                       partner_ids = ids_collided,
                                       partner_types = types_collided,
                                       partner_masses = masses_collided,
                                       disrupt = True
                                    )                

                continue

            ids_collided = bh_info['parents'][:nth] + bh_info['parents'][nth+1:]
            types_collided = bh_info['parent_types'][:nth] + bh_info['parent_types'][nth+1:]
            masses_collided = bh_info['parent_masses'][:nth] + bh_info['parent_masses'][nth+1:]
            
            wid = bh_id_to_wid[id_parent]
            
            bh_worldlines[wid].add_collision(
                                   time = bh_info['time'],
                                   mass = bh_info['mass_rem'],
                                   new_id = bh_id,
                                   partner_ids = ids_collided,
                                   partner_types = types_collided,
                                   partner_masses = masses_collided,
                                   disrupt = False
                               )

            bh_id_to_wid[bh_id] = wid

    for id_rem, bh_info in bh_mergers.items():

        if id_rem not in bh_id_to_wid.keys():

            if bh_info["id_host"][0] in bh_id_to_wid.keys():
            
                wid_rem = bh_id_to_wid[bh_info["id_host"][0]]

                bh_id_to_wid[id_rem] = wid_rem

            elif 'XC' + str(id_rem) in bh_id_to_wid.keys():

                id_rem = 'XC' + str(id_rem)

                wid_rem = bh_id_to_wid[id_rem]

                for ith, event in enumerate(bh_worldlines[wid_rem].events):

                    if event['event'] == "formation" and bh_info['time'][0] < event['time']:

                        bh_worldlines[wid_rem].events[ith]['time'] = bh_info['time'][0]

                        bh_worldlines[wid_rem].events[ith]['mass'] = bh_info['mass_host'][0]

                        logger.warning(f"Changed {id_rem} birth time from collisions to mergers.")

            else:
                
                if isinstance(id_rem, int):

                    logger.warning(f"{id_rem} in mergers doesn't have wid.")
                
                    id_rem = 'XM' + str(id_rem)

                    logger.info(f"{id_rem} added to formations.")

                wid = wid_counter

                bh_worldlines[wid] = BHWorldLine(
                                         wid = wid,
                                         birth_id = id_rem,
                                         birth_time = bh_info['time'][0],
                                         birth_mass = bh_info['mass_host'][0]
                                     )

                bh_id_to_wid[id_rem] = wid

                wid_rem = wid

                wid_counter += 1
        
        else:
            wid_rem = bh_id_to_wid[id_rem]

        for nth, id_merged in enumerate(bh_info['id_merged']):

            bh_worldlines[wid_rem].add_merger(
                                       time = bh_info['time'][nth],
                                       mass = bh_info["mass_rem"][nth],
                                       id_rem = id_rem,
                                       mass_host = bh_info["mass_host"][nth],
                                       id_host = bh_info["id_host"][nth],
                                       host_type = bh_info["type_host"][nth],
                                       partner_id = id_merged,
                                       partner_type = bh_info['type_merged'][nth],
                                       partner_mass = bh_info['mass_merged'][nth],
                                       host_spin = bh_info.get('spin_host', np.full(len(bh_info['time']), np.nan))[nth],
                                       partner_spin = bh_info.get('spin_merged', np.full(len(bh_info['time']), np.nan))[nth],
                                       semi_maj = bh_info.get('semi_maj', np.full(len(bh_info['time']), np.nan))[nth],
                                       eccentricity = bh_info.get('eccentricity', np.full(len(bh_info['time']), np.nan))[nth],
                                       disrupt = False
                                   )

            if id_merged not in bh_id_to_wid.keys():

                if isinstance(id_merged, int):
                    logger.warning(f"{id_merged} (merged partner) not found; creating merger-seeded worldline.")

                wid = wid_counter

                bh_worldlines[wid] = BHWorldLine(
                                         wid = wid,
                                         birth_id = id_merged,
                                         birth_time = bh_info['time'][nth],
                                         birth_mass = bh_info['mass_merged'][nth]
                                     )

                bh_id_to_wid[id_merged] = wid

                wid_counter += 1

            wid_merged = bh_id_to_wid[id_merged]
                 
            bh_worldlines[wid_merged].add_merger(
                                          time = bh_info['time'][nth],
                                          mass = bh_info['mass_rem'][nth],
                                          id_rem = id_rem,
                                          mass_host = bh_info['mass_merged'][nth],
                                          id_host = id_merged,
                                          host_type = bh_info['type_merged'][nth],
                                          partner_id = bh_info["id_host"][nth],
                                          partner_type = bh_info['type_host'][nth],
                                          partner_mass = bh_info['mass_host'][nth],
                                          host_spin = bh_info.get('spin_merged', np.full(len(bh_info['time']), np.nan))[nth],
                                          partner_spin = bh_info.get('spin_host', np.full(len(bh_info['time']), np.nan))[nth],
                                          disrupt = True
                                      )       
 
    escape_escapers = 0
    for id_esc, bh_info in bh_escapers.items():
        
        if id_esc not in bh_id_to_wid.keys():
            
            if 'XC' + str(id_esc) in bh_id_to_wid.keys():

                id_esc = 'XC' + str(id_esc)

            else:
                
                if isinstance(id_esc, int):

                    logger.info(
                        f"{id_esc} in escapers has no prior worldline; creating escape-seeded worldline"
                    )
              
                    id_esc = 'XE' + str(id_esc)

                    logger.info(f"{id_esc} added to formations.")

                wid = wid_counter

                bh_worldlines[wid] = BHWorldLine(
                                         wid = wid,
                                         birth_id = id_esc,
                                         birth_time = bh_info['time'],
                                         birth_mass = bh_info['mass']
                                     )

                bh_id_to_wid[id_esc] = wid

                wid_counter += 1
                escape_escapers += 1

        wid_esc = bh_id_to_wid[id_esc]

        bh_worldlines[wid_esc].add_escape(
                                   time = bh_info['time'],
                                   host_id = id_esc,
                                   mass = bh_info['mass'],
                                   binflag = bh_info['binflag'],
                                   a = bh_info['a'],
                                   e = bh_info['e'],
                                   partner_id = bh_info['companion_id'],
                                   partner_type = bh_info['companion_type'],
                                   spin = bh_info.get('spin', np.nan),
                                   partner_spin = bh_info.get('companion_spin', np.nan)
                               )
        
    logger.warning(
        f"{escape_escapers} in escapers has no prior worldline; created escape-seeded worldlines"
            )

    for bh_id, bh_info in bh_tracks.items():

        if bh_id not in bh_id_to_wid.keys():

            if 'XC' + str(bh_id) in bh_id_to_wid.keys():

                bh_id = 'XC' + str(bh_id)

                wid = bh_id_to_wid[bh_id]

                for ith, event in enumerate(bh_worldlines[wid].events):

                    if event['event'] == "formation" and bh_info['time'][0] < event['time']:

                        bh_worldlines[wid].events[ith]['time'] = bh_info['time'][0]

                        bh_worldlines[wid].events[ith]['mass'] = bh_info['mass'][0]

                        logger.warning(f"Changed {bh_id} birth time from collisions to tracks.")             

            elif 'XE' + str(bh_id) in bh_id_to_wid.keys():

                bh_id = 'XE' + str(bh_id)

                wid = bh_id_to_wid[bh_id]

                for ith, event in enumerate(bh_worldlines[wid].events):

                    if event['event'] == "formation" and bh_info['time'][0] < event['time']:

                        bh_worldlines[wid].events[ith]['time'] = bh_info['time'][0]  

                        bh_worldlines[wid].events[ith]['mass'] = bh_info['mass'][0] 

                        logger.warning(f"Changed {bh_id} birth time from escapers to tracks.")

            else:
                
                if isinstance(bh_id, int):

                    logger.warning(f"{bh_id} in tracks doesn't have wid")

                    bh_id = 'XT' + str(bh_id)

                    logger.info(f"{bh_id} added to formation.")

                wid = wid_counter

                bh_worldlines[wid] = BHWorldLine(
                                         wid = wid,
                                         birth_id = bh_id,
                                         birth_time = bh_info['time'][0],
                                         birth_mass = bh_info['mass'][0]
                                     )

                bh_id_to_wid[bh_id] = wid
 
                wid_counter += 1

        wid = bh_id_to_wid[bh_id]

        for ith, dmdt in enumerate(bh_info['dmdt']):
                
            if dmdt > 0:
                    
                try:
                    start_time = bh_info['time'][ith - 1]
                except IndexError:
                    start_time = bh_info['time'][ith]

                try:
                    end_time = bh_info['time'][ith + 1]
                except IndexError:
                    end_time = bh_info['time'][ith]

                bh_worldlines[wid].add_accretion(
                                       time = bh_info['time'][ith],
                                       mass = bh_info['mass'][ith],
                                       accr_time = end_time - start_time,
                                       snap_idx = bh_info['snap_idx'][ith],
                                       dmdt = bh_info['dmdt'][ith],
                                       partner_id = bh_info['companion_id'][ith],
                                       partner_type = bh_info['companion_type'][ith]
                                   )

    for wid in bh_worldlines.keys():

        bh_worldlines[wid].sort_history()
        bh_worldlines[wid].sort_events()

    with open(os.path.join(out_loc, "bh_worldlines.pkl"), 'wb') as f:
        pickle.dump(bh_worldlines, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open(os.path.join(out_loc, "bh_id_to_wid.pkl"), 'wb') as f:
        pickle.dump(bh_id_to_wid, f, protocol=pickle.HIGHEST_PROTOCOL)

    return bh_worldlines, bh_id_to_wid

def load_bh_worldlines(out_loc):

    bh_wl_file = os.path.join(out_loc, "bh_worldlines.pkl")
    bh_wid_file = os.path.join(out_loc, "bh_id_to_wid.pkl")

    try:

        with open(bh_wl_file, 'rb') as f:
            bh_worldlines = pickle.load(f)

        with open(bh_wid_file, 'rb') as f:
            bh_id_to_wid = pickle.load(f)

    except FileNotFoundError:

        logger.error("File \"bh_worldlines.pkl\" not found. Try generate_bh_worldlines()")
        raise

    return bh_worldlines, bh_id_to_wid

def invert_bh_id_map(bh_id_to_wid, sort_ids=False):
    wid_to_bh_ids = {}
    for bh_id, wid in bh_id_to_wid.items():
        wid_to_bh_ids.setdefault(wid, []).append(bh_id)
    if sort_ids:
        for wid in wid_to_bh_ids:
            wid_to_bh_ids[wid].sort(key=lambda x: (isinstance(x, str), str(x)))
    return wid_to_bh_ids

def get_all_related_wid(current_wid, bh_worldlines, bh_id_to_wid, visited=None):

    if visited is None:
        visited = set()

    if current_wid in visited:
        return visited

    visited.add(current_wid)

    partner_ids = bh_worldlines[current_wid].get_all_partner_ids()

    for partner_id in partner_ids:

        partner_wid = bh_id_to_wid[partner_id]

        get_all_related_wid(
            partner_wid,
            bh_worldlines,
            bh_id_to_wid,
            visited
        )

    return visited

def get_wid_from_id(bh_id, bh_id_to_wid):

    if bh_id in bh_id_to_wid.keys():
        return bh_id_to_wid[bh_id]
    if 'XE' + str(bh_id) in bh_id_to_wid.keys():
        return bh_id_to_wid['XE' + str(bh_id)]

def get_all_mergers(out_loc, bh_worldlines, bh_id_to_wid):

    wid_of_mergers = {}

    for wid, bhwl in bh_worldlines.items():

        n_mergers = 0

        merger_data = bhwl.get_mergers(only_bh=True)

        times = merger_data['times']
        rem_ids = merger_data['rem_ids']
        host_masses = merger_data['host_masses']
        host_ids = merger_data['host_ids']
        partner_ids = merger_data['partner_ids']
        partner_masses = merger_data['partner_masses']
        host_spins = merger_data['host_spins']
        partner_spins = merger_data['partner_spins']
        semi_majs = merger_data['semi_majs']
        eccentricitys = merger_data['eccentricitys']

        esc_data = bhwl.get_escape()

        if esc_data != None:

            esc_time = esc_data['time']

            if esc_data['partner_type'] == 14:
                
                mass = esc_data['mass']
                a = esc_data['a']
                e = esc_data['e']
                host_id = esc_data['host_id']
                partner_id = esc_data['partner_id']
                partner_wid = get_wid_from_id(partner_id, bh_id_to_wid)

                partner_escape = bh_worldlines[partner_wid].get_escape()
                partner_mass = partner_escape['mass']

                if mass > partner_mass:

                    host_spin = esc_data.get('spin', np.nan)
                    partner_spin = partner_escape.get('spin', np.nan)

                    # Query the line with mass id and partner_id to
                    # get orbit properties.

                    merg_time = esc_time + calc_peters_eqn(a, e, mass, partner_mass)

                    if merg_time < myr_to_code_unit(AGE_OF_UNIV):

                        times.append(merg_time)
                        rem_ids.append(host_id)
                        host_masses.append(mass)
                        host_ids.append(host_id)
                        partner_ids.append(partner_id)
                        partner_masses.append(partner_mass)
                        host_spins.append(host_spin)
                        partner_spins.append(partner_spin)
                        semi_majs.append(a)
                        eccentricitys.append(e)

        else:

            esc_time = None

        n_mergers = len(times)

        if n_mergers == 0:

            continue

        partner_wids = list({
            get_wid_from_id(partner_id, bh_id_to_wid)
            for partner_id in partner_ids
        })

        accr_data = bhwl.get_accretions()
        accr_times = accr_data['times']

        first_accr_time = (accr_times[0] 
                           if len(accr_times) != 0 
                           else 0
                           )

        coll_data = bhwl.get_collisions()
        coll_times = coll_data['times']

        first_coll_time = (coll_times[0]
                           if len(coll_times) != 0
                           else 0
                           )

        wid_of_mergers[wid] = {
            'num_mergers' : n_mergers,
            'times' : times,
            'host_masses' : host_masses,
            'partner_wid' : partner_wids,
            'partner_masses' : partner_masses,
            'host_spins' : host_spins,
            'partner_spins' : partner_spins,
            'first_accretion_time' : first_accr_time,
            'first_collision_time' : first_coll_time,
            'esc_time' : esc_time,
            'semi_majs' : semi_majs,
            'eccentricitys' : eccentricitys
        }

    for wid in wid_of_mergers:

        wid_of_mergers[wid]['merger_types'] = []
            
        for time, partner_wid in zip(
            wid_of_mergers[wid]['times'],
            wid_of_mergers[wid]['partner_wid']
            ):

            merger_str = set()

            if wid_of_mergers[wid]['esc_time'] != None:
                if time > wid_of_mergers[wid]['esc_time']:
                    merger_str.add('E')

            if time > wid_of_mergers[wid]['times'][0]:

                merger_str.add('HG')

            elif partner_wid in wid_of_mergers:

                if (wid_of_mergers[partner_wid]['times'][-1]
                    < time):

                    merger_str.add('HG')

            else:

                merger_str.add('1G')

            if (
                (wid_of_mergers[wid]['first_accretion_time']
                    > 0)
                and
                (wid_of_mergers[wid]['first_accretion_time']
                    < time)
            ):
                    
                merger_str.add('A')

            elif any(
                event['event'] == 'accretion'
                for event in bh_worldlines[partner_wid].events
                ):

                merger_str.add('A')

            if (
                (wid_of_mergers[wid]['first_collision_time']
                    > 0)
                and
                (wid_of_mergers[wid]['first_collision_time']
                    < time)
            ):
                    
                merger_str.add('C')

            elif any(
                event['event'] == 'collision'
                for event in bh_worldlines[partner_wid].events
                ):

                merger_str.add('C')

            wid_of_mergers[wid]['merger_types'].append(
                "".join(x for x in sorted(merger_str))
                )

        del wid_of_mergers[wid]['first_accretion_time']
        del wid_of_mergers[wid]['first_collision_time']
        del wid_of_mergers[wid]['esc_time']

    with open(os.path.join(out_loc, "all_mergers.pkl"), 'wb') as f:
        pickle.dump(wid_of_mergers, f, protocol=pickle.HIGHEST_PROTOCOL)

    return wid_of_mergers

def load_all_mergers(out_loc):

    all_mergers_file = os.path.join(out_loc, "all_mergers.pkl")

    try:

        with open(all_mergers_file, 'rb') as f:
            all_mergers = pickle.load(f)

    except FileNotFoundError:

        logger.error("File \"all_mergers.pkl\" not found. Try get_all_mergers()")
        raise

    return all_mergers

if __name__ == "__main__":

    print("This is a module named 'utils'")
