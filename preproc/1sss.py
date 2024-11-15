import mne
import os, glob
import yaml

os.chdir("/home/jerryjin/moth-meg-dataset/")  # change to your own path

# Load parameters
with open("preproc/config.yaml", "r") as f:
    config = yaml.safe_load(f)
SUBJECT = config["SUBJECT"]
SESSIONS = config["SESSIONS"]
# BADS_MANUAL = config["sss"]["BADS_MANUAL"]

# calibration and crosstalk files for maxfilter
calibration_file = "support_data/sss_cal.dat"
crosstalk_file = "support_data/ct_sparse.fif"

# run maxfilter
for SESSION in SESSIONS:
    
    print(f"Processing {SESSION}...")
    
    # Define paths
    LOC_RAW = f"/project_data/volume0/jerryjin/moth_meg/{SESSION}/raw/{SUBJECT}/"
    LOC_SAVE = f"/project_data/volume0/jerryjin/moth_meg/{SESSION}/sss/{SUBJECT}/"
    
    # make save directory if it doesn't exist
    if not os.path.exists(LOC_SAVE):
        os.makedirs(LOC_SAVE)
    
    # find the first raw file
    first_f = glob.glob(LOC_RAW + f"*{SUBJECT}_{SESSION}_01*raw.fif")[0]
    print(f"First file: {first_f}")
    
    # get BLOCKS
    BLOCKS = config[SESSION]["BLOCKS"]
    BLOCKS += ["EmptyRoom"]
    
    for BLOCK in BLOCKS:
        
        # load raw data
        raw_f = glob.glob(LOC_RAW + f"*{SUBJECT}_{SESSION}_{BLOCK}*raw.fif")[0]
        raw = mne.io.read_raw_fif(raw_f, preload=True)

        # add bad channels and delete existing proj
        # raw.info["bads"] += BADS_MANUAL
        raw.del_proj()

        # spatial-temporal SSS
        sss = mne.preprocessing.maxwell_filter(
            raw,
            calibration=calibration_file,
            cross_talk=crosstalk_file,
            st_duration=4,
            st_correlation=0.98,
            origin=(0.0, 0.0, 0.04),
            coord_frame="head",
            destination=first_f,
            verbose=True,
        )

        # save
        save_f = os.path.basename(raw_f).replace("_raw.fif", "_sss_raw.fif")
        sss.save(LOC_SAVE + save_f)