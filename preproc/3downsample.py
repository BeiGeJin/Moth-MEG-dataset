import mne
import os, glob
import yaml

os.chdir("/home/jerryjin/moth-meg-dataset/")  # change to your own path

# Load parameters
with open("preproc/config.yaml", "r") as f:
    config = yaml.safe_load(f)
SUBJECT = config["SUBJECT"]
SESSIONS = config["SESSIONS"]
DOWN_FS = config["downsample"]["DOWN_FS"]

# version slug
ER_NWINS = config["after_sss"]["ER_NWINS"]
ER_WINLEN = config["after_sss"]["ER_WINLEN"]
ER_NGRAD = config["after_sss"]["ER_NGRAD"]
ER_NMAG = config["after_sss"]["ER_NMAG"]
FILT_LFREQ = config["after_sss"]["FILT_LFREQ"] if config["after_sss"]["FILT_LFREQ"] != "None" else None
FILT_HFREQ = config["after_sss"]["FILT_HFREQ"] if config["after_sss"]["FILT_HFREQ"] != "None" else None
NOTCH_FREQS = config["after_sss"]["NOTCH_FREQS"]
ECG_METHOD = config["after_sss"]["ECG_METHOD"]
EOG_METHOD = config["after_sss"]["EOG_METHOD"]
VERSION = "sss"
VERSION += f"_er-{ER_NWINS}-{ER_WINLEN}-{ER_NGRAD}-{ER_NMAG}"
if FILT_LFREQ is not None and FILT_HFREQ is not None:
    VERSION += f"_band-{FILT_LFREQ}-{FILT_HFREQ}"
elif FILT_LFREQ is not None:
    VERSION += f"_highpass-{FILT_LFREQ}"
elif FILT_HFREQ is not None:
    VERSION += f"_lowpass-{FILT_HFREQ}"
if NOTCH_FREQS is not None:
    VERSION += f"_notch-{"-".join(map(str, NOTCH_FREQS))}"
if ECG_METHOD is not None:
    VERSION += f"_ecg-{ECG_METHOD}"
if EOG_METHOD is not None:
    VERSION += f"_eog-{EOG_METHOD}"

# run downsample
for SESSION in SESSIONS:
    
    print(f"Processing {SESSION}...")
    
    # Define paths
    LOC_RAW = f"/project_data/volume0/jerryjin/moth_meg/{SESSION}/{VERSION}/{SUBJECT}/"
    LOC_SAVE = LOC_RAW + f"downsampled-{DOWN_FS}/"
    
    # make save directory if it doesn't exist
    if not os.path.exists(LOC_SAVE):
        os.makedirs(LOC_SAVE)
    
    # get BLOCKS
    BLOCKS = config[SESSION]["BLOCKS"]
    
    for BLOCK in BLOCKS:
        
        # load raw data
        f = glob.glob(LOC_RAW + f"*{SUBJECT}_{SESSION}_{BLOCK}*raw.fif")[0]
        raw = mne.io.read_raw_fif(f, preload=True)  # preload helps accelerate

        # downsample
        raw.resample(DOWN_FS)

        # save downsampled data
        save_f = os.path.basename(f).replace("_raw.fif", f"_ds{DOWN_FS}_raw.fif")
        raw.save(LOC_SAVE + save_f)