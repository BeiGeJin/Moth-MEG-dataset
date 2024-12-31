import mne
import os, glob
import yaml
import numpy as np
import sys

os.chdir("/home/jerryjin/moth-meg-dataset/")  # change to your own path

# Load parameters
with open("preproc/config.yaml", "r") as f:
    config = yaml.safe_load(f)
SUBJECT = config["SUBJECT"]
SESSIONS = config["SESSIONS"]
ER_NWINS = config["after_sss"]["ER_NWINS"]
ER_WINLEN = config["after_sss"]["ER_WINLEN"]
ER_NGRAD = config["after_sss"]["ER_NGRAD"]
ER_NMAG = config["after_sss"]["ER_NMAG"]
FILT_LFREQ = config["after_sss"]["FILT_LFREQ"] if config["after_sss"]["FILT_LFREQ"] != "None" else None
FILT_HFREQ = config["after_sss"]["FILT_HFREQ"] if config["after_sss"]["FILT_HFREQ"] != "None" else None
NOTCH_FREQS = config["after_sss"]["NOTCH_FREQS"]
ECG_METHOD = config["after_sss"]["ECG_METHOD"] if config["after_sss"]["ECG_METHOD"] != "None" else None
EOG_METHOD = config["after_sss"]["EOG_METHOD"] if config["after_sss"]["EOG_METHOD"] != "None" else None

# file slug
slug = ""
if ER_NWINS > 0:
    slug += f"_er-{ER_NWINS}-{ER_WINLEN}-{ER_NGRAD}-{ER_NMAG}"
if FILT_LFREQ is not None and FILT_HFREQ is not None:
    slug += f"_band-{FILT_LFREQ}-{FILT_HFREQ}"
elif FILT_LFREQ is not None:
    slug += f"_highpass-{FILT_LFREQ}"
elif FILT_HFREQ is not None:
    slug += f"_lowpass-{FILT_HFREQ}"
if NOTCH_FREQS is not None:
    slug += f"_notch-{"-".join(map(str, NOTCH_FREQS))}"
if ECG_METHOD is not None:
    slug += f"_ecg-{ECG_METHOD}"
if EOG_METHOD is not None:
    slug += f"_eog-{EOG_METHOD}"
dir_name = "sss" + slug

# define bandpass filter parameters
l_trans_bandwidth = 0.5 if FILT_LFREQ is not None and FILT_LFREQ > 0.5 else "auto"
h_trans_bandwidth = 0.5 if FILT_HFREQ is not None and FILT_HFREQ > 0.5 else "auto"
bandpass_params = dict(
    l_freq=FILT_LFREQ,
    h_freq=FILT_HFREQ,
    filter_length="10s",
    l_trans_bandwidth=l_trans_bandwidth,
    h_trans_bandwidth=h_trans_bandwidth,
    phase="zero-double",
    fir_window="hann",
    fir_design="firwin2",
)

# define notch filter parameters
notch_params = dict(
    freqs=NOTCH_FREQS,
    filter_length="10s",
    phase="zero-double",
    fir_window="hann",
    fir_design="firwin2",
)

with open(f"/project_data/volume0/jerryjin/moth_meg/logs/{SUBJECT}{slug}.txt", "w") as f:  # redirect stdout to file
    
    original_stdout = sys.stdout
    sys.stdout = f
    
    # run after-sss processing
    for SESSION in SESSIONS:
        
        print(f"Processing {SESSION}...")
        
        # Define paths
        LOC_ROOT = f"/project_data/volume0/jerryjin/moth_meg/{SESSION}/"
        LOC_RAW = LOC_ROOT + f"raw/{SUBJECT}/"
        LOC_SSS = LOC_ROOT + f"sss/{SUBJECT}/"
        LOC_SAVE = LOC_ROOT + f"{dir_name}/{SUBJECT}/"
        
        # make save directory if it doesn't exist
        if not os.path.exists(LOC_SAVE):
            os.makedirs(LOC_SAVE)
        
        if ER_NWINS > 0:
            # load empty room
            print(">>Now calculating empty room projection...")
            er_f = glob.glob(LOC_SSS + f"*{SUBJECT}_{SESSION}_EmptyRoom*raw.fif")[0]
            er_sss = mne.io.read_raw_fif(er_f, preload=True)
            # determine intervals
            n_windows = ER_NWINS
            window_len = ER_WINLEN
            er_len = er_sss[0][0].shape[1]
            center_dist = np.floor(er_len / (1.0 + n_windows))
            curr_center = center_dist
            starts = []
            stops = []
            for i in range(n_windows):
                half_win_len = np.floor(er_sss.info["sfreq"] * window_len * 0.5)
                (win_start, win_stop) = (curr_center - half_win_len, curr_center + half_win_len)
                win_start = er_sss.times[int(win_start)]
                win_stop = er_sss.times[int(win_stop)]
                curr_center += center_dist
                starts.append(win_start)
                stops.append(win_stop)
            # calculate emptyroom projs
            er_projs = []
            for start, stop in zip(starts, stops):
                er_projs.extend(mne.compute_proj_raw(er_sss, start=start, stop=stop, n_grad=ER_NGRAD, n_mag=ER_NMAG, verbose=True))
        
        # load first recording
        if ECG_METHOD == "first" or EOG_METHOD == "first":
            first_f = glob.glob(LOC_SSS + f"*{SUBJECT}_{SESSION}_01*raw.fif")[0]
            first_sss = mne.io.read_raw_fif(first_f)

        # get BLOCKS
        BLOCKS = config[SESSION]["BLOCKS"]
        
        for BLOCK in BLOCKS:
            
            # load raw data
            sss_f = glob.glob(LOC_SSS + f"*{SUBJECT}_{SESSION}_{BLOCK}*raw.fif")[0]
            raw_sss = mne.io.read_raw_fif(sss_f, preload=True)

            # empty room correction
            if ER_NWINS > 0:
                print(">>Now applying empty room correction...")
                raw_sss.add_proj(er_projs)
                raw_sss.apply_proj()

            # bandpass filtering
            print(">>Now applying bandpass filtering...")
            raw_sss.filter(**bandpass_params)

            # notch filtering
            print(">>Now applying notch filtering...")
            raw_sss.notch_filter(**notch_params)

            # heartbeat removal
            if ECG_METHOD is not None:

                print(">>Now removing heartbeat artifacts...")

                if ECG_METHOD == "first":
                    # do all the steps for first recording
                    first_sss_sub = first_sss.copy().load_data()
                    first_sss_sub.add_proj(er_projs)
                    first_sss_sub.apply_proj()
                    first_sss_sub.filter(**bandpass_params, verbose=False)
                    first_sss_sub.notch_filter(**notch_params, verbose=False)
                    # compute ecg projs
                    ecg_projs, _ = mne.preprocessing.compute_proj_ecg(first_sss_sub, average=False)
                    # save memory
                    del first_sss_sub
                    # apply ecg projs
                    raw_sss.add_proj(ecg_projs)
                    raw_sss.apply_proj()

                elif ECG_METHOD == "self":
                    # compute ecg projs
                    ecg_projs, _ = mne.preprocessing.compute_proj_ecg(raw_sss, average=False)
                    # apply ecg projs
                    raw_sss.add_proj(ecg_projs)
                    raw_sss.apply_proj()

                elif ECG_METHOD == "ica":
                    # load raw data and filter (high-pass filter necessary for ica)
                    raw_f = glob.glob(LOC_RAW + f"*{SUBJECT}_{SESSION}_{BLOCK}*raw.fif")[0]
                    raw_ica = mne.io.read_raw_fif(raw_f, preload=True, verbose=False).pick(["meg", "ecg"]).filter(**bandpass_params, verbose=False)
                    # fit ICA
                    ica_ecg = mne.preprocessing.ICA(n_components=30, random_state=42)
                    ica_ecg.fit(raw_ica)
                    # find ecg components
                    ecg_indices, ecg_scores = ica_ecg.find_bads_ecg(raw_ica, l_freq=5, h_freq=35)
                    ica_ecg.exclude = ecg_indices
                    print(f"Detect {len(ecg_indices)} ECG components.")
                    # check
                    # assert len(ecg_indices) < 8
                    # apply ICA
                    ica_ecg.apply(raw_sss)
                    # save memory
                    del raw_ica

                else:
                    raise ValueError("ECG_METHOD not recognized.")

            # eyeblink removal
            if EOG_METHOD is not None:

                print(">>Now removing eyeblink artifacts...")

                if EOG_METHOD == "first":
                    # do all the steps for first recording
                    first_sss_sub = first_sss.copy().load_data()
                    first_sss_sub.add_proj(er_projs)
                    first_sss_sub.apply_proj()
                    first_sss_sub.filter(**bandpass_params, verbose=False)
                    first_sss_sub.notch_filter(**notch_params, verbose=False)
                    ecg_projs, _ = mne.preprocessing.compute_proj_ecg(first_sss_sub, average=False)
                    first_sss_sub.add_proj(ecg_projs)
                    first_sss_sub.apply_proj()
                    # compute eog projs
                    eog_projs, _ = mne.preprocessing.compute_proj_eog(first_sss_sub, average=False)
                    # save memory
                    del first_sss_sub
                    # apply eog projs
                    raw_sss.add_proj(eog_projs)
                    raw_sss.apply_proj()

                elif EOG_METHOD == "self":
                    # compute eog projs
                    eog_projs, _ = mne.preprocessing.compute_proj_eog(raw_sss, average=False)
                    # apply eog projs
                    raw_sss.add_proj(eog_projs)
                    raw_sss.apply_proj()

                elif EOG_METHOD == "ica":
                    # load filtered data with only MEG channels
                    raw_sss_ica = raw_sss.copy().pick(["meg", "eog"])
                    # fit ICA
                    ica_eog = mne.preprocessing.ICA(n_components=30, random_state=42)
                    ica_eog.fit(raw_sss_ica)
                    # find eog components
                    # ref_channel = "EOG002" if ("EOG002" in raw_sss_ica.ch_names) and ("EOG002" not in raw_sss_ica.info["bads"]) else "MEG1213"
                    good_eog_chs = [ch for ch in raw_sss_ica.info["ch_names"] if "EOG" in ch and ch not in raw_sss_ica.info["bads"]]
                    ref_channel = good_eog_chs[0] if len(good_eog_chs) > 0 else "MEG1213"
                    eog_indices, eog_scores = ica_eog.find_bads_eog(raw_sss_ica, ch_name=ref_channel)
                    ica_eog.exclude = eog_indices
                    print(f"Detect {len(eog_indices)} EOG components.")
                    # check
                    # assert len(eog_indices) < 5
                    # apply ICA
                    ica_eog.apply(raw_sss)
                    # save memory
                    del raw_sss_ica

                else:
                    raise ValueError("EOG_METHOD not recognized.")

            # save
            save_f = os.path.basename(sss_f).replace("_raw.fif", f"{slug}_raw.fif")
            raw_sss.save(LOC_SAVE + save_f)
            
    sys.stdout = original_stdout