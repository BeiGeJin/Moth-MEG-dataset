# Modified from Saaketh

import glob
import numpy as np
import mne
import matplotlib.pyplot as plt
from src.freq_analysis import get_envelope_phase
import librosa
from tqdm import tqdm


def determine_lag(story: str, acoustic_data: np.array, audio_ch_data: np.array, fs: int, start=0, duration=10, plot=False):
    """
    Determine the lag between the original audio file and the audio file played in the scanner and plot the results

    Parameters:
    -----------
    story: str
        story name
    acoustic_data: np.array
        original audio waveform
    fs: int
        sampling rate
    audio_ch_data: np.array
        meg audio channel for a story
    start: int
        start time of computing lag
    duration: int
        duration of signal for computing lag
    plot: bool
        whether to plot the results

    Returns
    -------
    xcorr_max: int
        lag between original audio file and audio file played in the scanner
    """

    # start and duration of the signal to be analyzed (in samples)
    if start < 0:
        start_samp_acoustic = len(acoustic_data) + start * fs
        start_samp_meg = len(audio_ch_data) + start * fs
    else:
        start_samp_acoustic = start * fs
        start_samp_meg = start * fs

    duration_samp = duration * fs

    # check whether start_samp is out of range of podcast length
    if start_samp_acoustic > len(acoustic_data) or start_samp_meg > len(audio_ch_data):
        print(f"{story} start_samp bounds are out of range of podcast length")
        return None

    # check if end_samp is out of range of podcast length
    if start_samp_acoustic + duration_samp > len(acoustic_data):
        print(f"{story} start_samp_audio + duration_samp is out of range of podcast length, shortening to the end of the acoustic audio")
        duration_samp_audio = len(acoustic_data) - start_samp_acoustic
    else:
        duration_samp_audio = duration_samp

    if start_samp_meg + duration_samp > len(audio_ch_data):
        print(f"{story} start_samp_meg + duration_samp is out of range of meg audio length, shortening to the end of the meg audio")
        duration_samp_meg = len(audio_ch_data) - start_samp_meg
    else:
        duration_samp_meg = duration_samp

    # calculate t separately, because the signal might be different length than the original audio
    t_slice_acoustic = np.arange(start_samp_acoustic, start_samp_acoustic + duration_samp_audio)
    t_slice_meg = np.arange(start_samp_meg, start_samp_meg + duration_samp_meg)

    # audio file as played in the scanner
    acoustic_sliced = acoustic_data[t_slice_acoustic]
    meg_sliced = audio_ch_data[t_slice_meg]
    meg_samplenum = len(meg_sliced)
    acoustic_samplenum = len(acoustic_sliced)

    # get and plot envelope for original audio
    _, _, _, envelope_acoustic, _ = get_envelope_phase(acoustic_sliced, fs, duration_samp_audio)
    if plot:
        fig, (ax0, ax1, ax2) = plt.subplots(nrows=3)
        ax0.plot(t_slice_acoustic / fs, acoustic_data[t_slice_acoustic], label="signal")
        ax0.plot(t_slice_acoustic / fs, envelope_acoustic, label="envelope")
        ax0.set_xlabel("time in seconds")
        # ax0.legend()
        ax0.set_title("original audio file")

    # get and plot envelope for audio file as played in the scanner
    _, _, _, envelope_meg, _ = get_envelope_phase(meg_sliced, fs, duration_samp_meg)
    if plot:
        ax1.plot(t_slice_meg / fs, meg_sliced, label="signal")
        ax1.plot(t_slice_meg / fs, envelope_meg, label="envelope")
        ax1.set_xlabel("time in seconds")
        # ax1.legend()
        ax1.set_title("audio file as played in the scanner")

    # if aligned, the envelopes should correlate the most at the center (i.e., 0), otherwise there will be non-zero delay
    xcorr = np.correlate(envelope_meg, envelope_acoustic, "full")
    # xcorr = np.correlate(meg_sliced, acoustic_sliced, "full")  # meg stay, acoustic move
    xcorr_max = np.argmax(xcorr) - acoustic_samplenum + 1

    # arguments of correlate in the order above so positive values correspond to delays
    # in playing the file at the scanner
    if plot:
        ax2.plot(np.arange(-(len(xcorr) // 2), (len(xcorr) + 1) // 2), xcorr)
        ax2.plot([0, 0], [0, np.max(xcorr)], ls='--')
        ax2.plot([xcorr_max, xcorr_max], [0, np.max(xcorr)])
        ax2.set_title("{}ms difference".format(xcorr_max))

        fig.suptitle(story)
        fig.tight_layout()
        plt.show()

    return xcorr_max


def compute_meg_acoustic_diff(
    story: str,
    acoustic_data: np.array,
    audio_ch_data: np.array,
    fs: int,
    stim_ch_data: np.array,
    STIM_ON_VAL: dict,
    initial_duration=10,
    final_duration=20,
):
    """
    Compute the discrepancy between the MEG audio and the original acoustic audio file

    Parameters:
    -----------
    acoustic_data: np.array
        original audio waveform
    audio_ch_data: np.array
        MEG audio waveform
    fs: int
        sampling rate
    stim_ch_data: np.array
        dictionary with story names as keys and the stimulus channel as values
    STIM_ON_VAL: dict
        dictionary with story names as keys and the maximum value of the stimulus channel as values

    Returns
    -------
    initial_lag: int
        dictionary with story names as keys and the initial lag as values
    rate: float
        dictionary with story names as keys and the scale factor as values
    """

    audio_ch_onstory = audio_ch_data[stim_ch_data == STIM_ON_VAL[story]]
    initial_lag = determine_lag(story=story, acoustic_data=acoustic_data, fs=fs, audio_ch_data=audio_ch_onstory, start=0, duration=initial_duration, plot=False)
    print(f"Computing initial lag for {story}: {initial_lag} ms")

    final_story_lag = determine_lag(
        story=story,
        acoustic_data=acoustic_data,
        fs=fs,
        audio_ch_data=audio_ch_onstory,
        start=int(len(acoustic_data) / fs - final_duration),
        duration=final_duration,
        plot=False,
    )
    print(f"Computing final story lag for {story}: {final_story_lag} ms")

    # compute new rate
    rate = 1 + (final_story_lag - initial_lag) / len(acoustic_data)
    print(f"Rate for {story}: {rate}\n")

    return initial_lag, rate


def stretch_shift_acoustic(acoustic_data: np.array, fs: int, initial_lag: int, rate: float, meg_sr=1000):
    """
    Stretch and shift the original acoustic audio file

    Parameters:
    -----------
    acoustic_data: np.array
        original audio file
    fs: int
        sampling rate
    initial_lag: int
        if None, no shift is performed
    rate: float
        if None, no stretch is performed

    Returns
    -------
    corrected_acoustic_data: np.array
        stretched and shifted audio file
    """

    scale_factor = fs / meg_sr

    # stretch signal
    if rate is not None:
        corrected_acoustic_data = librosa.resample(acoustic_data, orig_sr=fs, target_sr=fs * rate)

    # shift via zero padding initial lags onto stretched
    if initial_lag is not None:
        if initial_lag > 0:
            corrected_acoustic_data = np.concatenate([np.zeros(int(scale_factor * initial_lag)), corrected_acoustic_data])
        else:
            corrected_acoustic_data = corrected_acoustic_data[-int(scale_factor * initial_lag):]

    return corrected_acoustic_data


def align_meg_acoustic_end(acoustic_data: np.array, meg_acoustic_diff: int):
    """
    Trim/pad the acoustic audio to match the length of the MEG audio

    Parameters:
    -----------
    acoustic_data: np.array
        original audio file
    meg_acoustic_diff_end: int
        the difference between the number of MEG audio and the acoustic audio samples

    Returns
    -------
    corrected_acoustic_data: np.array
        trimmed/padded audio file
    """
    correction = meg_acoustic_diff

    # match to length of meg signal
    if correction < 0:
        corrected_acoustic_data = acoustic_data[: int(correction)]
    else:
        corrected_acoustic_data = np.concatenate([acoustic_data, np.zeros(int(correction))])

    return corrected_acoustic_data
