# Created by Saaketh and shared to me on 24/08/22

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import ShortTimeFFT as STFT
from scipy.stats import zscore
from scipy.fft import fft, fftfreq
from scipy.signal import hilbert, butter, filtfilt
import librosa
from typing import Tuple, List, Optional, Dict, Callable
from jaxtyping import Float, Int


def get_freq_indices(freq_range: tuple, freqs: np.array):
    """
    Get the indices of the lower and upper ends of a frequency range.

    Parameters:
    - freq_range (tuple): A tuple containing the lower and upper ends of the frequency range.
    - freqs (np.array): An array of frequencies.

    Returns:
    - min_freq_idx (int): The index of the lower end of the frequency range.
    - max_freq_idx (int): The index of the upper end of the frequency range.
    """

    # get the indices of the lower end of frequency range
    if freq_range[0] is not None:
        min_freq_idx = np.where(freqs <= freq_range[0])[0][-1]
    else:
        min_freq_idx = 0

    # get the indices of the upper end of frequency range
    if freq_range[1] is not None:
        max_freq_idx = np.where(np.round(freqs, decimals=5) >= freq_range[1])[0][0]
    else:
        max_freq_idx = len(freqs)

    return min_freq_idx, max_freq_idx


def convert_to_db(y, db, eps):
    """
    Converts the values in y to dB if db is True.

    Parameters:
    y: np.array
        values to convert to dB
    db: bool
        whether to convert to dB

    Returns:
    label: str
        label for y
    y: np.array
    """
    
    if db:
        return 'Magnitude (dB)', 10*np.log10(y+eps)
    else:
        return 'Magnitude', y


def normalize_min_max(y):
    """
    Normalizes the values in y.

    y: np.array
        values to normalize

    Returns:
    --------
    y: np.array

    """

    return (y - np.min(y)) / (np.max(y) - np.min(y))


def convert_to_mel(Sxx, SFT, mel: bool, num_mels: int, freq_range_idx=(None, None)):
    """
    Converts the values in Sxx to mel scale if mel is True.

    Parameters:
    -----------
    Sxx: np.array
        values to convert to mel scale
    SFT: ShortTimeFFT object
        object containing the sampling frequency and frequency resolution for doing STFT
    mel: bool
        whether to convert to mel scale
    num_mels: int
        number of mel bins
    freq_range_idx: tuple
        (min_freq_idx, max_freq_idx) to convert to mel scale

    Returns:
    --------
    ylabel: str
        label for y
    freqs: np.array
        mel frequencies
    Sxx: np.array
        new spectrogram
    """

    n_fft = Sxx.shape[0]
    
    if mel:
        # FIXME: should the mel conversion only be on the desired freqs, rather than the full thing?
        Sxx = librosa.feature.melspectrogram(S=Sxx, sr=SFT.fs, n_fft=n_fft, hop_length=SFT.hop, n_mels=num_mels)
        freqs = librosa.mel_frequencies(n_mels=num_mels, fmin=SFT.f[freq_range_idx[0]], fmax=SFT.f[freq_range_idx[1]])
        ylabel = f'Mel Freq. ('
    else:
        freqs = SFT.f[freq_range_idx[0]:freq_range_idx[1]]
        ylabel = f"Freq. $f$ in Hz (" + rf"$\Delta f = {SFT.delta_f:g}\,$Hz, "

    return ylabel, freqs, Sxx



def stft_analysis(data: np.array, story: str, SFT, channel_idx=None, freq_range=(None, None), 
                  mel=False, num_mels=128, db=False, normalize='minmax', plot=True, eps=1e-8, vmin=0, vmax=1
                  ) -> Tuple[np.array, np.array]:
    """
    Plots the spectrogram of the data using the ShortTimeFFT object SFT and returns the frequencies and spectrogram.

    Parameters:
    -----------
    data: np.array
        data to plot, shape (num_samples, num_channels)
    story: str
        name of the story
    SFT: ShortTimeFFT object
        object containing the sampling frequency and frequency resolution for doing STFT
    channel_idx: int
        index of the channel to plot (if None, average over channels)
    freq_range: tuple
        (min_freq, max_freq) to plot
    mel: bool
        whether to convert to mel scale
    num_mels: int
        number of mel bins 
    db: bool
        whether to convert to dB
    normalize: str
        'zscore': z-score normalization
        'minmax': min-max normalization (subtract off the min and divide by (max - min))
    plot: bool
        whether to plot the spectrogram
    eps: float
        small value to add to avoid log(0) in db calculation

    Returns:
    --------
    freqs: np.array
        frequencies
    Sxx: np.array
        spectrogram
    """
    
    if freq_range[0] and freq_range[1]:
        assert freq_range[0] < freq_range[1], "Invalid frequency range"

    if freq_range[1]:
        assert freq_range[1] < SFT.fs / 2, "Invalid maximum frequency (use None instead for max frequency)"

    num_samples = len(data[:, 0]) 
    tx = np.arange(num_samples, step=SFT.hop) * 1 / SFT.fs # max time will be lower bc stimulus off for some part

    # modify extent 
    extent = list(SFT.extent(n=num_samples))

    # get the indices of the lower end of frequency range
    extent[2] = freq_range[0] if freq_range[0] else 0
    extent[3] = freq_range[1] if freq_range[1] else SFT.fs // 2

    # calculate STFT for channel if specified, otherwise average over channels
    if channel_idx is not None:
        Sxx = np.abs(SFT.stft(data[:, channel_idx]))
        title = ""
    else:
        Sxx = np.abs(SFT.stft(np.mean(data, axis=1)))
        title = "Averaged "

    # get min, max freq idxs and index first
    min_freq_idx, max_freq_idx = get_freq_indices(extent[2:], SFT.f)
    Sxx = Sxx[min_freq_idx:max_freq_idx]

    # then convert to mel scale if specified
    ylabel, freqs, Sxx = convert_to_mel(Sxx, SFT, mel, num_mels, freq_range_idx=(min_freq_idx, max_freq_idx))
    ylabel += f"{len(Sxx)} bins)"

    # convert to dB scale if specified 
    label, Sxx = convert_to_db(Sxx, db, eps)

    # normalize values
    if normalize == 'minmax':
        Sxx = normalize_min_max(Sxx)
        label = f"{label}, Normalized"
    elif normalize == 'zscore':
        # Sxx = (Sxx - np.mean(Sxx)) / np.std(Sxx)  # TODO: need to check
        Sxx = zscore(Sxx, axis=None)
        label = f"{label}, Z-Score Normalized"

    if plot:
        # create figure
        plt.figure(figsize=(10,5))

        # plot spectrogram
        plt.imshow(Sxx, aspect='auto', extent=tuple(extent), origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)

        cbar = plt.colorbar()
        cbar.set_label(label)

        # if plotting by channel, add channel number to title
        title += f"Spectrogram: {story}, Window Length {len(SFT.win)}, Hop Length {SFT.hop}"
        if channel_idx is not None:
            title = f"Channel {channel_idx}, {title}"
        plt.title(title)
        
        # label axes
        plt.xlabel(f"Time $t$ in seconds ({SFT.p_num(num_samples)} slices, " + rf"$\Delta t = {SFT.delta_t:g}\,$s)")
        plt.ylabel(ylabel)

    return freqs, Sxx


def spectral_centroid(data, fs, channel_idx=None, freq_range=(None, None)) -> Float:
    """
    Returns the spectral centroid of the data.

    Parameters:
    -----------
    data: np.array
        data to analyze, shape (num_samples, num_channels)
    fs: int
        sampling frequency
    channel_idx: int
        index of the channel to analyze (if None, average over channels)
    freq_range: tuple
        (min_freq, max_freq) to analyze

    Returns:
    --------
    spectral_centroid: float
        spectral centroid

    """
    if freq_range[0] and freq_range[1]:
        assert freq_range[0] < freq_range[1], "Invalid frequency range"

    if freq_range[1]:
        assert freq_range[1] <= fs / 2, "Invalid maximum frequency (use None instead for max frequency)"

    num_samples = len(data[:, 0])

    # calculate the FFT of the signal
    if channel_idx is not None:
        y = np.abs(fft(data[:, channel_idx])[:(num_samples+1)//2])
    else:
        y = np.abs(fft(np.mean(data, axis=1))[:(num_samples+1)//2])

    # calculate the frequencies
    freqs = fftfreq(num_samples, d=1/fs)[:(num_samples+1)//2]

    # get the indices of the lower end of frequency range
    min_freq_idx, max_freq_idx = get_freq_indices(freq_range, freqs)

    # calculate the spectral centroid
    spectral_centroid = np.sum(freqs[min_freq_idx:max_freq_idx] * y[min_freq_idx:max_freq_idx]) / np.sum(y[min_freq_idx:max_freq_idx])

    return spectral_centroid


def spectrum_analysis(data, story, fs, channel_idx=None, freq_range=(None, None), db=False, normalize='minmax', plot=True, eps=1e-8
                      ) -> Tuple[np.array, np.array]:
    """
    Plots the spectrum of the data if specified and returns the frequencies and spectrum.

    Parameters:
    -----------
    data: np.array
        data to analyze, shape (num_samples, num_channels)
    story: str
        name of the story
    fs: int
        sampling frequency
    channel_idx: int
        index of the channel to analyze (if None, average over channels)
    freq_range: tuple  
        (min_freq, max_freq) to analyze
    db: bool
        whether to convert to dB
    normalize: str
        'zscore': z-score normalization, 'minmax': min-max normalization
    plot: bool
        whether to plot the spectrum
    eps: float
        small value to add to avoid log(0) in db calculation

    Returns:
    --------
    freqs: np.array
        frequencies
    spec: np.array
        spectrum
    """

    if freq_range[0] and freq_range[1]:
        assert freq_range[0] < freq_range[1], "Invalid frequency range"

    if freq_range[1]:
        assert freq_range[1] <= fs / 2, "Invalid maximum frequency (use None instead for max frequency)"

    num_samples = len(data[:, 0])

    # calculate the frequencies (index at the end since the function returns negatives frequencies too)
    freqs = fftfreq(num_samples, d=1/fs)[:(num_samples+1)//2]

    # get the indices of the frequency range
    min_freq_idx, max_freq_idx = get_freq_indices(freq_range, freqs)
    freqs = freqs[min_freq_idx:max_freq_idx]

    # calculate the FFT of the signal
    if channel_idx is not None:
        y = fft(data[:, channel_idx])[:(num_samples+1)//2]
        title = ""
    else:
        y = fft(np.mean(data, axis=1))[:(num_samples+1)//2 + 1]
        title = "Averaged "
    spec = np.abs(y)[min_freq_idx:max_freq_idx]

    # convert to db
    label, spec = convert_to_db(spec, db, eps)
    
    # normalize
    if normalize == 'minmax':
        spec = normalize_min_max(spec)
        label = f"{label}, Normalized"
    elif normalize == 'zscore':
        spec = zscore(spec, axis=None)
        label = f"{label}, Z-Score Normalized"

    if plot:
        # create figure
        plt.figure(figsize=(10,5))

        # plot spectrum
        plt.plot(freqs, spec, lw=0.5)

        # plot labels
        plt.xlabel("Frequency $f$ in Hz")
        plt.ylabel(label)
        plt.grid(True)

        # if plotting by channel, add channel number to title
        title += f"FFT Spectrum: {story}"
        if channel_idx is not None:
            title += f", Channel {channel_idx}"
        plt.title(title)

    return freqs, spec


# combine previous three functions into one
def filter_data(data, fs, cutoff_freqs, order=5) -> np.array:
    """
    Applies a low, high, or band pass filter to the data.

    Parameters:
    -----------
    data: np.array
        data to filter
    fs: int
        sampling frequency
    cutoff_freqs: tuple
        cutoff frequencies
    order: int
        order of the filter

    Returns:
    --------
    filtered_data: np.array
        filtered data
    """

    nyquist_freq = 0.5 * fs
    low_cutoff_freq, high_cutoff_freq = cutoff_freqs
    if high_cutoff_freq is None and low_cutoff_freq is None:
        print("No filter applied!")
        return data
    elif high_cutoff_freq is None:
        cutoff = low_cutoff_freq / nyquist_freq
        b, a = butter(order, cutoff, btype='high', analog=False)
    elif low_cutoff_freq is None:
        cutoff = high_cutoff_freq / nyquist_freq
        b, a = butter(order, cutoff, btype='low', analog=False)
    else:
        low_cutoff = low_cutoff_freq / nyquist_freq
        high_cutoff = high_cutoff_freq / nyquist_freq
        b, a = butter(order, [low_cutoff, high_cutoff], btype='band', analog=False)

    return filtfilt(b, a, data)    


def get_envelope_phase(orig_signal: np.array, fs: int, duration: int, start=0) -> Tuple[int, int, np.array, np.array, np.array]:
    """
    Returns the envelope of the signal.

    Parameters:
    -----------
    signal: np.array
        signal to analyze
    fs: int
        sampling frequency (Hz)
    duration: int
        duration of the signal (samples)
    start: int
        start of the signal (default 0 samples from start of signal)

    Returns:
    --------
    start: int
        start of the signal
    duration: int
        duration of the signal
    t: np.array
        time array
    amplitude_envelope: np.array
        amplitude envelope
    instantaneous_phase: np.array
        instantaneous phase
    """

    t = np.arange(start, start + duration) / fs
    signal = orig_signal[start:start+duration]
    analytic_signal = hilbert(signal)
    amplitude_envelope = np.abs(analytic_signal)
    instantaneous_phase = np.unwrap(np.angle(analytic_signal))

    return start, duration, t, amplitude_envelope, instantaneous_phase
    