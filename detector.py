#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Detects onsets, beats and tempo in WAV files.

For usage information, call with --help.

Author: Jan Schlüter
"""

from pathlib import Path
from argparse import ArgumentParser
import json

import numpy as np
from scipy.io import wavfile
import librosa
try:
    import tqdm
except ImportError:
    tqdm = None


def opts_parser():
    usage =\
"""Detects onsets, beats and tempo in WAV files.
"""
    parser = ArgumentParser(description=usage)
    parser.add_argument('indir',
            type=str,
            help='Directory of WAV files to process.')
    parser.add_argument('outfile',
            type=str,
            help='Output JSON file to write.')
    parser.add_argument('--plot',
            action='store_true',
            help='If given, plot something for every file processed.')
    return parser


def detect_everything(filename, options):
    """
    Computes some shared features and calls the onset, tempo and beat detectors.
    """
    # read wave file (this is faster than librosa.load)
    sample_rate, signal = wavfile.read(filename)

    # convert from integer to float
    if signal.dtype.kind == 'i':
        signal = signal / np.iinfo(signal.dtype).max

    # convert from stereo to mono (just in case)
    if signal.ndim == 2:
        signal = signal.mean(axis=-1)

    # compute spectrogram with given number of frames per second
    fps = 70
    hop_length = sample_rate // fps
    spect = librosa.stft(
            signal, n_fft=2048, hop_length=hop_length, window='hann')

    # only keep the magnitude
    magspect = np.abs(spect)

    # compute a mel spectrogram
    melspect = librosa.feature.melspectrogram(
            S=magspect, sr=sample_rate, n_mels=80, fmin=27.5, fmax=8000)

    # compress magnitudes logarithmically
    melspect = np.log1p(100 * melspect) 

    # compute onset detection function
    odf, odf_rate = onset_detection_function(
            sample_rate, signal, fps, spect, magspect, melspect, options)

    # detect onsets from the onset detection function
    onsets = detect_onsets(odf_rate, odf, options)

    # detect tempo from everything we have
    tempo = detect_tempo(
            sample_rate, signal, fps, spect, magspect, melspect,
            odf_rate, odf, onsets, options)

    # detect beats from everything we have (including the tempo)
    beats = detect_beats(
            sample_rate, signal, fps, spect, magspect, melspect,
            odf_rate, odf, onsets, tempo, options)

    # plot some things for easier debugging, if asked for it
    if options.plot:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(3, sharex=True)
        plt.subplots_adjust(hspace=0.3)
        plt.suptitle(filename)
        axes[0].set_title('melspect')
        axes[0].imshow(melspect, origin='lower', aspect='auto',
                       extent=(0, melspect.shape[1] / fps,
                               -0.5, melspect.shape[0] - 0.5))
        axes[1].set_title('onsets')
        axes[1].plot(np.arange(len(odf)) / odf_rate, odf)
        for position in onsets:
            axes[1].axvline(position, color='tab:orange')
        axes[2].set_title('beats (tempo: %r)' % list(np.round(tempo, 2)))
        axes[2].plot(np.arange(len(odf)) / odf_rate, odf)
        for position in beats:
            axes[2].axvline(position, color='tab:red')
        plt.show()

    return {'onsets': list(np.round(onsets, 3)),
            'beats': list(np.round(beats, 3)),
            'tempo': list(np.round(tempo, 2))}


def normalize(array):
    """Normalize array to range [0,1]"""
    if len(array) == 0 or np.max(array) == np.min(array):
        return array
    return (array - np.min(array)) / (np.max(array) - np.min(array))


def spectral_flux(spec):
    """
    Calculate spectral flux with half-wave rectification
    """
    flux = np.zeros(spec.shape[1])
    for frame in range(1, spec.shape[1]):
        diff = spec[:, frame] - spec[:, frame - 1]
        flux[frame] = np.sum(diff * (diff > 0))

    return normalize(flux)

def high_frequency_content(mag_spec):
    """
    Calculate high frequency content
    """
    freq_weights = np.arange(mag_spec.shape[0])
    hfc = np.zeros(mag_spec.shape[1])

    for frame in range(mag_spec.shape[1]):
        hfc[frame] = np.sum(mag_spec[:, frame] * freq_weights)
    return normalize(hfc)



def onset_detection_function(sample_rate, signal, fps, spect, magspect,
                             melspect, options):
    """
    Compute an onset detection function. Ideally, this would have peaks
    where the onsets are. Returns the function values and its sample/frame
    rate in values per second as a tuple: (values, values_per_second)
    """

    #TODO for ONSETS
    odfs = {}

    #spectral flux
    if melspect is not None:
        odfs['flux_mel'] = spectral_flux(melspect)
    else:
        print("no mel spect")

    if magspect is not None:
        odfs['flux_mag'] = spectral_flux(magspect)
        odfs['hfc'] = high_frequency_content(magspect)
    else:
        print("no mag spect")


    min_length = min(len(odf) for odf in odfs.values())

    weights = {
        'flux_mel': 0.1,  # Mel-based spectral flux
        'flux_mag': 0.3,  # Mag-based spectral flux
        'hfc': 0.5, # High-frequency content
    }

    available_weights = {k: weights[k] for k in odfs.keys()}
    total_weight = sum(available_weights.values())

    normalized_weights = {k: v / total_weight for k, v in available_weights.items()}

    # Combine available ODFs
    combined = np.zeros(min_length)
    for name, odf in odfs.items():
        combined += normalized_weights[name] * odf

    return combined, fps


def detect_onsets(odf_rate, odf, options):
    """
    Detect onsets in the onset detection function.
    Returns the positions in seconds.
    """
    window_size = int(odf_rate * 0.35)  # window for local average
    threshold_multiplier = 0.4  # Threshold above local average
    min_time_between_onsets = 0.08  # 80 ms minimum between consecutive onsets
    min_distance_samples = int(odf_rate * min_time_between_onsets)

    # adaptive threshold using moving average
    local_avg = np.zeros_like(odf)
    for i in range(len(odf)):
        start = max(0, i - window_size // 2)
        end = min(len(odf), i + window_size // 2 + 1)
        local_avg[i] = np.mean(odf[start:end])

    # adaptive threshold = local average and constant
    adaptive_threshold = local_avg * threshold_multiplier + 0.03

    # Find peaks that are above the threshold
    peaks = []
    for i in range(1, len(odf) - 1):
        if (odf[i] > odf[i - 1] and odf[i] > odf[i + 1] and
                odf[i] > adaptive_threshold[i]):
            peaks.append(i)

    # Apply minimum distance constraint
    if len(peaks) > 0:
        filtered_peaks = [peaks[0]]
        for peak in peaks[1:]:
            if peak - filtered_peaks[-1] >= min_distance_samples:
                filtered_peaks.append(peak)
        peaks = filtered_peaks

    # Convert peak indices to time in seconds
    onset_times = np.array(peaks) / odf_rate

    return onset_times


def detect_tempo(sample_rate, signal, fps, spect, magspect, melspect,
                 odf_rate, odf, onsets, options):
    """
    Detect tempo using onset detection function and spectral analysis.
    Returns one or two tempo estimations (in BPM).
    """

    # Method 1: Onset-based tempo estimation
    tempo_from_onsets = estimate_tempo_from_onsets(onsets)

    # Method 2: Autocorrelation-based tempo estimation
    tempo_from_autocorr = estimate_tempo_from_autocorrelation(odf, odf_rate)

    # Method 3: Beat histogram approach
    tempo_from_histogram = estimate_tempo_from_beat_histogram(odf, odf_rate)

    # Combine estimates (you can experiment with different weighting)
    candidate_tempos = []

    # Example values that each method might return:
    # tempo_from_onsets = [125.3]          # e.g., from inter-onset intervals
    # tempo_from_autocorr = [124.8]        # e.g., from autocorrelation peak
    # tempo_from_histogram = [126.1]       # e.g., from beat histogram

    if tempo_from_onsets is not None:
        candidate_tempos.extend(tempo_from_onsets)
        # candidate_tempos = [125.3]

    if tempo_from_autocorr is not None:
        candidate_tempos.extend(tempo_from_autocorr)
        # candidate_tempos = [125.3, 124.8]

    if tempo_from_histogram is not None:
        candidate_tempos.extend(tempo_from_histogram)
        # candidate_tempos = [125.3, 124.8, 126.1]

    # Alternative: Weighted combination approach
    # You could also weight the estimates based on confidence:
    weighted_tempos = []
    weights = []

    if tempo_from_onsets is not None:
        weighted_tempos.extend(tempo_from_onsets)
        weights.extend([0.5] * len(tempo_from_onsets))  # 40% weight for onset-based

    if tempo_from_autocorr is not None:
        weighted_tempos.extend(tempo_from_autocorr)
        weights.extend([0.35] * len(tempo_from_autocorr))  # 35% weight for autocorr

    if tempo_from_histogram is not None:
        weighted_tempos.extend(tempo_from_histogram)
        weights.extend([0.15] * len(tempo_from_histogram))  # 25% weight for histogram

    # Example: weighted_tempos = [125.3, 124.8, 126.1], weights = [0.4, 0.35, 0.25]
    # Weighted average: (125.3*0.4 + 124.8*0.35 + 126.1*0.25) = 125.225

    if weighted_tempos:
        weighted_avg = np.average(weighted_tempos, weights=weights)
        candidate_tempos.append(weighted_avg)
        # candidate_tempos = [125.3, 124.8, 126.1, 125.225]

    # Filter reasonable tempo range (typical music: 60-200 BPM)
    candidate_tempos = [t for t in candidate_tempos if 60 <= t <= 200]
    # Example after filtering: [125.3, 124.8, 126.1, 125.225] (all in valid range)

    if len(candidate_tempos) == 0:
        # Fallback to a reasonable default
        return [120]  # 120 BPM is a common tempo

    # Sort and pick most likely candidates
    candidate_tempos = sorted(candidate_tempos)
    # Example sorted: [124.8, 125.225, 125.3, 126.1]

    # Method 1: Use median as primary tempo
    primary_tempo = candidate_tempos[len(candidate_tempos) // 2]  # median
    # Example: primary_tempo = 125.225 (index 1 for 4 items)

    # Method 2: Alternative - use weighted average or most confident estimate
    # primary_tempo = weighted_avg  # Use the weighted average calculated above
    # Example: primary_tempo = 125.225

    # Method 3: Alternative - cluster similar tempos and pick strongest cluster
    # You could group tempos within ±3 BPM and pick the largest cluster

    # Common alternative: half or double tempo (handles tempo ambiguity)
    if primary_tempo > 120:
        secondary_tempo = primary_tempo / 2  # Example: 125.225 / 2 = 62.6
    else:
        secondary_tempo = primary_tempo * 2  # Example: if primary was 80, secondary = 160

    # Example with our values: primary_tempo = 125.225, secondary_tempo = 62.6

    # Ensure secondary tempo is in reasonable range
    if 60 <= secondary_tempo <= 200:
        return [primary_tempo, secondary_tempo]  # Example: [125.225, 62.6] -> [125.23, 62.61] after rounding
    else:
        return [primary_tempo]  # Example: [125.23] if secondary was out of range


def estimate_tempo_from_onsets(onsets):
    """
    Estimate tempo from inter-onset intervals.
    """
    if len(onsets) < 3:
        return None

    # Calculate inter-onset intervals
    intervals = np.diff(onsets)

    # Remove outliers (very short or very long intervals)
    intervals = intervals[(intervals > 0.2) & (intervals < 2.0)]

    if len(intervals) == 0:
        return None

    # Convert intervals to BPM
    tempos = 60.0 / intervals

    # Find most common tempo using histogram
    hist, bin_edges = np.histogram(tempos, bins=50, range=(60, 200))
    most_common_idx = np.argmax(hist)
    primary_tempo = (bin_edges[most_common_idx] + bin_edges[most_common_idx + 1]) / 2

    return [primary_tempo]


def estimate_tempo_from_autocorrelation(odf, odf_rate):
    """
    Estimate tempo using autocorrelation of the onset detection function.
    """
    # Calculate autocorrelation
    autocorr = np.correlate(odf, odf, mode='full')
    autocorr = autocorr[len(autocorr) // 2:]  # Keep only positive lags

    # Convert lag indices to time (seconds)
    lags_seconds = np.arange(len(autocorr)) / odf_rate

    # Focus on reasonable tempo range (0.3 to 1.0 seconds = 60-200 BPM)
    min_lag = int(0.3 * odf_rate)  # 200 BPM
    max_lag = int(1.0 * odf_rate)  # 60 BPM

    if max_lag >= len(autocorr):
        return None

    # Find peaks in autocorrelation within tempo range
    autocorr_segment = autocorr[min_lag:max_lag]
    lags_segment = lags_seconds[min_lag:max_lag]

    # Find local maxima
    peaks = []
    for i in range(1, len(autocorr_segment) - 1):
        if (autocorr_segment[i] > autocorr_segment[i - 1] and
                autocorr_segment[i] > autocorr_segment[i + 1]):
            peaks.append((lags_segment[i], autocorr_segment[i]))

    if not peaks:
        return None

    # Sort by autocorrelation strength
    peaks.sort(key=lambda x: x[1], reverse=True)

    # Convert best lag to tempo
    best_lag = peaks[0][0]
    tempo = 60.0 / best_lag

    return [tempo]


def estimate_tempo_from_beat_histogram(odf, odf_rate):
    """
    Estimate tempo using beat histogram approach.
    """
    # This is a simplified version - you might want to implement
    # a more sophisticated beat tracking algorithm

    # Find local maxima in ODF (potential beat positions)
    peaks = []
    for i in range(1, len(odf) - 1):
        if odf[i] > odf[i - 1] and odf[i] > odf[i + 1] and odf[i] > 0.1:
            peaks.append(i / odf_rate)  # Convert to seconds

    if len(peaks) < 3:
        return None

    # Calculate all possible inter-beat intervals
    intervals = []
    for i in range(len(peaks)):
        for j in range(i + 1, min(i + 5, len(peaks))):  # Look ahead max 4 beats
            interval = peaks[j] - peaks[i]
            if 0.3 <= interval <= 1.0:  # Reasonable beat interval range
                intervals.append(interval)

    if not intervals:
        return None

    # Create histogram of intervals
    hist, bin_edges = np.histogram(intervals, bins=30, range=(0.3, 1.0))

    # Find most common interval
    best_bin = np.argmax(hist)
    best_interval = (bin_edges[best_bin] + bin_edges[best_bin + 1]) / 2

    # Convert to BPM
    tempo = 60.0 / best_interval

    return [tempo]


def detect_beats(sample_rate, signal, fps, spect, magspect, melspect,
                 odf_rate, odf, onsets, tempo, options):
    """
    Detect beats using any of the input representations.
    Returns the positions of all beats in seconds.
    """
    # we only have a dumb dummy implementation here.
    # it returns every 10th onset as a beat.
    # this is not a useful solution at all, just a placeholder.
    # TODO for beats
    return onsets[::10]


def main():
    # parse command line
    parser = opts_parser()
    options = parser.parse_args()

    # iterate over the input directory
    indir = Path(options.indir)
    infiles = list(indir.glob('*.wav'))
    if tqdm is not None:
        infiles = tqdm.tqdm(infiles, desc='File')
    results = {}
    for filename in infiles:
        results[filename.stem] = detect_everything(filename, options)

    # write output file
    with open(options.outfile, 'w') as f:
        json.dump(results, f)


if __name__ == "__main__":
    main()

