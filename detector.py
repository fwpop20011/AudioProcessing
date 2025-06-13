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
    Detect tempo using any of the input representations.
    Returns one tempo or two tempo estimations.
    """    
    # we only have a dumb dummy implementation here.
    # it uses the time difference between the first two onsets to
    # define the tempo, and returns half of that as a second guess.
    # this is not a useful solution at all, just a placeholder.
    # TODO for tempo
    tempo = 60 / (onsets[1] - onsets[0])
    return [tempo / 2, tempo]


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
    if len(tempo) == 0:
        return[]

    bpm = tempo[0]
    beat_interval = 60.0 / bpm

    start_time = onsets[0] if len(onsets) > 0 else 0
    end_time = len(signal) / sample_rate

    raw_beats = []
    t = start_time
    while t < end_time:
        raw_beats.append(t)
        t += beat_interval

    beat_times = []
    search_window = 0.07

    for raw_beat in raw_beats:
        center = int(raw_beat*odf_rate)
        window = int(search_window*odf_rate)

        start = max(0, center - window)
        end = min(len(odf), center + window + 1)

        if end <= start:
            beat_times.append(raw_beat)
            continue

        local_odf = odf[start:end]
        if len(local_odf) == 0:
            beat_times.append(raw_beat)
            continue

        peak_offset = np.argmax(local_odf)
        snapped_index = start + peak_offset
        snapped_time = snapped_index/odf_rate

        if abs(snapped_time - raw_beat) <= search_window:
            beat_times.append(snapped_time)
        else:
            beat_times.append(raw_beat)




    return np.array(beat_times)


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

