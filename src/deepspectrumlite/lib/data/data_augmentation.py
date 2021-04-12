#                               DeepSpectrumLite
# ==============================================================================
# Edward Ma, "MLP Augmentation" (2019).
# @see https://github.com/makcedward/nlpaug
# ==============================================================================
import numpy as np
import librosa


def add_noise(data, noise_factor):
    noise = np.random.randn(len(data))
    augmented_data = data + noise_factor * noise
    # Cast back to same data type
    augmented_data = augmented_data.astype(type(data[0]))
    return augmented_data


def shift_time(data, sampling_rate, shift_max, shift_direction):
    shift = np.random.randint(sampling_rate * shift_max)
    if shift_direction == 'right':
        shift = -shift
    elif shift_direction == 'both':
        direction = np.random.randint(0, 2)
        if direction == 1:
            shift = -shift
    augmented_data = np.roll(data, shift)
    # Set to silence for heading/ tailing
    if shift > 0:
        augmented_data[:shift] = 0
    else:
        augmented_data[shift:] = 0
    return augmented_data


def shift_pitch(data, sampling_rate, pitch_factor):
    return librosa.effects.pitch_shift(data, sampling_rate, pitch_factor)


def change_speed(data, speed_factor):
    return librosa.effects.time_stretch(data, speed_factor)


def apply_hpss(data):
    return librosa.effects.hpss(data)[1]

