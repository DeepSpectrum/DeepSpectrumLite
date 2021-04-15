# code is token from https://gist.github.com/padoremu/8288b47ce76e9530eb288d4eec2e0b4d

# This file contains a collection of workarounds for missing TFLite support from:
# https://github.com/tensorflow/magenta/tree/master/magenta/music
# as posted in https://github.com/tensorflow/tensorflow/issues/27303
# Thanks a lot to github.com/rryan for his support!

# The function for testing MFCC computation given PCM input is:
# - test_mfcc_tflite
# Please not that the output has not yet been compared to the one produced by the respective TF functions.

# This file also contains test code for other problems in the context of audio processing with TF and TFLite: -
# test_rnn_gru_save_load (saving/loading of RNNs with more than one cell):
# https://github.com/tensorflow/tensorflow/issues/36093 - test_rnn_gru_tflite (missing TFLite support for some
# functions needed by RNN): https://github.com/tensorflow/tensorflow/issues/21526#issuecomment-577586202

# This code has been tested with tf-nightly-2.0-preview (2.0.0.dev20190731) on Ubuntu 18.04 with python 3.6.9.


import tensorflow as tf
import numpy as np

import fractions

# constants
padding = 'right'

desired_samples = 16000
window_size_samples = 640
window_stride_samples = 640
dct_coefficient_count = 10
sample_rate = 16000


# line of code for saving plot of model
# tf.keras.utils.plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)

# some converter flags that might help sometime
# converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
# converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
# converter.target_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
# converter.experimental_new_converter = True


def _dft_matrix(dft_length):
    """Calculate the full DFT matrix in numpy."""
    omega = (0 + 1j) * 2.0 * np.pi / float(dft_length)
    # Don't include 1/sqrt(N) scaling, tf.signal.rfft doesn't apply it.
    return np.exp(omega * np.outer(np.arange(dft_length), np.arange(dft_length)))


def _naive_rdft(signal_tensor, fft_length, padding='center'):
    """Implement real-input Fourier Transform by matmul."""
    # We are right-multiplying by the DFT matrix, and we are keeping
    # only the first half ("positive frequencies").
    # So discard the second half of rows, but transpose the array for
    # right-multiplication.
    # The DFT matrix is symmetric, so we could have done it more
    # directly, but this reflects our intention better.
    complex_dft_matrix_kept_values = _dft_matrix(fft_length)[:(fft_length // 2 + 1), :].transpose()
    real_dft_tensor = tf.constant(np.real(complex_dft_matrix_kept_values).astype(np.float32), name='real_dft_matrix')
    imag_dft_tensor = tf.constant(np.imag(complex_dft_matrix_kept_values).astype(np.float32),
                                  name='imaginary_dft_matrix')
    signal_frame_length = signal_tensor.shape[-1]  # .value
    half_pad = (fft_length - signal_frame_length) // 2

    if padding == 'center':
        # Center-padding.
        pad_values = tf.concat([
            tf.zeros([tf.rank(signal_tensor) - 1, 2], tf.int32),
            [[half_pad, fft_length - signal_frame_length - half_pad]]
        ], axis=0)
    elif padding == 'right':
        # Right-padding.
        pad_values = tf.concat([
            tf.zeros([tf.rank(signal_tensor) - 1, 2], tf.int32),
            [[0, fft_length - signal_frame_length]]
        ], axis=0)

    padded_signal = tf.pad(signal_tensor, pad_values)

    result_real_part = tf.matmul(padded_signal, real_dft_tensor)
    result_imag_part = tf.matmul(padded_signal, imag_dft_tensor)

    return result_real_part, result_imag_part


def _fixed_frame(signal, frame_length, frame_step, first_axis=False):
    """tflite-compatible tf.signal.frame for fixed-size input.
    Args:
        signal: Tensor containing signal(s).
        frame_length: Number of samples to put in each frame.
        frame_step: Sample advance between successive frames.
        first_axis: If true, framing is applied to first axis of tensor; otherwise,
        it is applied to last axis.
    Returns:
        A new tensor where the last axis (or first, if first_axis) of input
        signal has been replaced by a (num_frames, frame_length) array of individual
        frames where each frame is drawn frame_step samples after the previous one.
    Raises:
        ValueError: if signal has an undefined axis length.  This routine only
        supports framing of signals whose shape is fixed at graph-build time.
    """
    signal_shape = signal.shape.as_list()

    if first_axis:
        length_samples = signal_shape[0]
    else:
        length_samples = signal_shape[-1]

    if length_samples <= 0:
        raise ValueError('fixed framing requires predefined constant signal length')

    num_frames = max(0, 1 + (length_samples - frame_length) // frame_step)

    if first_axis:
        inner_dimensions = signal_shape[1:]
        result_shape = [num_frames, frame_length] + inner_dimensions
        gather_axis = 0
    else:
        outer_dimensions = signal_shape[:-1]
        result_shape = outer_dimensions + [num_frames, frame_length]
        # Currently tflite's gather only supports axis==0, but that may still
        # work if we want the last of 1 axes.
        gather_axis = len(outer_dimensions)

    subframe_length = fractions.gcd(frame_length, frame_step)  # pylint: disable=deprecated-method
    subframes_per_frame = frame_length // subframe_length
    subframes_per_hop = frame_step // subframe_length
    num_subframes = length_samples // subframe_length

    if first_axis:
        trimmed_input_size = [num_subframes * subframe_length] + inner_dimensions
        subframe_shape = [num_subframes, subframe_length] + inner_dimensions
    else:
        trimmed_input_size = outer_dimensions + [num_subframes * subframe_length]
        subframe_shape = outer_dimensions + [num_subframes, subframe_length]
    subframes = tf.reshape(
        tf.slice(
            signal,
            begin=np.zeros(len(signal_shape), np.int32),
            size=trimmed_input_size), subframe_shape)

    # frame_selector is a [num_frames, subframes_per_frame] tensor
    # that indexes into the appropriate frame in subframes. For example:
    # [[0, 0, 0, 0], [2, 2, 2, 2], [4, 4, 4, 4]]
    frame_selector = np.reshape(np.arange(num_frames) * subframes_per_hop, [num_frames, 1])

    # subframe_selector is a [num_frames, subframes_per_frame] tensor
    # that indexes into the appropriate subframe within a frame. For example:
    # [[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]]
    subframe_selector = np.reshape(np.arange(subframes_per_frame), [1, subframes_per_frame])

    # Adding the 2 selector tensors together produces a [num_frames,
    # subframes_per_frame] tensor of indices to use with tf.gather to select
    # subframes from subframes. We then reshape the inner-most subframes_per_frame
    # dimension to stitch the subframes together into frames. For example:
    # [[0, 1, 2, 3], [2, 3, 4, 5], [4, 5, 6, 7]].
    selector = frame_selector + subframe_selector
    frames = tf.reshape(tf.gather(subframes, selector.astype(np.int32), axis=gather_axis), result_shape)

    return frames


def _stft_tflite(signal, frame_length, frame_step, fft_length):
    """tflite-compatible implementation of tf.signal.stft.
    Compute the short-time Fourier transform of a 1D input while avoiding tf ops
    that are not currently supported in tflite (Rfft, Range, SplitV).
    fft_length must be fixed. A Hann window is of frame_length is always
    applied.
    Since fixed (precomputed) framing must be used, signal.shape[-1] must be a
    specific value (so "?"/None is not supported).
    Args:
        signal: 1D tensor containing the time-domain waveform to be transformed.
        frame_length: int, the number of points in each Fourier frame.
        frame_step: int, the number of samples to advance between successive frames.
        fft_length: int, the size of the Fourier transform to apply.
    Returns:
        Two (num_frames, fft_length) tensors containing the real and imaginary parts
        of the short-time Fourier transform of the input signal.
    """
    # Make the window be shape (1, frame_length) instead of just frame_length
    # in an effort to help the tflite broadcast logic.
    window = tf.reshape(
        tf.constant(
            (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(0, 1.0, 1.0 / frame_length))
             ).astype(np.float32),
            name='window'), [1, frame_length])

    framed_signal = _fixed_frame(signal, frame_length, frame_step, first_axis=False)
    framed_signal *= window

    real_spectrogram, imag_spectrogram = _naive_rdft(framed_signal, fft_length)

    return real_spectrogram, imag_spectrogram


def _stft_magnitude_tflite(waveform_input, window_length_samples, hop_length_samples, fft_length):
    """Calculate spectrogram avoiding tflite incompatible ops."""
    real_stft, imag_stft = _stft_tflite(waveform_input, frame_length=window_length_samples,
                                        frame_step=hop_length_samples, fft_length=fft_length)
    stft_magnitude = tf.sqrt(tf.add(real_stft * real_stft, imag_stft * imag_stft), name='magnitude_spectrogram')

    return stft_magnitude, real_stft.shape


def _stft_magnitude_full_tf(waveform_input, window_length_samples, hop_length_samples, fft_length):
    """Calculate STFT magnitude (spectrogram) using tf.signal ops."""
    sftfs = tf.signal.stft(waveform_input, frame_length=window_length_samples, frame_step=hop_length_samples,
                           fft_length=fft_length)
    stft_magnitude = tf.abs(sftfs, name='magnitude_spectrogram')

    return stft_magnitude, sftfs.shape


def tflite_mfcc(log_mel_spectrogram):
    axis_dim = log_mel_spectrogram.shape[-1]  # .value

    scale_arg = -tf.range(float(axis_dim)) * np.pi * 0.5 / float(axis_dim)
    scale_real = 2.0 * tf.cos(scale_arg)
    scale_imag = 2.0 * tf.sin(scale_arg)

    # Check for tf.signal compatibility.
    # scale = 2.0 * tf.exp(tf.complex(0.0, scale_arg))
    # np.testing.assert_allclose(scale, tf.complex(scale_real, scale_imag))

    rfft_real, rfft_imag = _naive_rdft(log_mel_spectrogram, fft_length=2 * axis_dim, padding=padding)
    # rfft_real = rfft_real[..., :axis_dim]
    rfft_real = rfft_real[:, :, :axis_dim]

    # Conjugate to match tf.signal convention:
    # rfft_imag = -rfft_imag[..., :axis_dim]
    rfft_imag = -rfft_imag[:, :, :axis_dim]

    # Check for tf.signal compatibility:
    # tf_rfft = tf.signal.rfft(log_mel_spectrogram, fft_length=[2*axis_dim])[..., :axis_dim]
    # np.testing.assert_allclose(tf_rfft, tf.complex(rfft_real, rfft_imag), atol=1e-4, rtol=1e-4)

    dct2 = rfft_real * scale_real - rfft_imag * scale_imag

    # Check for tf.signal compatibility:
    # tf_dct2 = tf.real(tf_rfft * scale)
    # np.testing.assert_allclose(dct2, tf_dct2, atol=1e-4, rtol=1e-4)

    mfcc = dct2 * tf.math.rsqrt(2 * float(axis_dim))

    # Check for tf.signal compatibility:
    # tf_mfcc = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)
    # np.testing.assert_allclose(mfcc, tf_mfcc, atol=1e-4, rtol=1e-4)

    return mfcc


def preprocess_audio(audio):
    reshaped_audio = tf.reshape(audio, [1, audio.shape[1]])

    # note: parameter fft_length will be automatically chosen smallest power of 2 > frame_length, e.g. 640 => 1024
    # stfts = tf.signal.stft(reshaped_audio, frame_length=window_size_samples, frame_step=window_stride_samples)
    # spectrograms_abs = tf.abs(stfts)

    spectrograms_abs, shape = _stft_magnitude_tflite(waveform_input=reshaped_audio,
                                                     window_length_samples=window_size_samples,
                                                     hop_length_samples=window_stride_samples, fft_length=1024)

    spectrograms = tf.square(spectrograms_abs)  # can this be done better?

    # warp the linear scale spectrograms into the mel-scale
    num_spectrogram_bins = shape[-1]  # sfts. #.value
    lower_edge_hertz, upper_edge_hertz, num_mel_bins = 80.0, 7600.0, 80
    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(num_mel_bins, num_spectrogram_bins, sample_rate,
                                                                        lower_edge_hertz, upper_edge_hertz)
    mel_spectrograms = tf.tensordot(spectrograms, linear_to_mel_weight_matrix, 1)
    mel_spectrograms.set_shape(spectrograms.shape[:-1].concatenate(linear_to_mel_weight_matrix.shape[-1:]))

    # compute a stabilized log to get log-magnitude mel-scale spectrograms
    log_mel_spectrograms = tf.math.log(mel_spectrograms + 1e-6)

    # compute MFCCs from log_mel_spectrograms and take the dct_coefficient_count coefficients
    # result_mfcc = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrograms)[..., :dct_coefficient_count]
    # result_mfcc = tflite_mfcc(log_mel_spectrograms)[..., :dct_coefficient_count]
    result_mfcc = tflite_mfcc(log_mel_spectrograms)[:, :, :dct_coefficient_count]

    # reshape to (1, x)
    reshaped_mfcc = tf.reshape(result_mfcc, [1, tf.size(result_mfcc)])

    return reshaped_mfcc


# slurmJobs model conversion to tensorflow lite and invocation
def test_tflite(model):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # converter.allow_custom_ops = True # does the trick
    # converter.target_spec.supported_ops = set([tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS])
    # converter.experimental_new_converter = True # does not help

    tflite_model = converter.convert()

    interpreter = tf.lite.Interpreter(model_content=tflite_model)

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.allocate_tensors()

    # interpreter.set_tensor(input_details[0]['index'], tf.convert_to_tensor(np.expand_dims(audio_data, 0), dtype=tf.float32))

    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]['index'])


# slurmJobs using RNN with GRU cell in tensorflow lite
def test_rnn_gru_tflite():
    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Input(shape=(1, 1,)))

    cell = tf.keras.layers.GRUCell(10)

    model.add(tf.keras.layers.RNN(cell))

    test_tflite(model)


# slurmJobs saving/loading RNN with more than one GRU cell
def test_rnn_gru_save_load(save_format):
    # saving succeeds for number_of_cells = 1, but fails for number_of_cells > 1
    number_of_cells = 2

    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Input(shape=(1, 1,)))

    cells = []

    for _ in range(number_of_cells):
        cells.append(tf.keras.layers.GRUCell(10))

    model.add(tf.keras.layers.RNN(cells))

    model_path = "rnn.h5" if save_format == "h5" else "rnn.tf"

    model.save(model_path, save_format=save_format)
    model2 = tf.keras.models.load_model(model_path)


# slurmJobs audio preprocessing (stft / abs / mfcc) in tensorflow lite
def test_mfcc_tflite():
    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Input(shape=16000, dtype=tf.float32))
    model.add(tf.keras.layers.Lambda(preprocess_audio))

    test_tflite(model)

# test_mfcc_tflite()
# test_rnn_gru_tflite()
# test_rnn_gru_save_load("h5")
# test_rnn_gru_save_load("tf")
