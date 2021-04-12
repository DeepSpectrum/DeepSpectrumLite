#                               DeepSpectrumLite
# ==============================================================================
# Copyright (C) 2020-2021 Shahin Amiriparian, Tobias Hübner, Maurice Gerczuk,
# Sandra Ottl, Björn Schuller: University of Augsburg. All Rights Reserved.
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.
# ==============================================================================

import tensorflow as tf
import numpy as np


# code is adapted from librosa
def speed_pert_single(decoded_audio, rate):
    with tf.name_scope("speed_pert_fn"):
        D = tf.signal.stft(decoded_audio, frame_length=2048, frame_step=512,
                           fft_length=2048)
        # phase_vocoder
        hop_length = 512

        time_steps = tf.range(0, D.shape[1], rate, dtype=tf.float32)
        pi = tf.constant(np.pi)

        # Expected phase advance in each bin
        phi_advance = tf.linspace(0.0, pi * hop_length, D.shape[0])
        # Phase accumulator; initialize to the first sample
        phase_acc = tf.math.angle(D[:, 0])

        current_stft_frame_size = 932  # todo dynamic value
        current_audio_frame_size = 15872  # todo dynamic value
        # Pad 0 columns to simplify boundary logic
        D = tf.pad(D, [(0, 0), (0, 2)], mode="CONSTANT")

        d_stretch2 = tf.zeros([0, 28], dtype=tf.complex64)

        def _modulo(x, y):
            '''
            A *mostly* differentiable modulo function! Builds and returns the ops for mod(x, y)
            '''
            with tf.name_scope("_modulo"):
                divided = x / y
                remainder = tf.round(
                    y * (divided - tf.cast(tf.cast(divided, tf.int32), tf.float32))
                )
                return remainder

        for t in range(current_stft_frame_size):
            step = time_steps[t]
            columns = D[:, tf.cast(step, dtype=tf.int32): tf.cast(step + 2, dtype=tf.int32)]

            # Weighting for linear magnitude interpolation
            alpha = _modulo(step, 1.0)
            mag = (1.0 - alpha) * tf.abs(columns[:, 0]) + alpha * tf.abs(columns[:, 1])

            # Store to output array
            tmp = tf.cast(mag, dtype=tf.complex64) * tf.exp(tf.cast(1.0j, dtype=tf.complex64) *
                                                            tf.cast(phase_acc, dtype=tf.complex64))
            tmp = tf.stack([tmp])
            d_stretch2 = tf.concat([d_stretch2, tmp], 0)

            # Compute phase advance
            dphase = tf.math.angle(columns[:, 1]) - tf.math.angle(columns[:, 0]) - phi_advance

            # Wrap to -pi:pi range
            dphase = dphase - 2.0 * pi * tf.math.round(dphase / (2.0 * pi))

            # Accumulate phase
            phase_acc += phi_advance + dphase
        d_stretch = tf.transpose(d_stretch2)
        output = tf.signal.inverse_stft(d_stretch, frame_length=2048, frame_step=512, fft_length=2048)

        paddings = tf.constant([[16000 - current_audio_frame_size, 0], ])
        output = tf.pad(output, paddings, "CONSTANT", 0.0)
        return output
