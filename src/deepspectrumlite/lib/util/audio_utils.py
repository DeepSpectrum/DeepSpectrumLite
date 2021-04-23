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


def amplitude_to_db(S, amin=1e-16, top_db=80.0): # pragma: no cover
    magnitude = tf.abs(S)
    ref_value = tf.reduce_max(magnitude)

    power = tf.square(magnitude)
    return power_to_db(power, ref=ref_value ** 2, amin=amin ** 2, top_db=top_db)


def power_to_db(S, ref=1.0, amin=1e-16, top_db=80.0): # pragma: no cover
    """Convert a power-spectrogram (magnitude squared) to decibel (dB) units.
    Computes the scaling ``10 * log10(S / max(S))`` in a numerically
    stable way.
    Based on:
    https://librosa.github.io/librosa/generated/librosa.core.power_to_db.html
    """

    # @tf.function
    def _tf_log10(x):
        numerator = tf.math.log(x)
        denominator = tf.math.log(tf.constant(10, dtype=numerator.dtype))
        return numerator / denominator

    # Scale magnitude relative to maximum value in S. Zeros in the output
    # correspond to positions where S == ref.
    ref_value = tf.abs(ref)

    log_spec = 10.0 * _tf_log10(tf.maximum(amin, S))
    log_spec -= 10.0 * _tf_log10(tf.maximum(amin, ref_value))

    if top_db is not None:
        log_spec = tf.maximum(log_spec, tf.reduce_max(log_spec) - top_db)

    return log_spec
