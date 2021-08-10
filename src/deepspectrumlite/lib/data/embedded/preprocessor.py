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
from tensorflow.python.keras.applications import imagenet_utils
from tensorflow.python.ops.image_ops_impl import ResizeMethod

from deepspectrumlite import power_to_db, amplitude_to_db
from deepspectrumlite.lib.data.plot import create_map_from_array, CividisColorMap, InfernoColorMap, MagmaColorMap, \
    PlasmaColorMap, ViridisColorMap
import numpy as np
import math


# TODO refactor
class PreprocessAudio(tf.Module):
    def __init__(self, hparams, *args, **kwargs):
        super(PreprocessAudio, self).__init__(*args, **kwargs)
        self.hparams = hparams

        self.resize_method = ResizeMethod.BILINEAR
        if 'resize_method' in self.hparams:
            self.resize_method = self.hparams['resize_method']

        self.anti_alias = True
        if 'anti_alias' in self.hparams:
            self.anti_alias = self.hparams['anti_alias']

        self.colormap = ViridisColorMap()

        available_color_maps = {
            "cividis": CividisColorMap,
            "inferno": InfernoColorMap,
            "magma": MagmaColorMap,
            "plasma": PlasmaColorMap,
            "viridis": ViridisColorMap
        }

        if self.hparams['color_map'] in available_color_maps:
            self.colormap = available_color_maps[self.hparams['color_map']]()

        self.preprocessors = {
            "vgg16":
                tf.keras.applications.vgg16.preprocess_input,
            "vgg19":
                tf.keras.applications.vgg19.preprocess_input,
            "resnet50":
                tf.keras.applications.resnet50.preprocess_input,
            "xception":
                tf.keras.applications.xception.preprocess_input,
            "inception_v3":
                tf.keras.applications.inception_v3.preprocess_input,
            "densenet121":
                tf.keras.applications.densenet.preprocess_input,
            "densenet169":
                tf.keras.applications.densenet.preprocess_input,
            "densenet201":
                tf.keras.applications.densenet.preprocess_input,
            "mobilenet":
                tf.keras.applications.mobilenet.preprocess_input,
            "mobilenet_v2":
                tf.keras.applications.mobilenet_v2.preprocess_input,
            "nasnet_large":
                tf.keras.applications.nasnet.preprocess_input,
            "nasnet_mobile":
                tf.keras.applications.nasnet.preprocess_input,
            "inception_resnet_v2":
                tf.keras.applications.inception_resnet_v2.preprocess_input,
            "squeezenet_v1":
                tf.keras.applications.imagenet_utils.preprocess_input,
        }

    def __preprocess_vgg(self, x, data_format=None):
        """
        Legacy function for VGG16 and VGG19 preprocessing without centering.
        """
        x = x[:, :, :, ::-1]
        return x

    @tf.function(input_signature=[tf.TensorSpec(shape=(1, 16000), dtype=tf.float32)])
    def preprocess(self, audio_signal):  # pragma: no cover
        decoded_audio = audio_signal * (0.7079 / tf.reduce_max(tf.abs(audio_signal)))

        frame_length = int(self.hparams['stft_window_size'] * self.hparams['sample_rate'])
        frame_step = int(self.hparams['stft_hop_size'] * self.hparams['sample_rate'])
        fft_length = int(self.hparams['stft_fft_length'] * self.hparams['sample_rate'])

        stfts = tf.signal.stft(decoded_audio, frame_length=frame_length, frame_step=frame_step,
                               fft_length=fft_length)
        spectrograms = tf.abs(stfts, name="magnitude_spectrograms") ** 2

        # Warp the linear scale spectrograms into the mel-scale.
        num_spectrogram_bins = stfts.shape[-1]
        lower_edge_hertz, upper_edge_hertz, num_mel_bins = self.hparams['lower_edge_hertz'], \
                                                           self.hparams['upper_edge_hertz'], \
                                                           self.hparams['num_mel_bins']

        linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins, num_spectrogram_bins, self.hparams['sample_rate'], lower_edge_hertz,
            upper_edge_hertz)

        mel_spectrograms = tf.tensordot(spectrograms, linear_to_mel_weight_matrix, 1)
        mel_spectrograms.set_shape(spectrograms.shape[:-1].concatenate(
            linear_to_mel_weight_matrix.shape[-1:]))
        num_mfcc = self.hparams['num_mfccs']

        if num_mfcc:
            if self.hparams['db_scale']:
                mel_spectrograms = amplitude_to_db(mel_spectrograms, top_db=None)
            else:
                mel_spectrograms = tf.math.log(mel_spectrograms + 1e-6)

            # Compute MFCCs from mel_spectrograms
            mfccs = tf.signal.mfccs_from_log_mel_spectrograms(mel_spectrograms)[..., :num_mfcc]

            # cep filter
            if self.hparams['cep_lifter'] > 0:
                cep_lifter = self.hparams['cep_lifter']

                (nframes, ncoeff) = mfccs.shape[-2], mfccs.shape[-1]
                n = tf.keras.backend.arange(start=0, stop=ncoeff, dtype=tf.float32)
                lift = 1 + (cep_lifter / 2) * tf.sin(math.pi * n / cep_lifter)

                mfccs *= lift

            output = mfccs
        else:
            if self.hparams['db_scale']:
                output = power_to_db(mel_spectrograms, top_db=None)
            else:
                output = mel_spectrograms

        if self.hparams['use_plot_images']:

            color_map = np.array(self.colormap.get_color_map(), dtype=np.float32)
            # color_map = tf.Variable(initial_value=self.colormap.get_color_map(), name="color_map")

            image_data = create_map_from_array(output, color_map=color_map)

            image_data = tf.image.resize(
                image_data, (self.hparams['image_width'], self.hparams['image_height']),
                method=self.resize_method, preserve_aspect_ratio=False, antialias=False
            )

            image_data = image_data * 255.
            image_data = tf.clip_by_value(image_data, clip_value_min=0., clip_value_max=255.)
            image_data = tf.image.rot90(image_data, k=1)

            def _preprocess(x):
                # values in the range [0, 255] are expected!!
                model_key = self.hparams['basemodel_name']

                if model_key in self.preprocessors:
                    return self.preprocessors[model_key](x, data_format='channels_last')

                return x

            # values in the range [0, 255] are expected!!
            image_data = _preprocess(image_data)

        else:
            image_data = tf.expand_dims(output, axis=3)

        return image_data
