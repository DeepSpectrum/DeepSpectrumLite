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
import math
import os
from numpy import random
import time
import numpy as np
import copy
import pandas as pd
import tensorflow as tf
import shutil
from tensorflow.python.keras.applications import imagenet_utils
from tensorflow.python.ops.image_ops_impl import ResizeMethod
from deepspectrumlite import power_to_db, amplitude_to_db
from deepspectrumlite.lib.data.plot import create_map_from_array, CividisColorMap, InfernoColorMap, MagmaColorMap, PlasmaColorMap, ViridisColorMap


class DataPipeline:
    def __init__(self, name: str, data_classes, hparams: dict, run_id: int,
                 verbose: bool = False, enable_gpu: bool = False,
                 enable_augmentation=False):
        self.name = name
        self.empty_list = {}

        for i, data_class in enumerate(data_classes):
            self.empty_list[data_class] = i

        self.num_classes = len(self.empty_list)

        self.augmented_wav = None
        self.verbose = verbose
        self._data = None
        self.chunk_length = int(hparams['chunk_size'] * hparams['sample_rate'])
        self.chunk_hop_length = int(hparams['chunk_hop_size'] * hparams['sample_rate'])
        self.filenames = None
        self.labels = None
        self.root_dir = ""
        self._data_classes = copy.deepcopy(self.empty_list)
        self._cache_dir = None
        self._run_id = run_id
        self.enable_gpu = enable_gpu
        self.enable_augmentation = enable_augmentation
        self.hparams = hparams
        self._sample_rate = hparams['sample_rate']
        self._batch_size = hparams['batch_size']

        self.prediction_type = 'categorical'
        if 'prediction_type' in self.hparams:
            self.prediction_type = self.hparams['prediction_type']

        self.resize_method = ResizeMethod.BILINEAR
        if 'resize_method' in self.hparams:
            self.resize_method = self.hparams['resize_method']

        self.anti_alias = True
        if 'anti_alias' in self.hparams:
            self.anti_alias = self.hparams['anti_alias']

        self.device = 'cpu'

        if self.enable_gpu:
            self.device = 'gpu'

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

    def __del__(self):
        self.destroy_cache()

    def get_class_id(self, class_name):
        if self.prediction_type == 'categorical':
            return self._data_classes.get(str(class_name), 'invalid class')
        else:
            return str(round(class_name, 2))

    def set_filename_prepend(self, prepend_filename_str: str):
        self.root_dir = prepend_filename_str

    def get_model_input_shape(self):
        if self.hparams['use_plot_images']:
            return self.hparams['image_width'], self.hparams['image_height'], 3
        else:
            window_length = int(self.hparams['stft_window_size'] * self._sample_rate)
            hop_length = int(self.hparams['stft_hop_size'] * self._sample_rate)

            frame_count = int(math.floor((self.chunk_length - window_length) / hop_length)) + 1
            input_shape = (frame_count, self.hparams['num_mfccs'], 1)

            return input_shape

    def set_data(self, data: pd.DataFrame):
        """
        :param data: must be a pandas DataFrame with the following columns:
         - filename
         - label
         - duration_frames: Number of frames of the audio file
        :return:
        """
        self._data = data

    def up_sample(self):
        if self.prediction_type == 'categorical':
            num_classes = []
            for _label in np.arange(0, self.num_classes):
                filenames_filter = self.filenames[self.labels == _label]
                num_classes.append(len(filenames_filter))
            num_classes = np.array(num_classes)
            max_num = max(num_classes)

            # from collections import Counter
            # hist = Counter(self.labels)
            # hist = dict(hist.most_common(7))
            # hist_np = []
            # max_num = 0
            # for item in hist:
            #     hist_np.append(hist[item])
            #
            #     if max_num < hist[item]:
            #         max_num = hist[item]
            # num_classes = np.array(hist_np)

            num_classes_to_add = max_num - num_classes
            class_id = 0
            for size in num_classes_to_add:
                filenames_class = self.filenames[self.labels == class_id]
                current_size = num_classes[class_id]
                if current_size > 1:
                    for i in range(size):
                        random_index = np.random.randint(0, current_size - 1)
                        self.filenames = np.append(self.filenames, copy.copy([filenames_class[random_index]]), axis=0)
                        self.labels = np.append(self.labels, [class_id], axis=0)
                del filenames_class
                del current_size

                class_id = class_id + 1
        else:
            from collections import Counter
            hist = Counter(self.labels)
            hist = dict(hist.most_common(7))
            hist_np = []
            max_num = 0
            for item in hist:
                hist_np.append(hist[item])

                if max_num < hist[item]:
                    max_num = hist[item]
            hist_np = np.array(hist_np)
            for item in hist:
                filenames_class = self.filenames[self.labels == item]
                current_size = hist[item]
                if current_size > 1:
                    size = max_num - current_size
                    for i in range(size):
                        random_index = np.random.randint(0, current_size - 1)
                        self.filenames = np.append(self.filenames, copy.copy([filenames_class[random_index]]), axis=0)
                        self.labels = np.append(self.labels, [item], axis=0)
                del filenames_class
                del current_size

    def preprocess(self):
        self.filenames = []
        self.labels = []

        for _, row in self._data.iterrows():
            label = row['label']
            max_length = row['duration_frames']
            filename = row['filename']

            frame_pos = 0
            if self.chunk_length > 0:
                while frame_pos + self.chunk_length <= max_length:
                    self.filenames.append([self.root_dir + filename, str(frame_pos)])
                    self.labels.append(self.get_class_id(label))

                    frame_pos = frame_pos + self.chunk_hop_length
            else:
                self.filenames.append([self.root_dir + filename, str(frame_pos)])
                self.labels.append(self.get_class_id(label))

        self.filenames = np.stack(self.filenames)
        self.labels = np.array(self.labels)

    def get_filenames(self):
        if self.filenames is None:
            raise AssertionError("Run preprocess() first to create a list of filenames")
        return self.filenames

    def get_labels(self):
        if self.labels is None:
            raise AssertionError("Run preprocess() first to create a list of labels")
        return self.labels

    def init_file_cache(self, cache_dir):
        timestamp = str(int(time.time())) + '_' + str(self._run_id)
        cache_dir_run = cache_dir + '/lite_audio_net/' + timestamp + '/'
        os.makedirs(cache_dir_run, exist_ok=True)
        self._cache_dir = cache_dir_run

    def destroy_cache(self):
        if self._cache_dir is not None:
            shutil.rmtree(self._cache_dir, ignore_errors=True)

    def pipeline(self, shuffle=True, cache=True, drop_remainder=True):
        if self.filenames is None:
            self.preprocess()

        # batch size used for computing spectrograms (speed acceleration)
        preprocessing_batch_size = self._batch_size*2

        dataset = tf.data.Dataset.from_tensor_slices((self.get_filenames(), self.get_labels()))

        dataset = dataset.map(self.read_file_function, num_parallel_calls=tf.data.AUTOTUNE)
        if self.chunk_length < 0:
            # batching > 1 does not work when input shapes are different
            dataset = dataset.batch(1)
        else:
            # create batches to improve speed
            dataset = dataset.batch(preprocessing_batch_size, drop_remainder=drop_remainder)

        dataset = dataset.map(self.generate_spectrogram, num_parallel_calls=tf.data.AUTOTUNE)

        dataset = dataset.unbatch()

        if cache:
            # import tempfile
            # cache_dir = tempfile.gettempdir()
            # self.init_file_cache(cache_dir=cache_dir)
            # dataset = dataset.cache(filename=self._cache_dir + 'pipeline_cache')
            dataset = dataset.cache()

        # important: call shuffle after caching
        if shuffle:
            dataset = dataset.shuffle(buffer_size=10000, seed=42)

        dataset = dataset.batch(self._batch_size, drop_remainder=drop_remainder)

        if self.enable_gpu:
            dataset = dataset.apply(tf.data.experimental.copy_to_device('gpu'))
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset

    def read_file_function(self, file_data, label): # pragma: no cover
        with tf.name_scope("read_file"):
            filename = file_data[0]
            segment = tf.strings.to_number(file_data[1], out_type=tf.dtypes.int32, name="segment_to_number")
            if self.prediction_type == 'categorical':
                label_one_hot = tf.one_hot(tf.cast(label, tf.int32), self.num_classes)
                label_one_hot = tf.cast(label_one_hot, tf.float32)
            else:
                label_one_hot = tf.strings.to_number(label, out_type=tf.dtypes.float32, name="label_to_number")
            audio_string = tf.io.read_file(filename)  # only on cpu possible

            decoded_audio, _ = tf.audio.decode_wav(audio_string, desired_channels=1)

            decoded_audio = tf.reshape(decoded_audio, [-1])

            if self.chunk_length > 0:

                check_op = tf.Assert(tf.greater_equal(tf.shape(decoded_audio)[0], segment + self.chunk_length),
                                     ["Could not parse wav audio file ", filename, tf.shape(decoded_audio), segment])
                with tf.control_dependencies([check_op]):
                    decoded_audio = tf.slice(decoded_audio, (segment,), (self.chunk_length,),
                                             name="decoded_audio_slice")

                    if 'normalize_audio' in self.hparams and self.hparams['normalize_audio']:
                        decoded_audio = decoded_audio * (0.7079 / tf.reduce_max(tf.abs(decoded_audio)))
                    # max_v = tf.int16.max
                    # decoded_audio = tf.cast(decoded_audio * max_v, tf.int16, name="decoded_audio_final_cast")

                    return decoded_audio, label_one_hot
            else:
                if 'normalize_audio' in self.hparams and self.hparams['normalize_audio']:
                    decoded_audio = decoded_audio * (0.7079 / tf.reduce_max(tf.abs(decoded_audio)))
                # max_v = tf.int16.max
                # decoded_audio = tf.cast(decoded_audio * max_v, tf.int16, name="decoded_audio_final_cast")

                return decoded_audio, label_one_hot

    def generate_spectrogram(self, decoded_audio, label): # pragma: no cover
        with tf.name_scope("generate_spectrogram"):
            with tf.device(self.device):

                def _preprocess(x):
                    # values in the range [0, 255] are expected!!
                    model_key = self.hparams['basemodel_name']

                    if model_key in self.preprocessors:
                        return self.preprocessors[model_key](x, data_format='channels_last')

                    return x

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

                image_data = []
                if self.hparams['use_plot_images']:

                    color_map = tf.Variable(initial_value=self.colormap.get_color_map(), name="color_map")

                    image_data = create_map_from_array(output, color_map=color_map)

                    image_data = tf.image.resize(
                        image_data, (self.hparams['image_width'], self.hparams['image_height']),
                        method=self.resize_method, preserve_aspect_ratio=False, antialias=self.anti_alias
                    )

                    image_data = image_data * 255.
                    image_data = tf.clip_by_value(image_data, clip_value_min=0., clip_value_max=255.)
                    image_data = tf.image.rot90(image_data, k=1)

                    # values in the range [0, 255] are expected!!
                    image_data = _preprocess(image_data)

                else:
                    image_data = tf.expand_dims(output, axis=3)

                return image_data, label


def preprocess_scalar_zero(model_name: str): # pragma: no cover
    available_modes = {
        "resnet50": 'caffee',
        "xception": 'tf',
        "inception_v3": 'tf',
        "densenet121": 'torch',
        "densenet169": 'torch',
        "densenet201": 'torch',
        "mobilenet": 'tf',
        "mobilenet_v2": 'tf',
        "nasnet_large": 'tf',
        "nasnet_mobile": 'tf',
        "inception_resnet_v2": 'tf',
        "squeezenet_v1": 'caffee'
    }
    mode = None
    if model_name in available_modes:
        mode = available_modes[model_name]

    if mode is None:
        return [0.5, 0.5, 0.5]
    elif mode == 'tf':
        return [0., 0., 0.]
    elif mode == 'torch':
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        mean = [103.939, 116.779, 123.68]
        std = None

    # return [-mean[0]/std[0], -mean[1]/std[1], -mean[2]/std[2]]
    return mean
