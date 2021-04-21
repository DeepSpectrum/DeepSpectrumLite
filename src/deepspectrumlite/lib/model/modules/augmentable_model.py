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
from tensorflow.python.eager import backprop
from tensorflow.python.keras.engine import data_adapter

from deepspectrumlite.lib.data.data_pipeline import preprocess_scalar_zero


'''
AugmentableModel implements a SapAugment data augmentation policy

Hu, Ting-yao et al. "SapAugment: Learning A Sample Adaptive Policy for Data Augmentation" (2020).
@see https://arxiv.org/abs/2011.01156
'''
class AugmentableModel(tf.keras.Model):

    def __init__(self, *args, **kwargs):
        super(AugmentableModel, self).__init__(*args, **kwargs)
        self.hy_params = {}
        self._batch_size = None
        self.lambda_sap = None
        self._sap_augment_a = None
        self._sap_augment_s = None

    def set_hyper_parameters(self, hy_params):
        self.hy_params = hy_params

        self._batch_size = self.hy_params['batch_size']

        self.lambda_sap = tf.zeros(shape=(self._batch_size,), dtype=tf.float32, name="lamda_sap")

        self._sap_augment_a = self.hy_params['sap_aug_a']
        self._sap_augment_s = self.hy_params['sap_aug_s']

    def set_batch_size(self, batch_size):
        self._batch_size = batch_size

    def set_sap_augment(self, a, s):
        self._sap_augment_a = a
        self._sap_augment_s = s

    def train_step(self, data):
        data = data_adapter.expand_1d(data)
        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)

        with backprop.GradientTape() as tape:
            y_pred = self(x, training=False)
            loss_list = []
            for i in range(self._batch_size):
                # TODO sample_weight None
                instance_loss = self.compiled_loss(
                    y[i,:], y_pred[i, :], None, regularization_losses=self.losses)
                loss_list.append(instance_loss)

            sorted_keys = tf.argsort(loss_list, axis=-1, direction='ASCENDING')

            a = self._sap_augment_a
            s = self._sap_augment_s

            alpha = s * (1 - a)
            beta = s * a
            new_lambda = tf.zeros(shape=(self._batch_size, 1))

            for i in range(self._batch_size):
                ranking_relative = (i + 1) / self._batch_size
                beta_inc = 1 - tf.math.betainc(alpha, beta, ranking_relative)  # D = 1 -> 0

                j = sorted_keys[i]
                p1 = tf.zeros(shape=(j, 1))
                p2 = tf.zeros(shape=(self._batch_size-1-j, 1))
                pf = tf.concat([p1, [(beta_inc,)]], 0)
                pf = tf.concat([pf, p2], 0)
                new_lambda = new_lambda + pf

            self.lambda_sap = new_lambda

            # import numpy as np
            #
            # for i in range(self._batch_size):
            #     image = x[i]
            #     image = np.array(image)
            #     lambda_value = self.lambda_sap[i][0].numpy()
            #     loss_value = loss_list[i].numpy()
            #     path = '/Users/tobias/Downloads/debug/' + str(i) + '-' + str(
            #         lambda_value) + '-' + str(
            #         loss_value) + '-orig.png'
            #     tf.keras.preprocessing.image.save_img(
            #         path=path, x=image, data_format='channels_last', scale=True
            #     )

            # tf.print("x=")
            # tf.print(tf.cast(tf.round(tf.squeeze(x[0])), tf.int32), summarize=-1, sep=",")
            if self.hy_params['augment_cutmix']:
                x, y = self.cutmix(x, y)

            # for i in range(self._batch_size):
            #     image = x[i]
            #     image = np.array(image)
            #     lambda_value = self.lambda_sap[i][0].numpy()
            #     loss_value = loss_list[i].numpy()
            #     path = '/Users/tobias/Downloads/debug/' + str(i) + '-' + str(
            #         lambda_value) + '-' + str(
            #         loss_value) + '-cutmix.png'
            #     tf.keras.preprocessing.image.save_img(
            #         path=path, x=image, data_format='channels_last', scale=True
            #     )

            # tf.print("cutmix=")
            # tf.print(tf.cast(tf.round(tf.squeeze(x[0])), tf.int32), summarize=-1, sep=",")
            if self.hy_params['augment_specaug']:
                x, y = self.apply_spec_aug(x, y)
            # tf.print("spec_aug=")
            # tf.print(tf.cast(tf.round(tf.squeeze(x[0])), tf.int32), summarize=-1, sep=",")
            # self.stop_training = True

            # for i in range(self._batch_size):
            #     image = x[i]
            #     image = np.array(image)
            #     lambda_value = self.lambda_sap[i][0].numpy()
            #     loss_value = loss_list[i].numpy()
            #     path = '/Users/tobias/Downloads/debug/' + str(i) + '-' + str(
            #         lambda_value) + '-' + str(
            #         loss_value) + '-both.png'
            #     tf.keras.preprocessing.image.save_img(
            #         path=path, x=image, data_format='channels_last', scale=True
            #     )
            # self.stop_training = True

            #tf.print("x_NEW=")
            #tf.print(x[0])

            y_pred = self(x, training=True)  # batch size, softmax output
            loss = self.compiled_loss(
                y, y_pred, sample_weight, regularization_losses=self.losses)

        self.optimizer.minimize(loss, self.trainable_variables, tape=tape)
        self.compiled_metrics.update_state(y, y_pred, sample_weight)
        return {m.name: m.result() for m in self.metrics}

    def cutmix(self, images, labels):
        """
        Implements cutmix data augmentation
        Yun, Sangdoo et al. "CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features" (2019).

        @param images: batch of images
        @param labels: batch of labels  (one hot encoded)
        @return: images, labels
        @see https://arxiv.org/abs/1905.04899
        """
        spectrogram_x = self.input_shape[1]
        spectrogram_y = self.input_shape[2]

        min_augment = self.hy_params['cutmix_min']
        max_augment = self.hy_params['cutmix_max']-self.hy_params['cutmix_min']

        probability_min = self.hy_params['da_prob_min']
        probability_max = self.hy_params['da_prob_max']

        image_list = []
        label_list = []
        for j in range(self._batch_size):
            probability_threshold = probability_min + probability_max * tf.squeeze(self.lambda_sap[j])
            P = tf.cast(tf.random.uniform([], 0, 1) <= probability_threshold, tf.int32)

            # works as we have square images only
            w = tf.cast(tf.round(min_augment * spectrogram_x + max_augment * spectrogram_x * tf.squeeze(self.lambda_sap[j])), tf.int32) * P

            k = tf.cast(tf.random.uniform([], 0, self._batch_size), tf.int32)
            x = tf.cast(tf.random.uniform([], 0, spectrogram_x), tf.int32)
            y = tf.cast(tf.random.uniform([], 0, spectrogram_y), tf.int32)

            xa = tf.math.maximum(0, x - w // 2)  # xa denotes the start
            xb = tf.math.minimum(spectrogram_x, x + w // 2)  # xb denotes the end

            ya = tf.math.maximum(0, y - w // 2)  # xa denotes the start
            yb = tf.math.minimum(spectrogram_y, y + w // 2)  # xb denotes the end

            piece = tf.concat([
                images[j, xa:xb, 0:ya, ],
                images[k, xa:xb, ya:yb, ],
                images[j, xa:xb, yb:spectrogram_y, ]
            ], axis=1)

            image = tf.concat([images[j, 0:xa, :, ],
                               piece,
                               images[j, xb:spectrogram_x, :, ]], axis=0)
            image_list.append(image)

            a = tf.cast((w**2) / (spectrogram_x*spectrogram_y), tf.float32)

            label_1 = labels[j,]
            label_2 = labels[k,]
            label_list.append((1 - a) * label_1 + a * label_2)


        x = tf.reshape(tf.stack(image_list), (self._batch_size, spectrogram_x, spectrogram_y, self.input_shape[3]))
        y = tf.reshape(tf.stack(label_list), (self._batch_size, self.output_shape[1]))

        return x, y

    def apply_spec_aug(self, images, labels):
        """
        Implements SpecAugment
        Park, Daniel et al. "SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition" (2019).

        @param images: batch of images
        @param labels: batch of labels
        @return: images, labels
        @see https://arxiv.org/abs/1904.08779
        """
        with tf.name_scope("apply_spec_aug"):
            n = self.input_shape[1]  # x = time
            v = self.input_shape[2]  # y = freq

            frequency_mask_num = self.hy_params['specaug_freq_mask_num']
            time_mask_num = self.hy_params['specaug_time_mask_num']

            freq_min_augment = self.hy_params['specaug_freq_min']
            freq_max_augment = self.hy_params['specaug_freq_max']-self.hy_params['specaug_freq_min']
            time_min_augment = self.hy_params['specaug_time_min']
            time_max_augment = self.hy_params['specaug_time_max']-self.hy_params['specaug_time_min']

            probability_min = self.hy_params['da_prob_min']
            probability_max = self.hy_params['da_prob_max']

            output_mel_spectrogram = []

            for j in range(self._batch_size):
                probability_threshold = probability_min + probability_max * tf.squeeze(self.lambda_sap[j])
                P = tf.cast(tf.random.uniform([], 0, 1) <= probability_threshold, tf.int32)

                f = tf.cast(tf.round((freq_min_augment * v + freq_max_augment * v * tf.squeeze(self.lambda_sap[j])) / frequency_mask_num), tf.int32) * P
                t = tf.cast(tf.round((time_min_augment * n + time_max_augment * n * tf.squeeze(self.lambda_sap[j])) / time_mask_num), tf.int32) * P

                tmp = images

                for i in range(frequency_mask_num):
                    f0 = tf.random.uniform([], minval=0, maxval=v - f, dtype=tf.int32)

                    mask = tf.concat((tf.ones(shape=(1, n, v - f0 - f, 1)),
                                      tf.zeros(shape=(1, n, f, 1)),
                                      tf.ones(shape=(1, n, f0, 1)),
                                      ), 2)
                    tmp = tmp * mask

                for i in range(time_mask_num):
                    t0 = tf.random.uniform([], minval=0, maxval=n - t, dtype=tf.int32)

                    mask = tf.concat((tf.ones(shape=(1, n - t0 - t, v, 1)),
                                      tf.zeros(shape=(1, t, v, 1)),
                                      tf.ones(shape=(1, t0, v, 1)),
                                      ), 1)
                    tmp = tmp * mask

                output_mel_spectrogram.append(tmp[j])

            images = tf.stack(output_mel_spectrogram)
            #tf.print(mel_spectrogram[0], summarize=-1, sep=",")
            # replace zero values by the mean
            preprocessed_mask = preprocess_scalar_zero(self.hy_params['basemodel_name'])
            images = tf.where(images == 0.0, preprocessed_mask, images)

            #tf.print(mel_spectrogram[0], summarize=-1, sep=",")
            #self.stop_training = True
            return images, labels