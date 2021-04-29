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
from tensorflow import keras
import tensorflow as tf
from tensorflow.python.keras.metrics import categorical_accuracy
from deepspectrumlite import Model
from .modules.augmentable_model import AugmentableModel
from .modules.squeeze_net import SqueezeNet
import logging

log = logging.getLogger(__name__)

class TransferBaseModel(Model):
    base_model = None

    def retrain_model(self):
        if self.hy_params['finetune_layer'] > 0:
            self.base_model.trainable = True

            layer_count = len(self.base_model.layers)
            keep_until = int(layer_count * self.hy_params['finetune_layer'])

            for layer in self.base_model.layers[:keep_until]:
                layer.trainable = False

            optimizer = self.get_optimizer_fn()
            optimizer._set_hyper('learning_rate', self.hy_params['fine_learning_rate'])
            self.get_model().compile(loss=self.hy_params['loss'], optimizer=optimizer,
                                     metrics=self._metrics)

        self.get_model().summary(print_fn=log.info)

    def create_model(self):
        hy_params = self.hy_params

        input = keras.Input(shape=self.input_shape, dtype=tf.float32)

        weights = None
        if hy_params['weights'] != '':
            weights = hy_params['weights']

        available_models = {
            "vgg16":
                tf.keras.applications.vgg16.VGG16,
            "vgg19":
                tf.keras.applications.vgg19.VGG19,
            "resnet50":
                tf.keras.applications.resnet50.ResNet50,
            "xception":
                tf.keras.applications.xception.Xception,
            "inception_v3":
                tf.keras.applications.inception_v3,
            "densenet121":
                tf.keras.applications.densenet.DenseNet121,
            "densenet169":
                tf.keras.applications.densenet.DenseNet169,
            "densenet201":
                tf.keras.applications.densenet.DenseNet201,
            "mobilenet":
                tf.keras.applications.mobilenet.MobileNet,
            "mobilenet_v2":
                tf.keras.applications.mobilenet_v2.MobileNetV2,
            "nasnet_large":
                tf.keras.applications.nasnet.NASNetLarge,
            "nasnet_mobile":
                tf.keras.applications.nasnet.NASNetMobile,
            "inception_resnet_v2":
                tf.keras.applications.inception_resnet_v2.InceptionResNetV2,
            "squeezenet_v1":
                SqueezeNet,
        }

        model_key = hy_params['basemodel_name']

        if model_key in available_models:
            self.base_model = available_models[model_key](weights=weights, include_top=False)
        else:
            raise ValueError(model_key + ' is not implemented')

        if hy_params['weights'] != '':
            training = False
        else:
            training = True

        self.base_model.trainable = training

        feature_batch_average = tf.keras.layers.GlobalAveragePooling2D()(self.base_model(input, training=training))
        flatten = keras.layers.Flatten()(feature_batch_average)
        dense_1 = keras.layers.Dense(hy_params['num_units'], activation=self.get_activation_fn())(flatten)
        dropout_1 = keras.layers.Dropout(rate=hy_params['dropout'])(dense_1)

        activation = 'softmax'
        if 'output_activation' in hy_params:
            activation = hy_params['output_activation']
        predictions = tf.keras.layers.Dense(len(self.data_classes), activation=activation)(dropout_1)

        model = AugmentableModel(inputs=input, outputs=predictions, name=hy_params['basemodel_name'])
        model.set_hyper_parameters(hy_params=hy_params)
        self.model = model
        self.compile_model()

    def train(self, train_dataset: tf.data.Dataset, devel_dataset: tf.data.Dataset):
        """
        trains the model with given train data.
        """
        epochs_first = self.hy_params['pre_epochs']
        if self.hy_params['weights'] == '':
            epochs_first = self.hy_params['epochs'] + self.hy_params['pre_epochs']

        history = self.get_model().fit(x=train_dataset, epochs=epochs_first,
                                       batch_size=self.hy_params['batch_size'],
                                       shuffle=True,
                                       validation_data=devel_dataset,
                                       callbacks=self.get_callbacks(), verbose=0)

        if self.hy_params['weights'] != '' and self.hy_params['finetune_layer'] > 0:
            self.retrain_model()

            self.get_model().fit(x=train_dataset, epochs=epochs_first + self.hy_params['epochs'],
                                 initial_epoch=history.epoch[-1],
                                 batch_size=self.hy_params['batch_size'],
                                 shuffle=True,
                                 validation_data=devel_dataset,
                                 callbacks=self.get_callbacks(), verbose=0)
