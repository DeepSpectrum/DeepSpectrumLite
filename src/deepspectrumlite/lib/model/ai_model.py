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
import shutil
import os
from tensorboard.plugins.hparams import api as hp
import numpy as np
from tensorflow.keras import backend as K
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from tensorflow.python.keras.metrics import categorical_accuracy
from deepspectrumlite import HyperParameterList
from .modules.arelu import ARelu
from .config import gridsearch
import logging

log = logging.getLogger(__name__)

class Model:
    model: tf.keras.Model = None

    def __init__(self,
                 hy_params: HyperParameterList,
                 input_shape: tuple,
                 data_classes,
                 run_id: int,
                 run_dir: str = None,
                 use_ram: bool = True,
                 verbose: int = 0
                 ):
        """
        Abstract model implementation
        Args:
            hy_params: HyperParameterList
                parameters for gridsearch
            input_shape: tuple
                size of input of the model
            run_dir: str (optional)
                log directory of tensorboard Default: None
            use_ram: bool (optional)
                If enabled, the whole train data set will be saved in memory.
                Otherwise only the current batch will be loaded to memory. Default: Truek
            verbose: int (optional)
                Verbosity for keras training and evaluation calls. Default: 0
        """
        self._run_id = run_id
        self.hy_params = hy_params.get_values(iteration_no=self._run_id)
        self.hy_params_tb = hy_params.get_values_tensorboard(iteration_no=self._run_id)
        self.use_ram = use_ram
        self.input_shape = input_shape

        # convert to keras verbosities
        self.verbose = 1 if verbose > 1 else 2 if verbose == 1 else verbose
        self.confusion_matrix = None
        self.run_dir = run_dir
        self.data_classes = data_classes

        self.prediction_type = 'categorical'
        if 'prediction_type' in self.hy_params:
            self.prediction_type = self.hy_params['prediction_type']

        if self.prediction_type == 'categorical':
            self._metrics = [keras.metrics.Precision(name="precision"),
                             keras.metrics.Recall(name="recall"),
                             categorical_accuracy]
            for i in range(len(self.data_classes)):
                self._metrics.append(keras.metrics.Recall(name="recall_class_" + str(i), class_id=i))
            for i in range(len(self.data_classes)):
                self._metrics.append(keras.metrics.Precision(name="precision_class_" + str(i), class_id=i))
        elif self.prediction_type == 'regression':
            self._metrics = [keras.metrics.MeanAbsoluteError(name="mae"),
                             keras.metrics.RootMeanSquaredError(name="rmse"),
                             keras.metrics.MeanSquaredError(name="mse")]
        else:
            raise ValueError('prediction_type "' + self.prediction_type + '" not implemented')

    def get_callbacks(self):
        return [keras.callbacks.TensorBoard(log_dir=self.run_dir),  # , histogram_freq=10
                hp.KerasCallback(self.run_dir, self.hy_params_tb)]

    def train(self, train_dataset: tf.data.Dataset, devel_dataset: tf.data.Dataset):
        """
        trains the model with given train data.
        """
        self.get_model().fit(x=train_dataset, epochs=self.hy_params['epochs'],
                             batch_size=self.hy_params['batch_size'],
                             shuffle=True,
                             validation_data=devel_dataset,
                             callbacks=self.get_callbacks(), verbose=self.verbose)

    '''
    def test_grouped(self, test_data_grouped):
        """
        operated a grouped test of the given model based on given test data. The difference between a "normal" test
        and a grouped test lies in the metrics. In a grouped test, one (1) whole wav file will be examined
        (i.e. one wav is either predicted truefully or not).
        In a normal test, each wav file will be chunked and every chunk will be individually examined
        (i.e. one wav could have several predictions).
        Args:
            test_data_grouped: np.ndarray of the data to test TODO explain structure of input

        Returns:
            accuracy
            precision
            recall
            fbeta_score

        """
        accuracy_total = precision_total = recall_total = 0.0
        wav_files_count = 0

        global_y_true = []
        global_y_pred = []

        for data_class in test_data_grouped:
            for test_bucket_list in test_data_grouped[data_class]:
                wav_files_count = wav_files_count + 1
                #  if wav_files_count >= 15:
                #      break
                chunk_count = np.shape(test_bucket_list)[0]
                labels = []
                label_name = 0
                y_true = []

                i = 0
                for data_class_validate in self.data_classes:
                    if data_class_validate == data_class:
                        label_name = i
                        y_true = np.zeros((len(self.data_classes)))
                        y_true[i] = 1
                        break

                    i = i + 1

                for i in range(chunk_count):
                    labels.append(label_name)

                test_labels = np.array(keras.utils.to_categorical(y=labels, num_classes=len(self.data_classes)))
                test_data = np.stack(test_bucket_list)

                # print("\n\n")

                results = self.get_model().evaluate(x=test_data, y=test_labels, verbose=0)
                pred = self.get_model().predict(x=test_data)

                # print(results)

                accuracy_total = accuracy_total + results[3]

                #   print(pred)

                y_pred = y_pred_r = []

                for prediction in pred:
                    y_pred.append(np.argmax(prediction, axis=-1))

                #   print(y_pred, y_true, most_frequent(y_pred), y_true[0])

                predicted_class = most_frequent(y_pred)
                y_pred_r = np.zeros(len(self.data_classes))
                y_pred_r[predicted_class] = 1

                global_y_true.append(y_true)
                global_y_pred.append(y_pred_r)

                # sys.exit()

        accuracy_total = accuracy_total / wav_files_count

        # print("accuracy_total", accuracy_total)
        # sys.exit()

        # print(global_y_true, global_y_pred)
        accuracy = accuracy_total

        cm = confusion_matrix(test_labels, pred)
        # confusion_matrix = self.create_confusion_matrix(labels=global_y_true, predictions=global_y_pred)
        precision, recall, fbeta_score, support = precision_recall_fscore_support(y_true=global_y_true,
                                                                                  y_pred=global_y_pred, average='macro')
        if self.verbose:
            print("confusion_matrix\n", cm)

            target_names = []
            for data_class in self.data_classes:
                target_names.append(data_class)

            print(classification_report(y_true=global_y_true, y_pred=global_y_pred, target_names=target_names,
                                        digits=4))
            print("precision, recall, fbeta_score, support", precision, recall, fbeta_score)

        # del test_data_grouped
        return accuracy, precision, recall, fbeta_score
    '''

    def test(self, test_dataset):
        """
        tests the trained model against given test data
        Args:
            test_dataset:

        Returns:
            accuracy
            precision
            recall
        """
        results = self.get_model().evaluate(x=test_dataset,
                                            batch_size=self.hy_params['batch_size'],
                                            verbose=self.verbose)

        if self.prediction_type == 'categorical':
            X_pred = self.get_model().predict(x=test_dataset)
            true_categories = tf.concat([y for x, y in test_dataset], axis=0)

            X_pred = tf.argmax(X_pred, axis=1)

            true_categories = tf.argmax(true_categories, axis=1)
            cm = tf.math.confusion_matrix(true_categories, X_pred)

            precision, recall, fbeta_score, support = precision_recall_fscore_support(y_true=true_categories.numpy(),
                                                                                      y_pred=X_pred.numpy(),
                                                                                      average='macro', zero_division=0)

            log.info("Confusion matrix:")
            log.info(cm)
            target_names = []
            for data_class in self.data_classes:
                target_names.append(data_class)

            log.info("\n" + classification_report(y_true=true_categories.numpy(), y_pred=X_pred.numpy(),
                                        target_names=target_names,
                                        digits=4, zero_division=0))
            log.info(f"precision: %.5f recall: %.5f fbeta_score: %.5f", precision, recall, fbeta_score)

            accuracy = results[2]
            return accuracy, precision, recall, fbeta_score
        else:
            # loss, mae, rmse, mse
            return results[0], results[1], results[2], results[2]

    def get_model(self) -> tf.keras.Model:
        """

        Returns:
            tf.keras.Model current model

        """
        return self.model

    def save_tl_lite_model(self, save_dir: str, model_name: str):
        """
        Saves the current model data in saved_model format
        Args:
            save_dir: str
                directory to save the saved_model model
            model_name: str
                name of the model
        """

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        converter = tf.lite.TFLiteConverter.from_keras_model(self.get_model())
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
                                               tf.lite.OpsSet.SELECT_TF_OPS]
        converter.experimental_new_converter = True
        tflite_quant_model = converter.convert()
        open(save_dir + model_name + ".tflite", "wb").write(tflite_quant_model)

        log.info("Model was saved as tflite as " + save_dir + model_name)

        interpreter = tf.lite.Interpreter(model_content=tflite_quant_model)

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        interpreter.allocate_tensors()

        # interpreter.set_tensor(input_details[0]['index'], tf.convert_to_tensor(np.expand_dims(audio_data, 0), dtype=tf.float32))

        interpreter.invoke()

        output = interpreter.get_tensor(output_details[0]['index'])

        # Test model on random input data.
        input_shape = input_details[0]['shape']
        log.info("input shape: ", input_shape)
        log.info("output shape: ", output_details[0]['shape'])
        input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
        interpreter.set_tensor(input_details[0]['index'], input_data)

        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])

        log.info(output_data)

    def save_model_saved_model(self, save_dir: str):
        """
        Saves the current model data in saved_model format
        Args:
            save_dir: directory to save the saved_model model
        """
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        #    keras.experimental.export_saved_model(model=self.get_model(), saved_model_path=save_dir) deprecated

        self.get_model().save(save_dir + "/model.h5", save_format="h5")

    def save_keras_model(self, save_dir: str, filename: str):
        self.get_model().save(save_dir + filename)
        log.info("Model was saved in " + save_dir + filename)

    def create_model(self):
        """
        creates a plain model
        """
        pass

    def get_activation_fn(self):
        if self.hy_params['activation'] == 'arelu':
            return ARelu
        else:
            return self.hy_params['activation']

    def get_optimizer_fn(self):
        if self.hy_params['optimizer'] == 'adadelta':
            return keras.optimizers.Adadelta(learning_rate=self.hy_params['learning_rate'])
        elif self.hy_params['optimizer'] == 'adam':
            return keras.optimizers.Adam(learning_rate=self.hy_params['learning_rate'])
        elif self.hy_params['optimizer'] == 'sgd':
            return keras.optimizers.SGD(learning_rate=self.hy_params['learning_rate'])
        elif self.hy_params['optimizer'] == 'adagrad':
            return keras.optimizers.Adagrad(learning_rate=self.hy_params['learning_rate'])
        elif self.hy_params['optimizer'] == 'ftrl':
            return keras.optimizers.Ftrl(learning_rate=self.hy_params['learning_rate'])
        elif self.hy_params['optimizer'] == 'nadam':
            return keras.optimizers.Nadam(learning_rate=self.hy_params['learning_rate'])
        elif self.hy_params['optimizer'] == 'rmsprop':
            return keras.optimizers.RMSprop(learning_rate=self.hy_params['learning_rate'])
        else:
            raise ValueError('Optimizer function ' + self.hy_params['optimizer'] + ' not implemented')

    def compile_model(self):
        self.get_model().compile(optimizer=self.get_optimizer_fn(), loss=self.hy_params['loss'],
                                 metrics=self._metrics)

        self.get_model().summary(print_fn=log.info)

    def run(self, train_dataset: tf.data.Dataset, test_dataset: tf.data.Dataset,
            devel_dataset: tf.data.Dataset, save_dir: str, save_model: bool = False):
        """
        Implementation of various steps:
            1. creates the model
            2. trains the model with given train_data
            3. slurmJobs the model against given test_data
            4. saves the model into model and model_tflite directory

        All training results will be saved through tensorboard

        Args:
            train_data: np.ndarray
                train data
            train_labels: np.ndarray
                corresponding train labels (ground truth)
            test_data: np.ndarray
                test data for final validation
            devel_data: np.ndarray
                devel data for validation while training
            devel_labels: np.ndarray
                corresponding devel labels (ground truth)
        """
        with tf.summary.create_file_writer(self.run_dir).as_default():
            hp.hparams(self.hy_params_tb)

            self.create_model()
            self.train(train_dataset, devel_dataset)


            log.info("Training finished")
            if save_model:
                self.save_model_saved_model(
                    save_dir=save_dir)
                log.info("Model saved to " + save_dir)
            '''
            if grouped_test:
                accuracy, precision, recall, f1_score = self.test_grouped(test_data)
            else:
            '''
            # final test
            if self.prediction_type == 'categorical':
                accuracy, precision, recall, f1_score = self.test(test_dataset)

                tf.summary.scalar(gridsearch.METRIC_ACCURACY, accuracy, step=1)
                tf.summary.scalar(gridsearch.METRIC_PRECISION, precision, step=1)
                tf.summary.scalar(gridsearch.METRIC_RECALL, recall, step=1)
                tf.summary.scalar(gridsearch.METRIC_F_SCORE, f1_score, step=1)
            else:
                _, mae, rmse, mse = self.test(test_dataset)

                tf.summary.scalar(gridsearch.METRIC_MAE, mae, step=1)
                tf.summary.scalar(gridsearch.METRIC_RMSE, rmse, step=1)
                tf.summary.scalar(gridsearch.METRIC_MSE, mse, step=1)
