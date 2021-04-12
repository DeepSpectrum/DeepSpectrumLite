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

'''
This script converts a h5 file to a tflite file
'''
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # print only error messages
import sys
import tensorflow as tf
from tensorflow import keras
import math
from tensorflow.keras import backend as K
from tensorflow.python.saved_model import loader_impl
from deepspectrumlite import AugmentableModel, DataPipeline, HyperParameterList, ARelu
import glob
from pathlib import Path
import numpy as np
import argparse
import importlib
import json
from sklearn.metrics import recall_score, classification_report, confusion_matrix
import csv
import json
import pandas as pd
from collections import Counter


def print_version():
    print(tf.version.GIT_VERSION, tf.version.VERSION)


if __name__ == "__main__":

    print_version()

    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--data-dir', type=str, dest='data_dir',
                        default='',
                        help='Directory of data class categories containing folders of each data class'
                             ' (default: %(default)s)')
    parser.add_argument('-hc', '--hyper-config', type=str, dest='hyper_config_file',
                        default='../config/hp_config.json',
                        help='Directory for the hyper parameter config file (default: %(default)s)')

    parser.add_argument('-cc', '--class-config', type=str, dest='class_config_file',
                        default='../config/class_config.json',
                        help='Directory for the class config file (default: %(default)s)')

    parser.add_argument('-l', '--label-file', type=str, dest='label_file',
                        default='',
                        help='Directory for the label file (default: %(default)s)')

    parser.add_argument('-md', '--model-dir', type=str, dest='model_dir',
                        default='',
                        help='Directory for the model files (default: %(default)s)')

    args = parser.parse_args()
    label_file = args.label_file
    data_dir = args.data_dir

    f = open(args.class_config_file)
    data = json.load(f)
    f.close()

    data_classes = data

    if data_classes is None:
        raise ValueError('no data classes defined')

    class_list = {}
    for i, data_class in enumerate(data_classes):
        class_list[data_class] = i

    hyper_parameter_list = HyperParameterList(config_file_name=args.hyper_config_file)

    print("Search by rule: " + args.model_dir)
    model_dir_list = glob.glob(args.model_dir)
    print("Found " + str(len(model_dir_list)) + " files")

    for model_filename in model_dir_list:
        print("Load " + model_filename)
        p = Path(model_filename)
        parent = p.parent
        directory = parent.name

        result_dir = os.path.join(parent, "evaluation")

        iteration_no = int(directory.split("_")[-1])

        print('--- Testing trial: %s' % iteration_no)
        hparam_values = hyper_parameter_list.get_values(iteration_no=iteration_no)
        print(hparam_values)

        label_parser_key = hparam_values['label_parser']

        if ":" not in label_parser_key:
            raise ValueError('Please provide the parser in the following format: path.to.parser_file.py:ParserClass')

        print(f'Using custom external parser: {label_parser_key}')
        path, class_name = label_parser_key.split(':')
        module_name = os.path.splitext(os.path.basename(path))[0]
        dir_path = os.path.dirname(os.path.realpath(__file__))
        path = os.path.join(dir_path, path)
        spec = importlib.util.spec_from_file_location(module_name, path)
        foo = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(foo)
        parser_class = getattr(foo, class_name)

        parser = parser_class(file_path=label_file)
        _, devel_data, test_data = parser.parse_labels()
        print("Successfully parsed labels: " + label_file)
        model = tf.keras.models.load_model(model_filename,
                                           custom_objects={'AugmentableModel': AugmentableModel, 'ARelu': ARelu},
                                           compile=False)
        print("Successfully loaded model: " + model_filename)

        dataset_list = ["devel", "test"]

        for dataset_name in dataset_list:
            print("===== Dataset Partition: " + dataset_name)
            data_raw = []
            if dataset_name == 'devel':
                data_raw = devel_data  # [:10]
            elif dataset_name == 'test':
                data_raw = test_data  # [:10]

            dataset_result_dir = os.path.join(result_dir, dataset_name)

            os.makedirs(dataset_result_dir, exist_ok=True)

            data_pipeline = DataPipeline(name=dataset_name + '_data_set', data_classes=data_classes,
                                         enable_gpu=True, verbose=True, enable_augmentation=False,
                                         hparams=hparam_values, run_id=iteration_no)
            data_pipeline.set_data(data_raw)
            data_pipeline.set_filename_prepend(prepend_filename_str=data_dir)
            data_pipeline.preprocess()
            filename_list = data_pipeline.filenames
            dataset = data_pipeline.pipeline(cache=False, shuffle=False, drop_remainder=False)

            X_pred = model.predict(x=dataset)
            true_categories = tf.concat([y for x, y in dataset], axis=0)

            X_pred = tf.argmax(X_pred, axis=1)
            X_pred_ny = X_pred.numpy()

            true_categories = tf.argmax(true_categories, axis=1)
            true_np = true_categories.numpy()
            cm = tf.math.confusion_matrix(true_categories, X_pred)
            print("Confusion Matrix (chunks):")
            print(cm.numpy())

            target_names = []
            for data_class in data_classes:
                target_names.append(data_class)

            print(classification_report(y_true=true_categories.numpy(), y_pred=X_pred_ny,
                                        target_names=target_names,
                                        digits=4))

            recall = recall_score(y_true=true_categories.numpy(), y_pred=X_pred_ny, average='macro')
            print("UAR: " + str(recall * 100))

            json_cm_dir = os.path.join(dataset_result_dir, dataset_name + ".chunks.metrics.json")
            with open(json_cm_dir, 'w') as f:
                json.dump({"cm": cm.numpy().tolist(), "uar": round(recall * 100, 4)}, f)

            X_pred_pd = pd.DataFrame(data=X_pred_ny, columns=["prediction"])
            pd_filename_list = pd.DataFrame(data=filename_list[..., 0], columns=["filename"])

            df = pd_filename_list.join(X_pred_pd, how='outer')
            df['filename'] = df['filename'].apply(lambda x: os.path.basename(x))

            df.to_csv(os.path.join(dataset_result_dir, dataset_name + ".chunks.predictions.csv"), index=False)

            ###### grouped #######

            grouped_data = df.groupby('filename', as_index=False).agg(lambda x: Counter(x).most_common(1)[0][0])
            grouped_data.to_csv(os.path.join(dataset_result_dir, dataset_name + ".grouped.predictions.csv"),
                                index=False)
            grouped_X_pred = grouped_data.values[..., 1].tolist()

            # test
            pd_filename_list = pd.DataFrame(data=filename_list[..., 0], columns=["filename"])
            true_pd = pd.DataFrame(data=true_np, columns=["label"])
            df = pd_filename_list.join(true_pd, how='outer')
            df['filename'] = df['filename'].apply(lambda x: os.path.basename(x))
            data_raw_labels = df.groupby('filename', as_index=False).agg(lambda x: Counter(x).most_common(1)[0][0])

            # data_raw_labels = data_raw
            # data_raw_labels['label'] = data_raw_labels['label'].apply(lambda x: class_list[x])
            grouped_true = data_raw_labels.values[..., 1].tolist()
            cm = confusion_matrix(grouped_true, grouped_X_pred)
            print("Confusion Matrix (grouped):")
            print(cm)

            print(classification_report(y_true=grouped_true, y_pred=grouped_X_pred,
                                        target_names=target_names,
                                        digits=4))

            recall = recall_score(y_true=grouped_true, y_pred=grouped_X_pred, average='macro')
            print("UAR: " + str(recall * 100))

            json_cm_dir = os.path.join(dataset_result_dir, dataset_name + ".grouped.metrics.json")
            with open(json_cm_dir, 'w') as f:
                json.dump({"cm": cm.tolist(), "uar": round(recall * 100, 4)}, f)
