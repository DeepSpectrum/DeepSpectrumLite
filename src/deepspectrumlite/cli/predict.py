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
import logging
import click
from .utils import add_options
import os
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
import importlib
import json
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report, confusion_matrix
import csv
import json
import pandas as pd
import librosa
from collections import Counter
from os.path import join, dirname, realpath

log = logging.getLogger(__name__)

_DESCRIPTION = 'Predict a file using an existing DeepSpectrumLite transer learning model.'

@add_options(
[
    click.option(
        "-d",
        "--data-dir",
        type=click.Path(exists=True),
        help="Directory of data class categories containing folders of each data class.",
        required=True
    ),
    click.option(
        "-md",
        "--model-dir",
        type=click.Path(exists=True, writable=False),
        help="HD5 file of the DeepSpectrumLite model",
        required=True
    ),
    click.option(
        "-hc",
        "--hyper-config",
        type=click.Path(exists=True, writable=False, readable=True),
        help="Directory for the hyper parameter config file.",
        default=join(dirname(realpath(__file__)), "config/hp_config.json"), show_default=True
    ),
    click.option(
        "-cc",
        "--class-config",
        type=click.Path(exists=True, writable=False, readable=True),
        help="Directory for the class config file.",
        default=join(dirname(realpath(__file__)), "config/class_config.json"), show_default=True
    )
]
)

@click.command(help=_DESCRIPTION)
@click.pass_context
def predict(ctx, model_dir, data_dir, class_config, hyper_config, **kwargs):
    verbose = ctx.obj['verbose']
    f = open(class_config)
    data = json.load(f)
    f.close()

    data_dir = os.path.join(data_dir, '')

    data_classes = data
    wav_files = sorted(glob.glob(f'{data_dir}/**/*.wav', recursive=True))
    filenames, labels, duration_frames = list(map(lambda x: os.path.relpath(x, start=data_dir), wav_files)), [list(data_classes.keys())[0]]*len(wav_files), []
    for fn in filenames:
        y, sr = librosa.load(os.path.join(data_dir, fn), sr=None)
        duration_frames.append(y.shape[0])

    log.info('Found %d wav files' % len(filenames))

    if data_classes is None:
        raise ValueError('no data classes defined')

    class_list = {}
    for i, data_class in enumerate(data_classes):
        class_list[data_class] = i

    hyper_parameter_list = HyperParameterList(config_file_name=hyper_config)
    log.info("Search within rule: " + model_dir)
    model_dir_list = glob.glob(model_dir)
    log.info("Found "+ str(len(model_dir_list)) + " files")

    for model_filename in model_dir_list:
        log.info("Load " + model_filename)
        p = Path(model_filename)
        parent = p.parent
        directory = parent.name

        result_dir = os.path.join(parent, "test")
        iteration_no = int(directory.split("_")[-1])

        log.info('--- Testing trial: %s' % iteration_no)
        hparam_values = hyper_parameter_list.get_values(iteration_no=iteration_no)
        log.info(hparam_values)

        test_data = pd.DataFrame({'filename': filenames, 'label': labels, 'duration_frames': duration_frames})

        print("Loading model: " + model_filename)
        model = tf.keras.models.load_model(model_filename,
                                           custom_objects={'AugmentableModel': AugmentableModel, 'ARelu': ARelu},
                                           compile=False)
        model.set_hyper_parameters(hparam_values)
        log.info("Successfully loaded model: " + model_filename)

        data_raw = test_data # [:10]
        dataset_name = 'test'

        dataset_result_dir = os.path.join(result_dir, dataset_name)

        os.makedirs(dataset_result_dir, exist_ok=True)

        data_pipeline = DataPipeline(name=dataset_name+'_data_set', data_classes=data_classes,
                                            enable_gpu=True, verbose=True, enable_augmentation=False,
                                            hparams=hparam_values, run_id=iteration_no)
        data_pipeline.set_data(data_raw)
        data_pipeline.set_filename_prepend(prepend_filename_str=data_dir)
        data_pipeline.preprocess()
        filename_list = data_pipeline.filenames
        dataset = data_pipeline.pipeline(cache=False, shuffle=False, drop_remainder=False)

        X_probs = model.predict(x=dataset, verbose=verbose)
        true_categories = tf.concat([y for x, y in dataset], axis=0)
        X_pred = tf.argmax(X_probs, axis=1)
        X_pred_ny = X_pred.numpy()


        target_names = []
        for data_class in data_classes:
            target_names.append(data_class)

        df = pd.DataFrame(data=filename_list[...,0], columns=["filename"])

        df['filename'] = df['filename'].apply(lambda x: os.path.basename(x))
        df['time'] = list(map(lambda x: int(x)/sr, filename_list[...,1]))
        for i, target in enumerate(target_names):
            df[f'prob_{target}'] = X_probs[:, i]
        df['prediction'] = list(map(lambda x: target_names[x], X_pred))

        df.to_csv(os.path.join(dataset_result_dir, dataset_name+".chunks.predictions.csv"), index=False)

        log.info("Finished testing")



