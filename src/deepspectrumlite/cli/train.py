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
from os import environ
from .utils import add_options
import shutil
import sys
import json
import os
from os.path import join, dirname, realpath

log = logging.getLogger(__name__)

_DESCRIPTION = 'Train a DeepSpectrumLite transer learning model.'

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
        type=click.Path(exists=False, writable=True),
        help="Directory for all training output (logs and final model files).",
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
    ),
    click.option(
        "-l",
        "--label-file",
        type=click.Path(exists=True, writable=False, readable=True),
        help="Directory for the label file.",
        required=True
    ),
    click.option(
        "-dc",
        "--disable-cache",
        type=click.Path(exists=True, writable=False, readable=True),
        help="Disables the in-memory caching."
    )
]
)

@click.command(help=_DESCRIPTION)
def train(model_dir, data_dir, class_config, hyper_config, label_file, disable_cache, **kwargs):
    import tensorflow as tf
    # tf.compat.v1.enable_eager_execution()
    # tf.config.experimental_run_functions_eagerly(True)
    from tensorboard.plugins.hparams import api as hp
    import numpy as np
    import importlib
    from deepspectrumlite import HyperParameterList, TransferBaseModel, DataPipeline, \
        METRIC_ACCURACY, METRIC_MAE, METRIC_RMSE, METRIC_RECALL, METRIC_PRECISION, METRIC_F_SCORE, METRIC_LOSS, METRIC_MSE
    import math

    enable_cache = not disable_cache
    data_dir = os.path.join(data_dir, '') # add trailing slash

    f = open(class_config)
    data = json.load(f)
    f.close()

    data_classes = data

    if data_classes is None:
        raise ValueError('no data classes defined')

    tensorboard_initialised = False

    log.info("Physical devices:")
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    log.info(physical_devices)
    del physical_devices

    hyper_parameter_list = HyperParameterList(config_file_name=hyper_config)

    max_iterations = hyper_parameter_list.get_max_iteration()
    log.info('Loaded hyperparameter configuration.')
    log.info("Recognised combinations of settings: " + str(max_iterations) + "")

    slurm_jobid = os.getenv('SLURM_ARRAY_TASK_ID')

    if slurm_jobid is not None:
        slurm_jobid = int(slurm_jobid)

        if slurm_jobid >= max_iterations:
            raise ValueError('slurm jobid ' + str(slurm_jobid) + ' is out of bound')

    for iteration_no in range(max_iterations):
        if slurm_jobid is not None:
            iteration_no = slurm_jobid
        hparam_values = hyper_parameter_list.get_values(iteration_no=iteration_no)
        hparam_values_tensorboard = hyper_parameter_list.get_values_tensorboard(iteration_no=iteration_no)

        run_identifier = hparam_values['tb_run_id'] + '_config_' + str(iteration_no)

        tensorboard_dir = hparam_values['tb_experiment']

        log_dir = os.path.join(model_dir, 'logs', tensorboard_dir)
        run_log_dir = os.path.join(log_dir, run_identifier)
        model_dir = os.path.join(model_dir, 'models', tensorboard_dir, run_identifier)
        # delete old log
        if os.path.isdir(run_log_dir):
            shutil.rmtree(run_log_dir)

        if not tensorboard_initialised:
            # create tensorboard
            with tf.summary.create_file_writer(log_dir).as_default():
                hp.hparams_config(
                    hparams=hyper_parameter_list.get_hparams(),
                    metrics=[hp.Metric(METRIC_ACCURACY, display_name='accuracy'),
                             hp.Metric(METRIC_PRECISION, display_name='precision'),
                             hp.Metric(METRIC_RECALL, display_name='unweighted recall'),
                             hp.Metric(METRIC_F_SCORE, display_name='f1 score'),
                             hp.Metric(METRIC_MAE, display_name='mae'),
                             hp.Metric(METRIC_RMSE, display_name='rmse')
                             ],
                )
                tensorboard_initialised = True

        # Use a label file parser to load data
        label_parser_key = hparam_values['label_parser']

        if ":" not in label_parser_key:
            raise ValueError('Please provide the parser in the following format: path.to.parser_file.py:ParserClass')

        log.info(f'Using custom external parser: {label_parser_key}')
        path, class_name = label_parser_key.split(':')
        module_name = os.path.splitext(os.path.basename(path))[0]
        dir_path = os.path.dirname(os.path.realpath(__file__))
        path = os.path.join(dir_path, path)
        spec = importlib.util.spec_from_file_location(module_name, path)
        foo = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(foo)
        parser_class = getattr(foo, class_name)

        parser = parser_class(file_path=label_file)
        train_data, devel_data, test_data = parser.parse_labels()

        # reset seed values to make keras reproducible
        np.random.seed(0)
        tf.compat.v1.set_random_seed(0)

        log.info('--- Starting trial: %s' % run_identifier)
        log.info({h.name: hparam_values_tensorboard[h] for h in hparam_values_tensorboard})

        log.info("Load data pipeline ...")

        ########### TRAIN DATA ###########
        train_data_pipeline = DataPipeline(name='train_data_set', data_classes=data_classes,
                                           enable_gpu=True, verbose=True, enable_augmentation=False,
                                           hparams=hparam_values, run_id=iteration_no)
        train_data_pipeline.set_data(train_data)
        train_data_pipeline.set_filename_prepend(prepend_filename_str=data_dir)
        train_data_pipeline.preprocess()
        train_data_pipeline.up_sample()
        train_dataset = train_data_pipeline.pipeline(cache=enable_cache)

        ########### DEVEL DATA ###########
        devel_data_pipeline = DataPipeline(name='devel_data_set', data_classes=data_classes,
                                           enable_gpu=True, verbose=True, enable_augmentation=False,
                                           hparams=hparam_values, run_id=iteration_no)
        devel_data_pipeline.set_data(devel_data)
        devel_data_pipeline.set_filename_prepend(prepend_filename_str=data_dir)
        devel_dataset = devel_data_pipeline.pipeline(cache=enable_cache, shuffle=False, drop_remainder=False)

        ########### TEST DATA ###########
        test_data_pipeline = DataPipeline(name='test_data_set', data_classes=data_classes,
                                          enable_gpu=True, verbose=True, enable_augmentation=False,
                                          hparams=hparam_values, run_id=iteration_no)
        test_data_pipeline.set_data(test_data)
        test_data_pipeline.set_filename_prepend(prepend_filename_str=data_dir)
        test_dataset = test_data_pipeline.pipeline(cache=enable_cache, shuffle=False, drop_remainder=False)

        log.info("All data pipelines have been successfully loaded.")
        log.info("Caching in memory is: " + str(enable_cache))

        model_name = hparam_values['model_name']

        available_ai_models = {
            'TransferBaseModel': TransferBaseModel
        }

        if model_name in available_ai_models:
            model = available_ai_models[model_name](hyper_parameter_list,
                                                    train_data_pipeline.get_model_input_shape(),
                                                    run_dir=run_log_dir,
                                                    data_classes=data_classes,
                                                    use_ram=True,
                                                    run_id=iteration_no)

            model.run(train_dataset=train_dataset,
                      test_dataset=test_dataset,
                      devel_dataset=devel_dataset,
                      save_model=True,
                      save_dir=model_dir)
        else:
            ValueError("Unknown model name: " + model_name)

        if slurm_jobid is not None:
            break
