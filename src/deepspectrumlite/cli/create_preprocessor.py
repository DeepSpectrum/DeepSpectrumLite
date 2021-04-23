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
from deepspectrumlite import HyperParameterList, PreprocessAudio
import time
import numpy as np
from os.path import join, dirname, realpath

log = logging.getLogger(__name__)

_DESCRIPTION = 'Creates a DeepSpectrumLite preprocessor TFLite file.'

@add_options(
[
    click.option(
        "-hc",
        "--hyper-config",
        type=click.Path(exists=True, writable=False, readable=True),
        help="Directory for the hyper parameter config file.",
        default=join(dirname(realpath(__file__)), "config/hp_config.json"), show_default=True
    ),
    click.option(
        "-d",
        "--destination",
        type=click.Path(exists=False, writable=True, readable=True),
        help="Destination of the TFLite preprocessor file",
        required=True
    )
]
)

@click.command(help=_DESCRIPTION)
def create_preprocessor(hyper_config, destination, **kwargs):
    hyper_parameter_list = HyperParameterList(config_file_name=hyper_config)
    hparam_values = hyper_parameter_list.get_values(iteration_no=0)
    working_directory = dirname(destination)

    preprocess = PreprocessAudio(hparams=hparam_values, name="dsl_audio_preprocessor")
    input = tf.convert_to_tensor(np.array(np.random.random_sample((1, 16000)), dtype=np.float32), dtype=tf.float32)
    result = preprocess.preprocess(input)

    # ATTENTION: antialias is not supported in tflite
    tmp_save_path = os.path.join(working_directory, "preprocessor")
    os.makedirs(tmp_save_path, exist_ok=True)
    tf.saved_model.save(preprocess, tmp_save_path)

    # new_model = preprocess
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir=tmp_save_path)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
                                           tf.lite.OpsSet.SELECT_TF_OPS]
    converter.experimental_new_converter = True
    tflite_quant_model = converter.convert()
    open(destination, "wb").write(tflite_quant_model)

    interpreter = tf.lite.Interpreter(model_path=destination)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    log.info(input_details)
    log.info(output_details)

    interpreter.allocate_tensors()

    interpreter.set_tensor(input_details[0]['index'], tf.convert_to_tensor(np.array(np.random.random_sample((1, 16000)), dtype=np.float32), dtype=tf.float32))

    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]['index'])

    # Test model on random input data.
    input_shape = input_details[0]['shape']
    log.info("input shape:")
    log.info(input_shape)
    log.info("output shape:")
    log.info(output_details[0]['shape'])
    input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    start_time = time.time()
    interpreter.invoke()
    stop_time = time.time()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    log.info(output_data)
    log.info('time: {:.3f}ms'.format((stop_time - start_time) * 1000))
    log.info("Finished creating the TFLite preprocessor")



