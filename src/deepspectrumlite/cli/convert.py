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
from deepspectrumlite import AugmentableModel, ARelu
import numpy as np


def print_version():
    print(tf.version.GIT_VERSION, tf.version.VERSION)


if __name__ == "__main__":

    print_version()

    if len(sys.argv) > 1:
        source = str(sys.argv[1])
    else:
        source = "model_run-0/model.h5"

    if len(sys.argv) > 1:
        destination = str(sys.argv[2])
    else:
        destination = 'converted_model.tflite'

    print("Load model: " + source)

    # loader_impl.parse_saved_model(source)

    new_model = tf.keras.models.load_model(source,
                                           custom_objects={'AugmentableModel': AugmentableModel, 'ARelu': ARelu},
                                           compile=False)
    print("Successfully loaded model: " + source)

    converter = tf.lite.TFLiteConverter.from_keras_model(new_model)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
                                           tf.lite.OpsSet.SELECT_TF_OPS]
    converter.experimental_new_converter = True
    tflite_quant_model = converter.convert()
    open(destination, "wb").write(tflite_quant_model)

    print("Model was saved as tflite as " + destination)

    interpreter = tf.lite.Interpreter(model_content=tflite_quant_model)

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.allocate_tensors()

    # interpreter.set_tensor(input_details[0]['index'], tf.convert_to_tensor(np.expand_dims(audio_data, 0), dtype=tf.float32))

    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]['index'])

    # Test model on random input data.
    input_shape = input_details[0]['shape']
    print("input shape: ", input_shape)
    print("output shape: ", output_details[0]['shape'])
    input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)

    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    print(output_data)
