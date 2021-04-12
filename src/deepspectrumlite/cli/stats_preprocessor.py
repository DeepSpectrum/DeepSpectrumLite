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
from deepspectrumlite import PreprocessAudio, HyperParameterList

# tf.compat.v1.enable_eager_execution()
tf.compat.v1.disable_eager_execution()
import numpy as np


def print_version():
    print(tf.version.GIT_VERSION, tf.version.VERSION)


def get_detailed_stats(model):
    session = tf.compat.v1.Session()
    graph = tf.compat.v1.get_default_graph()

    with graph.as_default():
        with session.as_default():
            new_model = tf.saved_model.load('preprocessor')
            # print(new_model.get_config())
            run_meta = tf.compat.v1.RunMetadata()
            input = tf.convert_to_tensor(np.array(np.random.random_sample((1, 16000)), dtype=np.float32),
                                         dtype=tf.float32)

            # result = new_model.preprocess(input)
            #
            # print(result)
            # {'serving_default_audio_signal:0': input},
            result2 = session.run(new_model.preprocess(input),
                                  options=tf.compat.v1.RunOptions(trace_level=tf.compat.v1.RunOptions.FULL_TRACE),
                                  run_metadata=run_meta)
            print(result2)

            ProfileOptionBuilder = tf.compat.v1.profiler.ProfileOptionBuilder
            opts = ProfileOptionBuilder(ProfileOptionBuilder.time_and_memory()
                                        ).with_timeline_output('test.json').build()

            tf.compat.v1.profiler.profile(
                tf.compat.v1.get_default_graph(),
                run_meta=run_meta,
                cmd='code',
                options=opts)
            # '''
            # Print to stdout an analysis of the memory usage and the timing information
            # broken down by operation types.
            json_export = tf.compat.v1.profiler.profile(
                tf.compat.v1.get_default_graph(),
                run_meta=run_meta,
                cmd='op',
                options=tf.compat.v1.profiler.ProfileOptionBuilder.time_and_memory())

            flops = tf.compat.v1.profiler.profile(graph=graph,
                                                  run_meta=run_meta, cmd='op', options=opts)
            print(json_export, flops)
    tf.compat.v1.reset_default_graph()


if __name__ == "__main__":
    print_version()
    hyper_parameter_list = HyperParameterList(
        config_file_name='/Users/tobias/PycharmProjects/liteAudioNets-huawei/config/hp_config_paper_css.json')
    hparam_values = hyper_parameter_list.get_values(iteration_no=0)
    preprocess = PreprocessAudio(hparams=hparam_values, name="dsl_audio_preprocessor")

    get_detailed_stats(preprocess)
