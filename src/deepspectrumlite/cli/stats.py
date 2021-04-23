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
import tensorflow as tf
import numpy as np
from deepspectrumlite import AugmentableModel, ARelu, HyperParameterList

def get_detailed_stats(model_h5_path):
    session = tf.compat.v1.Session()
    graph = tf.compat.v1.get_default_graph()

    parent_dir = dirname(model_h5_path)


    with graph.as_default():
        with session.as_default():
            new_model = tf.keras.models.load_model(model_h5_path, custom_objects={'AugmentableModel': AugmentableModel, 'ARelu': ARelu}, compile=False)
            new_model.summary(print_fn=log.info)
            run_meta = tf.compat.v1.RunMetadata()
            input_details = new_model.get_config()
            input_shape = input_details['layers'][0]['config']['batch_input_shape']

            _ = session.run(new_model.output, {
                'input_1:0': np.random.normal(size=(1, input_shape[1], input_shape[2], input_shape[3]))},
                            options=tf.compat.v1.RunOptions(trace_level=tf.compat.v1.RunOptions.FULL_TRACE),
                            run_metadata=run_meta)

            # '''
            ProfileOptionBuilder = tf.compat.v1.profiler.ProfileOptionBuilder
            opts = ProfileOptionBuilder(ProfileOptionBuilder.time_and_memory()
                                        ).with_step(1).with_timeline_output('test.json').build()

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

            text_file = open(os.path.join(parent_dir, "profiler.json"), "w")
            text_file.write(str(json_export))
            text_file.close()
            # print(json_export)
    tf.compat.v1.reset_default_graph()

# deprecated
def get_flops(model_h5_path): # pragma: no cover
    tf.compat.v1.enable_eager_execution()
    session = tf.compat.v1.Session()
    graph = tf.compat.v1.get_default_graph()


    with graph.as_default():
        with session.as_default():
            tf.keras.models.load_model(model_h5_path, custom_objects={'AugmentableModel': AugmentableModel, 'ARelu': ARelu}, compile=False)
            run_meta = tf.compat.v1.RunMetadata()
            opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()

            # Optional: save printed results to file
            # flops_log_path = os.path.join(tempfile.gettempdir(), 'tf_flops_log.txt')
            # opts['output'] = 'file:outfile={}'.format(flops_log_path)

            # We use the Keras session graph in the call to the profiler.
            flops = tf.compat.v1.profiler.profile(graph=graph,
                                                  run_meta=run_meta, cmd='op', options=opts)
    tf.compat.v1.reset_default_graph()
    return flops.total_float_ops

from os.path import join, dirname, realpath

log = logging.getLogger(__name__)

_DESCRIPTION = 'Retrieve statistics of a DeepSpectrumLite transer learning model.'

@add_options(
[
    click.option(
        "-md",
        "--model-dir",
        type=click.Path(exists=False, writable=True),
        help="Directory of a DeepSpectrumLite HD5 Model.",
        required=True
    )
]
)

@click.command(help=_DESCRIPTION)
def stats(model_dir, **kwargs):
    tf.compat.v1.enable_eager_execution()
    tf.config.run_functions_eagerly(True)
    # reset seed values
    np.random.seed(0)
    tf.compat.v1.set_random_seed(0)

    # new_model = tf.keras.models.load_model(model_dir, custom_objects={'AugmentableModel': AugmentableModel, 'ARelu': ARelu}, compile=False)
    # new_model.summary(print_fn=log.info)
    get_detailed_stats(model_dir)