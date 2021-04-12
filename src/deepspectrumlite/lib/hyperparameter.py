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
import itertools
import json
from tensorboard.plugins.hparams import api as hp


class HyperParameterList:
    def __init__(self, config_file_name: str):
        f = open(config_file_name)
        data = json.load(f)
        f.close()

        self._config = data
        self._param_list = {}

        self.load_configuration()

    def get_hparams(self):
        hparams = []

        for key in self._config:
            hparams.append(self._param_list[key])

        return hparams

    def load_configuration(self):
        self._param_list = {}

        for key in self._config:
            self._param_list[key] = hp.HParam(key, hp.Discrete(self._config[key]))

    def get_max_iteration(self):
        count = 1

        for key in self._config:
            count = count * len(self._config[key])

        return count

    def get_values_tensorboard(self, iteration_no: int):
        if iteration_no >= self.get_max_iteration():
            raise ValueError(str(self.get_max_iteration()) + ' < iteration_no >= 0')

        configuration_space = []
        for key in self._config:
            configurations = []
            for v in self._config[key]:
                configurations.append({key: v})
            configuration_space.append(configurations)
        perturbations = list(itertools.product(*configuration_space))

        perturbation = perturbations[iteration_no]
        hparams = {}
        for param in perturbation:
            for key in param:
                k = self._param_list[key]
                hparams[k] = param[key]

        return hparams

    def get_values(self, iteration_no: int):
        if iteration_no >= self.get_max_iteration():
            raise ValueError(str(self.get_max_iteration()) + ' < iteration_no >= 0')

        configuration_space = []
        for key in self._config:
            configurations = []
            for v in self._config[key]:
                configurations.append({key: v})
            configuration_space.append(configurations)
        perturbations = list(itertools.product(*configuration_space))

        perturbation = perturbations[iteration_no]
        hparams = {}
        for param in perturbation:
            for key in param:
                hparams[key] = param[key]

        return hparams
