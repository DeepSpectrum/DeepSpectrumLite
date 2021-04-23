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
import re
import pandas as pd
import numpy as np


class ComParEParser: # pragma: no cover

    def __init__(self, file_path: str, delimiter=','): # pragma: no cover
        self._file_path = file_path
        self._delimiter = delimiter

    def parse_labels(self): # pragma: no cover
        complete = pd.read_csv(self._file_path, sep=self._delimiter)
        complete.columns = ['filename', 'label', 'duration_frames']

        train_data = complete[complete.filename.str.startswith('train')]  # 1-3
        devel_data = complete[complete.filename.str.startswith('devel')]  # 4
        test_data = complete[complete.filename.str.startswith('test')]  # 5

        return train_data, devel_data, test_data
