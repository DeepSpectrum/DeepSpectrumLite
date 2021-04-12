import abc


class AbstractColorMap(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        self.color_map = None

    def set_color_map(self, color_map):
        self.color_map = color_map

    def get_color_map(self):
        return self.color_map
