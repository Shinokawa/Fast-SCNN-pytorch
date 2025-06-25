from .cityscapes import CitySegmentation
from .tusimple import TUSimpleSegmentation

datasets = {
    'citys': CitySegmentation,
    'tusimple': TUSimpleSegmentation,
}


def get_segmentation_dataset(name, **kwargs):
    """Segmentation Datasets"""
    return datasets[name.lower()](**kwargs)
