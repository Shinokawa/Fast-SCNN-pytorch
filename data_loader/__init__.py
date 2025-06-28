from .cityscapes import CitySegmentation
from .tusimple import TUSimpleSegmentation

datasets = {
    'citys': CitySegmentation,
    'tusimple': TUSimpleSegmentation,
}


def get_segmentation_dataset(name, **kwargs):
    """Segmentation Datasets"""
    return datasets[name.lower()](**kwargs)

def get_dataset(dataset, root, split=None, mode=None, **kwargs):
    if dataset == 'citys':
        return CitySegmentation(root, split=split, mode=mode, **kwargs)
    elif dataset == 'tusimple':
        return TUSimpleSegmentation(root, split=split, mode=mode, **kwargs)
    elif dataset == 'custom':
        from .custom import CustomDataset
        return CustomDataset(root, split=split, mode=mode, **kwargs)
    else:
        raise NotImplementedError(f"Dataset {dataset} is not implemented.")
