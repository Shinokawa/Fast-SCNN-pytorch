from .cityscapes import CitySegmentation
from .tusimple import TUSimpleSegmentation
from .bdd100k import BDD100KSegmentation
from .custom import CustomDataset

datasets = {
    'citys': CitySegmentation,
    'tusimple': TUSimpleSegmentation,
    'bdd100k': BDD100KSegmentation,
    'custom': CustomDataset,
}


def get_segmentation_dataset(name, **kwargs):
    """Segmentation Datasets"""
    return datasets[name.lower()](**kwargs)

def get_dataset(dataset, root, split=None, mode=None, **kwargs):
    if dataset == 'citys':
        return CitySegmentation(root, split=split, mode=mode, **kwargs)
    elif dataset == 'tusimple':
        return TUSimpleSegmentation(root, split=split, mode=mode, **kwargs)
    elif dataset == 'bdd100k':
        return BDD100KSegmentation(root, split=split, mode=mode, **kwargs)
    elif dataset == 'custom':
        from .custom import CustomDataset
        return CustomDataset(root, split=split, mode=mode, **kwargs)
    else:
        raise NotImplementedError(f"Dataset {dataset} is not implemented.")
