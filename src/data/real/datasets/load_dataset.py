import os


def load_dataset(name, root, val=False, transform=None, **kwargs):
    '''
    '''
    if name in ('pfpascal', 'pfwillow', 'spair'):
        return load_semantic_pairs(name, root, val, **kwargs)
    elif name == 'imagenet':
        return load_imagenet(root, val, transform, **kwargs)
    else:
        raise Exception(f'Unsupported dataset "{name}"')


def load_semantic_pairs(name, root, val=False, **kwargs):
    from .semantic_pairs import PFPascalDataset, PFWillowDataset, SPairDataset
    datasets = {
        'pfpascal': PFPascalDataset,
        'pfwillow': PFWillowDataset,
        'spair': SPairDataset,
    }

    dataset = datasets.get(name, None)
    split = 'tst' if val else 'trn'
    return dataset(benchmark=name, datapath=root, split=split, **kwargs)


def load_imagenet(root, val, transform, **kwargs):
    '''
    '''
    from .img_datasets import Hdf5ImageDataset
    root = os.path.join(root, 'val' if val else 'train')
    return Hdf5ImageDataset(root, transform=transform)