from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class GaofenImageDataset(CustomDataset):
    """
    Gaofen Image Dataset from WHU.

    In segmentation map annotation for Gaofen Image Dataset,
    The ``img_suffix`` and
    ``seg_map_suffix`` are both fixed to '.npy'.
    """
    CLASSES = ('built_up', 'farmland', 'forest', 'meadow',
               'water', 'background')

    PALETTE = [[255, 0, 0], [0, 255, 0], [0, 255, 255], [255, 255, 0],
               [0, 0, 255], [0, 0, 0]]

    def __init__(self, **kwargs):
        super(GaofenImageDataset, self).__init__(
            img_suffix='.npy',
            seg_map_suffix='.npy',
            reduce_zero_label=False,
            **kwargs)
