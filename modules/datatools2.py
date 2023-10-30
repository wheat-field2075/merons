import torch
import numpy as np

"""
Functions/classes defined:
    - MapDataset
    - transform_data
"""

""" ##############################
MapDataset: a simple map-style class built upon the abstract class torch.utils.data.Dataset

__init__: captures and saves ordered sequences of images and segmentation masks
Args:
    image_mask_seq: a sequence of image-mask pairs

__len__: returns the number of image-mask pairs in the dataset
Returns:
    *unnamed*: the number of image-mask pairs in the dataset

__getitem__: returns the image-mask pair that corresponds to a given index
Args:
    index: the index corresponding to the desired image-mask pair
Returns:
    *unnamed*: the corresponding image-mask pair
############################## """
class MapDataset(torch.utils.data.Dataset):

    def __init__(self, image_mask_seq):
        self.image_mask_seq = image_mask_seq

    def __len__(self):
        return len(self.image_mask_seq)
    
    def __getitem__(self, index):
        return self.image_mask_seq[index]
""" ############################## """

""" ##############################
transform_data: apply the same transformation to batched images and masks. should be used with
    albumentations

Args:
    images: a batched set of inputs
    masks: a batched set of masks
    transform: the transformation that should be applied to all of the batched image-mask pairs
Returns:
    images: a batched set of images as pytorch Tensors and with shape (B, C, H, W)
    masks: a batched set of masks as pytorch Tensors and with shape (B, C, H, W)
############################## """
def transform_data(images, masks, transform):
    # ensure that data is of correct type and shape
    if type(images) != np.ndarray:
        images = np.array(images).astype(np.float32)
    if type(masks) != np.ndarray:
        masks = np.array(masks).astype(np.float32)
    if len(images.shape) == 3:
        images = np.expand_dims(images, 3)
    if len(masks.shape) == 3:
        masks = np.expand_dims(masks, 3)

    # iteratively transform, normalize, and replace along the batch dimension
    for batch_index, image, mask in zip(range(images.shape[0]), images, masks):
        transformed = transform(image=image, mask=mask)
        images[batch_index] = transformed['image'] / 255
        masks[batch_index] = transformed['mask'] / 255
        
    # ensure that data is of correct type and shape
    images, masks = np.moveaxis(images, -1, 1), np.moveaxis(masks, -1, 1)
    images, masks = torch.Tensor(images), torch.Tensor(masks)
    return images, masks
""" ######################### """