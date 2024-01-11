import numpy as np
from PIL import Image
import albumentations as A
import numpy.typing as npt


# Functions defined:
#     - open_image
#     - MapDataset
#     - transform_data

def open_image(image_path: str, reshape: bool=True) -> npt.NDArray[np.uint8]:
   """
   open_image: given an image path, returns a JPEG image represented as a numpy array

   Inputs:
        image_path (str): the path to the JPEG image
        reshape (bool; default=True): if True, crops the image into a square based on the shorter dimenion
   Outputs:
      image (npt.NDArray): the corresponding image as a 2D numpy array
   """
   image = Image.open(image_path).convert('L')
   image = np.array(image)
   if reshape:
       length = np.min(image.shape)
       image = image[(image.shape[0]-length)//2:(image.shape[0]-length)//2+length, (image.shape[1]-length)//2:(image.shape[1]-length)//2+length]
   return image

import torch
import numpy as np
import albumentations as A
import numpy.typing as npt

class MapDataset(torch.utils.data.Dataset):
    """
    MapDataset: a simple map-style class built upon the abstract class torch.utils.data.Dataset
    """

    def __init__(self, image_mask_pairs: list):
        """
        __init__: captures and saves ordered sequences of images and segmentation masks
        
        Inputs:
            image_mask_seq (list): a sequence of image-mask pairs
        """
        self.image_mask_pairs = image_mask_pairs

    def __len__(self) -> int:
        """
        __len__: returns the number of image-mask pairs in the dataset

        Outputs:
            *unnamed* (int): the number of image-mask pairs in the dataset
        """
        return len(self.image_mask_pairs)
    
    def __getitem__(self, index):
        """
        __getitem__: returns the image-mask pair that corresponds to a given index

        Inputs:
            index (int): the index corresponding to the desired image-mask pair
        Outputs:
            *unnamed* (list): the corresponding image-mask pair
        """
        return self.image_mask_pairs[index]

def transform_data(images: npt.NDArray[np.float32], masks: npt.NDArray[np.float32], transform: A.core.composition.Compose):
    """
    transform_data: apply the same transformation to batched images and masks. should be used with albumentations
    
    Inputs:
        images (npt.NDArray[np.float32]): a batched set of inputs
        masks (npt.NDArray[np.float32]): a batched set of masks
        transform (A.core.composition.Compose): the transformation that should be applied to all of the batched image-mask pairs
    Outputs:
        images (npt.NDArray[np.float32]): a batched set of images as pytorch Tensors and with shape (B, C, H, W)
        masks (npt.NDArray[np.float32]): a batched set of masks as pytorch Tensors and with shape (B, C, H, W)
    """
    # ensure that data is of correct type and shape
    images, masks = np.array(images).astype(np.float32), np.array(masks).astype(np.float32)

    # iteratively transform, normalize, and replace along the batch dimension
    for batch_index, image, mask in zip(range(images.shape[0]), images, masks):
        transformed = transform(image=image.copy(), mask=mask.copy())
        images[batch_index] = transformed['image'] / 255
        masks[batch_index] = transformed['mask']
        
    # ensure that data is of correct type and shape
    images, masks = np.moveaxis(images, -1, 1), np.moveaxis(masks, -1, 1)
    images, masks = torch.Tensor(images), torch.Tensor(masks)
    return images, masks
