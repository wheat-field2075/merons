import cv2
import scipy
import torch
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

def get_stats(prediction: npt.NDArray[np.float32], target: npt.NDArray[np.float32]) -> tuple[float, float, float]:
    """
    get_stats: calcualte precision, recall, and F1 score given a prediction and a target
    
    Inputs:
        prediction (npt.NDArray[np.float32]): the model prediction as a numpy array
        target (npt.NDArray[np.float32]): the target image as a numpy array
    Outputs:
        precision (float): the proportion of detections that are accurate
        recall (float): the proportion of points of interest that the model correctly detected
        f1 (float): the F1 score
    """
    # convert to uint8
    prediction = (prediction * 255).astype(np.uint8)
    target = (target * 255).astype(np.uint8)

    # threshold images
    prediction = cv2.threshold(prediction,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1].astype(np.uint8) * 255
    target = cv2.threshold(target,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1].astype(np.uint8) * 255
    
    # get centroids for each component
    p_centroids = cv2.connectedComponentsWithStats(prediction)[3][1:]
    t_centroids = cv2.connectedComponentsWithStats(target)[3][1:]

    # compute number of true positives using Hungarian matching
    d_matrix = scipy.spatial.distance_matrix(p_centroids, t_centroids)
    row, col = scipy.optimize.linear_sum_assignment(d_matrix)
    matches = [d_matrix[i, j] for i, j in zip(row, col) if d_matrix[i, j] < 70]

    # calculate and return precision, recall, and F1 score
    precision = len(matches) / len(p_centroids)
    recall = len(matches) / len(t_centroids)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    return precision, recall, f1


    