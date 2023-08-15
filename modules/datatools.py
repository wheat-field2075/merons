"""
datatools: tools for creating temporary copies of a dataset, to facilitate changing data hyper-
parameters in between runs, and transforming data
"""

import os
import json
import torch
import shutil
import itertools
import numpy as np
from PIL import Image

""" ##############################
create_temp_dir: Create a set of empty directories, enclosed within a parent, for
    a temporary dataset
    
Args:
    root_dir: the path to the directory that contains source images
Returns:
    parent_path: the path to the temporary dataset's parent directory
############################## """
def create_temp_dir(root_dir):
    # find a valid path for a temporary dataset and create necessary
    # (empty) directories
    counter = 0
    if os.path.exists('./temp_datasets') == False:
        os.mkdir('./temp_datasets')
    while True:
        parent_dir = './temp_datasets/ds_{}'.format(counter)
        if os.path.exists(parent_dir):
            counter += 1
            continue
        break
    os.mkdir(parent_dir)
    for child_dir in ['images', 'masks', 'image_patches', 'mask_patches']:
        os.mkdir(os.path.join(parent_dir, child_dir))

    return parent_dir
""" ############################## """

""" ##############################
draw_gaussian: Given a greyscale image and a point, place a two-dimensional
    Gaussian distribution with specified sigma at that point.
    
Args:
    array: the greyscale image at the point
    array_point: the centerpoint of the intended Gaussian distribution
    sigma: the standard deviation of the intended Gaussian distribution
    dim: the dimension of the supplanted bbox that contains the Gaussian distribution
############################## """
def draw_gaussian(array, array_point, sigma, dim):
    # create a kernel containing a Gaussian distribution
    y = np.expand_dims(np.linspace(-1 * (dim // 2), dim // 2, num=dim), 0)
    x = np.linspace(-1 * (dim // 2), dim // 2, num=dim)
    xx, yy = np.meshgrid(y, x)
    gaussian_kernel = (1 / (2 * np.pi * sigma ** 2)) * np.exp(-1 * (xx ** 2 + yy ** 2)
                                                              / (2 * sigma ** 2))
    gaussian_kernel /= gaussian_kernel.max()

    # supplant the above Gaussian kernel into array
    for y, x in itertools.product(range(array_point[0] - dim // 2, array_point[0] + dim // 2),
                                 range(array_point[1] - dim // 2, array_point[1] + dim // 2)):
        if y in range(0, array.shape[0]) and x in range(array.shape[1]):
            array[y, x] = np.maximum(array[y, x],
                                     gaussian_kernel[int(y - (array_point[0] - dim // 2)),
                                                    int(x - (array_point[1] - dim // 2))])
""" ############################## """

""" ##############################
create_data: Given the parent directory, fill in the enclosed 'images' and 'masks' folders with
    the corresponding information
    
Args:
    root_dir: the path to the folder containing the source dataset
    parent_dir: the path to the temporary dataset's parent directory
    json_name: the name of the file containing annotations. should be in (plain) COCO format.
        Default: annotations.json
    sigma: the standard deviation for Gaussian distributions in masks. Default: 15
    dim: the dimension for supplanted bboxes in masks. Default: 71
############################## """
def create_data(root_dir, parent_dir, json_name='annotations.json', sigma=15, dim=71):
    # copy source images from root_dir into temporary dataset
    print('##### Images #####')
    for file_name in sorted(os.listdir(root_dir)):
        if file_name[-5:] == '.json':
            shutil.copyfile(os.path.join(root_dir, file_name),
                           os.path.join(parent_dir, 'images', file_name)) 
        if (file_name[-4:] in ['.jpg', '.png']) == False:
            continue
        print('Copying: {}...'.format(file_name), end='')
        image = Image.open(os.path.join(root_dir, file_name))
        image.save(os.path.join(parent_dir, 'images', file_name))
        print('done')

    # load annotation data from JSON
    print('##### Creating masks...', end='')
    json_path = os.path.join(parent_dir, 'images', json_name)
    json_data = json.load(open(json_path, 'r'))

    # read JSON annotations and initialize masks
    json_images = [[image['id'], image['file_name']] for image in json_data['images']]
    json_annotations = json_data['annotations']
    mask_list = []
    for image_id, image_name in json_images:
        image = Image.open(os.path.join(parent_dir, 'images', image_name))
        mask = np.zeros(image.size[::-1])
        mask_list.append([image_id, image_name, mask])

    # iterate through bboxes and add to relevant masks
    for annotation in json_annotations:
        bbox_image_id = annotation['image_id']
        x, y, width, height = annotation['bbox']
        center = [int(y + height // 2), int(x + width // 2)]
        for image_id, image_name, mask in mask_list:
            if bbox_image_id == image_id:
                draw_gaussian(mask, center, sigma, dim)
                break

    # save completed masks into temporary dataset
    for image_id, image_name, mask in mask_list:
        mask = Image.fromarray(mask * 255).convert('L')
        mask.save(os.path.join(parent_dir, 'masks', image_name), mode='.jpg')
    print('done #####')
""" ############################## """

""" ##############################
create_patches: Given the parent directory, create image and mask patches based on provided
    specifications
    
Args:
    parent_dir: the path to the temporary dataset's parent directory
    json_name: the name of the file containing annotations. should be in (plain) COCO format.
        Default: annotations.json
    full_size: whether to use the whole image/mask as an input/target, respectively. ignores
        patch_size if True. Default: False
    patch_size: the size of each patch. Default: 200
    create_grid: seperate each sample into evenly spaced grids. Default: True
    grid_step: the distance between each patch. tied to make_grid. Default: 100
    create_centered: make a patch centered around each Gaussian distribution. Default: False
############################## """
def create_patches(parent_dir,
                json_name='annotations.json',
                full_size=False,
                patch_size=200,
                create_grid=True,
                grid_step=100,
                create_centered=False):
    if full_size == False:
        assert True in [create_grid, create_centered]

    # load image annotation data
    json_path = os.path.join(parent_dir, 'images', json_name)
    json_data = json.load(open(json_path, 'r'))
    images = [[image['id'], image['file_name']] for image in json_data['images']]
    annotations = json_data['annotations']

    # prepare folders for each sample
    for image_id, image_name in images:
        os.mkdir(os.path.join(parent_dir, 'image_patches', image_name))
        os.mkdir(os.path.join(parent_dir, 'mask_patches', image_name))

    # copy entire image if full_size
    if full_size:
        for image_name in sorted(os.listdir(os.path.join(parent_dir, 'images'))):
            shutil.copyfile(os.path.join(parent_dir, 'images', image_name),
                            os.path.join(parent_dir, 'image_patches', image_name, image_name))
            shutil.copyfile(os.path.join(parent_dir, 'masks', image_name),
                            os.path.join(parent_dir, 'mask_patches', image_name, image_name))
        return

    # create grid samples
    if create_grid:
        print('##### Grid samples #####')
        for image_id, image_name in images:
            print('Making samples for: {}...'.format(image_name), end='')
            temp_image = Image.open(os.path.join(parent_dir, 'images', image_name))
            temp_mask = Image.open(os.path.join(parent_dir, 'masks', image_name))
            patch_counter = 0
            for y, x in itertools.product(range(0, temp_image.size[0] - patch_size, grid_step),
                                         range(0, temp_image.size[1] - patch_size, grid_step)):
                image_patch = temp_image.crop([x, y, x + patch_size, y + patch_size])
                mask_patch = temp_mask.crop([x, y, x + patch_size, y + patch_size])
                image_patch.save(os.path.join(parent_dir, 'image_patches', image_name,
                                             'gp_{:03}.jpg'.format(patch_counter)), mode='.jpg')
                mask_patch.save(os.path.join(parent_dir, 'mask_patches', image_name,
                                            'gp_{:03}.jpg'.format(patch_counter)), mode='.jpg')
                patch_counter += 1
            print('done')

    # create centered samples
    if create_centered:
        print('##### Centered samples #####')
        for image_id, image_name in images:
            print('Making samples for: {}...'.format(image_name), end='')
            temp_image = Image.open(os.path.join(parent_dir, 'images', image_name))
            temp_mask = Image.open(os.path.join(parent_dir, 'masks', image_name))
            patch_counter = 0
            for annotation in annotations:
                bbox_image_id = annotation['image_id']
                if bbox_image_id != image_id:
                    continue
                x, y = annotation['bbox'][0:2]
                x = np.clip(x, 0, temp_image.size[0] - patch_size)
                y = np.clip(y, 0, temp_image.size[1] - patch_size)
                image_patch = temp_image.crop([x, y, x + patch_size, y + patch_size])
                mask_patch = temp_mask.crop([x, y, x + patch_size, y + patch_size])
                image_patch.save(os.path.join(parent_dir, 'image_patches', image_name,
                                             'cp_{:03}.jpg'.format(patch_counter)), mode='.jpg')
                mask_patch.save(os.path.join(parent_dir, 'mask_patches', image_name,
                                            'cp_{:03}.jpg'.format(patch_counter)), mode='.jpg')
                patch_counter += 1
            print('done')
""" ############################## """

""" ##############################
make_dataset: wrapper function to create a temporary dataset at a valid location

Args:
    - create_temp_dir
        root_dir: the path to the directory that contains source images
    - create_data
        parent_dir: the path to the temporary dataset's parent directory
        json_name: the name of the file containing annotations. should be in (plain) COCO
            format. Default: annotations.json
        sigma: the standard deviation for Gaussian distributions in masks. Default: 15
        dim: the dimension for supplanted bboxes in masks. Default: 71
            parent_dir: the path to the temporary dataset's parent directory
        json_name: the name of the file containing annotations. should be in (plain) COCO
            format. Default: annotations.json
    - create_patches
        full_size: whether to use the whole image/mask as an input/target, respectively. ignores
        patch_size if True. Default: False
        patch_size: the size of each patch. Default: 200
        create_grid: seperate each sample into evenly spaced grids. Default: True
        grid_step: the distance between each patch. tied to make_grid. Default: 100
        create_centered: make a patch centered around each Gaussian distribution.
            Default: False
Returns:
    parent_path: the path to the temporary dataset's parent directory
############################## """

def make_dataset(root_dir,
                json_name='annotations.json',
                sigma=15,
                dim=71,
                full_size=False,
                patch_size=200,
                create_grid=True,
                grid_step=100,
                create_centered=False):
    parent_dir = create_temp_dir(root_dir)
    create_data(root_dir, parent_dir, json_name=json_name, sigma=sigma, dim=dim)
    create_patches(parent_dir,
                  json_name=json_name,
                  full_size=full_size,
                  patch_size=patch_size,
                  create_grid=create_grid,
                  grid_step=grid_step,
                  create_centered=create_centered)
    return parent_dir
""" ############################## """

""" ##############################
load_temp_dataset: a function to load image and mask patches from .jpg files to numpy arrays

Args:
    parent_dir: the parent directory of the temporary dataset
    exclude_list: the names of images to exclude from the training set (and hence, include in
        the validation set)
Returns:
    train_list: a list of image-mask pairs, deisgnated as training data
    val_list: a list of image-mask pairs, deisgnated as validation data
############################## """

""" ############################## """
def load_temp_dataset(parent_dir, exclude_list):
    # prepare directory paths
    print('##### Loading dataset #####')
    image_patches_dir = os.path.join(parent_dir, 'image_patches')
    mask_patches_dir = os.path.join(parent_dir, 'mask_patches')
    subdirs = sorted(os.listdir(image_patches_dir))

    # create lists to store images and masks
    train_list = []
    val_list = []

    # iterate through patch directories and store patches as numpy arrays
    for subdir in subdirs:
        print('Loading samples for: {}...'.format(subdir), end='')
        target = val_list if subdir in exclude_list else train_list
        image_subdir = os.path.join(image_patches_dir, subdir)
        mask_subdir = os.path.join(mask_patches_dir, subdir)
        patch_names = os.listdir(image_subdir)
        for patch_name in patch_names:
            image_patch = Image.open(os.path.join(image_subdir, patch_name)).convert('L')
            mask_patch = Image.open(os.path.join(mask_subdir, patch_name)).convert('L')
            image_mask_pair = tuple([np.array(image_patch), np.array(mask_patch)])
            target.append(image_mask_pair)
        print('done')

    return train_list, val_list

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

"""
transform_data: apply the same transformation to batched images and masks. should be used with
    albumentations

Args:
    images: a batched set of inputs
    masks: a batched set of masks
    transform: the transformation that should be applied to all of the batched image-mask pairs
Returns:
    images: a batched set of images as pytorch Tensors and with shape (B, C, H, W)
    masks: a batched set of masks as pytorch Tensors and with shape (B, C, H, W)
"""

def transform_data(images, masks, transform):
    # ensure that data is of correct type and shape
    if type(images) != np.ndarray:
        images = np.array(images)
    if type(masks) != np.ndarray:
        masks = np.array(masks)
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
""" ############################## """