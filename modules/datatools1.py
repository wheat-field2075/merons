import os
import json
import shutil
import itertools
import numpy as np
from PIL import Image
from datatools0 import pil_to_numpy, numpy_to_pil

"""
Functions defined:
    - create_temporary_directory
    - create_data
    - create_patches
    - make_dataset
    - load_temp_dataset
"""

""" ##############################
create_temporary_directory: Create a set of empty directories, enclosed within a parent, for a
    temporary dataset
    
Returns:
    parent_path: the path to the temporary dataset's parent directory
############################## """
def create_temporary_directory(parent_path):
    # find a valid path for a temporary dataset
    counter = 0
    if os.path.exists('./temp_datasets') == False:
        os.mkdir('./temp_datasets')
    while True:
        parent_path = './temp_datasets/ds_{}'.format(counter)
        if os.path.exists(parent_path):
            counter += 1
            continue
        break
    os.mkdir(parent_path)

    # create empty subdirectories for later uses
    for child_dir in ['images', 'masks', 'image_patches', 'mask_patches']:
        os.mkdir(os.path.join(parent_path, child_dir))
        
    return parent_path
""" ############################## """

""" ##############################
create_data: Given the parent directory, fill in the enclosed 'images' and 'masks' folders with
    the corresponding information
    
Args:
    root_directory: the path to the folder containing the source dataset
    parent_directory: the path to the temporary dataset's parent directory
    json_name: the name of the file containing annotations. should be in (plain) COCO format.
        Default: annotations.json
    sigma: the standard deviation for Gaussian distributions in masks. Default: 15
    side_length: the side length for bboxes in masks. must be odd. Default: 71
############################## """
def create_data(root_directory, parent_directory, json_name='annotations.json',
               sigma=15, side_length=71):
    # side_length must be odd
    assert side_length % 2 == 1
    
    # copy image data from root_directory into parent_directory
    for file_name in os.listdir(root_directory):
        if os.path.isfile(os.path.join(root_directory, file_name)) == False:
            continue
        shutil.copy(os.path.join(root_directory, file_name),
                   os.path.join(parent_directory, 'images', file_name))

    # load annotation data from JSON
    json_data = json.load(open(os.path.join(parent_directory, 'images',json_name)))
    json_images = json_data['images']
    json_annotations = json_data['annotations']

    # read JSON annotations and create/save masks
    image_annotation_pairs = [[img, [annotation for annotation in json_annotations
                                    if annotation['image_id'] == img['id']]] for img in json_images]
    for image, annotations in image_annotation_pairs:
        mask = np.pad(np.zeros([image['height'], image['width']]), side_length//2)
        for annotation in annotations:
            x, y, width, height = annotation['bbox']
            center = np.array([int(y+height//2), int(x+width//2)])
            center = center + side_length//2
            yy = np.expand_dims(np.linspace(-1 * side_length//2, side_length//2+1, 
                                           num=side_length), 0)
            xx = np.linspace(-1 * side_length//2, side_length//2+1, num=side_length)
            yy, xx = np.meshgrid(yy, xx)
            gaussian_kernel = (1/(2*np.pi*sigma**2))*np.exp(-1*(xx**2+yy**2)/(2*sigma**2))
            gaussian_kernel /= gaussian_kernel.max()
            # this formatting is a plague upon my eyes
            mask[center[0]-side_length//2:center[0]+side_length//2+1, 
                center[1]-side_length//2:center[1]+side_length//2+1] = np.maximum(
                    mask[center[0]-side_length//2:center[0]+side_length//2+1, 
                    center[1]-side_length//2:center[1]+side_length//2+1],
                    gaussian_kernel
                )
        mask = mask[side_length//2:side_length//2+image['height'], 
                   side_length//2:side_length//2+image['width']]
        mask = numpy_to_pil(mask)
        mask.save(os.path.join(parent_directory, 'masks', image['file_name']))
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
    grid_step: the distance between each patch. tied to make_grid. Default: 100
############################## """
def create_patches(parent_dir,
                json_name='annotations.json',
                full_size=False,
                patch_size=200,
                grid_step=100):

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
            if image_name[-4:] in ['.jpg', '.png']:
                shutil.copyfile(os.path.join(parent_dir, 'images', image_name),
                                os.path.join(parent_dir, 'image_patches', image_name, image_name))
                shutil.copyfile(os.path.join(parent_dir, 'masks', image_name),
                                os.path.join(parent_dir, 'mask_patches', image_name, image_name))
        return

    # create samples
    print('##### Making samples #####')
    for image_id, image_name in images:
        print('Making samples for: {}...'.format(image_name), end='')
        temp_image = pil_to_numpy(Image.open(os.path.join(parent_dir, 'images', image_name)))
        temp_mask = pil_to_numpy(Image.open(os.path.join(parent_dir, 'masks', image_name)))
        patch_counter = 0
        for y, x in itertools.product(range(0, temp_image.shape[0] - patch_size, grid_step),
                                     range(0, temp_image.shape[1] - patch_size, grid_step)):
            image_patch = temp_image[y:y+patch_size, x:x+patch_size]
            mask_patch = temp_mask[y:y+patch_size, x:x+patch_size]
            image_patch = numpy_to_pil(image_patch)
            mask_patch = numpy_to_pil(mask_patch)
            image_patch.save(os.path.join(parent_dir, 'image_patches', image_name,
                                         'gp_{:03}.jpg'.format(patch_counter)), mode='.jpg')
            mask_patch.save(os.path.join(parent_dir, 'mask_patches', image_name,
                                        'gp_{:03}.jpg'.format(patch_counter)), mode='.jpg')
            patch_counter += 1
        print('done')
""" ############################## """

""" ##############################
make_dataset: wrapper function to create a temporary dataset at a valid location

Args:
    - create_temp_directory
        root_directory: the path to the directory that contains source images
    - create_data
        parent_dir: the path to the temporary dataset's parent directory
        json_name: the name of the file containing annotations. should be in (plain) COCO
            format. Default: annotations.json
        sigma: the standard deviation for Gaussian distributions in masks. Default: 15
        side_length: the dimension for supplanted bboxes in masks. Default: 71
        parent_dir: the path to the temporary dataset's parent directory
        json_name: the name of the file containing annotations. should be in (plain) COCO
            format. Default: annotations.json
    - create_patches
        full_size: whether to use the whole image/mask as an input/target, respectively. ignores
        patch_size if True. Default: False
        patch_size: the size of each patch. Default: 200
        grid_step: the distance between each patch. tied to make_grid. Default: 100
Returns:
    parent_path: the path to the temporary dataset's parent directory
############################## """
def make_dataset(root_directory,
                json_name='annotations.json',
                sigma=15,
                side_length=71,
                full_size=False,
                patch_size=200,
                grid_step=100):
    parent_directory = create_temporary_directory(root_directory)
    create_data(root_directory, parent_directory, json_name=json_name, sigma=sigma,
               side_length=side_length)
    create_patches(parent_directory,
                  json_name=json_name,
                  full_size=full_size,
                  patch_size=patch_size,
                  grid_step=grid_step)
    return parent_directory
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
""" ############################## """