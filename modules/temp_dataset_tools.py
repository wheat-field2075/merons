import os
import json
import shutil
import itertools
import numpy as np
from PIL import Image
import numpy.typing as npt
from data_tools import open_image

# Functions defined:
#     - create_temporary_directory
#     - create_data
#     - make_dataset


def create_temporary_directory() -> str:
    """
    create_temporary_directory: Create a set of empty directories, enclosed within a parent, for a temporary dataset

    Outputs:
        source_dir (str): the path to the temporary dataset's parent directory
    """
    # find a valid path for a temporary dataset
    counter = 0
    if os.path.exists('./temp_ds') == False:
        os.mkdir('./temp_ds')
    while True:
        parent_dir = './temp_ds/ds_{}'.format(counter)
        if os.path.exists(parent_dir):
            counter += 1
            continue
        break
    os.mkdir(parent_dir)
    # create empty subdirectories for later uses
    for child_dir in ['images', 'masks']:
        os.mkdir(os.path.join(parent_dir, child_dir))  
    return parent_dir

def create_data(root_dir: str, parent_dir: str, json_name='annotations.json', label: int=1, sigma: int=15, side_length: int=71, reshape: bool=True) -> None:
    """
    create_data: Given the parent directory, fill in the enclosed 'images' and 'masks' folders with
        the corresponding information
        
    Inputs:
        root_dir (str): the path to the folder containing the source dataset
        parent_dir (str): the path to the temporary dataset's parent directory
        json_name (str; default=annotations.json): the name of the file containing annotations. should be in (plain) COCO format.
        label (int; default=1): the label to be considered. use 1 for merons, 2 for antimerons.
        sigma (int; default=15): the standard deviation for Gaussian distributions in masks.
        side_length (int; default=71): the side length for bboxes in masks. must be odd.
        reshape (bool; default=True): if True, crops images and masks into a square based on the shorter dimenion
    """
    # side_length must be odd
    assert side_length % 2 == 1
    # copy image data from root_dir into parent_dir
    for file_name in os.listdir(root_dir):
        if os.path.isfile(os.path.join(root_dir, file_name)) == False:
            continue
        if file_name[-5:] == '.json':
            shutil.copy(os.path.join(root_dir, file_name), os.path.join(parent_dir, 'images', file_name))
        if file_name[-4:] == '.jpg':
            np.save(os.path.join(parent_dir, 'images', file_name), open_image(os.path.join(root_dir, file_name), reshape=reshape))
    # load annotation data from JSON
    json_data = json.load(open(os.path.join(parent_dir, 'images',json_name)))
    json_images = json_data['images']
    json_annotations = json_data['annotations']
    # read JSON annotations and create/save masks
    image_annotation_pairs = [[img, [annotation for annotation in json_annotations
                                    if annotation['image_id'] == img['id']]] for img in json_images]
    for image, annotations in image_annotation_pairs:
        mask = np.pad(np.zeros([image['height'], image['width']]), side_length//2)
        for annotation in annotations:
            if annotation['category_id'] != label:
                continue
            x, y, width, height = annotation['bbox']
            # exclude if annotation is outside of mask
            if y + height > image['height'] or x + width > image['width']:
                continue
            center = np.array([int(y+height//2), int(x+width//2)])
            center = center + side_length//2
            yy = np.expand_dims(np.linspace(-1 * side_length//2, side_length//2+1, num=side_length), 0)
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
        mask = mask[side_length//2:side_length//2+image['height'], side_length//2:side_length//2+image['width']]
        if reshape:
            length = np.min(mask.shape)
            mask = mask[(mask.shape[0]-length)//2:(mask.shape[0]-length)//2+length, (mask.shape[1]-length)//2:(mask.shape[1]-length)//2+length]
        np.save(os.path.join(parent_dir, 'masks', image['file_name']), mask)

def make_dataset(root_dir: str, json_name: str='annotations.json', label: int=1, sigma: int=15, side_length: int=71) -> str:
    """
    make_dataset: wrapper function to create a temporary dataset at a valid location

    Inputs:
        create_temp_directory
            root_dir: the path to the directory that contains source images
        create_data
            json_name (str; default=annotations.json): the name of the file containing annotations. should be in (plain) COCO format.
            label (int; default=1): the label to be considered. use 1 for merons, 2 for antimerons.
            sigma (int; default=15): the standard deviation for Gaussian distributions in masks.
            side_length (int; default=71): the side length for bboxes in masks. must be odd.

    Outputs:
        parent_dir (str): the path to the temporary dataset's parent directory
    """
    parent_dir = create_temporary_directory()
    create_data(root_dir, parent_dir, json_name=json_name, label=label, sigma=sigma, side_length=side_length)
    return parent_dir