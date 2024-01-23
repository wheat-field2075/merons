import os
import cv2
import shutil
import numpy as np
from PIL import Image
from tqdm import tqdm

root_folder = './all_data'
target_folder = './dataset'
data_name = '20190822_movie_01_SampleOldA1_120kV_81x2048x2048_30sec_Aligned'
include_list = range(0, 81, 5)

if os.path.exists(target_folder) == False:
    os.mkdir(target_folder)

shutil.copy(os.path.join(root_folder, 'annotations.json'), os.path.join(target_folder, 'annotations.json'))

data = np.load(os.path.join(root_folder, data_name+'.npy'))
for idx, sample in tqdm(enumerate(data)):
    low, high = np.percentile(sample, [5, 95])
    sample = 255 - ((np.clip(sample, low, high) - low) / (high - low) * 255).astype(np.uint8)
    sample = cv2.medianBlur(sample, 9)
    # sample = sample[:1592, :1592]
    Image.fromarray(sample).convert('L').save(os.path.join(root_folder, data_name+' patch_{:03}.jpg'.format(idx)))
    if idx in include_list:
        Image.fromarray(sample).convert('L').save(os.path.join(target_folder, data_name+' patch_{:03}.jpg'.format(idx)))