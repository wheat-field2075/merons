# Outside packages
import os
import cv2
import sys
import json
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

# Custom classes
sys.path.append('./modules')
from data_tools import open_image
from model_tools import Hourglass

# make folder if necessary
if os.path.exists('./all_data_pred') == False:
    os.mkdir('./all_data_pred')

# model initialization
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = Hourglass(depth=1).to(device)

# creating predictions for each label
for label in [1, 2]:
    results = []
    results_t = []
    annotations = dict({"licenses":[{"name":"","id":0,"url":""}], 
                        "info":{"contributor":"","date_created":"","description":"","url":"","version":"","year":""}, 
                        "categories":[{"id":1,"name":"TP","supercategory":""},{"id":2,"name":"FN","supercategory":""}]})
    model.load_state_dict(torch.load("./active/2023.12.24 I shouldn't be working right now/model_saves/gcel, label={} epoch=10000.pth".format(label), map_location=device))
    image_list = []
    annotation_list = []
    annotation_counter = 1

    idx = 0
    for file in tqdm(sorted(os.listdir('./all_data'))):
        if file[-4:] != '.jpg':
            continue
        idx += 1
        temp = open_image('./all_data/'+file)
        pred = model(torch.Tensor(np.expand_dims(temp, [0, 1])).to(device))
        pred = pred.detach().cpu().numpy().squeeze() * 255
        results.append(pred.copy())

        image_list.append({"id":idx+1,"width":int(temp.shape[1]),"height":int(temp.shape[0]),"file_name":file,"license":0,"flickr_url":"","coco_url":"","date_captured":0})

        temp = cv2.threshold(pred.astype(np.uint8), 0,255,cv2.THRESH_OTSU)[1]

        numLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(temp, 8, cv2.CV_32S)
        for x, y, w, h, area in stats[1:]:
            if area <= 560:
                temp[y:y+h, x:x+w] = 0
            else:
                annotation_list.append({"id":annotation_counter,"image_id":idx+1,"category_id":1,"segmentation":[],"area":int(area),"bbox":list(map(int, [x, y, w, h])),"iscrowd":0,"attributes":{"occluded":False,"rotation":0}})
                annotation_counter += 1
        results_t.append(temp)

    print("Writing annotations...")
    annotations['images'] = image_list
    annotations['annotations'] = annotation_list
    with open('./all_data_pred/annotations label={}.json'.format(label), 'w') as fp:
        json.dump(annotations, fp)
 
    # print('Writing gif...')
    # temp = []
    # for result in results_t:
    #     temp.append(Image.fromarray(result).convert('L'))
    #     temp[0].save(os.path.join('./all_data_predictions/', 'label={}'.format(label)), save_all=True, append_images=temp[1:], optimize=False, duration=500, loop=0)
    print('Writing npy...')
    results_t = np.array(results_t)
    np.save('./all_data_pred/label={}, thresh.npy'.format(label), results_t)

print("Done!")