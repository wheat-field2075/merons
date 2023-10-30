import os
import torch
import itertools
import numpy as np
from PIL import Image

"""
Functions defined:
    - write_to_pred
    - predict_on_image
    - get_pr
    - get_pr_image
"""

""" ##############################
write_to_pred: helper function for predict_on_image. writes predicted patches into a mask

Args:
    y: the y coordinate of the top-left corner
    x: the x coordinate of the top-left corner
    model: the model to evaluate
    source: the source image
    prediction: the predicted mask to write into
    patch_size: the relevant patch size
    device: the device to send Tensors to
############################## """
def write_to_pred(y, x, model, source, prediction, patch_size, device):
    with torch.no_grad():
        model.eval()
        source_patch = np.expand_dims(source[y:y+patch_size, x:x+patch_size], [0, 1])
        pred_patch = model(torch.Tensor(source_patch).to(device) / 255)
        pred_patch = pred_patch.detach().cpu().numpy().squeeze()
        prediction[y:y+patch_size, x:x+patch_size] = np.maximum(prediction[y:y+patch_size, x:x+patch_size],
                                                               pred_patch)
""" ######################### """

""" ##############################
predict_on_image: predicts the mask for an entire image

Args:
    model: the model to be evaluated
    source: the source image
    patch_size: the patch size that the model was trained on
    device: the device to send Tensors to
Returns:
    prediction: the predicted mask
############################## """
def predict_on_image(model, source, patch_size, device):
    prediction = np.zeros(source.shape)

    # part 1 (most of image)
    for y, x in itertools.product(range(0, source.shape[0] - patch_size, patch_size // 2),
                                 range(0, source.shape[1] - patch_size, patch_size // 2)):
        write_to_pred(y, x, model, source, prediction, patch_size, device)

    # part 2 (along lower border)
    y = source.shape[0] - patch_size
    for x in range(0, source.shape[1] - patch_size, patch_size // 2):
        write_to_pred(y, x, model, source, prediction, patch_size, device)

    # part 3 (along rightmost border)
    x = source.shape[1] - patch_size
    for y in range(0, source.shape[0] - patch_size, patch_size // 2):
        write_to_pred(y, x, model, source, prediction, patch_size, device)

    # part 4 (lower right corner)
    y, x = [source.shape[0] - patch_size, source.shape[1] - patch_size]
    write_to_pred(y, x, model, source, prediction, patch_size, device)

    return prediction
""" ######################### """

""" ##############################
get_pr: calculate the precision and recall, in that order, betwen a target and prediction

Args:
    target: a binary target mask
    prediction: a binary prediction mask
Returns:
    precision: TP / (TP + FP), calculated per-pixel
    recall: TP / (TP + FN), calculated per-pixel
############################## """
def get_pr(target, prediction):
    precision = (target * prediction).sum() / prediction.sum()
    recall = (target * prediction).sum() / target.sum()
    return precision, recall
""" ######################### """

""" ##############################
get_pr_image: calculate the precision and recall, in that order, between a target and prediction
    given only the name of the image, as opposed to the target abd rpediction masks

Args:
    dir_name: the parent directory that the image resides in
    image_name: the name of the image
    model: the model to use to generate the prediction
    patch_size: the patch size that the model is designed to take
    device: the device to send Tensors to
Returns:
    precision: TP / (TP + FP), calculated per-pixel
    recall: TP / (TP + FN), calculated per-pixel
############################## """
def get_pr_image(dir_name, image_name, model, patch_size, device):
    image = Image.open(os.path.join(dir_name, 'images', image_name)).convert('L')
    mask = Image.open(os.path.join(dir_name, 'masks', image_name)).convert('L')

    image = np.array(image) / 255
    mask = np.array(mask) / 255

    prediction = predict_on_image(model, image, patch_size, device)
    precision, recall = get_pr(mask, prediction)
    return precision, recall
""" ######################### """