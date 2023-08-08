"""
losstools: multiple loss functions and a wrapper class to allow for easier switching between
loss functions
"""

import torch

""" ##############################
binary_cross_entropy_loss: calculate the binary cross-entropy loss

Args:
    pred: the (batched set of) predictions
    target: the (batched set of) targets for above predictions
    reduction: the reduction method. Default: mean
    epsilon: the epsilon value to counteract floating-point precision issues. Default: 2e-7
Returns:
    loss: the loss, which may be an entire Tensor or reduced to a single number
############################## """

def binary_cross_entropy_loss(pred, target, reduction='mean', epsilon=2e-7):
    assert reduction in ['mean', 'none']
    
    p_t = (-1 * (target == 0) + target) * pred + (1 - target)
    ce_term = (p_t + epsilon).log()
    loss = -1 * ce_term
    
    if reduction == 'none':
        return loss
    return loss.mean()
""" ############################## """


""" ##############################
generalized_cross_entropy_loss: calculate the generalized cross-entropy loss. can handle floats.

Args:
    pred: the (batched set of) predictions
    target: the (batched set of) targets for above predictions
    reduction: the reduction method. Default: mean
    epsilon: the epsilon value to counteract floating-point precision issues. Default: 2e-7
Returns:
    loss: the loss, which may be an entire Tensor or reduced to a single number
############################## """

def generalized_cross_entropy_loss(pred, target, beta=4, reduction='mean', epsilon=2e-7):
    assert reduction in ['mean', 'none']
    
    ce_term = (1 - target) * torch.log(1 - pred + epsilon) + target * torch.log(pred + epsilon)
    loss = -1 * ce_term
        
    if reduction == 'none':
        return loss
    return loss.mean()
""" ############################## """

""" ##############################
binary_focal_loss: calculate the binary focal loss

Args:
    pred: the (batched set of) predictions
    target: the (batched set of) targets for above predictions
    beta: the exponentiating factor for focal loss. Default: 4
    reduction: the reduction method. Default: mean
    epsilon: the epsilon value to counteract floating-point precision issues. Default: 2e-7
Returns:
    loss: the loss, which may be an entire Tensor or reduced to a single number
############################## """

def binary_focal_loss(pred, target, gamma=4, reduction='mean'):
    assert reduction in ['mean', 'none']
    
    p_t = (-1 * (target == 0) + target) * pred + (1 - target)
    focal_term = (1 - p_t) ** gamma
    ce_term = p_t.log()
    loss = -1 * focal_term * ce_term
    
    if reduction == 'none':
        return loss
    return loss.mean()
""" ############################## """

""" ##############################
generalized_focal_loss: calculate the generalized focal loss. can handle floats. adapted from
    https://arxiv.org/abs/2006.04388

Args:
    pred: the (batched set of) predictions
    target: the (batched set of) targets for above predictions
    gamma: the exponentiating factor for focal loss. Default: 4
    reduction: the reduction method. Default: mean
    epsilon: the epsilon value to counteract floating-point precision issues. Default: 2e-7
Returns:
    loss: the loss, which may be an entire Tensor or reduced to a single number
############################## """

def generalized_focal_loss(pred, target, gamma=4, reduction='mean'):
    assert reduction in ['mean', 'none']
    
    p_t = (-1 * (target == 0) + target) * pred + (1 - target)
    focal_term = (1 - p_t) ** gamma
    ce_term = p_t.log()
    loss = -1 * focal_term * ce_term
    
    if reduction == 'none':
        return loss
    return loss.mean()
""" ############################## """
    
""" ##############################
loss_function_wrapper: a wrapper class to allow for quicker swapping between different loss
    functions

__init__: specifies the loss function to be encapsulated in the wrapper
Args:
    name: the name of the loss function to use
    kargs: any keyword arguments specific to the loss function being used
    
__repr__: presents a clean(er) representation of the loss function and any specific keyword
    arguments
Returns:
    *unnamed*: a formatted string that represents an instance of this wrapper class

__call__: the forward function that calculates loss given a prediction and a target
Args:
    pred: the (batched set of) predictions
    target: the (batched set of) targets for above predictions
Returns:
    *unnamed*: the loss according to the predetermind loss function
############################## """

class loss_function_wrapper():
    def __init__(self, name, **kargs):
        assert name in ['bcel', 'gcel', 'bfl', 'gfl']
        self.name = name
        self.kargs = kargs

        if (self.name == 'bcel'):
            self.func = binary_cross_entropy_loss
        if (self.name == 'gcel'):
            self.func = generalized_cross_entropy_loss
        if (self.name == 'bfl'):
            self.func = binary_focal_loss
        if (self.name == 'gfl'):
            self.func = generalized_focal_loss

    def __repr__(self):
        return "loss_function_wrapper(type={}, kargs={})".format(self.name, self.kargs)

    def __call__(self, pred, target):
        return self.func(pred, target, **(self.kargs))
""" ############################## """