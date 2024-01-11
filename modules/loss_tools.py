import torch
from typing import Any, Union

# functions defined:
#     - binary_cross_entropy_loss
#     - generalized_cross_entropy_loss
#     - binary_focal_loss
#     - generalized_focal_loss

def binary_cross_entropy_loss(pred: torch.Tensor, target: torch.Tensor, reduction: str='mean', epsilon: float=2e-7) -> Union[torch.Tensor, float]:
    """
    binary_cross_entropy_loss: calculate the binary cross-entropy loss

    Inputs:
        pred (torch.Tensor): the (batched set of) predictions
        target (torch.Tensor): the (batched set of) targets for above predictions
        reduction (str; default=mean): the reduction method
        epsilon (float; default=2e-7): the epsilon value to counteract floating-point precision issues
    Outputs:
        loss (torch.Tensor or float): the loss, which may be an entire Tensor or reduced to a single number
    """
    assert reduction in ['mean', 'none']
    
    p_t = (-1 * (target == 0) + target) * pred + (1 - target)
    ce_term = (p_t + epsilon).log()
    loss = -1 * ce_term
    
    if reduction == 'none':
        return loss
    return loss.mean()

def generalized_cross_entropy_loss(pred: torch.Tensor, target: torch.Tensor, reduction: str='mean', epsilon: float=2e-7) -> Union[torch.Tensor, float]:
    """
    generalized_cross_entropy_loss: calculate the generalized cross-entropy loss. can handle floats.

    Inputs:
        pred (torch.Tensor): the (batched set of) predictions
        target (torch.Tensor): the (batched set of) targets for above predictions
        reduction (str; default=mean): the reduction method
        epsilon (float; default=2e-7): the epsilon value to counteract floating-point precision issues.
    Outputs:
        loss (torch.Tensor or float): the loss, which may be an entire Tensor or reduced to a single number
    """
    assert reduction in ['mean', 'none']
    
    ce_term = (1 - target) * torch.log(1 - pred + epsilon) + target * torch.log(pred + epsilon)
    loss = -1 * ce_term
        
    if reduction == 'none':
        return loss
    return loss.mean()

def binary_focal_loss(pred: torch.Tensor, target: torch.Tensor, gamma: float=4, reduction: str='mean') -> Union[torch.Tensor, float]:
    """
    binary_focal_loss: calculate the binary focal loss

    Args:
        pred (torch.Tensor): the (batched set of) predictions
        target (torch.Tensor): the (batched set of) targets for above predictions
        beta (float; default=4): the exponentiating factor for focal loss
        reduction (str; default=mean): the reduction method
        epsilon (float; default=2e-7): the epsilon value to counteract floating-point precision issues
    Returns:
        loss (torch.Tensor or float): the loss, which may be an entire Tensor or reduced to a single number
    """
    assert reduction in ['mean', 'none']
    
    p_t = (-1 * (target == 0) + target) * pred + (1 - target)
    focal_term = (1 - p_t) ** gamma
    ce_term = p_t.log()
    loss = -1 * focal_term * ce_term

    if reduction == 'none':
        return loss
    return loss.mean()

def generalized_focal_loss(pred: torch.Tensor, target: torch.Tensor, gamma: float=4, reduction: str='mean') -> Union[torch.Tensor, float]:
    """
    generalized_focal_loss: calculate the generalized focal loss. can handle floats. adapted from https://arxiv.org/abs/2006.04388
    Args:
        pred (torch.Tensor): the (batched set of) predictions
        target (torch.Tensor): the (batched set of) targets for above predictions
        beta (float; default=4): the exponentiating factor for focal loss
        reduction (str; default=mean): the reduction method
        epsilon (float; default=2e-7): the epsilon value to counteract floating-point precision issues
    Returns:
        loss (torch.Tensor or float): the loss, which may be an entire Tensor or reduced to a single number
    """
    assert reduction in ['mean', 'none']
    
    p_t = (-1 * (target == 0) + target) * pred + (1 - target)
    focal_term = (1 - p_t) ** gamma
    ce_term = p_t.log()
    loss = -1 * focal_term * ce_term
    
    if reduction == 'none':
        return loss
    return loss.mean()
    
class loss_function_wrapper():
    """
    loss_function_wrapper: a wrapper class to allow for quicker swapping between different loss functions
    """
    def __init__(self, name: str, **kargs: Any) -> None:
        """
        __init__: specifies the loss function to be encapsulated in the wrapper

        Inputs:
            name (str): the name of the loss function to use
            kargs (Any): any keyword arguments specific to the loss function being used
        """
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

    def __repr__(self) -> str:
        """
        __repr__: presents a clean(er) representation of the loss function and any specific keyword arguments
        
        Outputs:
            *unnamed* (str): a formatted string that represents an instance of this wrapper class
        """
        return "loss_function_wrapper(type={}, kargs={})".format(self.name, self.kargs)

    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> Union[torch.Tensor, float]:
        """
        __call__: the forward function that calculates loss given a prediction and a target

        Inputs:
            pred (torch.Tensor): the (batched set of) predictions
            target (torch.Tensor): the (batched set of) targets for above predictions
        Outputs:
            *unnamed* (Union[torch.Tensor, float]): the loss according to the predetermind loss function
        """
        return self.func(pred, target, **(self.kargs))