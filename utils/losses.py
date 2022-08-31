
import numpy as np
from monai.losses import DiceLoss, DiceCELoss, GeneralizedDiceLoss
from monai.utils.enums import Weight
import torch
import math

def calculate_pixels(data,num_classes,background):
    '''
    This function calculates the total number of pixels associated with each class in the entire dataset. 
    '''
    if not background:
        num_classes_n=num_classes - 1
    else:
        num_classes_n=num_classes
    
    val = np.zeros(num_classes)

    for batch in data:      
        for i in range(num_classes_n):
            if not background:
                j=i+1
            else:
                j=i
            batch_label = batch[1][0,j,:,:,:]
            _, count = np.unique(batch_label, return_counts=True)
            if len(count)>1:
                val[j] +=count[1]            
    return val

def calculate_weights(val):
    '''
    This function takes the number of pixels of each class in the dataset and returns the weight
    associated with each class for the cross-entropy loss.
    '''
    count = np.array(val)
    summ = count.sum() #Total number of pixels in the dataset
    weights = count/summ #Relative frequency of each class
    weights = 1/weights #Inverse of the relative frequency of each class
    if weights[0]==math.inf:
        weights[0]=0
    summ = weights.sum() #Summation of all weights
    weights = weights/summ #To make the weights add up to 1
    return torch.tensor(weights, dtype=torch.float32)

def LossFunction(params: dict, data: torch.utils.data.DataLoader, device:torch.device):
    num_classes =       int(params.get("num_classes", 1))
    weighted    =       bool(params.get('weighted_loss',True))
    loss        =       params.get("loss","categorical_crossentropy")
    background  =       params.get('background',True)
    loss_name = loss
    
    if "jaccard_loss" in loss:
        loss = DiceLoss(include_background=background, to_onehot_y=False, softmax=True, squared_pred=False,jaccard=True),
        loss=loss[0]
    elif "dice_loss" in loss:
        loss = DiceLoss(include_background=background, to_onehot_y=False, softmax=True, squared_pred=False),
        loss=loss[0]
    elif "generalized_dice" in loss: 
        loss= GeneralizedDiceLoss(include_background=background, to_onehot_y=False, softmax=True, w_type=Weight.SQUARE),
        loss=loss[0]
    elif "categorical_crossentropy" in loss: 
            if weighted:
                val=calculate_pixels(data, num_classes, background)
                weights=calculate_weights(val)
                weights=weights.to(device)
            else:
                weights=None
            dice=params.get("weight_dice", 0.0)
            ce = params.get("weight_ce", 1.0)
            jaccard = params.get("jaccard_ce", False)
            loss= DiceCELoss(include_background=background, to_onehot_y=False,softmax=True, squared_pred=False,
                                    jaccard = jaccard, ce_weight=weights, lambda_dice=dice, lambda_ce=ce)
    
    loss.__name__ = loss_name

    return loss
