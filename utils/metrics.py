import numpy as np
import torch
import segmentation_models_pytorch as sm
import monai
from numpy.core.numeric import NaN
from warnings import warn
from monai.networks.utils import one_hot


def handle_zero_division(x, zero_division):
    nans = torch.isnan(x)
    if torch.any(nans) and zero_division == "warn":
        warn("Zero division in metric calculation!")
    value = zero_division if zero_division != "warn" else 0
    value = torch.tensor(value, dtype=x.dtype).to(x.device)
    x = torch.where(nans, value, x)
    return x

# Metrics to compute
def f1(true: torch.Tensor, pred: torch.Tensor, num_classes:int, background:bool, th=None, device='cpu') -> dict:
    f1_s_nan={}
  
    if th is None:
        if device != 'cpu':
            data=np.copy(pred.cpu().detach().numpy())
        else:
            data=np.copy(pred)
        data=np.argmax(data, axis=1)
        data=np.expand_dims(data,axis=1)
        data=torch.Tensor(data)
        data=one_hot(data,pred.shape[1], dim=1)
        data=data.type(torch.long)
        if device != 'cpu':
            data=data.to(device)
    else:
        data=torch.clone(pred)

    if background==False:
        data=data[:,1:,:,:,:]
        true=true[:,1:,:,:,:]

    tp, fp, fn, tn = sm.metrics.get_stats(data, true, mode='multilabel', threshold=th)
    f1_score_nan=sm.metrics.f1_score(tp,fp,fn,tn,zero_division=NaN)
    f1_label_nan=torch.nanmean(f1_score_nan,0) #Compute average over all patients for each class
    labels = np.array(range(num_classes))
    for label in labels:
        f1_s_nan[f"F1_{label}"] = f1_label_nan[label].item() 

    f1_dict= {"F1_mean":np.nanmean(list(f1_s_nan.values()))} #Compute average over all classes    
    f1_dict.update(f1_s_nan)
    return f1_dict

def pre_rec_spec(true: torch.Tensor, pred: torch.Tensor, num_classes: int, background:bool, th= None, device='cpu') -> dict:
    pre = {}
    rec = {}
    spec={}

    if th is None:
        if device != 'cpu':
            data=np.copy(pred.cpu().detach().numpy())
        else:
            data=np.copy(pred)
        data=np.argmax(data, axis=1)
        data=np.expand_dims(data,axis=1)
        data=torch.Tensor(data)
        data=one_hot(data,pred.shape[1], dim=1)
        data=data.type(torch.long)
        if device != 'cpu':
            data=data.to(device)
    else:
        data=torch.clone(pred)

    if background==False:
        data=data[:,1:,:,:,:]
        true=true[:,1:,:,:,:]
    tp, fp, fn, tn = sm.metrics.get_stats(data, true, mode='multilabel', threshold=th)
    precision= sm.metrics.precision(tp,fp,fn,tn, zero_division=1.0)
    recall= sm.metrics.recall(tp,fp,fn,tn, zero_division=1.0)
    specificity=sm.metrics.specificity(tp, fp, fn, tn, zero_division=1.0)
    pre_label=torch.nanmean(precision,0)  #Compute average over all patients for each class
    rec_label=torch.nanmean(recall,0)
    spec_label=torch.nanmean(specificity,0)
    labels = np.array(range(num_classes))
    for label in labels:
        rec[f"rec_{label}"] = rec_label[label].item() 
        pre[f"pre_{label}"] = pre_label[label].item() 
        spec[f"spec_{label}"]=spec_label[label].item() 
    
    dic= {"rec":np.nanmean(list(rec.values())),"pre":np.nanmean(list(pre.values())),"specificity":np.nanmean(list(spec.values()))}  #Compute average over all classes    
    dic.update(rec)
    dic.update(pre)
    dic.update(spec)
    return dic

    

def IOU(true: torch.Tensor, pred: torch.Tensor, num_classes: int, background:bool, th=None, device='cpu') -> dict:
    iou_l={}

    if th is None:
        if device != 'cpu':
            data=np.copy(pred.cpu().detach().numpy())
        else:
            data=np.copy(pred)
        data=np.argmax(data, axis=1)
        data=np.expand_dims(data,axis=1)
        data=torch.Tensor(data)
        data=one_hot(data,pred.shape[1], dim=1)
        data=data.type(torch.long)
        if device != 'cpu':
            data=data.to(device)
    else:
        data=torch.clone(pred)

    if background==False:
        data=data[:,1:,:,:,:]
        true=true[:,1:,:,:,:]
    tp, fp, fn, tn = sm.metrics.get_stats(data, true, mode='multilabel', threshold=th)
    iou= sm.metrics.iou_score(tp,fp,fn,tn,zero_division=1.0)
    iou_label=torch.nanmean(iou,0)
    labels = np.array(range(num_classes))
    for label in labels:
        iou_l[f"IoU_{label}"] = iou_label[label].item() 
    
    iou_dict= {"IoU_mean":np.nanmean(list(iou_l.values()))}
    
    iou_dict.update(iou_l)
    return iou_dict


def AUC(true: torch.Tensor, pred: torch.Tensor, num_classes: int, background:bool, th=None, device='cpu') -> dict:
    """
        Formula:
            AUC = 1 - 1/2 * (FP/(FP+TN) + FN/(FN+TP))
    """
    if th is None:
        if device != 'cpu':
            data=np.copy(pred.cpu().detach().numpy())
        else:
            data=np.copy(pred)
        data=np.argmax(data, axis=1)
        data=np.expand_dims(data,axis=1)
        data=torch.Tensor(data)
        data=one_hot(data,pred.shape[1], dim=1)
        data=data.type(torch.long)
        if device != 'cpu':
            data=data.to(device)
    else:
        data=torch.clone(pred)

    if background==False:
        data=data[:,1:,:,:,:]
        true=true[:,1:,:,:,:]
    tp, fp, fn, tn = sm.metrics.get_stats(data, true, mode='multilabel', threshold=th)
        
    auc_d={}

    x = fp/(fp+tn)
    x = handle_zero_division(x, 0.0)
    y = fn/(fn+tp)
    y=handle_zero_division(y,0.0)
    auc = 1 - (1/2)*(x + y)
   
    value = torch.tensor(1.0, dtype=auc.dtype).to(auc.device)
    auc=torch.where(x+y==0, value, auc)
    auc_label = torch.nanmean(auc,0)
    labels = np.array(range(num_classes))
    for label in labels:
        auc_d[f"AUC_{label}"] = auc_label[label].item() 
    
    auc_dict= {"AUC_mean":np.nanmean(list(auc_d.values()))}
    
    auc_dict.update(auc_d)

    return auc_dict

def kappa(true: torch.Tensor, pred: torch.Tensor, num_classes: int, background:bool, th=None, device='cpu') -> dict:
    """
        Cohen's  Kappa  
    """
    if th is None:
        if device != 'cpu':
            data=np.copy(pred.cpu().detach().numpy())
        else:
            data=np.copy(pred)
        data=np.argmax(data, axis=1)
        data=np.expand_dims(data,axis=1)
        data=torch.Tensor(data)
        data=one_hot(data,pred.shape[1], dim=1)
        data=data.type(torch.long)
        if device != 'cpu':
            data=data.to(device)
    else:
        data=torch.clone(pred)

    if background==False:
        data=data[:,1:,:,:,:]
        true=true[:,1:,:,:,:]
    tp, fp, fn, tn = sm.metrics.get_stats(data, true, mode='multilabel', threshold=th)
        
    kapp={}

    # Compute kappa
    fa = tp + tn
    fc = ((tn+fn)*(tn+fp) + (fp+tp)*(fn+tp)) / (tp+tn+fp+fn)
    kappa = (fa-fc) / ((tp+tn+fp+fn)-fc)
    kappa=handle_zero_division(kappa,1.0)
    kappa_label = torch.nanmean(kappa,0)


    labels = np.array(range(num_classes))
    for label in labels:
        kapp[f"Kappa_{label}"] = kappa_label[label].item() 
    
    kappa_dict= {"Kappa_mean":np.nanmean(list(kapp.values()))}
    
    kappa_dict.update(kapp)
    return kappa_dict

def conf_matrix(true: torch.Tensor, pred: torch.Tensor,num_classes: int,background:bool, th=None, device='cpu') -> dict:

    if th is None:
        if device != 'cpu':
            data=np.copy(pred.cpu().detach().numpy())
        else:
            data=np.copy(pred)
        data=np.argmax(data, axis=1)
        data=np.expand_dims(data,axis=1)
        data=torch.Tensor(data)
        data=one_hot(data,pred.shape[1], dim=1)
        data=data.type(torch.long)
        if device != 'cpu':
            data=data.to(device)
    else:
        data=pred>th
    
    
    #The third dimension represents the number of true positive, false positive// true negative and false negative 
    cfm = monai.metrics.get_confusion_matrix(data, true, include_background=background)
    cfm_l={}
    cfm_label=torch.nansum(cfm,0)
    labels = np.array(range(num_classes))
    for label in labels:
        cfm_l[f"Conf_matrix_{label}"] = np.reshape(cfm_label[label].numpy().tolist(),(2,2)).tolist()
    cfm_dict= {"Conf_matrix":np.reshape(np.nansum(cfm_label,0).tolist(),(2,2)).tolist()}
   
    cfm_dict.update(cfm_l)

    return cfm_dict


def hausdorff(true: torch.Tensor, pred: torch.Tensor,num_classes: int,background:bool, th: float, device='cpu') -> dict:
    """
        Hausdorff distance
    """
    h_d={}
    if th is None:
        if device != 'cpu':
            data=np.copy(pred.cpu().detach().numpy())
        else:
            data=np.copy(pred)
        data=np.argmax(data, axis=1)
        data=np.expand_dims(data,axis=1)
        data=torch.Tensor(data)
        data=one_hot(data,pred.shape[1], dim=1)
        data=data.type(torch.long)
        if device != 'cpu':
            data=data.to(device)
    else:
        data=pred>th
    
    hausdorff = monai.metrics.compute_hausdorff_distance(data, true, include_background=background)
    hausdorff_label=torch.nanmean(hausdorff,0)
    labels = np.array(range(num_classes))
    for label in labels:
        h_d[f"Hausdorff_distance_{label}"] = hausdorff_label[label].item() 
    
    dict= {"Hausdorff_distance_mean":np.nanmean(list(h_d.values()))}

    dict.update(h_d)

    return dict

def dice(true: torch.Tensor, pred: torch.Tensor,num_classes: int,background:bool, th=None, device='cpu') -> dict:
    """
        Dice Enhanced: 
            when a patient contains no lesions, and the model predicts correctly that there are no lesions,
            then it returns a value of 1. 
    """

    dice_d={}
    if th is None:
        if device != 'cpu':
            data=np.copy(pred.cpu().detach().numpy())
        else:
            data=np.copy(pred)
        data=np.argmax(data, axis=1)
        data=np.expand_dims(data,axis=1)
        data=torch.Tensor(data)
        data=one_hot(data,pred.shape[1], dim=1)
        data=data.type(torch.long)
        if device != 'cpu':
            data=data.to(device)
    else:
        data=pred>th
    
    dice= monai.metrics.compute_meandice(data, true, include_background=background,ignore_empty=False)
  
    dice_label = torch.nanmean(dice,0)
    labels = np.array(range(num_classes))
    for label in labels:
        dice_d[f"Dice_Enhanced_{label}"] = dice_label[label].item() 
    
    dict= {"Dice_Enhanced_mean":np.nanmean(list(dice_d.values()))}
    dict.update(dice_d)

    return dict

def accuracy(true: torch.Tensor, pred: torch.Tensor,num_classes: int, background:bool, th=None, device='cpu') -> dict:
    acc={}
    if th is None:
        if device != 'cpu':
            data=np.copy(pred.cpu().detach().numpy())
        else:
            data=np.copy(pred)
        data=np.argmax(data, axis=1)
        data=np.expand_dims(data,axis=1)
        data=torch.Tensor(data)
        data=one_hot(data,pred.shape[1], dim=1)
        data=data.type(torch.long)
        if device != 'cpu':
            data=data.to(device)
    else:
        data=torch.clone(pred)

    if background==False:
        data=data[:,1:,:,:,:]
        true=true[:,1:,:,:,:]
  
    tp, fp, fn, tn = sm.metrics.get_stats(data, true, mode='multilabel', threshold=th)
    accuracy = sm.metrics.accuracy(tp, fp, fn, tn, zero_division=1.0)
    acc_label=torch.nanmean(accuracy,0)
    labels = np.array(range(num_classes))
    for label in labels:
        acc[f"accuracy_{label}"] = acc_label[label].item()     
    acc_dict= {"Accuracy_mean":np.nanmean(list(acc.values()))}   
    acc_dict.update(acc)
    return acc_dict



def evaluation(true: torch.Tensor, pred: torch.Tensor,num_classes: int, background:bool, th: float, device='cpu') -> dict:
    y_true = true
    y_pred = pred
    
    measures = {}
    measures.update(pre_rec_spec(y_true, y_pred,num_classes,background,th,device))
    measures.update(accuracy(y_true, y_pred, num_classes,background,th,device))
    measures.update(IOU(y_true, y_pred,num_classes,background,th,device))
    measures.update(kappa(y_true, y_pred,num_classes,background,th,device))
    measures.update(AUC(y_true, y_pred,num_classes,background,th,device))
    #measures.update(hausdorff(y_true,y_pred, num_classes,background,th,device))
    measures.update(conf_matrix(y_true,y_pred,num_classes,background,th,device))
    measures.update(dice(y_true,y_pred,num_classes,background,th,device))
    measures.update(f1(y_true, y_pred,num_classes,background,th,device))
    return measures