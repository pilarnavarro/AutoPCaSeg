#!/usr/bin/python3
import os
import numpy as np
from metrics import AUC, evaluation, dice, IOU, accuracy, f1
from torch.optim.lr_scheduler import CosineAnnealingLR
from monai.networks.nets import UNet, BasicUnet, VNet, DynUNet
import numpy as np
from monai.networks.layers.factories import Conv
import torch
from torch.optim import Adam, SGD
from torch.utils.tensorboard import SummaryWriter
import math
import torch.nn as nn
from monai.networks.blocks.convolutions import Convolution, ResidualUnit
from typing import Union
from monai.networks.nets.vnet import  OutputTransition
from preprocess import postprocessing
from losses import LossFunction
from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.data import TestTimeAugmentation
from functools import partial
from monai.transforms import(
    Compose,
    RandRotated,
    RandFlipd,
)

def get_model(params):
    model = params.get("model")
    num_classes = params.get("num_classes")
    in_channels = params.get("in_channels",1)
    dims=params.get("dims",3)
    dropout=params.get("dropout",0.0) 
    modeloFinal = None
    if model == "Unet":
        modeloFinal = UNet(
        spatial_dims=dims,
        in_channels=in_channels,
        out_channels=num_classes,
        channels= params.get('channels'), 
        strides= params.get('strides'),
        kernel_size=params.get('kernel_size', 3),
        up_kernel_size=params.get('kernel_size', 3),
        dropout=dropout,
        num_res_units= params.get('res_units',2),
        act=params.get('act','PRELU'),
        norm=params.get('norm','INSTANCE'), 
        bias=True)
        
    elif model == "Vnet":
        modeloFinal =  VNet(spatial_dims=dims, 
                       in_channels=in_channels,
                       out_channels=num_classes,
                       act=params.get('act','PRELU'),
                       dropout_prob=dropout, 
                       dropout_dim=3, 
                       bias=False)
    elif model == "BasicUnet":
        modeloFinal =  BasicUnet(spatial_dims=dims, 
                                 in_channels=in_channels, 
                                 out_channels=num_classes, 
                                 features= params.get('channels'), 
                                 act=params.get('act',('LeakyReLU', {'negative_slope': 0.1, 'inplace': True})),
                                 norm=params.get('norm','BATCH'), 
                                 bias=False, 
                                 dropout=dropout, 
                                 upsample='deconv')
    elif model == "Dyn":
        strides=params.get('strides')
        modeloFinal = DynUNet(   
            spatial_dims=dims,
            in_channels=in_channels,
            out_channels=num_classes,
            filters=params.get('channels'),
            kernel_size=params.get('kernel_size', 3),
            strides=strides,
            upsample_kernel_size=strides[1:],
            dropout=dropout,
            deep_supervision=False, 
            deep_supr_num=1,
            res_block=True,
            norm_name=params.get('norm',('INSTANCE', {'affine': True})),
            act_name=params.get('act',('leakyrelu', {'inplace': True, 'negative_slope': 0.01})),
            trans_bias=False
        )
    else:
        print("No model named",model)
    return modeloFinal

def get_pretrained_model(params:dict):
    if params.get("model_pre",False):
        paramsPre = {
        "model":params.get("model"),
        "num_classes":params.get("pre_num_classes"),
        "pre_num_classes":params.get("pre2_num_classes"),
        "in_channels":params.get("in_channels",1),
        "dims":params.get("dims",3),
        "dropout":params.get("dropout",0.0),
        "pretrained_folder":params.get("pretrained_folder2"),
        "channels": params.get('channels'),
        "strides" :params.get('strides'),
        "kernel_size":params.get('kernel_size', 3),       
        "res_units" : params.get('res_units',2),
        "act": params.get('act','PRELU'),
        "norm": params.get('norm','INSTANCE'), 
        }

        model_pre=get_pretrained_model(paramsPre)
    
    model = params.get("model")
    num_classes = params.get("num_classes")
    in_channels = params.get("in_channels",1)
    dims=params.get("dims",3)
    dropout=params.get("dropout",0.0)
    pre_num_classes  = params.get("pre_num_classes")
    pretrained_folder = params.get("pretrained_folder")
    pretrained_weights  = f"{pretrained_folder}/weights.best.pt"
    paramsPretrained = {
        "model":model,
        "num_classes":pre_num_classes,
        "weights_file":pretrained_weights,
        "in_channels":in_channels,
        "dims":dims,
        "dropout":dropout,
        "channels": params.get('channels'),
        "strides" :params.get('strides'),
        "kernel_size":params.get('kernel_size', 3),       
        "res_units" : params.get('res_units',2),
        "act": params.get('act','PRELU'),
        "norm": params.get('norm','INSTANCE'),       
    }
    if params.get("model_pre"):
        pre_model = load_model(paramsPretrained, model_pre)
    else:
        pre_model = load_model(paramsPretrained)
    
    if model=='Unet':
        layers = list(pre_model.children())
        if len(layers)>1:
            sublayers=layers
        else:
            orig_fc = layers[0]
            sublayers=list(orig_fc.children())
        num_res_units=params.get('res_units',2)          
        conv: Union[Convolution, nn.Sequential]
        conv = Convolution(
            dims,
            params.get('channels')[1],
            num_classes,
            strides=params.get('strides')[0],
            kernel_size=params.get('kernel_size')[0],
            act=params.get('act','PRELU'),
            norm=params.get('norm','INSTANCE'), 
            dropout=dropout,
            bias=True,
            conv_only=num_res_units == 0,
            is_transposed=True,
        )
        up=conv
        if num_res_units > 0:
            ru = ResidualUnit(
                dims,
                num_classes,
                num_classes,
                strides=1,
                kernel_size=params.get('kernel_size')[0],
                subunits=1,
                act=params.get('act','PRELU'),
                norm=params.get('norm','INSTANCE'), 
                dropout=dropout,
                bias=True,
                last_conv_only=True,
            )
            up = nn.Sequential(conv, ru)
        final_model=nn.Sequential(sublayers[0],sublayers[1], up)
    elif model == "Vnet":       
        pre_model.out_tr=OutputTransition(dims, 32, num_classes, params.get('act','PRELU'), bias=False)
        final_model=pre_model

    elif model == "Dyn":
        pre_model.output_block  = UnetOutBlock(pre_model.spatial_dims, pre_model.filters[0], num_classes, dropout=pre_model.dropout)
        final_model=pre_model
    elif model == "BasicUnet":
        features=params.get('channels')
        pre_model.final_conv = Conv["conv", dims](features[5], num_classes, kernel_size=1)
        final_model=pre_model
    return final_model


def Optimizer(params: dict,model):
    opt             = params.get("optimizer")
    learning_rate   = params.get("learning_rate")
    optimizer = None
    if opt == "adam":
        optimizer = Adam(model.parameters(),lr=learning_rate,betas=(float(params.get("beta1", 0.9)),float(params.get("beta2", 0.999))), weight_decay=params.get('wd',0))
    elif opt == "sgd":
        optimizer = SGD(model.parameters(),lr=learning_rate)
    return optimizer

def Metrics():
    metrics = [
        dice, IOU,AUC, accuracy, f1
    ]
    return metrics
    

#This function returns the trained model together with the training and validation loss
#obtained at the best epoch (that is, the epoch when the lowest validation loss was reached)
def train_model(params: dict, x: np.ndarray, y: np.ndarray, x_val: np.ndarray, y_val: np.ndarray):
    background  =       params.get('background',True)
    writer_train = SummaryWriter(log_dir=os.path.join(params['name'], 'logs_train'))
    writer_valid = SummaryWriter(log_dir=os.path.join(params['name'], 'logs_valid'))  
    device      =       torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size  =       params.get("batch_size",4)
    save_dir    =       params.get('name')
    epochs      =       int(params.get("epochs",200))
    save_file   =       os.path.join(save_dir, 'weights.best.pt')
    scheduler = params.get("scheduler", None )
    num_classes = params.get("num_classes",2)
    pretrained = params.get("pretrained",False)
    ncs = params.get("ncs",True)
    cancer = params.get('cancer',False)

    if pretrained:
        model=get_pretrained_model(params)
    else:
        model =  get_model(params)
    
    opt = Optimizer(params,model)
    metrics = Metrics()

    x = torch.Tensor(x.copy())
    y = torch.Tensor(y.copy()).type(torch.long)

    x_val = torch.Tensor(x_val.copy())
    y_val = torch.Tensor(y_val.copy()).type(torch.long)
    
    trainDS = torch.utils.data.TensorDataset(x,y)       
    valDS = torch.utils.data.TensorDataset(x_val,y_val)
    
    trainLoader  = torch.utils.data.DataLoader(trainDS, batch_size=batch_size, shuffle=True, drop_last=True, )
    valLoader = torch.utils.data.DataLoader(valDS,shuffle=False)

    lossF = LossFunction(params,trainLoader,device)
    th=params.get('th',None)
  
    model.to(device)
    best_val_loss = None
    best_val_loss_train = None
    train_logs = {}
    valid_logs = {}
    if scheduler=='CosineAnnealingLR':
            scheduler = CosineAnnealingLR(opt, T_max=100)
    
    # Actually training during a fixed number of epochs
    for i in range(epochs):
        print("-" * 10)
        print('\nEpoch: {}'.format(i))
        model.train()
        train_epoch_loss = 0 #Training loss at each epoch 
        train_step = 0
        epoch_metric_train = {"Accuracy_mean":0, "IoU_mean":0, 'Dice_Enhanced_mean':0, "AUC_mean":0, "F1_mean":0,"F1_cs":0,"F1_ncs":0,"Dice_cs":0,"Dice_ncs":0}
        
        #Training dataset
        for batch_data in trainLoader:         
            train_step += 1  

            volume = batch_data[0]
            label = batch_data[1]
            volume, label = (volume.to(device), label.to(device))
            opt.zero_grad()
            outputs = model(volume)

            train_loss = lossF(outputs, label)
            
            train_loss.backward()
            opt.step()

            if scheduler is not None:
                scheduler.step()

            train_epoch_loss += train_loss.item()
            # print(
            #     f"{train_step}/{len(trainLoader)}, "
            #     f"Train_loss: {train_loss.item():.4f}")
           
            label=label.type(torch.long)
            softmax_activation = torch.nn.Softmax(dim=1)
            outputs = softmax_activation(outputs)
            
            if background==False:
                num_classes_m=num_classes-1
            else:
                num_classes_m=num_classes
               
            for metric in metrics:
                if metric==dice:
                    train_metric = metric(label, outputs,num_classes_m,background,th=th,device=device)
                    train_metric_mean = train_metric['Dice_Enhanced_mean']
                    epoch_metric_train['Dice_Enhanced_mean']+=train_metric_mean
                    #print(f'Train_dice: {train_metric_mean:.4f}')
                    if cancer:
                        if background: k=1
                        else: k=0
                        train_metric_cs = train_metric["Dice_Enhanced_"+str(k)]
                        epoch_metric_train["Dice_cs"]+=train_metric_cs
                        if ncs:
                            train_metric_ncs = train_metric["Dice_Enhanced_"+str(k+1)]
                            epoch_metric_train["Dice_ncs"]+=train_metric_ncs  
                elif metric==accuracy:
                    train_metric = metric(label, outputs,num_classes_m,background,th=th,device=device)
                    train_metric = train_metric["Accuracy_mean"]
                    epoch_metric_train["Accuracy_mean"]+=train_metric
                elif metric==f1:
                    train_metric = metric(label, outputs,num_classes_m,background,th=th,device=device)
                    train_metric_mean = train_metric["F1_mean"]
                    epoch_metric_train["F1_mean"]+=train_metric_mean
                    if cancer:
                        if background: k=1
                        else: k=0
                        train_metric_cs = train_metric["F1_"+str(k)]
                        epoch_metric_train["F1_cs"]+=train_metric_cs
                        if ncs:
                            train_metric_ncs = train_metric["F1_"+str(k+1)]
                            epoch_metric_train["F1_ncs"]+=train_metric_ncs       
                elif metric==IOU:
                    train_metric = metric(label, outputs,num_classes_m, background,th=th,device=device)
                    train_metric = train_metric["IoU_mean"]
                    epoch_metric_train["IoU_mean"]+=train_metric
                elif metric==AUC:
                    train_metric = metric(label, outputs,num_classes_m,background,th=th,device=device)
                    train_metric= train_metric["AUC_mean"]     
                    epoch_metric_train["AUC_mean"]+=train_metric
            

        #print('-'*20)

        #Update loss logs
        train_epoch_loss /= train_step
        #print(f'Epoch_loss: {train_epoch_loss:.4f}')
        loss_logs = {lossF.__name__: train_epoch_loss}
        train_logs.update(loss_logs)
        
        #Update metrics logs
        for key in epoch_metric_train.keys():
            epoch_metric_train[key] /= train_step
            metrics_logs = {key: epoch_metric_train[key]}
            train_logs.update(metrics_logs)
            #if key == 'Dice_Enhanced_mean':
                #print(f'Epoch_metric_dice: { epoch_metric_train[key]:.4f}')
 

        # Validation dataset
        model.eval()
        with torch.no_grad():
            test_epoch_loss = 0
            test_metric = 0
            epoch_metric_test = {"Accuracy_mean":0, "IoU_mean":0, 'Dice_Enhanced_mean':0, "AUC_mean":0,"F1_mean":0,"F1_cs":0,"F1_ncs":0,"Dice_cs":0,"Dice_ncs":0}
            test_step = 0

            for val_data in valLoader:

                test_step += 1

                test_volume = val_data[0]
                test_label = val_data[1]
                test_volume, test_label = (test_volume.to(device), test_label.to(device),)
                
                test_outputs = model(test_volume)
                
                test_loss = lossF(test_outputs, test_label)
                test_epoch_loss += test_loss.item()
                test_label=test_label.type(torch.long)
                softmax_activation = torch.nn.Softmax(dim=1)
                
                test_outputs = softmax_activation(test_outputs)

                if background==False:
                    num_classes_m=num_classes-1
                else:
                    num_classes_m=num_classes
                           
                for metric in metrics:
                    if metric==dice:
                        test_metric = metric(test_label, test_outputs,num_classes_m,background,th=th,device=device)
                        test_metric_mean = test_metric['Dice_Enhanced_mean']
                        epoch_metric_test['Dice_Enhanced_mean']+=test_metric_mean
                        if cancer:
                            if background: k=1
                            else: k=0
                            test_metric_cs = test_metric["Dice_Enhanced_"+str(k)]
                            epoch_metric_test["Dice_cs"]+=test_metric_cs
                            if ncs:
                                test_metric_ncs = test_metric["Dice_Enhanced_"+str(k+1)]
                                epoch_metric_test["Dice_ncs"]+=test_metric_ncs  
                    elif metric==f1:
                        test_metric = metric(test_label, test_outputs,num_classes_m,background,th=th,device=device)
                        test_metric_mean = test_metric["F1_mean"]
                        epoch_metric_test["F1_mean"]+=test_metric_mean
                        if cancer:
                            if background: k=1
                            else: k=0
                            test_metric_cs = test_metric["F1_"+str(k)]
                            epoch_metric_test["F1_cs"]+=test_metric_cs
                            if ncs:
                                test_metric_ncs = test_metric["F1_"+str(k+1)]
                                epoch_metric_test["F1_ncs"]+=test_metric_ncs  
                    elif metric==accuracy:
                        test_metric = metric(test_label, test_outputs,num_classes_m,background,th=th,device=device)
                        test_metric = test_metric["Accuracy_mean"]
                        epoch_metric_test["Accuracy_mean"]+=test_metric
                    elif metric==IOU:
                        test_metric = metric(test_label, test_outputs,num_classes_m,background,th=th,device=device)
                        test_metric = test_metric["IoU_mean"]
                        epoch_metric_test["IoU_mean"]+=test_metric
                    elif metric==AUC:
                        test_metric = metric(test_label, test_outputs,num_classes_m,background,th=th,device=device)
                        test_metric= test_metric["AUC_mean"]  
                        epoch_metric_test["AUC_mean"]+=test_metric
                       
                
            #Update loss logs
            test_epoch_loss /= test_step
            #print(f'Test_loss_in_epoch: {test_epoch_loss:.4f}')
            loss_logs = {lossF.__name__: test_epoch_loss}
            valid_logs.update(loss_logs)

            #Update metric logs
            for key in epoch_metric_test.keys():
                epoch_metric_test[key] /= test_step
                metrics_logs = {key: epoch_metric_test[key]}
                valid_logs.update(metrics_logs)
                #if key == 'Dice_Enhanced_mean':
                    #print(f'Test_dice_in_epoch: {epoch_metric_test[key]:.4f}')
          
        writer_train.add_scalar('Loss', train_logs[lossF.__name__], i)
        writer_valid.add_scalar('Loss', valid_logs[lossF.__name__], i)
        writer_train.add_scalar('Dice', train_logs['Dice_Enhanced_mean'], i)
        writer_valid.add_scalar('Dice', valid_logs['Dice_Enhanced_mean'], i)
        writer_train.add_scalar('F1_mean', train_logs['F1_mean'], i)
        writer_valid.add_scalar('F1_mean', valid_logs['F1_mean'], i)
        if cancer and ncs:
            writer_train.add_scalars('F1_lesions', {'F1_cs':train_logs['F1_cs'],'F1_ncs':train_logs['F1_ncs']}, i)
            writer_valid.add_scalars('F1_lesions', {'F1_cs':valid_logs['F1_cs'],'F1_ncs':valid_logs['F1_ncs']}, i)
            writer_train.add_scalars('Dice_lesions', {'Dice_cs':train_logs['Dice_cs'],'Dice_ncs':train_logs['Dice_ncs']}, i)
            writer_valid.add_scalars('Dice_lesions', {'Dice_cs':valid_logs['Dice_cs'],'Dice_ncs':valid_logs['Dice_ncs']}, i)
        elif cancer:
            writer_train.add_scalar('F1_cs', train_logs['F1_cs'], i)
            writer_valid.add_scalar('F1_cs', valid_logs['F1_cs'], i)
            writer_train.add_scalar('Dice_cs', train_logs['Dice_cs'], i)
            writer_valid.add_scalar('Dice_cs', valid_logs['Dice_cs'], i)
        writer_train.add_scalar('IOU', train_logs['IoU_mean'], i)
        writer_valid.add_scalar('IOU', valid_logs['IoU_mean'], i)
        writer_train.add_scalar('Accuracy', train_logs["Accuracy_mean"], i)
        writer_valid.add_scalar('Accuracy', valid_logs["Accuracy_mean"], i)
        writer_train.add_scalar('AUC', train_logs["AUC_mean"], i)
        writer_valid.add_scalar('AUC', valid_logs["AUC_mean"], i)

        if best_val_loss is None or best_val_loss > valid_logs[lossF.__name__]:
            best_val_loss = valid_logs[lossF.__name__]
            best_val_loss_train = train_logs[lossF.__name__]
            best_metric_epoch = i
            best_metric = valid_logs['Dice_Enhanced_mean']
            torch.save(model.state_dict(), save_file)
            #print(f"New best model reached. Best val_loss:{best_val_loss:.4f}. Best mean Dice: {best_metric:.4f}")
    state_dict = torch.load(save_file)
    model.load_state_dict(state_dict)
    model.eval()

    print(
        f"Train completed, Best_metric: {best_metric:.4f} "
        f"at epoch: {best_metric_epoch}")

    return model, ( best_val_loss_train , best_val_loss )


def inferrer_func(patient: torch.Tensor, model):
    prediction = model(patient)
    softmax_activation = torch.nn.Softmax(dim=1)
    prediction = softmax_activation(prediction)
    return prediction


#This function takes the trained model (result), the images whose predictions we want to compute,
#and their ground truth masks, and uses the trained model to make the predictions, which are then returned by this function.
def predict(result, x: np.ndarray, y:np.ndarray, params: dict, postp=False,tta=False):
    device      =       torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if type(result)==list and len(result)==params.get('kfold',3):
        preds=[]
        iters=len(result)
        ensemble=True
    else:
        iters=1
        ensemble=False

    for i in range(iters):
        if ensemble:
            model=result[i]
        else:
            model=result

        num_classes = params.get("num_classes")

        x = torch.Tensor(x)
        y = torch.Tensor(y)
        predictions=[]

        with torch.no_grad(): # Do not compute the gradients during inference 
            model = model.to(device)
            for patient,label in zip(x,y):
                if tta: #Apply test-time augmentation
                    tta_transforms = Compose([ 
                        RandFlipd(keys=["image","label"], prob=0.7),
                        RandRotated(keys=["image","label"], prob=0.7, range_z=0.349066, mode=('bilinear','nearest'), padding_mode='border'), #Rotate 20 degrees at most.
                    ])

                    tt_aug = TestTimeAugmentation(tta_transforms, batch_size=1, num_workers=0, inferrer_fn=partial(inferrer_func,model=model), device=device)
                    d={'image':patient,'label':label}

                    mode, mean, std, vvc = tt_aug(d, num_examples=params.get('num_tta',10))
        
                    prediction=torch.Tensor(mean)
                else:
                    patient=torch.unsqueeze(patient, 0)
                    patient = patient.to(device)
                    prediction=inferrer_func(patient,model)

                prediction = prediction.to("cpu")
                predictions.append(prediction.numpy())
        
                # Free up memory
                del patient,label
           
            if np.array(predictions).shape[1]==1:
                predictions=np.squeeze(predictions, axis=1)
           
            torch.cuda.empty_cache()
                     
            if ensemble:
                preds.append(predictions)
            else:
                preds=predictions
    if ensemble: #Cross-validation ensemble
        preds=np.mean(preds,0)
 
    if postp: #Apply postprocessing
        cc=[3] 
        if params.get('ncs',True): cc.append(3)
        if num_classes>3:
            cc.append(1)
            cc.append(1)
        preds=postprocessing(preds, labels=[i for i in range(1, num_classes)], num_c=cc, th=params.get('th',None), spatial_dims=(0.5,0.5,1.25), min_vol=params.get('min_vol',40))
    return torch.Tensor(preds)

#To evaluate the given model. 
def evaluate(model, x: np.ndarray, y: np.ndarray, params: dict, postp=False,tta=False):
    model_alone ,(tr_loss,val_loss) = model    
       
    background = params.get('background',True)
    num_classes = params.get('num_classes',2)
    preds = predict(model_alone, x, y,params, postp,tta) 
  
    y = torch.Tensor(y)
    y=y.type(torch.long)

    if background==False:
        num_classes=num_classes-1
    eval_dict = evaluation(y,preds,num_classes,background, params.get('th',None))
    eval_dict.update(loss=tr_loss, val_loss=val_loss)

    return eval_dict

def load_model(params, model=None): 
    weights_file = params.get("weights_file")
    if model is None:
        model = get_model(params)
    state_dict = torch.load(weights_file)
    model.load_state_dict(state_dict,strict=False)
    model.eval()
    return model