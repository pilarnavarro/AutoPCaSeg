import numpy as np
import random
import os
from sklearn.model_selection import  train_test_split
from tensorflow.keras.backend import clear_session
from train import *

from save import *
import json
import torch
from preprocess import augmentation, prepare_2
from monai.transforms import(
    RandFlipd,
    RandAffined,
    RandZoomd,
    RandRotated,
    Rand3DElasticd,
    RandAdjustContrastd,
    RandGaussianNoised,
    ToTensord
)

RANDOM_SEED=123456789
np.random.seed(RANDOM_SEED)

def save_folds(folds,foldFile):
    foldDict = {}
    for i, (train_index, val_index) in enumerate(folds):
        foldDict["fold"+str(i)] = {"train":train_index.tolist(),"test":val_index.tolist()}
    
    with open(foldFile,"w") as openFile:
        json.dump(foldDict, openFile)

def load_folds(foldFile):
    folds = []
    with open(foldFile, "r") as openFile:
        newsFolds = json.load(openFile)

    for key in newsFolds:
        folds.append((np.array(newsFolds[key]["train"]),np.array(newsFolds[key]["test"])))

    return folds

#Function to train and evaluate a model.
#It returns the value of the evaluation metrics obtained by the trained model and its predictions on the test data.
def train_test(x_train: np.ndarray, y_train: np.ndarray, params: dict, train_f, predict_f, evaluate_f=None, x_test=None,y_test=None, x_val=None, y_val=None, validation_ratio=None, postp=False, seed=RANDOM_SEED, name="%08x" % random.getrandbits(32)) -> dict:
    params["name"] = name
    if validation_ratio is not None:
        x_train_in, x_val, y_train_in, y_val = train_test_split(x_train, y_train, test_size=validation_ratio, random_state=seed)
    else:
        x_train_in=np.copy(x_train)
        y_train_in=np.copy(y_train)
        if x_val is None and x_test is not None:
            x_val=x_test
            y_val=y_test

    model = train_f(params, x_train_in, y_train_in, x_val, y_val)
    model_alone,_ = model
    
    metrics = None
    predictions = None
    if x_test is not None:
        predictions = predict_f(model_alone, x_test, y_test, params,postp,False)
        if evaluate_f  is not None:
        
            metrics = evaluate_f(model, x_test, y_test, params,postp,False)

    del model
    torch.cuda.empty_cache()
    return metrics, predictions

# Function that implements cross-validation.
# It returns a dictionary with the average metrics over all folds and the corresponding standard deviations. 
def crossval(x: np.ndarray, y: np.ndarray, params: dict, train_f, predict_f, foldsFolder, evaluate_f=None, k=5, seed=RANDOM_SEED, name="%08x" % random.getrandbits(32), postp=False) -> np.ndarray:
    if not os.path.exists(name):
        os.mkdir(name)  
    metrics = []
    if params.get('pretrained_folder') is not None:
        pretrained_folder=params.get('pretrained_folder')
    if params.get("model_pre",False):
        pretrained_folder2=params.get('pretrained_folder2')
    zones=params.get('zones')
    cancer=params.get('cancer', False)
    ncs=params.get('ncs')

    #Transformations for data augmentation
    if cancer:
        aug_transforms = Compose(
        [
            RandFlipd(keys=["vol", "seg"], prob=0.7),
            RandRotated(keys=["vol","seg"], prob=0.7, range_z=0.349066, mode=('bilinear','nearest'), padding_mode='border'), #Rotate 20 degrees at most
            Rand3DElasticd(keys=["vol", "seg"], sigma_range=(3,5), magnitude_range=(5,10), prob=0.7, mode=('bilinear','nearest'), padding_mode='border'),
            RandAdjustContrastd(keys=["vol"],prob=0.5, gamma=(0.5, 2.0)),
            ToTensord(keys=["vol", "seg"]),
        ])
    else:
        aug_transforms = Compose(
            [
                RandFlipd(keys=["vol", "seg"], prob=0.7),
                RandAffined(keys=["vol", "seg"], prob=0.7, translate_range=(20,20,0), mode=('bilinear','nearest'), padding_mode='border'), 
                RandZoomd(keys=["vol", "seg"], prob=0.7, min_zoom=0.9, max_zoom=1.1, mode=("trilinear", "nearest"), padding_mode='reflect'),
                RandRotated(keys=["vol","seg"], prob=0.7, range_z=0.349066, mode=('bilinear','nearest'), padding_mode='border'), #Rotate 20 degrees at most
                Rand3DElasticd(keys=["vol", "seg"], sigma_range=(3,5), magnitude_range=(10,20), prob=0.5, mode=('bilinear','nearest'), padding_mode='border'),
                RandAdjustContrastd(keys=["vol"],prob=0.5, gamma=(0.5, 2.0)),
                RandGaussianNoised(keys=["vol"], prob=0.5, mean=0.0, std=0.1),
                ToTensord(keys=["vol", "seg"]),
            ])

    for i in range(k):
        if params.get('pretrained_folder') is not None:
            params['pretrained_folder']=os.path.join(pretrained_folder,f"fold{i}")
        if params.get("model_pre",False):
            params['pretrained_folder2']=os.path.join(pretrained_folder2,f"fold{i}")
        
        partial_name = f"{name}/fold{i}"
        if not os.path.exists(partial_name):
            os.mkdir(partial_name)
        clear_session()
        if cancer:
            if params.get("model_pre",False):
                paramsPre = {
                    "model":params.get("pre_model",params.get('model')),
                    "num_classes":params.get("pre_num_classes"),
                    "pre_num_classes":params.get("pre2_num_classes"),
                    "in_channels":params.get("pre_in_channels", params.get("in_channels",1)),
                    "dims":params.get("dims",3),
                    "dropout":params.get("pre_dropout",params.get("dropout",0.0)),
                    "pretrained_folder":params.get("pretrained_folder2"),
                    "channels": params.get("pre_channels",params.get("channels")),
                    "strides" :params.get("pre_strides",params.get("strides")),
                    "kernel_size":params.get("pre_kernel_size",params.get("kernel_size",3)),       
                    "res_units" : params.get("pre_res_units",params.get("res_units",2)),
                    "act":params.get("pre_act",params.get("act",'PRELU')),
                    "norm": params.get("pre_norm",params.get("norm",'INSTANCE')),   
                }
                
                model_pre = get_pretrained_model(paramsPre)
            pretrained_folder_aux = params.get("pretrained_folder")
            pretrained_weights    = os.path.join(pretrained_folder_aux,'weights.best.pt')
            in_channels=params.get("pre_in_channels", params.get("in_channels",1))
            paramspre2 = {
                "model":params.get("pre_model",params.get('model')),
                "num_classes":params.get("pre_num_classes"),
                "weights_file":pretrained_weights,
                "in_channels":in_channels,
                "dims":params.get("dims",3),   
                "dropout":params.get("pre_dropout",params.get("dropout",0.0)),
                "channels": params.get("pre_channels",params.get("channels")),
                "strides" :params.get("pre_strides",params.get("strides")),
                "kernel_size":params.get("pre_kernel_size",params.get("kernel_size",3)),       
                "res_units" : params.get("pre_res_units",params.get("res_units",2)),
                "act":params.get("pre_act",params.get("act",'PRELU')),
                "norm": params.get("pre_norm",params.get("norm",'INSTANCE')),  
            }
            
            if params.get("model_pre",False):
                model = load_model(paramspre2, model_pre)
            else:
                model = load_model(paramspre2)

            dataset=params.get('dataset_only',False)
            if dataset is not False: #Only the data from a single dataset is used
                folds=load_folds(os.path.join(foldsFolder[0],dataset+'.json'))
                train_index=folds[i][0]
                test_index=folds[i][1]
                if dataset=='X':
                    j,n=2,2
                elif dataset=='GE':
                    j,n=3,6
                elif dataset=='Siemens':
                    j,n=4,6
                elif dataset=='prostate158':
                    j,n=5,2
                x_train, y_train= x[j],y[j]

                pred=predict_f(model,x_train,y_train,params,False,False).numpy()
                x_train_2,y_train_2=prepare_2(x_train, params.get('num_classes'), y_train, pred, th=params.get('th',None), shuffle=False) 
                x_train_aux,y_train_aux=x_train_2[train_index], y_train_2[train_index]
                if params.get('aug',True):
                    d=[{} for _ in range(len(x_train_aux))]
                    for l in range(len(x_train_aux)):
                        d[l]['vol']=x_train_aux[l,:,:,:,:]
                        d[l]['seg']=y_train_aux[l,:,:,:,:]
                    x_train_aug,y_train_aug=augmentation(d,aug_transforms,n)
                    x_train=np.concatenate((x_train_aug,x_train_aux))
                    y_train=np.concatenate((y_train_aug,y_train_aux))
                else:
                    x_train=x_train_aux
                    y_train=y_train_aux
                x_test, y_test= x_train_2[test_index],y_train_2[test_index]
            else:
                folds=[]             
                for j in range(3,len(foldsFolder)):                    
                    folds.append(load_folds(os.path.join(foldsFolder[0],foldsFolder[j])))
                for j in range(len(folds)):
                    train_index=folds[j][i][0]
                    test_index=folds[j][i][1]
                    n=2 if j==0 or j==3 else 6 #Data from I2CVB are augmented 6 times, the rest is agumented only 2 times. 
                  
                    x_train_aux, y_train_aux= x[j+2],y[j+2]
                 
                    pred=predict_f(model,x_train_aux,y_train_aux,params,False,False).numpy()
                  
                    x_train_aux_2,y_train_aux_2=prepare_2(x_train_aux, params.get('num_classes'), y_train_aux, pred, th=params.get('th',None), shuffle=False) 
                    x_train_ng,y_train_ng=x_train_aux_2[train_index], y_train_aux_2[train_index]
                    x_test_aux, y_test_aux= x_train_aux_2[test_index],y_train_aux_2[test_index]
                    if params.get('aug',True):
                        d=[{} for _ in range(len(x_train_ng))]
                        for l in range(len(x_train_ng)):
                            d[l]['vol']=x_train_ng[l,:,:,:,:]
                            d[l]['seg']=y_train_ng[l,:,:,:,:]
                        x_train_aug,y_train_aug=augmentation(d,aug_transforms,n)
                        x_train_aux=np.concatenate((x_train_aug,x_train_ng))
                        y_train_aux=np.concatenate((y_train_aug,y_train_ng))
                    else:
                        x_train_aux=x_train_ng
                        y_train_aux=y_train_ng
                    if j==0:
                        x_train=x_train_aux
                        y_train=y_train_aux
                        x_test=x_test_aux
                        y_test=y_test_aux
                    else:
                        x_train=np.concatenate((x_train,x_train_aux))
                        y_train=np.concatenate((y_train,y_train_aux))
                        x_test=np.concatenate((x_test,x_test_aux))
                        y_test=np.concatenate((y_test,y_test_aux))
        elif zones:
            folds=[]
            for j in range(2,len(foldsFolder)):
                folds.append(load_folds(os.path.join(foldsFolder[0],foldsFolder[j])))
            for j in range(len(folds)):
                train_index=folds[j][i][0]
                test_index=folds[j][i][1]
                n=6 if j==2 or j==3 else 2 #Data from I2CVB are augmented 6 times, the rest is agumented only 2 times. 
                x_train_aux_2, y_train_aux_2= x[j+1],y[j+1]
                x_train_ng,y_train_ng=x_train_aux_2[train_index], y_train_aux_2[train_index]
                x_test_aux, y_test_aux= x_train_aux_2[test_index],y_train_aux_2[test_index]
                
                if params.get('aug',True):
                    d=[{} for _ in range(len(x_train_ng))]  
                    for l in range(len(x_train_ng)):
                        d[l]['vol']=x_train_ng[l,:,:,:,:]
                        d[l]['seg']=y_train_ng[l,:,:,:,:]
                    x_train_aug,y_train_aug=augmentation(d,aug_transforms,n)
                    x_train_aux=np.concatenate((x_train_aug,x_train_ng))
                    y_train_aux=np.concatenate((y_train_aug,y_train_ng))
                else:
                    x_train_aux=x_train_ng
                    y_train_aux=y_train_ng
                if j>0:
                    if ncs:
                        y_train_aux=y_train_aux[:,[0,3,4],:,:,:]
                        y_test_aux=y_test_aux[:,[0,3,4],:,:,:]
                    else:
                        y_train_aux=y_train_aux[:,[0,2,3],:,:,:]
                        y_test_aux=y_test_aux[:,[0,2,3],:,:,:]
                    x_train=np.concatenate((x_train,x_train_aux))
                    x_test=np.concatenate((x_test,x_test_aux))
                    y_train=np.concatenate((y_train,y_train_aux))
                    y_test=np.concatenate((y_test,y_test_aux))
                else:
                    x_train=x_train_aux
                    y_train=y_train_aux
                    x_test=x_test_aux
                    y_test=y_test_aux
        else:
            folds=[]
            for j in range(1,len(foldsFolder)):
                folds.append(load_folds(os.path.join(foldsFolder[0],foldsFolder[j])))
            for j in range(len(folds)):
                n=2 if j==1 or j==2 or j==5 else 6  #Data from promise12 and I2CVB are augmented 6 times, the rest is augmented only 2 times. 
                train_index=folds[j][i][0]
                test_index=folds[j][i][1]
                x_train_aux_2, y_train_aux_2= x[j],y[j]
                x_train_ng,y_train_ng=x_train_aux_2[train_index], y_train_aux_2[train_index]
                x_test_aux, y_test_aux= x_train_aux_2[test_index],y_train_aux_2[test_index]
                if params.get('aug',True):
                    d=[{} for _ in range(len(x_train_ng))]
                    for l in range(len(x_train_ng)):
                        d[l]['vol']=x_train_ng[l,:,:,:,:]
                        d[l]['seg']=y_train_ng[l,:,:,:,:]
                    x_train_aug,y_train_aug=augmentation(d,aug_transforms,n)
                    x_train_aux=np.concatenate((x_train_aug,x_train_ng))
                    y_train_aux=np.concatenate((y_train_aug,y_train_ng))
                else:
                    x_train_aux=x_train_ng
                    y_train_aux=y_train_ng
                if j>0:
                    y_train_aux=np.stack([y_train_aux[:,0,:,:,:], np.logical_not(y_train_aux[:,0,:,:,:])], axis=1)
                    y_test_aux=np.stack([y_test_aux[:,0,:,:,:], np.logical_not(y_test_aux[:,0,:,:,:])], axis=1)
                    x_train=np.concatenate((x_train,x_train_aux))
                    x_test=np.concatenate((x_test,x_test_aux))
                    y_train=np.concatenate((y_train,y_train_aux))
                    y_test=np.concatenate((y_test,y_test_aux))
                else:
                    x_train=x_train_aux
                    y_train=y_train_aux
                    x_test=x_test_aux
                    y_test=y_test_aux
        #Shuffle 
        p_train = np.random.permutation(len(x_train))
        
        x_train=x_train[p_train]
        y_train=y_train[p_train]

        m, fold_predictions = train_test(
            x_train = x_train, 
            y_train = y_train, 
            params = params, 
            train_f = train_f, 
            predict_f = predict_f, 
            evaluate_f = evaluate_f,
            x_test = x_test, 
            y_test = y_test,
            postp=postp,
            seed = seed, name = partial_name)
        save_experiment(params, {"test":m}, name =partial_name, save_dir=params.get("modeldir", params.get("save_dir", "experiments")))
        metrics.append(m)
        if params.get('print_val',False): 
            save_val_dir=os.path.join(partial_name, "test_results")
            if not os.path.exists(save_val_dir):
                os.mkdir(save_val_dir)
            plot_save_predictions(x_test, fold_predictions, th=params.get('th',None), y_val=y_test, save_dir=save_val_dir, show=False)
        mdict = {}

        for mname in metrics[0].keys(): #Compute average metrics over all folds and the corresponding standard deviations
            #print(mname)
            #print([fold[mname] for fold in metrics])
            mdict[mname] = np.nanmean([fold[mname] for fold in metrics]) if None not in [fold[mname] for fold in metrics] else None
            mdict[mname+"_std"] = np.nanstd([fold[mname] for fold in metrics]) if None not in [fold[mname] for fold in metrics] else None
    if params.get('pretrained_folder') is not None:
        params['pretrained_folder']=pretrained_folder
    if params.get("model_pre",False):
        params['pretrained_folder2']=pretrained_folder2

    return mdict