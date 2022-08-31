#!/usr/bin/python
from utils.train import *
import os
import datetime
import torch
import numpy as np
from utils.save import *
from utils.preprocess import *
from numpy.random import seed

RANDOM_SEED = 123456789
seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.multiprocessing.set_sharing_strategy('file_system')

def test(conf, training=False):
    # Set up folders
    modeldir = conf.get("model_dir", conf.get("save_dir", "experiments"))
    os.makedirs(modeldir, exist_ok=True)
    name = os.path.join(modeldir, conf.get("name", f"sm_{conf['model']}_"f"{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"))
    name_save=os.path.join(name, conf.get("name_save", f"sm_{conf['model']}_"f"{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"))
    os.makedirs(name, exist_ok=True)
    os.makedirs(name_save, exist_ok=True)
    conf["name"] = name

    torch.cuda.empty_cache()

    zones=conf.get('zones', False)
    cancer=conf.get('cancer', False)
    test_cancer=conf.get('test_cancer', False)
    test_zones=conf.get('test_zones', False)
    test_prostate=conf.get('test_prostate', False)
    k=conf.get('kfold')

    if conf.get('pretrained_folder') is not None:
        pretrained_folder=conf.get('pretrained_folder')
    if conf.get("model_pre",False):
        pretrained_folder2=conf.get('pretrained_folder2')

    models=[]
    models_test=[]
    #Get trained models
    for i in range(k):
        if conf.get('pretrained_folder') is not None:
            conf['pretrained_folder']=os.path.join(pretrained_folder,f"fold{i}")
        if conf.get("model_pre",False):
            conf['pretrained_folder2']=os.path.join(pretrained_folder2,f"fold{i}")
        if test_cancer:     
            if conf.get("model_pre",False):
                paramsPre = {
                    "model":conf.get("pre_model",conf.get('model')),
                    "num_classes":conf.get("pre_num_classes"),
                    "pre_num_classes":conf.get("pre2_num_classes"),
                    "in_channels":conf.get("pre_in_channels", conf.get("in_channels",1)),
                    "dims":conf.get("dims",3),
                    "dropout":conf.get("pre_dropout",conf.get("dropout",0.0)),
                    "pretrained_folder":conf.get("pretrained_folder2"),
                    "channels": conf.get("pre_channels",conf.get("channels")),
                    "strides" :conf.get("pre_strides",conf.get("strides")),
                    "kernel_size":conf.get("pre_kernel_size",conf.get("kernel_size",3)),       
                    "res_units" : conf.get("pre_res_units",conf.get("res_units",2)),
                    "act":conf.get("pre_act",conf.get("act",'PRELU')),
                    "norm": conf.get("pre_norm",conf.get("norm",'INSTANCE')),   
                }
                
                model_pre=get_pretrained_model(paramsPre)
        
            pretrained_folder_aux     = conf.get('pretrained_folder')
            pretrained_weights    = f"{pretrained_folder_aux}/weights.best.pt"
            in_channels=conf.get("pre_in_channels", conf.get("in_channels",1))
            params = {
                "model":conf.get("pre_model",conf.get('model')),
                "num_classes":conf.get("pre_num_classes"),
                "weights_file":pretrained_weights,
                "in_channels":in_channels,
                "dims":conf.get("dims",3),   
                "dropout":conf.get("pre_dropout",conf.get("dropout",0.0)),
                "channels": conf.get("pre_channels",conf.get("channels")),
                "strides" :conf.get("pre_strides",conf.get("strides")),
                "kernel_size":conf.get("pre_kernel_size",conf.get("kernel_size",3)),       
                "res_units" : conf.get("pre_res_units",conf.get("res_units",2)),
                "act":conf.get("pre_act",conf.get("act",'PRELU')),
                "norm": conf.get("pre_norm",conf.get("norm",'INSTANCE')),     
            }
            
            if conf.get("model_pre"):
                models.append(load_model(params, model_pre)) 
            else:
                models.append(load_model(params))

        conf['weights_file']=os.path.join(name,'fold'+str(i), "weights.best.pt")
        if conf.get('pretrained', False):  
            model_test=get_pretrained_model(conf)   
            models_test.append(load_model(conf, model_test))
        else:
            models_test.append(load_model(conf))
    if conf.get('pretrained_folder') is not None:
        conf['pretrained_folder']=pretrained_folder
    if conf.get("model_pre",False):
        conf['pretrained_folder2']=pretrained_folder2
	
   
    if cancer:
        seg_dirs=conf.get('seg_dirs')                   
    else:
        seg_dirs=None

    #Preprocess data
    if training: #Predict on the training dataset instead of the test set. 
        x_test, y_test = prepare(conf.get('datadir'),'ImagesTr', 'LabelsTr',seg_dirs=seg_dirs,cache=False, zones=zones, cancer=cancer,ncs=ncs)
    else:    
        x_test, y_test = prepare(conf.get('datadir'),'ImagesTest', 'LabelsTest',seg_dirs=seg_dirs,cache=False, zones=zones, cancer=cancer,ncs=ncs)

    if test_cancer: 
        pred=predict(models,x_test,y_test,conf, postp=False,tta=conf.get('tta',False)).numpy()
        x_test,y_test=prepare_2(x_test, conf.get('num_classes'), y_test, pred, postp=True, th=conf.get('th',None))

    elif test_zones and cancer:
        if ncs:
            y_test=y_test[:,[0,3,4],:,:,:]
        else:
            y_test=y_test[:,[0,2,3],:,:,:]

    elif test_prostate:   
        if cancer or zones:
            y_test=np.stack([y_test[:,0,:,:,:], np.logical_not(y_test[:,0,:,:,:])], axis=1) 
       
    #Get predictions and metrics
    model=models_test,(None,None)
    metrics = evaluate(model, x_test, y_test, conf, postp=conf.get('postp', False),tta=conf.get('tta',False))
    eval_dict = {}
    eval_dict.update(testing_test=metrics)
    save_experiment(conf, eval_dict, os.path.basename(name_save), save_dir=name)
    pred=predict(models_test,x_test,y_test,conf, postp=conf.get('postp', False),tta=conf.get('tta',False))
    save_test_dir=os.path.join(name_save, "test_results")
    if not os.path.exists(save_test_dir):
        os.mkdir(save_test_dir)
    if conf.get('print_pred',False):
        plot_save_predictions(x_test, pred,th=conf.get('th',None),y_val=y_test, save_dir=save_test_dir, show=False)



name_prostate='Unet_dice_loss_prostate_exp_2_nuevo'
name_zones='Unet_dice_loss_zones_exp_2_nuevo'
name_m1 ='Unet_dice_loss_m1_exp_2_nuevo_96'
name_m2='Unet_dice_loss_m2_exp_2_nuevo_96'
model='Unet'
seg_dirs1=['lesions/cs','lesions/ncs','pz','tz']
seg_dirs2=['lesions','','pz','tz']
channels=(16, 32, 64, 128, 256)
strides=(2,2,2,2,2)
kernel_size=(3,3,3,3,3)
res_units=2
dropout=0.2
pre_model='Unet'
pre_channels=(16, 32, 64, 128, 256)
pre_kernel=(3,3,3,3,3,3)
pre_strides=(2,2,2,2,2)
pre_units=2
pre_dropout=0.2
pre_norm='INSTANCE'
pre_act='PRELU'
act='PRELU'
norm='INSTANCE'
ncs=True
th=None
min_vol=40
tta=True
num_tta=5
postp=True

#Test the segmentation performance of the entire prostate gland on each separate dataset
conf1={'num_classes':2, 'model':model, 'name': name_prostate, 'name_save':'test_prostate', 'in_channels':1, 'dims':3, 'dropout':dropout,'background':False,'zones':False, 'print_pred':True,'ncs':ncs,'th':th,'min_vol':min_vol,'num_tta':num_tta,
        'model_dir':'experiments', 'datadir': 'seg_prostate', 'postp':postp, 'test_prostate':True, 'channels':channels,'strides':strides, 'kernel_size':kernel_size, 'act':act, 'norm':norm, 'res_units':res_units,'kfold':3,'tta':tta}

conf2={'num_classes':2, 'model':model, 'name': name_prostate, 'name_save':'test_zones',  'in_channels':1, 'dims':3, 'dropout':dropout,'background':False,'zones':True, 'print_pred':True,'ncs':ncs,'th':th,'min_vol':min_vol,'num_tta':num_tta,
    'model_dir':'experiments', 'datadir': 'seg_zones', 'postp':postp, 'test_prostate':True,'channels':channels,'strides':strides, 'kernel_size':kernel_size, 'act':act, 'norm':norm, 'res_units':res_units,'kfold':3,'tta':tta}

conf3={'num_classes':2, 'model':model, 'name': name_prostate, 'name_save':'test_cancer',  'in_channels':1, 'dims':3, 'dropout':dropout,'background':False,'zones':False, 'cancer':True,  'print_pred':True,'ncs':ncs,'th':th,'min_vol':min_vol,'tta':tta,
    'model_dir':'experiments', 'datadir': 'seg_cancer', 'seg_dirs':seg_dirs1,'postp':postp, 'test_prostate':True,'channels':channels,'strides':strides, 'kernel_size':kernel_size, 'act':act, 'norm':norm, 'res_units':res_units,'kfold':3,'num_tta':num_tta,}

conf31={'num_classes':2, 'model':model, 'name': name_prostate, 'name_save':'test_ge',  'in_channels':1, 'dims':3, 'dropout':dropout,'background':False,'zones':False, 'cancer':True,  'print_pred':True,'ncs':ncs,'th':th,'min_vol':min_vol,'tta':tta,
    'model_dir':'experiments', 'datadir': 'GE', 'seg_dirs':seg_dirs2,'postp':postp, 'test_prostate':True,'channels':channels,'strides':strides, 'kernel_size':kernel_size, 'act':act, 'norm':norm, 'res_units':res_units,'kfold':3,'num_tta':num_tta,}

conf32={'num_classes':2, 'model':model, 'name': name_prostate, 'name_save':'test_X',  'in_channels':1, 'dims':3, 'dropout':dropout,'background':False,'zones':False, 'cancer':True,  'print_pred':True,'ncs':ncs,'th':th,'min_vol':min_vol,'tta':tta,
    'model_dir':'experiments', 'datadir': 'X', 'seg_dirs':seg_dirs1,'postp':postp, 'test_prostate':True,'channels':channels,'strides':strides, 'kernel_size':kernel_size, 'act':act, 'norm':norm, 'res_units':res_units,'kfold':3,'num_tta':num_tta,}

conf33={'num_classes':2, 'model':model, 'name': name_prostate, 'name_save':'test_siemens',  'in_channels':1, 'dims':3, 'dropout':dropout,'background':False,'zones':False, 'cancer':True,  'print_pred':True,'ncs':ncs,'th':th,'min_vol':min_vol,'tta':tta,
    'model_dir':'experiments', 'datadir': 'Siemens', 'seg_dirs':seg_dirs2,'postp':postp, 'test_prostate':True,'channels':channels,'strides':strides, 'kernel_size':kernel_size, 'act':act, 'norm':norm, 'res_units':res_units,'kfold':3,'num_tta':num_tta,}

conf34={'num_classes':2, 'model':model, 'name': name_prostate, 'name_save':'test_p158',  'in_channels':1, 'dims':3, 'dropout':dropout,'background':False,'zones':False, 'cancer':True,  'print_pred':True,'ncs':ncs,'th':th,'min_vol':min_vol,'tta':tta,
    'model_dir':'experiments', 'datadir': 'prostate158', 'seg_dirs':seg_dirs2,'postp':postp, 'test_prostate':True,'channels':channels,'strides':strides, 'kernel_size':kernel_size, 'act':act, 'norm':norm, 'res_units':res_units,'kfold':3,'num_tta':num_tta,}


#Test the segmentation performance of the prostate zones on each separate dataset
conf4={'num_classes':3, 'model':model, 'name':name_zones, 'optimizer': 'adam','name_save':'test_zones', 'in_channels':1, 'dims':3, 'dropout':dropout,'background':False,'zones':True,'print_pred':True,'kfold':3,'num_tta':num_tta,
    'model_dir':'experiments', 'datadir': 'seg_zones', 'pretrained':True,'pretrained_folder':os.path.join('experiments',name_prostate),'pre_num_classes':2,'test_zones':True,'ncs':ncs,'th':th,'min_vol':min_vol,'tta':tta,
    'channels':channels,'strides':strides, 'kernel_size':kernel_size, 'act':act, 'norm':norm, 'res_units':res_units,'postp':postp,}

conf5={'num_classes':3, 'model':model, 'name':name_zones, 'optimizer': 'adam','name_save':'test_cancer', 'in_channels':1, 'dims':3, 'dropout':dropout,'background':False,'zones':False, 'cancer':True, 'print_pred':True,'tta':tta,
    'model_dir':'experiments', 'datadir': 'seg_cancer', 'seg_dirs':seg_dirs1, 'pretrained':True,'pretrained_folder':os.path.join('experiments',name_prostate),'pre_num_classes':2,'test_zones':True,'ncs':ncs,'th':th,'kfold':3,
    'channels':channels,'strides':strides, 'kernel_size':kernel_size, 'act':act, 'norm':norm, 'res_units':res_units,'min_vol':min_vol,'postp':postp,'num_tta':num_tta,}

conf51={'num_classes':3, 'model':model, 'name':name_zones, 'optimizer': 'adam','name_save':'test_ge', 'in_channels':1, 'dims':3, 'dropout':dropout,'background':False,'zones':False, 'cancer':True, 'print_pred':True,'tta':tta,
    'model_dir':'experiments', 'datadir': 'GE', 'seg_dirs':seg_dirs2, 'pretrained':True,'pretrained_folder':os.path.join('experiments',name_prostate),'pre_num_classes':2,'test_zones':True,'ncs':ncs,'th':th,'kfold':3,
    'channels':channels,'strides':strides, 'kernel_size':kernel_size, 'act':act, 'norm':norm, 'res_units':res_units,'min_vol':min_vol,'postp':postp,'num_tta':num_tta,}

conf52={'num_classes':3, 'model':model, 'name':name_zones, 'optimizer': 'adam','name_save':'test_siemens', 'in_channels':1, 'dims':3, 'dropout':dropout,'background':False,'zones':False, 'cancer':True, 'print_pred':True,'tta':tta,
    'model_dir':'experiments', 'datadir': 'Siemens', 'seg_dirs':seg_dirs2, 'pretrained':True,'pretrained_folder':os.path.join('experiments',name_prostate),'pre_num_classes':2,'test_zones':True,'ncs':ncs,'th':th,'kfold':3,
    'channels':channels,'strides':strides, 'kernel_size':kernel_size, 'act':act, 'norm':norm, 'res_units':res_units,'min_vol':min_vol,'postp':postp,'num_tta':num_tta,}

conf53={'num_classes':3, 'model':model, 'name':name_zones, 'optimizer': 'adam','name_save':'test_X', 'in_channels':1, 'dims':3, 'dropout':dropout,'background':False,'zones':False, 'cancer':True, 'print_pred':True,'tta':tta,
    'model_dir':'experiments', 'datadir': 'X', 'seg_dirs':seg_dirs1, 'pretrained':True,'pretrained_folder':os.path.join('experiments',name_prostate),'pre_num_classes':2,'test_zones':True,'ncs':ncs,'th':th,'kfold':3,
    'channels':channels,'strides':strides, 'kernel_size':kernel_size, 'act':act, 'norm':norm, 'res_units':res_units,'min_vol':min_vol,'postp':postp,'num_tta':num_tta,}

conf54={'num_classes':3, 'model':model, 'name':name_zones, 'optimizer': 'adam','name_save':'test_p158', 'in_channels':1, 'dims':3, 'dropout':dropout,'background':False,'zones':False, 'cancer':True, 'print_pred':True,'tta':tta,
    'model_dir':'experiments', 'datadir': 'prostate158', 'seg_dirs':seg_dirs2, 'pretrained':True,'pretrained_folder':os.path.join('experiments',name_prostate),'pre_num_classes':2,'test_zones':True,'ncs':ncs,'th':th,'kfold':3,
    'channels':channels,'strides':strides, 'kernel_size':kernel_size, 'act':act, 'norm':norm, 'res_units':res_units,'min_vol':min_vol,'postp':postp,'num_tta':num_tta,}


#for min_vol in [5,10,20,30,40]:
  #  for num_tta in [5,7,10,12,15,17,20]:
   #     for tta in [True,False]:
    #        for postp in [True, False]:


#Test the segmentation performance of PCa-Seg model 1 on each separate dataset
if tta and postp: 
    name6='test_ge_tta'+str(num_tta)+'_postp'+str(min_vol)
elif tta:
    name6='test_ge_tta'+str(num_tta)
elif postp:
    name6='test_ge_postp'+str(min_vol)
else:
    name6='test_ge'

conf6={'num_classes':5, 'model':model, 'model_pre':(model=='Unet'),'name': name_m1, 'name_save':name6,  'in_channels':1, 'dims':3, 'dropout':dropout,'background':True, 'cancer':True, 'zones':False,'kfold':3,
    'model_dir':'experiments', 'datadir': 'GE', 'seg_dirs':seg_dirs2, 'print_pred':True,'postp':postp, 'pretrained':True,'pretrained_folder':os.path.join('experiments',name_zones),'ncs':ncs,'th':th,'min_vol':min_vol,
    'pretrained_folder2':os.path.join('experiments', name_prostate), 'pre_num_classes':3, 'pre2_num_classes':2, 'test_cancer':True, 'channels':channels,'strides':strides, 'kernel_size':kernel_size,'tta':tta,'num_tta':num_tta,
    'act':act, 'norm':norm, 'res_units':res_units}

if tta and postp: 
    name7='test_siemens_tta'+str(num_tta)+'_postp'+str(min_vol)
elif tta:
    name7='test_siemens_tta'+str(num_tta)
elif postp:
    name7='test_siemens_postp'+str(min_vol)
else:
    name7='test_siemens'

conf7={'num_classes':5, 'model':model, 'model_pre':(model=='Unet'),'name': name_m1, 'name_save':name7,  'in_channels':1, 'dims':3, 'dropout':dropout,'background':True, 'cancer':True, 'zones':False,
    'model_dir':'experiments', 'datadir': 'Siemens', 'seg_dirs':seg_dirs2, 'print_pred':True,'postp':postp, 'pretrained':True,'pretrained_folder':os.path.join('experiments',name_zones),'ncs':ncs,'th':th,'kfold':3,
    'pretrained_folder2':os.path.join('experiments', name_prostate), 'pre_num_classes':3, 'pre2_num_classes':2, 'test_cancer':True, 'channels':channels,'strides':strides, 'kernel_size':kernel_size, 'min_vol':min_vol,
    'act':act, 'norm':norm, 'res_units':res_units,'tta':tta,'num_tta':num_tta,}

if tta and postp: 
    name8='test_X_tta'+str(num_tta)+'_postp'+str(min_vol)
elif tta:
    name8='test_X_tta'+str(num_tta)
elif postp:
    name8='test_X_postp'+str(min_vol)
else:
    name8='test_X'


conf8={'num_classes':5, 'model':model, 'model_pre':(model=='Unet'),'name': name_m1, 'name_save':name8,   'in_channels':1, 'dims':3, 'dropout':dropout,'background':True, 'cancer':True, 'zones':False,
    'model_dir':'experiments', 'datadir': 'X', 'seg_dirs':seg_dirs1, 'print_pred':True,'postp':postp, 'pretrained':True,'pretrained_folder':os.path.join('experiments',name_zones),'ncs':ncs,'th':th,'kfold':3,
    'pretrained_folder2':os.path.join('experiments', name_prostate), 'pre_num_classes':3, 'pre2_num_classes':2, 'test_cancer':True, 'channels':channels,'strides':strides, 'kernel_size':kernel_size, 'min_vol':min_vol,
    'act':act, 'norm':norm, 'res_units':res_units,'tta':tta,'num_tta':num_tta,}


if tta and postp: 
    name9='test_p158_tta'+str(num_tta)+'_postp'+str(min_vol)
elif tta:
    name9='test_p158_tta'+str(num_tta)
elif postp:
    name9='test_p158_postp'+str(min_vol)
else:
    name9='test_p158'

conf9={'num_classes':5, 'model':model, 'model_pre':(model=='Unet'),'name': name_m1, 'name_save':name9,  'in_channels':1, 'dims':3, 'dropout':dropout,'background':True, 'cancer':True, 'zones':False,
    'model_dir':'experiments', 'datadir': 'prostate158', 'seg_dirs':seg_dirs2, 'print_pred':True,'postp':postp, 'pretrained':True,'pretrained_folder':os.path.join('experiments',name_zones),'ncs':ncs,'th':th,'kfold':3,
    'pretrained_folder2':os.path.join('experiments', name_prostate), 'pre_num_classes':3, 'pre2_num_classes':2, 'test_cancer':True, 'channels':channels,'strides':strides, 'kernel_size':kernel_size, 'min_vol':min_vol,
    'act':act, 'norm':norm, 'res_units':res_units,'tta':tta,'num_tta':num_tta,}

if tta and postp: 
    name10='test_cancer_tta'+str(num_tta)+'_postp'+str(min_vol)
elif tta:
    name10='test_cancer_tta'+str(num_tta)
elif postp:
    name10='test_cancer_postp'+str(min_vol)
else:
    name10='test_cancer'

conf10={'num_classes':5, 'model':model, 'model_pre':(model=='Unet'),'name': name_m1, 'name_save':name10,   'in_channels':1, 'dims':3, 'dropout':dropout,'background':True, 'cancer':True, 'zones':False,'kfold':3,
    'model_dir':'experiments', 'datadir': 'seg_cancer', 'seg_dirs':seg_dirs1, 'print_pred':False,'postp':postp, 'pretrained':True,'pretrained_folder':os.path.join('experiments',name_zones),'ncs':ncs,'th':th,
    'pretrained_folder2':os.path.join('experiments', name_prostate), 'pre_num_classes':3, 'pre2_num_classes':2, 'test_cancer':True, 'channels':channels,'strides':strides, 'kernel_size':kernel_size, 'min_vol':min_vol,
    'act':act, 'norm':norm, 'res_units':res_units,'tta':tta,'num_tta':num_tta,}

#Test the segmentation performance of PCa-Seg model 2 on each separate dataset
if tta and postp: 
    name11='test_ge_tta'+str(num_tta)+'_postp'+str(min_vol)
elif tta:
    name11='test_ge_tta'+str(num_tta)
elif postp:
    name11='test_ge_postp'+str(min_vol)
else:
    name11='test_ge'


conf11={'num_classes':3, 'model':model, 'model_pre':(model=='Unet'),'name': name_m2, 'name_save':name11, 'pre_in_channels':1, 'in_channels':3, 'dims':3, 'dropout':dropout,'background':False, 'cancer':True, 'zones':False,
    'model_dir':'experiments', 'datadir': 'GE', 'seg_dirs':seg_dirs2, 'print_pred':True,'postp':postp, 'pretrained':False,'pretrained_folder':os.path.join('experiments',name_zones),'ncs':ncs,'th':th,'min_vol':min_vol,
    'pretrained_folder2':os.path.join('experiments', name_prostate), 'pre_num_classes':3, 'pre2_num_classes':2, 'test_cancer':True, 'channels':channels,'strides':strides, 'kernel_size':kernel_size,'kfold':3,
    'act':act, 'norm':norm, 'res_units':res_units,'tta':tta,'num_tta':num_tta,'pre_dropout':pre_dropout, 'pre_channels':pre_channels, 'pre_strides':pre_strides, 'pre_kernel_size':pre_kernel, 'pre_res_units':pre_units,
    'pre_act':pre_act,'pre_norm':pre_norm,'pre_model':pre_model,}


if tta and postp: 
    name12='test_siemens_tta'+str(num_tta)+'_postp'+str(min_vol)
elif tta:
    name12='test_siemens_tta'+str(num_tta)
elif postp:
    name12='test_siemens_postp'+str(min_vol)
else:
    name12='test_siemens'

conf12={'num_classes':3, 'model':model, 'model_pre':(model=='Unet'),'name': name_m2, 'name_save':name12,'pre_in_channels':1,   'in_channels':3, 'dims':3, 'dropout':dropout,'background':False, 'cancer':True, 'zones':False,
    'model_dir':'experiments', 'datadir': 'Siemens', 'seg_dirs':seg_dirs2, 'print_pred':True,'postp':postp, 'pretrained':False,'pretrained_folder':os.path.join('experiments',name_zones),'ncs':ncs,'th':th,'kfold':3,
    'pretrained_folder2':os.path.join('experiments', name_prostate), 'pre_num_classes':3, 'pre2_num_classes':2, 'test_cancer':True, 'channels':channels,'strides':strides, 'kernel_size':kernel_size, 'min_vol':min_vol,
    'act':act, 'norm':norm, 'res_units':res_units,'tta':tta,'num_tta':num_tta,'pre_dropout':pre_dropout, 'pre_channels':pre_channels, 'pre_strides':pre_strides, 'pre_kernel_size':pre_kernel, 'pre_res_units':pre_units,
    'pre_act':pre_act,'pre_norm':pre_norm,'pre_model':pre_model,}

if tta and postp: 
    name13='test_X_tta'+str(num_tta)+'_postp'+str(min_vol)
elif tta:
    name13='test_X_tta'+str(num_tta)
elif postp:
    name13='test_X_postp'+str(min_vol)
else:
    name13='test_X'


conf13={'num_classes':3, 'model':model, 'model_pre':(model=='Unet'),'name': name_m2, 'name_save':name13, 'pre_in_channels':1,  'in_channels':3, 'dims':3, 'dropout':dropout,'background':False, 'cancer':True, 'zones':False,
    'model_dir':'experiments', 'datadir': 'X', 'seg_dirs':seg_dirs1, 'print_pred':True,'postp':postp, 'pretrained':False,'pretrained_folder':os.path.join('experiments',name_zones),'ncs':ncs,'th':th,'kfold':3,
    'pretrained_folder2':os.path.join('experiments', name_prostate), 'pre_num_classes':3, 'pre2_num_classes':2, 'test_cancer':True, 'channels':channels,'strides':strides, 'kernel_size':kernel_size, 'min_vol':min_vol,
    'act':act, 'norm':norm, 'res_units':res_units,'tta':tta,'num_tta':num_tta,'pre_dropout':pre_dropout, 'pre_channels':pre_channels, 'pre_strides':pre_strides, 'pre_kernel_size':pre_kernel, 'pre_res_units':pre_units,
    'pre_act':pre_act,'pre_norm':pre_norm,'pre_model':pre_model,}

if tta and postp: 
    name14='test_p158_tta'+str(num_tta)+'_postp'+str(min_vol)
elif tta:
    name14='test_p158_tta'+str(num_tta)
elif postp:
    name14='test_p158_postp'+str(min_vol)
else:
    name14='test_p158'

conf14={'num_classes':3, 'model':model, 'model_pre':(model=='Unet'),'name': name_m2, 'name_save':name14,'pre_in_channels':1,   'in_channels':3, 'dims':3, 'dropout':dropout,'background':False, 'cancer':True, 'zones':False,
    'model_dir':'experiments', 'datadir': 'prostate158', 'seg_dirs':seg_dirs2, 'print_pred':True,'postp':postp, 'pretrained':False,'pretrained_folder':os.path.join('experiments',name_zones),'ncs':ncs,'th':th,'kfold':3,
    'pretrained_folder2':os.path.join('experiments', name_prostate), 'pre_num_classes':3, 'pre2_num_classes':2, 'test_cancer':True, 'channels':channels,'strides':strides, 'kernel_size':kernel_size, 'min_vol':min_vol,
    'act':act, 'norm':norm, 'res_units':res_units,'tta':tta,'num_tta':num_tta,'pre_dropout':pre_dropout, 'pre_channels':pre_channels, 'pre_strides':pre_strides, 'pre_kernel_size':pre_kernel, 'pre_res_units':pre_units,
    'pre_act':pre_act,'pre_norm':pre_norm,'pre_model':pre_model,}


if tta and postp: 
    name15='test_cancer_tta'+str(num_tta)+'_postp'+str(min_vol)
elif tta:
    name15='test_cancer_tta'+str(num_tta)
elif postp:
    name15='test_cancer_postp'+str(min_vol)
else:
    name15='test_cancer'

conf15={'num_classes':3, 'model':model, 'model_pre':(model=='Unet'),'name': name_m2, 'name_save':name15, 'pre_in_channels':1,  'in_channels':3, 'dims':3, 'dropout':dropout,'background':False, 'model_dir':'experiments', 
    'datadir': 'seg_cancer', 'seg_dirs':seg_dirs1, 'print_pred':False,'postp':postp, 'pretrained':False,'pretrained_folder':os.path.join('experiments',name_zones),'ncs':ncs,'th':th,'kfold':3,
    'pretrained_folder2':os.path.join('experiments', name_prostate), 'pre_num_classes':3, 'pre2_num_classes':2, 'test_cancer':True, 'channels':channels,'strides':strides, 'kernel_size':kernel_size,
    'min_vol':min_vol, 'act':act, 'norm':norm, 'res_units':res_units,'cancer':True, 'zones':False,'tta':tta,'num_tta':num_tta,'pre_dropout':pre_dropout, 'pre_channels':pre_channels, 'pre_strides':pre_strides, 'pre_kernel_size':pre_kernel, 
    'pre_res_units':pre_units,'pre_act':pre_act,'pre_norm':pre_norm,'pre_model':pre_model,}


configs=[conf11,conf12,conf13,conf14]


for conf in configs:
    test(conf,training=False)