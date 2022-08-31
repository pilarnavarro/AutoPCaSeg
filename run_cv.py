#!/usr/bin/python
import os
import datetime
import torch
from utils.preprocess import *
from utils.train import *
from utils.save import *
from utils.cross_val import *
from sklearn.model_selection import KFold
from numpy.random import seed

RANDOM_SEED = 123456789
seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.multiprocessing.set_sharing_strategy('file_system')

sch='CosineAnnealingLR'
model='Unet'
opt='adam'
pre_model='Unet'
pre_channels=(16, 32, 64, 128, 256)
pre_kernel=(3,3,3,3,3,3)
pre_strides=(2,2,2,2,2)
pre_units=2
pre_dropout=0.2
pre_norm='INSTANCE'
pre_act='PRELU'
ncs=True
th=None
aug=True
act='PRELU'

losses=["dice_loss", "ce_dice","categorical_crossentropy"]
lrs=[1e-3,1e-4,1e-5],
wds=[1e-2,1e-3,1e-4,1e-5]
channels=[(16,32,64,128,256),(16, 32, 64, 128, 256,512),(16,32,64,128,128),(32,64,128,256,320,320),(16,32,64,128,256,256),(8,16,32,64,128,128)] 
strides=[[(2,2,2),(2,2,2),(2,2,2),(2,2,2),(1,2,2)],(2,2,2,2,2),[(1,2,2),(2,2,2),(2,2,2),(2,2,2),(1,2,2)],[(1,1,1),(1,2,2),(1,2,2),(2,2,2),(2,2,2)],(1,2,2,2,2),(1,1,1,1,1)]
kernel_size=[(3,3,3,3,3,3),[(1,3,3),(3,3,3),(3,3,3),(3,3,3),(3,3,3)],[(1,3,3),(1,3,3),(3,3,3),(3,3,3),(3,3,3)]]
res_units=[4,2,3]
dropouts=[0.3,0.2,0.4]
norms=['INSTANCE','BATCH']

i=1

for loss in losses:        
    for norm in norms:
        for k in kernel_size:
            for s in strides:
                for c in channels:
                    for unit in res_units:
                        for dropout in dropouts:   
                            for lr in lrs:
                                for wd in wds:
                                    name1="Unet_" +loss + '_prostate_exp_'+str(i)
                                    conf1={'num_classes':2, 'model':model, 'name':name1, 'optimizer': opt, 'in_channels':1, 'dims':3, 'dropout':dropout,'background':True,'kfold':3,'aug':aug,'zones':False, 
                                            'learning_rate':lr, 'loss':loss, 'scheduler':sch,'batch_size': 2, 'model_dir':'experiments', 'datadir': 'seg_prostate','epochs': 400, 'postp':True, 'wd':wd,
                                            'channels':c,'strides':s, 'kernel_size':k, 'act':act, 'norm':norm, 'res_units':unit,'ncs':ncs,'th':th,'first_cv':False }
                                    
                                    name2="Unet_" +loss + '_zones_exp_'+str(i)
                                    conf2={'num_classes':3, 'model':model, 'name':name2, 'optimizer': opt, 'in_channels':1, 'dims':3, 'dropout':dropout,'background':True,'kfold':3,'aug':aug,'zones':True, 'th':th,
                                            'learning_rate':lr, 'loss':loss, 'scheduler':sch,'batch_size': 2, 'model_dir':'experiments', 'datadir': 'seg_zones','epochs': 300, 'postp':True,'ncs':ncs ,'wd':wd,
                                            'pretrained':True,'pretrained_folder':os.path.join('experiments',name1),'pre_num_classes':2, 'channels':c,'strides':s, 'kernel_size':k, 'act':act, 'norm':norm, 'res_units':unit}
                                    
                                    name3="Unet_" +loss + '_m1_exp_'+str(i)
                                    conf3={'num_classes':4, 'model':model, 'model_pre':(model=='Unet'),'name':name3, 'optimizer': opt, 'in_channels':1, 'dims':3, 'dropout':dropout,'background':True,'kfold':3,'aug':aug, 'cancer':True, 'zones':False,
                                        'learning_rate':lr, 'loss':loss, 'scheduler':sch,'batch_size': 2, 'model_dir':'experiments', 'datadir': 'seg_cancer','epochs':300,  'postp':True,'th':th,'wd':wd,
                                        'pretrained':True,'pretrained_folder':os.path.join('experiments',name2),'pre_num_classes':3,'pretrained_folder2':os.path.join('experiments',name1), 'pre2_num_classes':2,'ncs':ncs,
                                        'channels':c,'strides':s, 'kernel_size':k, 'act':act, 'norm':norm, 'res_units':unit }
                                    
                                    name4="Unet_" +loss + '_m2_exp_'+str(i)
                                    conf4={'num_classes':3, 'model':model, 'model_pre':(pre_model=='Unet'),'name':name4, 'optimizer': opt, 'in_channels':3, 'pre_in_channels':1, 'dims':3, 'dropout':dropout,'background':True,'kfold':3,'aug':aug, 'cancer':True, 'zones':False,
                                        'learning_rate':lr, 'loss':loss, 'scheduler':sch,'batch_size': 2, 'model_dir':'experiments', 'datadir': 'seg_cancer','epochs': 500,  'ncs':ncs,'th':th,
                                        'pretrained':False,'pretrained_folder':os.path.join('experiments',name2),'pre_num_classes':3,'pretrained_folder2':os.path.join('experiments',name1), 'pre2_num_classes':2,'postp':True,
                                        'channels':c,'strides':s, 'kernel_size':k, 'act':act, 'norm':norm, 'res_units':unit ,'pre_model':pre_model,'wd':wd,
                                        'pre_dropout':pre_dropout, 'pre_channels':pre_channels, 'pre_strides':pre_strides, 'pre_kernel_size':pre_kernel, 'pre_res_units':pre_units,'pre_act':pre_act,'pre_norm':pre_norm}
                                    
                                    name5="Unet_" +loss + '_m1_X_exp_'+str(i)
                                    conf5={'num_classes':5, 'model':model, 'model_pre':(model=='Unet'),'name':name5, 'optimizer': opt, 'in_channels':1, 'dims':3, 'dropout':dropout,'background':True,'kfold':3,'aug':aug, 'cancer':True, 'zones':False,
                                        'learning_rate':lr, 'loss':loss, 'scheduler':sch,'batch_size': 2, 'model_dir':'experiments', 'datadir': 'X', 'dataset_only':'X', 'epochs':500,  'postp':True,'wd':wd,
                                        'pretrained':True,'pretrained_folder':os.path.join('experiments',name2),'pre_num_classes':3,'pretrained_folder2':os.path.join('experiments',name1), 'pre2_num_classes':2,'ncs':ncs,'th':th,
                                        'channels':c,'strides':s, 'kernel_size':k, 'act':act, 'norm':norm, 'res_units':unit }

                                    configs=[conf1,conf2,conf3,conf4]

                                    for conf in configs:
                                        if loss=="ce_dice":
                                            conf['loss']="categorical_crossentropy"
                                            conf.update({"weight_dice":0.5, "weight_ce":0.5})
                                        #print("==> CURRENT CONFIGURATION:\n", str(conf), file=sys.stderr)

                                        modeldir = conf.get("modeldir", conf.get("save_dir", "experiments")) 
                                        os.makedirs(modeldir, exist_ok=True)

                                        zones=conf.get('zones')
                                        cancer=conf.get('cancer', False)
                                        postp=conf.get('postp', False)
                                        ncs=conf.get('ncs',True)
                                        seg_dirs=['lesions/cs', 'lesions/ncs','pz','tz'] #Folders with the segmentation masks corresponding to the different classes
                                        seg_dirs2=['lesions', '','pz','tz']
                                        
                                        #Preprocess the data
                                        x_train, y_train = prepare('seg_prostate','ImagesTr', 'LabelsTr', cache=True, zones=False, cancer=False,shuffle=False)
                                        x_train_2, y_train_2= prepare('seg_zones','ImagesTr', 'LabelsTr',cache=True, zones=True, cancer=False,shuffle=False)
                                        x_train_3, y_train_3= prepare('X','ImagesTr', 'LabelsTr',seg_dirs=seg_dirs,cache=True, zones=False, cancer=True,ncs=ncs,shuffle=False) 
                                        x_train_4, y_train_4= prepare('GE','ImagesTr', 'LabelsTr',seg_dirs=seg_dirs2,cache=True, zones=False, cancer=True,ncs=ncs,shuffle=False) 
                                        x_train_5, y_train_5= prepare('Siemens','ImagesTr', 'LabelsTr',seg_dirs=seg_dirs2,cache=True, zones=False, cancer=True,ncs=ncs,shuffle=False)  
                                        x_train_6, y_train_6= prepare('prostate158','ImagesTr', 'LabelsTr',seg_dirs=seg_dirs2,cache=True, zones=False, cancer=True,ncs=ncs,shuffle=False)                                                          
                                       
                                        x_train=[x_train,x_train_2,x_train_3,x_train_4,x_train_5,x_train_6]
                                        y_train=[y_train,y_train_2,y_train_3,y_train_4,y_train_5,y_train_6]
                                       
                                        #Files and folder to save the folds of the different datasets
                                        foldfile=['prostate.json','zones.json','X.json','GE.json','Siemens.json','prostate158.json']
                                        foldfolder='folds'
                                        
                                        if not os.path.exists(foldfolder):
                                            os.mkdir(foldfolder)
                                        if conf.get('first_cv',False):
                                            for i in range(len(x_train)): #Create and save the folds for each dataset
                                                save_folds(list(KFold(conf.get('kfold',3), shuffle=True, random_state=RANDOM_SEED).split(x_train[i])),os.path.join(foldfolder,foldfile[i]))
                                        
                                        name = os.path.join(modeldir, conf.get("name", f"sm_{conf['model']}_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"))
                                        eval_dict = {}
                                        foldfile.insert(0,foldfolder)
                                        x_total, y_total = x_train, y_train
                                        #Run cross-validation
                                        eval_val = crossval(x_total, y_total, dict(conf), train_model, predict, foldsFolder=foldfile, evaluate_f=evaluate, name=name, k=conf.get("kfold",3),postp=postp)
                                        eval_dict.update(validation=eval_val)

                                        save_experiment(conf, eval_dict, os.path.basename(name), save_dir=modeldir)
                                        
                                    i=i+1
