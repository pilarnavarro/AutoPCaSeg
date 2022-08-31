from glob import glob 
import os
import numpy as np
from copy import deepcopy
from monai.transforms import(
    Compose,
    LoadImaged,
    Resized,
    ToTensord,   
    Spacingd,
    MapTransform,  
    CenterSpatialCropd,
    NormalizeIntensityd,
    FillHolesd,
    KeepLargestConnectedComponentd,

)

from monai.data import DataLoader, Dataset, CacheDataset
from monai.utils import set_determinism
import torch
from skimage.measure import label,regionprops
from monai.networks.utils import one_hot
from sklearn.model_selection import train_test_split
from monai.config import KeysCollection

RANDOM_SEED=12345678

def select_data(in_dir, images_dir,labels_dir, seg_dirs, leng):
    files=[]
    j=0
    l=0
    path_volumes = sorted(glob(os.path.join(in_dir, images_dir, "*.nii.gz")))
    path_segmentation_cs = sorted(glob(os.path.join(in_dir, labels_dir,seg_dirs[0], "*.nii.gz")))
    if len(seg_dirs[1])>0:
        path_segmentation_ncs = sorted(glob(os.path.join(in_dir, labels_dir,seg_dirs[1], "*.nii.gz")))
    #Length of each path
    indices = [len(os.path.join(in_dir, images_dir)), len(os.path.join(in_dir, labels_dir,seg_dirs[0])),len(os.path.join(in_dir, labels_dir,seg_dirs[1]))]
    
    for i, img in enumerate(path_volumes):
        labels_cs=[]
        labels_ncs=[]
        k=j
        m=l
        while k<len(path_segmentation_cs):
            if path_volumes[i][indices[0]:indices[0]+leng]==path_segmentation_cs[k][indices[1]:indices[1]+leng]:
                labels_cs.append(path_segmentation_cs[k])
                j=k
            k=k+1 
        if len(seg_dirs[1])>0:
            while m<len(path_segmentation_ncs):
                if path_volumes[i][indices[0]:indices[0]+leng]==path_segmentation_ncs[m][indices[2]:indices[2]+leng]:
                    labels_ncs.append(path_segmentation_ncs[m])
                    l=m
                m=m+1
        d={"vol": img}
        if len(labels_cs)>0:
            d["seg"]= labels_cs
        if len(labels_ncs)>0:
            d["seg_ncs"]= labels_ncs
        files.append(d)
    return files

#Function to delete empty slices from the image volumes until a number of 'slices' is reached. 
def delete_empty(d,keys,slices):
    empty_front_t=[]
    empty_back_t=[]
    i=0
    for key in keys:
        empty_front=[]
        empty_back=[]
        if key not in d: 
            i=i+1
        if key!='vol' and key in d:
            #For each mask we select the empty slices from the beginning
            for slice in range(d[key].shape[3]):  
                if len(np.unique(d[key][0,:,:,slice])) == 1:
                    empty_front.append(slice)
                if len(np.unique(d[key][0,:,:,slice])) > 1:
                    break
            #For each mask we select the empty slices from the end
            for slice in range(d[key].shape[3]):     
                if len(np.unique(d[key][0,:,:,d[key].shape[3]-1-slice])) == 1:
                    empty_back.append(d[key].shape[3]-1-slice)
                if len(np.unique(d[key][0,:,:,d[key].shape[3]-1-slice])) > 1:
                    break
            empty_front_t.append(empty_front)
            empty_back_t.append(empty_back)  
    #If all slices from all segmentation masks are empty, we delete slices at each end alternatively until the desired number of slices is reached.
    if i==(len(keys)-1):
        joined=np.array([i for i in range(d['vol'].shape[3])])
        k=0
        while d['vol'].shape[3]>slices:
            if k%2==0:
                d['vol']=np.delete(d['vol'],np.argwhere(joined==np.min(joined)),axis=3)
                joined=joined[1:]
            else:
                d['vol']=np.delete(d['vol'],np.argwhere(joined==np.max(joined)),axis=3)
                joined=joined[:-1]
            k=k+1
    else:           
        #We look for empty slices that are common to all segmentation masks at both ends. 
        if len(empty_front_t)>1:
            joined_front=np.intersect1d(empty_front_t[0],empty_front_t[1]) 
            for i in range(len(empty_front_t)-2):
                joined_front=np.intersect1d(joined_front,empty_front_t[i+2]) 
        else:
            joined_front=np.array(empty_front_t)
            if len(joined_front.shape)>1:
                joined_front=np.squeeze(joined_front, axis=0)

        if len(empty_back_t)>1:
            joined_back=np.intersect1d(empty_back_t[0],empty_back_t[1]) 
            for i in range(len(empty_back_t)-2):
                joined_back=np.intersect1d(joined_back,empty_back_t[i+2])  
        else:       
            joined_back=np.array(empty_back_t)
            if len(joined_back.shape)>1:
                joined_back=np.squeeze(joined_back, axis=0)

        #The first empty slice at each end is not deleted.   
        if len(joined_front)>0:
            joined_front_2 = np.delete(joined_front, np.argwhere(joined_front == np.max(joined_front)))
        else:
            joined_front_2=[]
        if len(joined_back)>0:
            joined_back_2=np.delete(joined_back, np.argwhere(joined_back == np.min(joined_back)))
        else:
            joined_back_2=[]

        change=False

        if len(joined_front_2)>0 and len(joined_back_2)>0:
            joined=np.concatenate((joined_front_2,joined_back_2))
            #If the resulting depth is larger than the target one, we also delete the first empty slice at each end
            if (d['vol'].shape[3]-len(joined))>slices:
                change=True

                joined=np.concatenate((joined_front,joined_back))                
        elif len(joined_front_2)>0:
            joined=joined_front_2
            if (d['vol'].shape[3]-len(joined))>slices:
                change=True
                if len(joined_back)>0:      
                    joined=np.concatenate((joined_front,joined_back))
                else:
                    joined=joined_front                             
        else:                           
            joined=joined_back_2
            if (d['vol'].shape[3]-len(joined))>slices:
                change=True
                if len(joined_front)>0:  
                    joined=np.concatenate((joined_front,joined_back))
                else:
                    joined=joined_back                            
        if len(joined)>0:     
            for key in keys:
                aux_joined=joined
                if key in d:          
                    if d[key].shape[3]>slices:
                        if change:
                            del_front_v=joined_front
                            del_back_v=joined_back
                        else:
                            del_front_v=joined_front_2
                            del_back_v=joined_back_2
                        #If the resulting depth when removing the empty slices is smaller than the desired depth, we delete fewer slices. 
                        #To this end, we remove from the set of slices to delete one slice at each end alternatively.
                        k=0
                        while (d[key].shape[3]-len(aux_joined))<slices:
                            if (k%2==0 and len(del_front_v)>0) or (len(del_back_v)==0 and len(del_front_v)>0):
                                del_e=np.max(del_front_v)
                                del_front_v=np.delete(del_front_v,np.argwhere(del_front_v == np.max(del_front_v)))
                            elif len(del_back_v)>0:
                                del_e=np.min(del_back_v)
                                del_back_v=np.delete(del_back_v,np.argwhere(del_back_v == np.min(del_back_v)))
                            else:
                                break
                            aux_joined = np.delete(aux_joined,np.argwhere(aux_joined == del_e))
                            k=k+1
                        if len(aux_joined)>0:
                            d[key] = np.delete(d[key],aux_joined, axis=3) 
    return d

class DeleteEmptySlicesd(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        num_slices: np.int32,
    ):
        super().__init__(keys)
        self.slices = num_slices
    def __call__(self, data):
        d = dict(data)    
        d=delete_empty(d,self.keys,self.slices)             
        return d

#Join all segmentation masks in a single ground truth multi-channel segmentation mask.
def join(d,keys, zones, cancer):
    if 'tz' not in d and 'pz' not in d:
        d['seg']=np.squeeze(d['seg'])
        background=np.array(np.logical_not(d['seg']), dtype=np.float32)
        #We are in the case of segmenting the prostate zones, but there is only one zone visible in the patient.
        #We add the background channel and the channel corresponding to the non-visible zone is empty. 
        if zones:
            aux=np.zeros(d['seg'].shape)
            #Only PZ is visible
            if 2. in np.unique(d['seg'][:,:,:]):
                    d['seg']=np.stack([background,aux,np.array(d['seg']>0, dtype=np.float32)],axis=0)
            # Only TZ (or CG) is visible
            else:
                d['seg']=np.stack([background,d['seg'],aux],axis=0)
        #We are in the case of segmenting only the whole prostate gland. We just need to add the background channel.
        else: 
            d['seg']=np.stack([background,d['seg']],axis=0)           
    else:  
        if cancer:
            for key in keys:
                if key not in d:              
                    d[key]=np.zeros(d[keys[0]].shape)
                d[key]=np.squeeze(d[key])  
        else:
            d['tz']=np.squeeze(d['tz'])
            d['pz']=np.squeeze(d['pz'])
        #As we are preparing the data to segment the prostate anatomy, only the prostate zones are considered to compute the background
        background=np.logical_or(d['tz'],d['pz']) 
        background=np.array(np.logical_not(background), dtype=np.float32)
        if cancer:
            d['seg']=np.stack([background,d[keys[1]],d[keys[2]],d[keys[3]],d[keys[0]]],axis=0)
        else:
            d['seg']=np.stack([background,d['tz'],d['pz']],axis=0)
            
    for key in keys:
        if not key=='seg' and key in d:
            del d[key]
    return d
    
class JoinChannelsd(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        zones: bool = False,
        cancer: bool = False,
    ) -> None:
        super().__init__(keys)
        self.zones = zones
        self.cancer = cancer
    
    def __call__(self, data):
        d = dict(data)
        keys=sorted(self.keys)        
        d=join(d,keys,self.zones, self.cancer)       
        return d

#This function is to separate the segmentation masks corresponding to the different prostate zones in different binary masks. 
class ConvertMultiClassd(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        zones: bool = False,
    ):
        super().__init__(keys)
        self.zones = zones
        
    def __call__(self, data):
        d = dict(data)   
        if self.zones:
            for key in self.keys:
                if key in d:
                    if len(np.unique(d[key][0,:,:,:]))>2:
                        d['pz']=np.array(d[key]==1,dtype=np.float32)  
                        d['tz']=np.array(d[key]==2,dtype=np.float32)    
        return d


#Add a new channel to both the images and segmentation masks
class AddChanneld(MapTransform):
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:        
            if key in d:
                if len(d[key].shape)<4:
                    d[key]=np.expand_dims(d[key],0) 
                #When there are two channels in the images, corresponding to different MRI modalities (t2w and adc map),
                #we delete the channel corresponding to the modality that we do not use (the adc map).
                elif key=='vol' and d[key].shape[3]==2:
                    d['vol']=d['vol'][:,:,:,0]
                    d[key]=np.expand_dims(d[key],0)          
        return d
    
#Join the masks of different lesions that belong to the same patient in a single segmentation mask.
class JoinMasksd(MapTransform):    
    def __init__(
    self,
    keys: KeysCollection,
    apply: bool,
    ):
        super().__init__(keys)
        self.apply = apply
        
    def __call__(self, data):
        d = dict(data)
        if self.apply:
            for key in self.keys:
                if key in d:
                    if len(np.array(d[key]).shape)>3:
                        num_masks=np.array(d[key]).shape[0]
                        if num_masks>1:
                            list_masks = [[] for i in range(num_masks)]
                            for i,patient in enumerate(d[key]):
                                patient=np.swapaxes(patient,0,2)
                                for mask in patient:
                                    list_masks[i].append(mask)
                            joined=np.logical_or(list_masks[0],list_masks[1])
                            for i in range(len(list_masks)-2):
                                joined=np.logical_or(joined,list_masks[i+2])                          
                            joined=np.swapaxes(joined,0,2)
                            d[key]=np.array(joined,dtype=np.float32)     
        return d


#For the segmentation of prostate cancer lesions, this function concatenates the image and the probability maps of the prostate zones.
class ConcatenateChannelsd(MapTransform):
    def __init__(
    self,
    keys: KeysCollection,
    apply: bool,
    ):
        super().__init__(keys)
        self.apply = apply

    def __call__(self, data):
        d = dict(data)
        if self.apply:
            d['vol']=np.squeeze(d['vol'], axis=0)
            d['vol']=np.stack([d['vol'],d['pred'][1,:,:,:],d['pred'][2,:,:,:]],axis=0)
        return d

#Adjust the channels of the ground truth segmentation mask for the case of segmenting cancer lesions. 
class AdjustSegChannelsd(MapTransform):
    def __init__(
    self,
    keys: KeysCollection,
    num_classes: int,
    ):
        super().__init__(keys)
        self.num_classes = num_classes

    def __call__(self, data):
        d = dict(data)
        if self.num_classes==3:
            #If we want to segment only the PCa lesions, then the background is just the complement of the union of the two types of PCa lesions. 
            background=np.logical_or(d['seg'][1,:,:,:],d['seg'][2,:,:,:])
            background=np.array(np.logical_not(background), dtype=np.float32)
            #The ground truth mask has three channels, for the background and the two types of lesions
            d['seg']=np.stack([background, d['seg'][1,:,:,:],d['seg'][2,:,:,:]], axis=0)
        elif self.num_classes==5: #The segmentation mask contains all classes, including the prostate zones
            background1=np.logical_or(d['seg'][1,:,:,:],d['seg'][2,:,:,:])
            background2=np.logical_or(d['seg'][3,:,:,:],d['seg'][4,:,:,:])
            background=np.logical_or(background1,background2)
            background=np.array(np.logical_not(background), dtype=np.float32)
            d['seg'][0,:,:,:]=background     
        elif self.num_classes==4: #The segmentation mask contains all classes except for the NCSPCa lesions
            background=np.logical_or(np.logical_not(d['seg'][0,:,:,:]),d['seg'][1,:,:,:])
            background=np.array(np.logical_not(background), dtype=np.float32)
            d['seg'][0,:,:,:]=background 
        else: #Only clinically significant lesions are going to be segmented.
            background=np.array(np.logical_not(d['seg'][1,:,:,:]), dtype=np.float32)
            d['seg']=np.stack([background, d['seg'][1,:,:,:]], axis=0)                  
        return d
    
#To delete the intersections between the segmentation masks of different classes, prioritazing the masks corresponding to PCa lesions. 
class DeleteIntersectionsd(MapTransform):  
    def __init__(
    self,
    keys: KeysCollection,
    num_classes: int = None,
    remove: str = None,
    cancer: bool = False,
    zones: bool = False,
    ):
        super().__init__(keys)
        self.num_classes = num_classes
        self.cancer = cancer
        self.zones = zones
        self.remove = remove
        
    def __call__(self, data):
        d = dict(data)
        keys=self.keys
        if self.num_classes is not None:
            if self.num_classes!=2 and self.num_classes!=4:
                #Set the pixels in the segmentation mask corresponding to NCSPCa lesions that also have associated CSPCa lesions to 0.
                same=np.logical_and(d['seg'][1,:,:,:],d['seg'][2,:,:,:]) 
                d['seg'][2,:,:,:][same==True]=False 
            if self.num_classes>3: #The prostate zones are also segmented
                #Set to 0 the pixels in the segmentation masks of the prostate zones that have an associated PCa lesion.
                if self.num_classes==4:
                    num_lesions=1
                else: num_lesions=2
                for i in range(num_lesions):
                    same=np.logical_and(d['seg'][i+1,:,:,:],d['seg'][num_lesions+2,:,:,:])
                    d['seg'][num_lesions+2,:,:,:][same==True]=False 
                    same=np.logical_and(d['seg'][i+1,:,:,:],d['seg'][num_lesions+1,:,:,:])
                    d['seg'][num_lesions+1,:,:,:][same==True]=False     
        else: #For the case of segmenting the prostate anatomy only, not the PCa lesions. 
            if self.cancer or self.zones:
                if len(keys)>1 and keys[0] in d and keys[1] in d :
                    same=np.logical_and(d[keys[0]],d[keys[1]]) #Pixels that have two associated classes, corresponding to the classes given by 'keys'
                    d[self.remove][same==True]=False #Set to 0 the pixels in the segmentation mask of the class determined by the 'remove' parameter. 
                                                    #This only occurs in segmentation masks that are wrong, where both prostate zones overlap.
        return d

#Convert the segmentation masks to binary masks, with only 0 and 1 values. 
class NormalizeLabelsd(MapTransform):
    def __init__(
    self,
    keys: KeysCollection,
    apply: bool,
    ):
        super().__init__(keys)
        self.apply = apply
    
    def __call__(self, data):
        d = dict(data)
        if self.apply:            
            keys=self.keys
            for key in keys:
                if key in d:
                    d[key]=np.array(d[key]>0, dtype=np.float32)        
        return d

#Remove the channel corresponding to not clinically significant PCa lesions from the ground truth segmentation mask,
#for the case of segmenting only clinically significant lesions 
class IgnoreNCSCad(MapTransform):
    def __init__(
    self,
    keys: KeysCollection,
    apply: bool,
    ):
        super().__init__(keys)
        self.apply = apply
    
    def __call__(self, data):
        d = dict(data)
        if self.apply:            
            key=self.keys[0]
            d[key]=d[key][[0,1,3,4],:,:,:]
        return d

#Compute the bounding box around the prostate
def calculate_bbox(pred):
    minr_abs, minc_abs = pred.shape[1],pred.shape[2]
    maxr_abs,maxc_abs = 0,0
    for slice in range(pred.shape[3]):
        lab=np.logical_or(pred[1,:,:,slice],pred[2,:,:,slice])            
        lab=label(lab)
        regions=regionprops(lab)
        if len(regions)>0:
            for region in regions:
                if region.bbox_area > 13:
                    minr, minc, maxr, maxc = region.bbox
                    if minr<minr_abs: minr_abs=minr
                    if minc<minc_abs: minc_abs=minc
                    if maxr>maxr_abs: maxr_abs=maxr
                    if maxc>maxc_abs: maxc_abs=maxc
    minr_abs=minr_abs-10 if minr_abs-10>=0 else 0
    minc_abs=minc_abs-10 if minc_abs-10>=0 else 0
    maxr_abs=maxr_abs+10 if maxr_abs+10<=pred.shape[1] else pred.shape[1]
    maxc_abs=maxc_abs+10 if maxc_abs+10<=pred.shape[2] else pred.shape[2]
    return  minr_abs, minc_abs, maxr_abs,maxc_abs


#Crop the images and segmentation masks according the the calculated bounding box around the prostate.
class CropBBoxd(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        th: float,
    ):
        super().__init__(keys)
        self.th = th
    def __call__(self, data):
        d = dict(data)
        if 'pred_det' in d:
            minr, minc, maxr, maxc=calculate_bbox(d['pred_det'])   
            vol=[]                  
            for key in self.keys[1:]:
                seg=[[] for i in range(d[key].shape[0])]
                for i in range(d[key].shape[0]):
                    for slice in range(d[key].shape[3]):
                        seg[i].append(d[key][i,minr:maxr,minc:maxc,slice])
                seg=np.swapaxes(np.array(seg),1,2)
                seg=np.swapaxes(seg,2,3)
                d[key]=seg
            for slice in range(d['vol'].shape[3]):
                vol.append(d['vol'][0,minr:maxr,minc:maxc,slice])      
            vol=np.expand_dims(np.array(vol),0)
            vol=np.swapaxes(vol,1,2)
            vol=np.swapaxes(vol,2,3)            
            d['vol']=vol
            
        return d


def prepare(in_dir, images_dir, labels_dir, seg_dirs=None, leng=15, zones=False, cancer=False, ncs=True, pixdim=(0.5, 0.5, 1.25), spatial_size=(160,160,32), 
                validation_rate=None, cache=False, seed=RANDOM_SEED, shuffle=True):
    """
    This function is for preprocessing the data
    """
    set_determinism(seed=seed)

    if cancer: 
        files=select_data(in_dir,images_dir,labels_dir, seg_dirs, leng)
        path_pz = sorted(glob(os.path.join(in_dir, labels_dir, seg_dirs[2], "*.nii.gz")))
        path_tz = sorted(glob(os.path.join(in_dir, labels_dir, seg_dirs[3], "*.nii.gz")))
        i=0
        for pz,tz in zip(path_pz,path_tz):
            files[i]["pz"] = pz
            files[i]["tz"] = tz
            i=i+1
        keys=["vol", "seg_ncs", "seg","pz","tz"]
        mode=("bilinear", "nearest","nearest","nearest","nearest")
    else:
        path_volumes = sorted(glob(os.path.join(in_dir, images_dir, "*.nii.gz")))
        path_segmentation = sorted(glob(os.path.join(in_dir, labels_dir, "*.nii.gz")))
        files=[{"vol": image_name, "seg": label_name} for image_name, label_name in zip(path_volumes, path_segmentation)]
        keys=['vol','seg']
        mode=("bilinear", "nearest")  

    transforms = Compose(
    [
        LoadImaged(keys=keys, allow_missing_keys=True),    
        JoinMasksd(keys=["seg_ncs", "seg"], apply=cancer),
        NormalizeLabelsd(keys=keys[1:], apply=cancer), 
        AddChanneld(keys=keys),
        Spacingd(keys=keys, pixdim=pixdim, mode=mode, allow_missing_keys=True),
        DeleteEmptySlicesd(keys=keys, num_slices=spatial_size[2]),
        ConvertMultiClassd(keys=keys[1:], zones=zones),
        DeleteIntersectionsd(keys=['pz','tz'], cancer=cancer, zones=zones, remove='pz'),
        JoinChannelsd(keys=keys[1:], zones=zones, cancer=cancer),
        NormalizeIntensityd(keys='vol'),
        IgnoreNCSCad(keys='seg', apply=(cancer and not ncs)),
        Resized(keys=["vol", "seg"], spatial_size=(-1,-1, 32),mode=("trilinear", "nearest")),  
        CenterSpatialCropd(keys=["vol", "seg"], roi_size=[192,192,-1]),    
        Resized(keys=["vol", "seg"], spatial_size=spatial_size,mode=("trilinear", "nearest")),   
        ToTensord(keys=["vol","seg"])])   

    img = []
    labels = []
    if cache:
        ds = CacheDataset(data=files, transform=transforms,cache_rate=1.0)
    else:
        ds = Dataset(data=files, transform=transforms)

    loader = DataLoader(ds, batch_size=1,shuffle=shuffle, num_workers=2)     
 
    for patient in loader:    
        img.append(patient['vol'].numpy())           
        labels.append(patient['seg'].numpy())

    img=np.squeeze(img,axis=1)   
    labels=np.squeeze(labels,axis=1)

    if validation_rate is not None:
        x_train, x_val, y_train, y_val = train_test_split(img, labels, test_size=validation_rate, random_state=seed)
        return np.array(x_train,dtype=np.float32),np.array(y_train,dtype=np.float32),np.array(x_val,dtype=np.float32),np.array(y_val,dtype=np.float32)
    else:
        return np.array(img,dtype=np.float32),np.array(labels, dtype=np.float32)
        

def prepare_2(x, num_classes, y, pred_prostate, th=None, spatial_size=(96,96,-1), cache=False, postp=False, seed=RANDOM_SEED, shuffle=True):

    """
    This function is for further preprocessing the data for the segmentation of PCa lesions. 
    """

    set_determinism(seed=seed)
    files=[{} for i in range(len(x))]
    
    for i in range(len(x)):
        files[i]['vol']=x[i,:,:,:,:]
        files[i]['seg']=y[i,:,:,:,:]
        
        files[i]['pred']=pred_prostate[i,:,:,:,:]
        if th is None:
            data=np.argmax(pred_prostate[i,:,:,:,:], axis=0)
            data=np.expand_dims(data,axis=0)
            data=torch.Tensor(data)
            data=one_hot(data,pred_prostate.shape[1], dim=0)
            data=data.numpy()
        else:
            data=pred_prostate[i,:,:,:,:]>th
        files[i]['pred_det']=np.array(data, dtype=np.float32)
        
    
    if postp:
        transforms = Compose([
            FillHolesd(keys=['pred_det'], applied_labels=[1,2], connectivity=1),
            KeepLargestConnectedComponentd(keys=['pred_det'], applied_labels=[1,2], is_onehot=True, independent=True, connectivity=3), 
            AdjustSegChannelsd(keys=['seg'], num_classes=num_classes),   
            DeleteIntersectionsd(keys=['seg'], num_classes=num_classes),                          
            CropBBoxd(keys=["vol", "seg","pred"],th=th),                     
            ConcatenateChannelsd(keys=["vol", "pred"], apply=(num_classes==3 or num_classes==2)),
            Resized(keys=["vol", "seg"], spatial_size=spatial_size,mode=("trilinear", "nearest")),   
            ToTensord(keys=["vol","seg"]),
            ]
        )
    else:        
        transforms = Compose(
        [
            AdjustSegChannelsd(keys=['seg'], num_classes=num_classes),
            DeleteIntersectionsd(keys=['seg'], num_classes=num_classes),
            CropBBoxd(keys=["vol", "seg","pred"],th=th),        
            ConcatenateChannelsd(keys=["vol", "pred"], apply=(num_classes==3 or num_classes==2)),
            Resized(keys=["vol", "seg"], spatial_size=spatial_size,mode=("trilinear", "nearest")),   
            ToTensord(keys=["vol","seg"]),
        ])
    
    img = []
    labels = []
    
    if cache:
        ds = CacheDataset(data=files, transform=transforms,cache_rate=1.0)
    else:
        ds = Dataset(data=files, transform=transforms)

    loader = DataLoader(ds, batch_size=1,shuffle=shuffle, num_workers=2)     

    for patient in loader:
        img.append(patient['vol'].numpy())            
        labels.append(patient['seg'].numpy())

    img=np.squeeze(img,axis=1)
    labels=np.squeeze(labels,axis=1)         
   

    return np.array(img,dtype=np.float32),np.array(labels, dtype=np.float32)
    

def augmentation(d,transforms, num_examples,seed=RANDOM_SEED):
    """
    This function is for augmenting the data.
    """

    #set_determinism(seed=seed)

    data_in = [deepcopy(d) for _ in range(num_examples)]
    ds = Dataset(data_in, transforms)
    loader = DataLoader(ds, num_workers=2, batch_size=1, shuffle=False)

    img=[]
    labels=[]

    for patient in loader:
        img.append(patient['vol'].numpy())      
        labels.append(patient['seg'].numpy())

    img=np.array(img,dtype=np.float32)
    labels= np.array(labels, dtype=np.float32)
    img=np.reshape(img,(img.shape[0]*img.shape[1],img.shape[2],img.shape[3],img.shape[4],img.shape[5]))
    labels=np.reshape(labels,(labels.shape[0]*labels.shape[1],labels.shape[2],labels.shape[3],labels.shape[4],labels.shape[5]))

    return img, labels