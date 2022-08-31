import numpy as np
from monai.transforms import(
    Compose,
    ToTensor,
    Transform,
    FillHoles,
)

from monai.config.type_definitions import NdarrayOrTensor
from monai.data import DataLoader, Dataset
import torch
from skimage.measure import label
from monai.networks.utils import one_hot
from typing import Sequence, Union


#Keep only the 'num_components' largerst connected components and remove the rest, for each class given in 'applied_labels'.
class KeepLargestConnectedComponents(Transform):
    def __init__(
        self,
        applied_labels: Union[Sequence[int], int],
        connectivity: int,
        num_components: Sequence[int]
    ) -> None:

        super().__init__()
        self.applied_labels = applied_labels
        self.connectivity = connectivity
        self.num_c=num_components
    
    def __call__(self, img: NdarrayOrTensor) -> NdarrayOrTensor:
        n=0
        for l in self.applied_labels:
            lab=img[l,:,:,:]
            cc,count=label(lab, connectivity=self.connectivity, return_num=True)  
            if self.num_c[n]<count: #If the number of connected components found is smaller than the desired number to keep, this function has no effect. 
                bincounts=np.bincount(cc.flat) #Count the number of voxels that each connected component has.
                idx=(-bincounts).argsort()
                largest_cc=idx[1:self.num_c[n]+1] #Contains the connected components sorted in decreasing order of their size. 
                if self.num_c[n]==1:
                    largest=cc==(largest_cc[0])
                else:
                    largest=np.logical_or(cc==(largest_cc[0]),cc==(largest_cc[1]))
                    for i in range(2, len(largest_cc)):
                        largest=np.logical_or(largest,cc==(largest_cc[i]))
                img[l,:,:,:]=largest
            n=n+1
        return img
    
#Remove detected PCa lesions that have a volume smaller than the value given by the parameter 'min_vol'
class RemoveSmallLesions(Transform):
    def __init__(
        self,
        applied_labels: Union[Sequence[int], int],
        connectivity: int,
        spatial_dims: Sequence[int],
        min_vol: int
    ) -> None:

        super().__init__()
        self.applied_labels = applied_labels
        self.connectivity = connectivity
        self.spatial_dims=spatial_dims
        self.min_vol=min_vol
    
    def __call__(self, img: NdarrayOrTensor) -> NdarrayOrTensor:
        pix_vol=self.spatial_dims[0]*self.spatial_dims[1]*self.spatial_dims[2] #Volume of a voxel
        for l in self.applied_labels:
            lab=img[l,:,:,:]
            cc=label(lab, connectivity=self.connectivity, return_num=False)  
            bincounts=np.bincount(cc.flat) #Count the number of voxels that each connected component has.
            remove=[] #Connected components to remove
            for i in range(1, len(bincounts)):
                vol=bincounts[i]*pix_vol #Volume of the connected component. If it is too small it is considered a false positive and it is deleted. 
                if vol<self.min_vol:
                    remove.append(i)
            for j in range(len(remove)):             
                same=(cc==remove[j])
                img[l,:,:,:][same==True]=False
        return img


def postprocessing(pred, labels, num_c, spatial_dims, min_vol, th=None):
    """
    This function is for postprocessing the predicted segmentation masks.
    """
    if th is None:
        data=np.argmax(pred, axis=1)
        data=np.expand_dims(data,axis=1)
        data=torch.Tensor(data)
        data=one_hot(data,pred.shape[1], dim=1)
        data=data.numpy()
    else:
        data=np.array(pred>th,dtype=np.float32)

    transforms = Compose(
    [
        FillHoles(applied_labels=labels, connectivity=3),
        KeepLargestConnectedComponents(applied_labels=labels, connectivity=3, num_components=num_c),
        RemoveSmallLesions(applied_labels=labels, spatial_dims=spatial_dims, connectivity=3, min_vol=min_vol),
        ToTensor()
    ])

    img = []
    
    ds = Dataset(data=data, transform=transforms)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2)     

    for patient in loader: 
        img.append(patient.numpy())     
    img=np.squeeze(img,axis=1)

    return img