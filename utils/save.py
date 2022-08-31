import json
import os
os.environ['MPLCONFIGDIR'] = "./cancer/configuration"
import matplotlib.pyplot as plt
import numpy as np
import torch
from monai.networks.utils import one_hot


def save_experiment(config: dict, metrics: dict, name: str, save_dir="./experiments"):
    """
    Function used to save the configuration used for a certain experiment in a json file
    and the values of the different metrics obtained by the model during that experiment in another file.
    """
    os.makedirs(save_dir, exist_ok=True)
    name = name.replace(save_dir,".")
    base_dir = os.path.join(save_dir, name)
    os.makedirs(base_dir, exist_ok=True)
    with open(os.path.join(base_dir, "parameters.json"), "w") as save_file:
        json.dump(config, save_file)
    
    for metric_type in metrics.keys():
        with open(os.path.join(base_dir, f"{metric_type}.json"), "w") as save_file:
            json.dump(metrics[metric_type], save_file, indent=4)


def plot_save_predictions(x_val: np.ndarray, pred: np.ndarray, th=None, y_val=None, save_dir="./test_preds",show=False):
    """
    Function used to plot the predicted segmentation masks together with the ground truth masks and save the plotted images. 
    One image will be plotted for each slice. 
    """

    os.makedirs(save_dir, exist_ok=True)
    maps=['spring','summer','autumn','winter','cool']
    for num, patient in enumerate(x_val):
        test_patient = patient
        test_outputs = pred[num]
        
        if th is None:
            data=np.argmax(test_outputs, axis=0)
            data=np.expand_dims(data,axis=0)
            data=torch.Tensor(data)
            data=one_hot(data,test_outputs.shape[0], dim=0)
            test_outputs=data.numpy()
        else:
            test_outputs = test_outputs > th
        
        for slice in range(patient.shape[3]):     
            if y_val is not None:
                fig,axs=plt.subplots(1,2, figsize=(18, 6))
                fig.tight_layout() 
                fig.suptitle(f"Patient {num+1}, slice {slice+1}")
                axs[0].set_title("Output")
                axs[0].imshow(test_patient[0,:, :, slice], cmap="gray")
                for i in range(test_outputs.shape[0]-1):
                    mask=np.ma.masked_where(test_outputs[test_outputs.shape[0]-1-i,:, :, slice] == 0, test_outputs[test_outputs.shape[0]-1-i,:, :, slice])
                    axs[0].imshow(mask, cmap=maps[i], alpha=0.3+i/10)
                axs[1].set_title("Label")
                axs[1].imshow(test_patient[0,:, :, slice], cmap="gray")
                for i in range(y_val[num].shape[0]-1):
                    mask=np.ma.masked_where(y_val[num][y_val[num].shape[0]-1-i,:, :, slice] == 0, y_val[num][y_val[num].shape[0]-1-i,:, :, slice])
                    axs[1].imshow(mask, cmap=maps[i], alpha=0.3+i/10)
                fig.savefig(f'{save_dir}/Prediction Image {num+1}_{slice+1}', facecolor='white')
                if show:
                    plt.show()
                
                fig,axs=plt.subplots(2,test_outputs.shape[0], figsize=(18, 6))
                
                fig.subplots_adjust(top=0.8)
                fig.suptitle(f"Patient {num+1}, slice {slice+1}")
                for i in range(test_outputs.shape[0]):
                    axs[0,i].set_title(f"output channel {i}")
                    axs[0,i].imshow(test_outputs[i, :, :, slice], cmap='cividis')
                    axs[1,i].set_title(f"label channel {i}")          
                    axs[1,i].imshow(y_val[num][i, :, :, slice], cmap='cividis')
                fig.tight_layout() 
                fig.savefig(f'{save_dir}/Prediction Masks {num+1}_{slice+1}', facecolor='white')
                if show:
                    plt.show()               
            else:
                fig,ax=plt.subplots(1, figsize=(8, 8))
                fig.suptitle(f"Patient {num+1}, slice {slice+1}")
                ax.imshow(test_patient[0,:, :, slice], cmap="gray")
                for i in range(test_outputs.shape[0]-1):
                    mask=np.ma.masked_where(test_outputs[test_outputs.shape[0]-1-i,:, :, slice] == 0, test_outputs[test_outputs.shape[0]-1-i,:, :, slice])
                    ax.imshow(mask, cmap=maps[i], alpha=0.3+i/10)
                fig.savefig(f'{save_dir}/Prediction Image {num+1}_{slice+1}', facecolor='white')
                if show:
                    plt.show()