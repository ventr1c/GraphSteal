import torch
from torch_geometric.data import Data, Dataset
import numpy as np

def label_regression2classification(y, num_classes, device):
    y = y.cpu()
    tar_task = 0
    regression_values = [y[:, tar_task]]
    all_values = torch.cat(regression_values, dim=0)
    q = np.arange(0,1,1/num_classes)
    q = torch.FloatTensor(q[1:])
    quantiles = all_values.quantile(q=q)
    thresholds = quantiles.tolist()
    # Step 3: Create a new dataset
    value = y[:, tar_task]
    update_y = value.clone()
    
    for i in range(num_classes):
        if(i == num_classes - 1):
            update_y[torch.where(value > thresholds[i-1])[0]] = i
        elif(i == 0):
            update_y[torch.where(value <= thresholds[i])[0]] = i
        else:
            condition = torch.logical_and(value <= thresholds[i], value > thresholds[i-1])
            update_y[torch.where(condition)[0]] = i
    update_y = update_y.to(device)
    update_y = update_y.long()
    return update_y