import numpy as np
import torch
from torch_geometric.data import batch
from datasets.dataset_utils import label_regression2classification
import utils
import torch.nn.functional as F
from diffusion.diffusion_utils import partial_sample

@torch.no_grad()
def label_wise_sorted_by_conf(cfg, model, dataset, device):
    model.eval()
    label_wise_indices = {}
    
    target = dataset.data.y.clone()
    ori_y = torch.zeros(dataset.data.y.shape[0], 0).type_as(dataset.data.y)
    y = label_regression2classification(target, cfg.classifier.num_classes, device)
    unique_classes = np.unique(y.cpu().numpy())

    for c in unique_classes:
        # Filter samples that belong to the current class.
        y_idx_list = [i for i in range(len(dataset)) if y[i] == c]
        y_idx_list = np.array(y_idx_list)
        conf_scores = []
        for i in y_idx_list:
            try:
                data = dataset[i].to(device)
                dense_data, node_mask = utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch)
                dense_data = dense_data.mask(node_mask)
                X, E = dense_data.X, dense_data.E
                noisy_data = model.apply_noise(X, E, ori_y[i], node_mask)
                extra_data = model.compute_extra_data(noisy_data)
                # print(noise_data.y, extra_data.y)
                pred = model.forward(noisy_data, extra_data, node_mask)
                output = F.softmax(pred.y)
                # print(pred.y, output)

                conf_score = output[0][c].item()
                conf_scores.append(conf_score)
            except:
                data = dataset[i].to(device)
                dense_data, node_mask = utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch)
                dense_data = dense_data.mask(node_mask)
                X, E = dense_data.X, dense_data.E
                print(X.shape, E.shape)
        conf_scores = np.array(conf_scores)
        # sort conf_scores in descending order and get corresponding indices
        sorted_y_idx = y_idx_list[np.argsort(conf_scores)[::-1]]
        label_wise_indices[c] = sorted_y_idx #sorted_y_idx[:n]
    
    return label_wise_indices

def pyg_to_discrete(probX,probE,node_mask):
    ''' Sample features from multinomial distribution with given probabilities (probX, probE, proby)
        :param probX: bs, n, dx_out        node features
        :param probE: bs, n, n, de_out     edge features
        :param proby: bs, dy_out           global features.
    '''
    probX = probX
    probE = probE
    bs, n, _ = probX.shape
    # Noise X
    # The masked rows should define probability distributions as well
    probX[~node_mask] = 1 / probX.shape[-1]

    # Flatten the probability tensor to sample with multinomial
    probX = probX.reshape(bs * n, -1)       # (bs * n, dx_out)

    # Sample X
    # X_t = probX.multinomial(1)                                  # (bs * n, 1)
    X_t = partial_sample(probX)
    X_t = X_t.reshape(bs, n)     # (bs, n)

    # Noise E
    # The masked rows should define probability distributions as well
    inverse_edge_mask = ~(node_mask.unsqueeze(1) * node_mask.unsqueeze(2))
    diag_mask = torch.eye(n).unsqueeze(0).expand(bs, -1, -1)

    probE[inverse_edge_mask] = 1 / probE.shape[-1]
    probE[diag_mask.bool()] = 1 / probE.shape[-1]

    probE = probE.reshape(bs * n * n, -1)    # (bs * n * n, de_out)

    # Sample E
    # E_t = probE.multinomial(1)
    E_t = partial_sample(probE)
    E_t = E_t.reshape(bs, n, n)   # (bs, n, n)
    E_t = torch.triu(E_t, diagonal=1)
    E_t = (E_t + torch.transpose(E_t, 1, 2))
    # return PlaceHolder(X=X_t, E=E_t, y=torch.zeros(bs, 0).type_as(X_t))
    return X_t, E_t