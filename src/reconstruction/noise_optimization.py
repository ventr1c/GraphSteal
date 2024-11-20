import numpy as np
import scipy.sparse as sp
import torch
from torch import optim
from torch.nn import functional as F
from torch.nn.parameter import Parameter
import torch.nn as nn
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from tqdm import tqdm
from deeprobust.graph.utils import normalize_adj_tensor
import copy
from torch.autograd import Variable
from torch_geometric.data import Data, Batch
import copy

class NoiseGeneration(torch.nn.Module):
    def __init__(self, aux_dataset, target_model, device):
        super(NoiseGeneration, self).__init__()
        self.aux_dataset = copy.deepcopy(aux_dataset)
        self.aux_dataset = self.edge_index_to_full_edge_index(aux_dataset)
        self.recons_dataset = []
        self.target_model = target_model
        self.device = device

        self.if_reconstruct_feat = True
        self.extraction_edge_lr = 0.001
        self.ce_weight = 1
        
        self.ori_feats = [data['x'].to(device) for data in self.aux_dataset]
        self.ori_labels = [data['y'].item() for data in self.aux_dataset]
        self.ori_labels = torch.tensor(self.ori_labels).to(device)
        self.ori_edge_indexs = [data['edge_index'].to(device) for data in self.aux_dataset]
        self.ori_edge_weights = [data['edge_weight'].to(device) for data in self.aux_dataset]
        self.ori_edge_attrs = None # [data['edge_attr'].to(device) for data in self.aux_dataset]
        for data in self.aux_dataset:
            if(self.ori_edge_attrs == None):
                self.ori_edge_attrs = data.edge_attr
            else:
                self.ori_edge_attrs = torch.concat((self.ori_edge_attrs, data.edge_attr), dim = 0)
        self.ori_edge_attrs = self.ori_edge_attrs.to(self.device)
        self.modified_feats = copy.deepcopy(self.ori_feats)
        self.modified_edge_weights = copy.deepcopy(self.ori_edge_weights)
        
        self.complementary_edge_weights = []
        self.edge_weights_change = []
        for i, edge_weight in enumerate(self.ori_edge_weights):
            all_one_edge_weight = torch.ones_like(edge_weight).to(self.device)
            for j in range(self.ori_feats[i].shape[0]):
                self_loop_index_j = j * self.ori_feats[i].shape[0] + (j+1) - 1
                all_one_edge_weight[self_loop_index_j] = 0
            compementary_edge_weight = (all_one_edge_weight - edge_weight) - edge_weight
            self.complementary_edge_weights.append(compementary_edge_weight)
            edge_weight_change = Parameter(torch.FloatTensor(edge_weight.shape[0])).to(self.device)
            edge_weight_change.data.fill_(0)
            self.edge_weights_change.append(edge_weight_change)
        
        self.feats_change = []
        for feat in self.ori_feats:
            feats_change = Parameter(feat.clone().detach()).to(self.device)
            feats_change.data.fill_(0)
            self.feats_change.append(feats_change)

        self.edge_weights_change = nn.ParameterList(self.edge_weights_change)
        self.feats_change = nn.ParameterList(self.feats_change)
        
        
        '''Set the number of perturbations'''
        self.n_perturbations = 4
        '''Initialize the reconstruction'''
        self.initialize_reconstruction() 
            
    def edge_index_to_full_edge_index(self, aux_dataset):
        full_aux_dataset = []
        for i in range(len(aux_dataset)):
            data = aux_dataset[i]
            nnodes = data.x.shape[0]
            '''add unobserved edge with edge_weight 0 into data.edge_index'''
            full_adj = torch.ones((nnodes,nnodes))
            full_edge_index, full_edge_weight = dense_to_sparse(full_adj)
            full_edge_weight = full_edge_weight.type(torch.float32)
            full_edge_weight = full_edge_weight * 0

            def func(i,j, nnodes):
                i +=0
                j +=0
                # nnodes
                out = (i) * nnodes + j
                # out -= 1
                # print((i), nnodes, j)
                # print(out)
                return out
            # print(data.edge_index)
            for (i,j) in zip(data.edge_index[0], data.edge_index[1]):
                full_edge_weight[func(i,j, data.x.shape[0])] = 1
            full_data = Data(x=data.x, edge_index=full_edge_index, edge_weight=full_edge_weight, y=data.y, edge_attr= data.edge_attr)
            full_aux_dataset.append(full_data)
        return full_aux_dataset
    
    def get_modified_edge_weight(self, ori_edge_weight, complementary_edge_weight, edge_weight_change):
        modified_edge_weight = ori_edge_weight + complementary_edge_weight * edge_weight_change
        return modified_edge_weight
    
    def get_modified_feat(self, ori_feat, feat_change):
        modified_feat = ori_feat + feat_change

        return modified_feat
    
    def projection(self, adj_change, n_perturbations = None):
        if n_perturbations is None:
            adj_change.data.copy_(torch.clamp(adj_change.data, min=0, max=1))
        else:
            if torch.clamp(adj_change, 0, 1).sum() > n_perturbations:
                # left = (adj_change - 1).min()
                # right = adj_change.max()
                # miu = self.bisection(adj_change, left, right, n_perturbations, epsilon=1e-5)
                # adj_change.data.copy_(torch.clamp(adj_change.data - miu, min=0, max=1))
                
                adj_change.data.copy_(torch.clamp(adj_change.data, min=0, max=1))
                '''# find the top-k largest values, k = n_perturbations'''
                idx = torch.argsort(adj_change, descending=True)[:n_perturbations]
                mask = torch.zeros_like(adj_change)
                mask[idx] = 1
                adj_change.data.copy_(adj_change * mask)


            else:
                adj_change.data.copy_(torch.clamp(adj_change.data, min=0, max=1))
        return adj_change
    
    def projection_edge_weight(self, edge_weight_change, n_perturbations = None):
        if n_perturbations is None:
            edge_weight_change.data.copy_(torch.clamp(edge_weight_change.data, min=0, max=1))
        else:
            if torch.clamp(edge_weight_change, 0, 1).sum() > n_perturbations:
                edge_weight_change.data.copy_(torch.clamp(edge_weight_change.data, min=0, max=1))
                idx = torch.argsort(edge_weight_change, descending=True)[:n_perturbations]
                mask = torch.zeros_like(edge_weight_change)
                mask[idx] = 1
                edge_weight_change.data.copy_(edge_weight_change * mask)
            else:
                edge_weight_change.data.copy_(torch.clamp(edge_weight_change.data, min=0, max=1))
        return edge_weight_change
    
    def projection_feat(self, modified_feat, n_perturbations = None):
        modified_feat.data.copy_(torch.clamp(modified_feat.data, min=0, max=1))

        return modified_feat
    
    def bisection(self, adj_change, a, b, n_perturbations, epsilon):
        def func(x):
            return torch.clamp(adj_change-x, 0, 1).sum() - n_perturbations
        miu = a
        while ((b-a) >= epsilon):
            miu = (a+b)/2
            # Check if middle point is root
            if (func(miu) == 0.0):
                break
            # Decide the side to repeat the steps
            if (func(miu)*func(a) < 0):
                b = miu
            else:
                a = miu
        # print("The value of root is : ","%.4f" % miu)
        return miu
    
    def _loss(outputs, labels):
        loss = F.cross_entropy(outputs, labels)

        return loss
    
    def reconstruct(self, epochs = 40, verbose = True):
        if verbose:
            print("Optimizing input graphs...")
        print("self.ori_edge_attrs",self.ori_edge_attrs.shape)
        target_model = self.target_model
        target_model.eval()

        pre_epoch_loss = 1e10
        patience = 1e10
        patience_cnt = 0
        print(self.complementary_edge_weights[0])
        for epoch in range(epochs):
            epoch_loss = 0
            degree_loss = 0
            for i in range(len(self.ori_edge_indexs)):
                modified_edge_weight = self.get_modified_edge_weight(self.ori_edge_weights[i], self.complementary_edge_weights[i], self.edge_weights_change[i])
                modified_edge_weight = self.projection_edge_weight(modified_edge_weight, n_perturbations=None)
                if(self.if_reconstruct_feat):
                    modified_feat = self.get_modified_feat(self.ori_feats[i], self.feats_change[i])
                    modified_feat = self.projection_feat(modified_feat)
                else:
                    modified_feat = self.ori_feats[i]
                output, _, _ = target_model(modified_feat, self.ori_edge_indexs[i], None, modified_edge_weight)
                output = output[0]
                loss = F.cross_entropy(output, self.ori_labels[i])
                epoch_loss += loss
                
                if(self.if_reconstruct_feat):
                    pass
                else:
                    adj_grad = torch.autograd.grad(loss, self.edge_weights_change[i])[0]
                    lr = self.extraction_edge_lr if self.extraction_edge_lr else 10/np.sqrt(epoch+1)
                    self.edge_weights_change[i].data.add_(-1 * lr * adj_grad)
                
                    # TODO: add SVD
                    self.edge_weights_change[i] = self.projection_edge_weight(self.edge_weights_change[i], n_perturbations=self.n_perturbations)
                    self.modified_edge_weights[i] = self.get_modified_edge_weight(self.ori_edge_weights[i], self.complementary_edge_weights[i], self.edge_weights_change[i])

            self.member_weight = self.ce_weight
            total_loss = self.member_weight * epoch_loss 

            if(verbose and epoch % 1 == 0):
                print('Epoch: {:03d}, Training Loss: {:.5f}'.format(epoch, epoch_loss))
            # early stop: if the loss does not decrease for 10 epochs, stop training
            if(epoch_loss >= pre_epoch_loss):
                patience_cnt += 1
            else:
                patience_cnt = 0

            if(patience_cnt >= patience):
                break
            pre_epoch_loss = epoch_loss
        
        # self.random_sample(self.n_perturbations)

        for i in range(len(self.modified_edge_weights)):
            modified_edge_weight = self.get_modified_edge_weight(self.ori_edge_weights[i], self.complementary_edge_weights[i], self.edge_weights_change[i])
            output, em, _ = target_model(self.ori_feats[i], self.ori_edge_indexs[i], None, modified_edge_weight)
            self.edge_weights_change[i] = self.dot_product_decode(em)
            self.modified_edge_weights[i] = self.get_modified_edge_weight(self.ori_edge_weights[i], self.complementary_edge_weights[i], self.edge_weights_change[i])
            

        # save final modified adj
        for i in range(len(self.modified_edge_weights)):
            if(self.if_reconstruct_feat):
                self.modified_edge_weights[i] = self.get_modified_edge_weight(self.ori_edge_weights[i], self.complementary_edge_weights[i], self.edge_weights_change[i])
                self.modified_feats[i] = self.get_modified_feat(self.ori_feats[i], self.feats_change[i])
                self.modified_feats[i] = self.projection_feat(self.modified_feats[i])
                
                modified_edge_weight = self.modified_edge_weights[i]
                modified_feats = self.modified_feats[i]
                modified_edge_index, modified_edge_weight = self.update_edge_index_with_edge_weight(self.ori_edge_indexs[i], modified_edge_weight, self.device)


                idx = np.random.randint(self.ori_edge_attrs.shape[0], size = modified_edge_index.shape[1])
                modified_edge_attr = self.ori_edge_attrs[idx]
                self.recons_dataset.append(Data(x=modified_feats, edge_index=modified_edge_index, edge_weight=modified_edge_weight, edge_attr = modified_edge_attr, y=self.ori_labels[i]))
            else:
                self.edge_weights_change[i] = self.projection_edge_weight(self.edge_weights_change[i], n_perturbations=self.n_perturbations)
                self.modified_edge_weights[i] = self.get_modified_edge_weight(self.ori_edge_weights[i], self.complementary_edge_weights[i], self.edge_weights_change[i])
                modified_edge_weight = self.modified_edge_weights[i]
                modified_edge_index, modified_edge_weight = self.update_edge_index_with_edge_weight(self.ori_edge_indexs[i], modified_edge_weight, self.device)

                idx = np.random.randint(self.ori_edge_attrs.shape[0], size = modified_edge_index.shape[1])
                modified_edge_attr = self.ori_edge_attrs[idx]
                self.recons_dataset.append(Data(x=self.ori_feats[i], edge_index=modified_edge_index, edge_weight=modified_edge_weight, edge_attr = modified_edge_attr, y=self.ori_labels[i]))
        return self.recons_dataset
    
    def random_sample(self, n_perturbations):
        K = 20
        best_loss = -1000
        target_model = self.target_model
        target_model.eval()
        with torch.no_grad():
            for i in range(len(self.ori_edge_indexs)):
                best_s = None
                best_loss = 1e10
                s = self.edge_weights_change[i].cpu().detach().numpy()
                for j in range(K):
                    sampled = np.random.binomial(1, s)
                    if((n_perturbations != None) and (sampled.sum() > n_perturbations)):
                        continue
                    self.edge_weights_change[i].data.copy_(torch.tensor(sampled))
                    if(self.if_reconstruct_feat):
                        modified_edge_weights = self.get_modified_edge_weight(self.ori_edge_weights[i], self.complementary_edge_weights[i], self.edge_weights_change[i])
                        modified_feats = self.get_modified_feat(self.ori_feats[i], self.feats_change[i])
                        modified_feats = self.projection_feat(modified_feats)
                        output, _, _ = target_model(modified_feats, self.ori_edge_indexs[i], None, modified_edge_weights)
                    else:
                        modified_edge_weights = self.get_modified_edge_weight(self.ori_edge_weights[i], self.complementary_edge_weights[i], self.edge_weights_change[i])
                        output, _, _ = target_model(self.ori_feats[i], self.ori_edge_indexs[i], None, modified_edge_weights)
                    output = output[0]
                    loss = F.cross_entropy(output, self.ori_labels[i])
                    # loss = F.nll_loss(output[idx_train], labels[idx_train])
                    if best_loss > loss:
                        best_loss = loss
                        best_s = sampled
                self.edge_weights_change[i].data.copy_(torch.tensor(best_s))
    
    def update_edge_index_with_edge_weight(self, edge_index, edge_weight, device):
        if(edge_weight is not None):
            edge_index = edge_index[:, edge_weight > 0]
            edge_weight = torch.ones(edge_index.shape[1]).to(device)
        else:
            edge_weight = torch.ones(edge_index.shape[1]).to(device)
        return edge_index, edge_weight
        
    def initialize_reconstruction(self):
        '''Construct the label vector for reconstruction'''
        self.num_classes = np.max(self.ori_labels.cpu().detach().numpy()) + 1
        if(self.num_classes == 2):
            # crate binary labels (1/-1)
            self.recons_labels = copy.deepcopy(self.ori_labels.cpu().detach().numpy())
            for i in range(len(self.recons_labels)):
                if(self.recons_labels[i] == 0):
                    self.recons_labels[i] = -1
            self.recons_labels = torch.tensor(self.recons_labels).to(self.device)
        else:
            self.recons_labels = self.ori_labels.clone().to(self.device)
        
    def dot_product_decode(self, Z):
        nnodes = Z.shape[0]
        Z = F.normalize(Z, p=2, dim=1)
        A_pred = torch.relu(torch.matmul(Z, Z.t()))
        tril_indices = torch.tril_indices(row=nnodes, col=nnodes, offset=-1)
        edge_index, edge_weight = self.adj_to_full_edge_index(A_pred)
        edge_weight = edge_weight.to(self.device)
        return edge_weight
    
    def adj_to_full_edge_index(self, adj):
        edge_index, edge_weight = dense_to_sparse(adj)
        nnodes = adj.shape[0]
        '''add unobserved edge with edge_weight 0 into data.edge_index'''
        full_adj = torch.ones((nnodes,nnodes))
        full_edge_index, full_edge_weight = dense_to_sparse(full_adj)
        # to float32
        full_edge_weight = full_edge_weight.type(torch.float32)
        full_edge_weight = full_edge_weight * 0

        def func(i,j, nnodes):
            i +=1
            j +=1
            out = (i-1) * nnodes + j
            out -= 1
            return out
        
        for (i,j) in zip(edge_index[0], edge_index[1]):
            full_edge_weight[func(i,j, adj.shape[0])] = 1
        return full_edge_index, full_edge_weight