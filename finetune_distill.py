from model import GNN, GNN_graphpred

# import pandas as pd
from evaluation_distill.config import cfg, update_cfg
import time
import torch
import torch.nn as nn
import numpy as np
# from models import GCN, DiffPool
from torch_geometric.datasets import MoleculeNet
import torch_geometric.transforms as T
from torchvision import transforms as visionT
from torch_geometric.loader import DataLoader, DenseDataLoader
from torchmetrics import AUROC
from torchmetrics.regression import MeanSquaredError as MSE
from evaluation_distill.utils import init_path, set_seed, change_dtype, ToDense, valid_smiles_filter, change_target
from functools import partial
from evaluation_distill.loader import split_loader, batch_loader
import pandas as pd
LOG_FREQ = 20

from loader import MoleculeDataset
from splitters import scaffold_split, random_scaffold_split

class GNNTrainer():
    def __init__(self, cfg):
        self.seed = cfg.seed
        set_seed(cfg.seed)
        print(self.seed)
        
        self.device = cfg.device
        self.dataset_name = cfg.dataset.name
        self.target_task = cfg.dataset.target_task
        self.split_method = cfg.dataset.split_method
        
        self.gnn_model_name = cfg.gnn.model.name.lower()
        self.hidden_dim = cfg.gnn.model.hidden_dim
        self.num_layers = cfg.gnn.model.num_layers
        self.dropout = cfg.gnn.train.dropout
        self.lr = cfg.gnn.train.lr
        self.weight_decay = cfg.gnn.train.weight_decay
        self.epochs = cfg.gnn.train.epochs
        self.max_nodes = cfg.gnn.model.max_nodes
        self.batch_size = cfg.gnn.train.batch_size
        
        if self.dataset_name in ['esol', 'lipo', 'freesolv']:
            self.metrics = 'rmse'
        else:
            self.metrics = 'auc'

        if self.dataset_name == "hiv":
            num_tasks = 1
        elif self.dataset_name == "bace":
            num_tasks = 1
        elif self.dataset_name == "bbbp":
            num_tasks = 1
        elif self.dataset_name == "clintox":
            num_tasks = 2
        elif self.dataset_name == "esol":
            num_tasks = 1   # Regression
        elif self.dataset_name == "freesolv":
            num_tasks = 1   # Regression
        elif self.dataset_name == "lipo":
            num_tasks = 1   # Regression
        else:
            raise ValueError("Invalid dataset name.")
        
        # to_dense = ToDense(self.max_nodes)
        tsfm = partial(change_target, self.target_task) if self.metrics == 'rmse' else partial(change_dtype, self.target_task)
        # dataset = MoleculeNet(name=self.dataset_name, root='./dataset/', transform=tsfm, pre_filter=valid_smiles_filter)
        dataset = MoleculeDataset("./dataset/" + self.dataset_name,  dataset=self.dataset_name, transform=tsfm)

        split = 'random_scaffold'
        if split == "scaffold":
            smiles_list = pd.read_csv('./dataset/' + self.dataset_name + '/processed/smiles.csv', header=None)[0].tolist()
            train_dataset, valid_dataset, test_dataset = scaffold_split(dataset, smiles_list, null_value=0, frac_train=0.8,frac_valid=0.1, frac_test=0.1)
            print("scaffold")
        elif split == "random":
            train_dataset, valid_dataset, test_dataset = random_split(dataset, null_value=0, frac_train=0.8,frac_valid=0.1, frac_test=0.1, seed = self.seed)
            print("random")
        elif split == "random_scaffold":
            smiles_list = pd.read_csv('./dataset/' + self.dataset_name + '/processed/smiles.csv', header=None)[0].tolist()
            train_dataset, valid_dataset, test_dataset = random_scaffold_split(dataset, smiles_list, null_value=0, frac_train=0.8,frac_valid=0.1, frac_test=0.1, seed = self.seed)
            print("random scaffold")
        else:
            raise ValueError("Invalid split option.")

        self.dataset = dataset
        self.num_graphs = len(self.dataset)
        self.num_classes = self.dataset.y.max().long().item() + 1
        self.num_features = self.dataset.x.shape[1]
        print(self.dataset, f'has {self.num_graphs} graphs.')
        

        # train_idx, valid_idx, test_idx = split_loader(dataset=self.dataset, 
        #                                             split_method=self.split_method,
        #                                             frac_train=cfg.dataset.train_prop, 
        #                                             frac_val=cfg.dataset.val_prop, 
        #                                             frac_test=cfg.dataset.test_prop,
        #                                             seed=self.seed)
        # self.train_dataset = torch.utils.data.Subset(self.dataset, train_idx)
        # self.val_dataset = torch.utils.data.Subset(self.dataset, valid_idx)
        # self.test_dataset = torch.utils.data.Subset(self.dataset, test_idx)

        self.train_dataset = train_dataset
        self.val_dataset = valid_dataset
        self.test_dataset = test_dataset
        
        self.train_loader, self.val_loader, self.test_loader = batch_loader(self.train_dataset, 
                                                                            self.val_dataset, 
                                                                            self.test_dataset, 
                                                                            self.gnn_model_name, 
                                                                            self.batch_size)
        # self.model = model_loader(model_name=self.gnn_model_name,
        #                             in_channels=self.num_features,
        #                             hidden_channels=self.hidden_dim,
        #                             out_channels=1 if self.metrics=='rmse' else self.num_classes,
        #                             num_layers=self.num_layers,
        #                             dropout=self.dropout,
        #                             max_nodes=self.max_nodes).to(self.device)
        graph_pooling = 'mean'
        gnn_type = 'gin'
        self.model = GNN_graphpred(num_layer=5, 
                                    emb_dim = 300, 
                                    num_tasks = 1 if self.metrics=='rmse' else self.num_classes, 
                                    JK = 'last', 
                                    drop_ratio = 0.5, 
                                    graph_pooling = graph_pooling, 
                                    gnn_type = gnn_type)
        lr = 0.001
        lr_scale = 1
        decay = 0
        model_param_group = []
        model_param_group.append({"params": self.model.gnn.parameters()})
        if graph_pooling == "attention":
            model_param_group.append({"params": self.model.pool.parameters(), "lr":lr*lr_scale})
        model_param_group.append({"params": self.model.graph_pred_linear.parameters(), "lr":lr*lr_scale})
        self.optimizer = torch.optim.Adam(model_param_group, lr=lr, weight_decay=decay)

        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.loss_func = nn.MSELoss(reduction='mean') if self.metrics == 'rmse' else nn.CrossEntropyLoss()

        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"\nNumber of parameters: {trainable_params}")
        self.ckpt = f"gnn_output/{self.dataset_name}/{self.split_method}_{self.gnn_model_name}_{self.hidden_dim}_{self.seed}.pt"
        print(f'Model will be saved: {self.ckpt}', '\n')
        

    def _forward(self, data):
        if self.gnn_model_name == 'diffpool':
            logits = self.model(data.x, data.adj)
        else:
            logits = self.model(data.x, data.edge_index, data.edge_attr, data.batch)
        return logits

    def _train(self):
        self.model.train()
        train_evaluate = MSE(squared=False).to(self.device) if self.metrics == 'rmse' else AUROC(task="multiclass", num_classes=self.num_classes)
        for data in self.train_loader:
            data = data.to(self.device)
            self.optimizer.zero_grad()
            logits, _ = self._forward(data)
            logits = logits.squeeze() if self.metrics == 'rmse' else logits
            target_y = data.y.T.squeeze() if self.gnn_model_name == 'diffpool' else data.y
            loss = self.loss_func(input=logits, target=target_y)
            train_evaluate.update(preds=logits, target=target_y)
            loss.backward()
            self.optimizer.step()
        train_metric = train_evaluate.compute()
        return loss.item(), train_metric
             

    @ torch.no_grad()
    def _evaluate(self):
        self.model.eval()
        
        val_evaluate = MSE(squared=False).to(self.device) if self.metrics == 'rmse' else AUROC(task="multiclass", num_classes=self.num_classes)
        for data in self.val_loader:
            data = data.to(self.device)
            logits, _ = self._forward(data)
            logits = logits.squeeze() if self.metrics == 'rmse' else logits
            target_y = data.y
            val_evaluate.update(preds=logits, target=target_y)
        val_metric = val_evaluate.compute()
        
        test_evaluate = MSE(squared=False).to(self.device) if self.metrics == 'rmse' else AUROC(task="multiclass", num_classes=self.num_classes)
        for data in self.test_loader:
            data = data.to(self.device)
            logits, _ = self._forward(data)
            logits = logits.squeeze() if self.metrics == 'rmse' else logits
            target_y = data.y
            test_evaluate.update(preds=logits, target=target_y)
        test_metric = test_evaluate.compute()            
        return val_metric, test_metric, logits
    

    def train(self):
        if self.metrics == 'rmse':
            best_val_metric = 1e8
            best_test_metric = 1e8
            for epoch in range(self.epochs + 1):
                loss, train_rmse = self._train()
                val_rmse, test_rmse, _ = self._evaluate()
                
                if val_rmse < best_val_metric:
                    best_val_metric = val_rmse
                    best_test_metric = test_rmse
                    torch.save(self.model.state_dict(), self.ckpt)
                    
                if epoch % LOG_FREQ == 0:
                    print(f'Epoch: {epoch}, Loss: {loss:.4f}, TrainRMSE: {train_rmse:.4f}, ValRMSE: {val_rmse:.4f}, TestRMSE: {test_rmse:.4f}')
                    print(f'                BestValRMSE: {best_val_metric:.4f}, BestTestRMSE: {best_test_metric:.4f}')
            
        else:
            best_val_metric = 0
            best_test_metric = 0
            for epoch in range(self.epochs + 1):
                loss, train_auc = self._train()
                val_auc, test_auc, _ = self._evaluate()
                
                if val_auc > best_val_metric:
                    best_val_metric = val_auc
                    best_test_metric = test_auc
                    torch.save(self.model.state_dict(), self.ckpt)
                    
                if epoch % LOG_FREQ == 0:
                    print(f'Epoch: {epoch}, Loss: {loss:.4f}, TrainAuc: {train_auc:.4f}, ValAuc: {val_auc:.4f}, TestAuc: {test_auc:.4f}')
                    print(f'                BestValAuc: {best_val_metric:.4f}, BestTestAuc: {best_test_metric:.4f}')
             
        print(f'[{self.gnn_model_name}] model saved: {self.ckpt}, with best_val_acc:{best_val_metric:.4f} and corresponding test_acc:{best_test_metric:.4f}', '\n')
        return self.model

    @ torch.no_grad()
    def eval_and_save(self):
        self.model.load_state_dict(torch.load(self.ckpt))
        print(f'[{self.gnn_model_name}] model saved: {self.ckpt}')
        val_acc, test_acc, logits = self._evaluate()
        print(f'[{self.gnn_model_name}] ValAuc: {val_acc:.4f}, TestAuc: {test_acc:.4f}\n')
        res = {'val_auc': val_acc.detach().cpu().numpy(), 'test_auc': test_acc.detach().cpu().numpy()}
        return logits, res



def run_train_gnn(cfg):
    seeds = cfg.seed
    all_acc = []
    print(seeds) 
    for seed in seeds:
        cfg.seed = seed
        trainer = GNNTrainer(cfg)
        trainer.train()
        _, acc = trainer.eval_and_save()
        all_acc.append(acc)
        print("-"*100, '\n')

    if len(all_acc) > 1:
        df = pd.DataFrame(all_acc)
        print(df)
        for k, v in df.items():
            print(f"{k}: {v.mean():.2f}±{v.std():.2f}")
            
        path = f'prt_results/prt_gnn_results/{cfg.dataset.name}/{cfg.dataset.split_method}_{cfg.gnn.model.name}_{cfg.gnn.model.hidden_dim}.csv'
        df.to_csv(path, index=False)



if __name__ == '__main__':
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
    
    cfg = update_cfg(cfg)
    run_train_gnn(cfg)

