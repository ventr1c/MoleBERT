from evaluation_distill.splitter import scaffold_split, random_scaffold_split, random_split
from torch_geometric.loader import DataLoader, DenseDataLoader
# from models import GCN, DiffPool, MLP, GAT, ChebNet, GIN, GraphSage


def split_loader(dataset, split_method, frac_train, frac_val, frac_test, seed):
    assert split_method in ['random', 'scaffold', 'random_scaffold']
    if split_method == 'random':
        train_dataset, val_dataset, test_dataset = random_split(dataset=dataset, 
                                                                frac_train=frac_train, 
                                                                frac_valid=frac_val, 
                                                                frac_test=frac_test)
    elif split_method == 'scaffold':
        train_dataset, val_dataset, test_dataset = scaffold_split(dataset=dataset, 
                                                                frac_train=frac_train, 
                                                                frac_valid=frac_val, 
                                                                frac_test=frac_test)
    else:
        train_dataset, val_dataset, test_dataset = random_scaffold_split(dataset=dataset, 
                                                                        frac_train=frac_train, 
                                                                        frac_valid=frac_val, 
                                                                        frac_test=frac_test,
                                                                        seed=seed)
    return train_dataset, val_dataset, test_dataset



def batch_loader(train_dataset, val_dataset, test_dataset, model_name, batch_size):
    if model_name == 'diffpool':
        train_loader = DenseDataLoader(train_dataset, batch_size=batch_size)
        val_loader = DenseDataLoader(val_dataset, batch_size=batch_size)
        test_loader = DenseDataLoader(test_dataset, batch_size=batch_size)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
    return train_loader, val_loader, test_loader



# def model_loader(model_name, in_channels, hidden_channels, out_channels, num_layers, dropout, **kwargs):
#     if model_name == 'gcn':
#         model = GCN(in_channels=in_channels,
#                     hidden_channels=hidden_channels,
#                     out_channels=out_channels,
#                     num_layers=num_layers,
#                     dropout=dropout)
#     elif model_name == 'diffpool':
#         model = DiffPool(in_channels=in_channels,
#                         hidden_channels=hidden_channels,
#                         out_channels=out_channels,
#                         num_layers=num_layers,
#                         dropout=dropout,
#                         max_nodes=kwargs['max_nodes'])
#     elif model_name == 'mlp':
#         model = MLP(in_channels=in_channels,
#                     hidden_channels=hidden_channels,
#                     out_channels=out_channels,
#                     num_layers=num_layers,
#                     dropout=dropout)
#     elif model_name == 'gat':
#         model = GAT(in_channels=in_channels,
#                     hidden_channels=hidden_channels,
#                     out_channels=out_channels,
#                     num_layers=num_layers,
#                     dropout=dropout,
#                     gat_heads=8)
#     elif model_name == 'chebnet':
#         model = ChebNet(in_channels=in_channels,
#                     hidden_channels=hidden_channels,
#                     out_channels=out_channels,
#                     num_layers=num_layers,
#                     dropout=dropout,
#                     K=2)
    
#     elif model_name == 'gin':
#         model = GIN(in_channels=in_channels,
#                     hidden_channels=hidden_channels,
#                     out_channels=out_channels,
#                     num_layers=num_layers,
#                     dropout=dropout)
#     elif model_name == 'sage':
#         model = GraphSage(in_channels=in_channels,
#                     hidden_channels=hidden_channels,
#                     out_channels=out_channels,
#                     num_layers=num_layers,
#                     dropout=dropout)
#     return model