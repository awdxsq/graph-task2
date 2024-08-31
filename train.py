import torch
from torch.optim import Adam
from torch_geometric.data import DataLoader
from dgl.dataloading import GraphDataLoader
from load_cora import load_cora_data
from model import GCN
from torch_geometric.nn import global_mean_pool

def train(dataset, model, optimizer, n_gpu):
    model.train()
    loader = GraphDataLoader(dataset, batch_size=32, shuffle=True, num_workers=n_gpu)
    for data in loader:
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.binary_cross_entropy_with_logits(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

def evaluate(dataset, model, n_gpu):
    model.eval()
    loader = GraphDataLoader(dataset, batch_size=32, shuffle=False, num_workers=n_gpu)
    with torch.no_grad():
        correct = 0
        for data in loader:
            out = model(data.x, data.edge_index)
            pred = out.sigmoid() > 0.5
            correct += (pred[data.test_mask] == data.y[data.test_mask]).sum().item()
    return correct / len(dataset)

def main():
    dataset = load_cora_data()
    model = GCN(dataset.num_features, 16)
    optimizer = Adam(model.parameters(), lr=0.01)
    n_gpu = torch.cuda.device_count()

    for epoch in range(200):
        train(dataset, model, optimizer, n_gpu)
        print(f'Epoch {epoch}, Test Accuracy: {evaluate(dataset, model, n_gpu)}')

if __name__ == "__main__":
    main()