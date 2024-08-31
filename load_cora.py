from dgl.data import CoraGraphDataset

def load_cora_data():
    dataset = CoraGraphDataset()
    return dataset[0]