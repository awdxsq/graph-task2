from dgl.data import CiteseerGraphDataset

def load_citeseer_data():
    dataset = CiteseerGraphDataset()
    return dataset[0]