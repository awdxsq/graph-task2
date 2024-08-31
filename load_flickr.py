from dgl.data import FlickrGraphDataset

def load_flickr_data():
    dataset = FlickrGraphDataset()
    return dataset[0]