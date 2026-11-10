import random
from torch.utils.data import Dataset

class FacePairDataset(Dataset):
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset