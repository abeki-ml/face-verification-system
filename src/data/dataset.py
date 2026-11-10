from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class CalebADataset(Dataset):
    def __init__(self, image_dir, label_file):
        self.image_dir = Path(image_dir)
        self.label_file = Path(label_file)
        with open (self.label_file, "r") as file:
            lines = file.readlines()
            self.samples = []
            for line in lines:
                image_name, identity = line.strip().split()
                self.samples.append((image_name, int(identity)))
            self.transform = transforms.Compose([
                transforms.Resize((224,224)),
                transforms.ToTensor()
            ])
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        image_name, identity = self.samples[index]
        image_path = self.image_dir / image_name
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        return image, identity



