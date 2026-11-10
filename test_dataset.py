from src.data.dataset import CalebADataset
from torch.utils.data import DataLoader
dataset = CalebADataset(image_dir="data/raw/celeba/img_align_celeba", label_file="data/raw/celeba/identity_celebA.txt")

print("Total images:", len(dataset))

image, identity = dataset[0]


print("Image shape:", image.shape)
print("Identity:", identity)

loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True
)
print(len(loader))

images, identities = next(iter(loader))

print("Images shape:", images.shape)
print("Identities shape:", identities.shape)