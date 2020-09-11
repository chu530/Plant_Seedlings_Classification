from torch.utils.data import Dataset
import os
from PIL import Image


class PlantSeedlingDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.images = []
        self.labels = []
        self.num_classes = 0
        self.transform = transform

        class_names = os.walk(self.root_dir).__next__()[1]
        self.num_classes = len(class_names)

        for index, class_name in enumerate(class_names):
            label = index
            img_dir = os.path.join(self.root_dir, class_name)
            img_files = os.walk(img_dir).__next__()[2]

            for img_file in img_files:
                img_file = os.path.join(img_dir, img_file)
                img = Image.open(img_file)
                if img is not None:
                    self.images.append(img_file)
                    self.labels.append(label)

    def __getitem__(self, index):
        image = Image.open(self.images[index]).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image, self.labels[index]

    def __len__(self):
        return len(self.images)
