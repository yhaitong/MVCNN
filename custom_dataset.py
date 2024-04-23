from torch.utils.data.dataset import Dataset
import os
from PIL import Image
from collections import defaultdict


class MultiViewDataSet(Dataset):

    def find_classes(self, dir):
        classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}

        return classes, class_to_idx

    def __init__(self, root, data_type, transform=None, target_transform=None):
        self.x = []
        self.y = []
        self.root = root

        self.classes, self.class_to_idx = self.find_classes(root)

        self.transform = transform
        self.target_transform = target_transform

        item_views = defaultdict(list)
        for label in os.listdir(root):
            label_dir = os.path.join(root, label)
            if os.path.isdir(label_dir) and not label.startswith("."):
                for item_file in os.listdir(os.path.join(label_dir, data_type)):
                    if not item_file.endswith("png"):
                        continue
                    c, item_id, view_number = item_file.rsplit("_", 2)
                    item_views[(label, item_id)].append(
                        os.path.join(label_dir, data_type, item_file)
                    )
        for (label, item_id), files in item_views.items():
            self.x.append(files)
            self.y.append(self.class_to_idx[label])

    # Override to give PyTorch access to any image on the dataset
    def __getitem__(self, index):
        orginal_views = self.x[index]
        views = []

        for view in orginal_views:
            im = Image.open(view)
            im = im.convert("RGB")
            if self.transform is not None:
                im = self.transform(im)
            views.append(im)

        return views, self.y[index]

    # Override to give PyTorch size of dataset
    def __len__(self):
        return len(self.x)
