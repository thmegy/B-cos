from __future__ import print_function
import numpy as np
import os
import json
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from torchvision.datasets.folder import ImageFolder, VisionDataset
from torchvision.datasets import CIFAR10, MNIST

from data.imagenet_classnames import name_map, folder_label_map
from data.cracks_classnames import cracks_name_map, cracks_to_idx
from data.voc_classnames import voc_name_map

# Ignore imports, just importing here to consistently get all datasets from this file.
_ = CIFAR10, MNIST


class TinyImagenet(ImageFolder):

    base_folder = "tiny-imagenet-200"

    def __init__(self, root, train=True, transform=None, target_transform=None, **kwargs):
        _ = kwargs  # Just for consistency with other datasets.
        path = os.path.join(root, self.base_folder, "train" if train else "val")
        super().__init__(path, transform=transform, target_transform=target_transform)
        self.class_labels = {i: folder_label_map[folder] for i, folder in enumerate(self.classes)}


class Imagenet(ImageFolder):

    base_folder = "ILSVRC2012"
    classes = [name_map[i] for i in range(1000)]
    name_map = name_map

    def __init__(self, root, train=True, transform=None, target_transform=None, class_idcs=None,
                 **kwargs):
        _ = kwargs  # Just for consistency with other datasets.
        path = os.path.join(root, self.base_folder, "train" if train else "val")
        super().__init__(path, transform=transform, target_transform=target_transform)
        if class_idcs is not None:
            class_idcs = list(sorted(class_idcs))
            tgt_to_tgt_map = {c: i for i, c in enumerate(class_idcs)}
            self.classes = [self.classes[c] for c in class_idcs]
            self.samples = [(p, tgt_to_tgt_map[t]) for p, t in self.samples if t in tgt_to_tgt_map]
            self.class_to_idx = {k: tgt_to_tgt_map[v] for k, v in self.class_to_idx.items() if v in tgt_to_tgt_map}

        self.class_labels = {i: folder_label_map[folder] for i, folder in enumerate(self.classes)}
        self.targets = np.array(self.samples)[:, 1]


class Cracks(ImageFolder):

    base_folder = "cracks"
    classes = [cracks_name_map[i] for i in range(12)]
    name_map = cracks_name_map

    def __init__(self, root, train=True, transform=None, target_transform=None, class_idcs=None,
                 **kwargs):
        _ = kwargs  # Just for consistency with other datasets.
        path = os.path.join(root, self.base_folder, "train" if train else "val")
        super().__init__(path, transform=transform, target_transform=target_transform)
        if class_idcs is not None:
            class_idcs = list(sorted(class_idcs))
            tgt_to_tgt_map = {c: i for i, c in enumerate(class_idcs)}
            self.classes = [self.classes[c] for c in class_idcs]
            self.samples = [(p, tgt_to_tgt_map[t]) for p, t in self.samples if t in tgt_to_tgt_map]
            self.class_to_idx = {k: tgt_to_tgt_map[v] for k, v in self.class_to_idx.items() if v in tgt_to_tgt_map}

#        self.class_labels = {i: folder_label_map[folder] for i, folder in enumerate(self.classes)}
        self.targets = np.array(self.samples)[:, 1]
        

class CracksFull(VisionDataset):
    def __init__(self, root, train=True, transform=None, target_transform=None, class_idcs=None,
                 **kwargs):

        path = os.path.join('/home/theo/workdir/mmdet/data/cracks_12_classes/cracks_annotations_sorted_by_mission/', "train" if train else "val")
        super().__init__(path, transform=transform, target_transform=None)

        self.class_to_idx = cracks_to_idx
        self.classes = list(self.class_to_idx.keys())

        if train:
            ann_path = '/home/theo/workdir/mmcls/data/cracks/from_detection/cracks_train.json'
        else:
            ann_path = '/home/theo/workdir/mmcls/data/cracks/from_detection/cracks_val.json'
        with open(ann_path, 'r') as f_in:
            ann_dict = json.load(f_in)

        samples = [(f'{path}/{k}', self.get_label_idx(v)) for k,v in ann_dict.items()]

        self.loader = self.pil_loader

        self.samples = samples
        self.imgs = self.samples
        self.targets = [s[1] for s in samples]
        
    def __len__(self):
        return len(self.samples)
   
    def __getitem__(self,index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target

    def pil_loader(self, path: str) -> Image.Image:
        # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        with open(path, "rb") as f:
            img = Image.open(f)
            img.load()
        return img.convert("RGB")

    def get_label_idx(self, ann_list):
        if ann_list[0] == 'Background':
            gt_label = np.zeros(len(self.classes))
        else:
            labels = []
            for ann in set(ann_list):
                label = self.class_to_idx[ann]
                labels.append(label)

            gt_label = np.zeros(len(self.classes))
            gt_label[labels] = 1

        return gt_label
        
    
        
class voc(ImageFolder):

    base_folder = "voc"
    classes = [voc_name_map[i] for i in range(20)]
    name_map = voc_name_map

    def __init__(self, root, train=True, transform=None, target_transform=None, class_idcs=None,
                 **kwargs):
        _ = kwargs  # Just for consistency with other datasets.
        path = os.path.join(root, self.base_folder, "train" if train else "val")
        super().__init__(path, transform=transform, target_transform=target_transform)
        if class_idcs is not None:
            class_idcs = list(sorted(class_idcs))
            tgt_to_tgt_map = {c: i for i, c in enumerate(class_idcs)}
            self.classes = [self.classes[c] for c in class_idcs]
            self.samples = [(p, tgt_to_tgt_map[t]) for p, t in self.samples if t in tgt_to_tgt_map]
            self.class_to_idx = {k: tgt_to_tgt_map[v] for k, v in self.class_to_idx.items() if v in tgt_to_tgt_map}

#        self.class_labels = {i: folder_label_map[folder] for i, folder in enumerate(self.classes)}
        self.targets = np.array(self.samples)[:, 1]
        
        
