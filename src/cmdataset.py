# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from logging import getLogger

from PIL import ImageFilter, Image
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.utils.data as data
import scipy.io as sio
import os
import h5py

logger = getLogger()

class Sampler():
    def __init__(self, root, paths):
        self.root = root
        if isinstance(paths, np.ndarray):
            if len(paths.shape) == 1 or paths.shape[0] == 1 or paths.shape[1] == 1:
                paths = paths.reshape([-1]).tolist()
        self.paths = paths

    def __getitem__(self, item):
        path = self.paths[item]
        if isinstance(path, np.ndarray):
            if len(path.shape) >= 2:
                return Image.fromarray(path, mode='RGB')
            else:
                path = path[0]
        return Image.open(os.path.join(self.root, path))

    def __len__(self):
        return len(self.paths)

def text_transform(text):
    return text

class CMDataset(data.Dataset):
    """
    Load precomputed captions and image features
    Possible options: f30k_precomp, coco_precomp
    """

    def __init__(
        self,
        data_name,
        return_index=False,
        partition='train'
    ):
        self.data_name = data_name
        self.partition = partition
        training = 'train' in partition.lower()
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        trans = []
        if training:
            trans.extend([transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomResizedCrop(224),
                    # transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean, std=std)])
                ])
        else:
            trans.extend([transforms.Compose([
                    # transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean, std=std)])
                ])
        self.trans = trans
        self.return_index = return_index
        self.open_data()

    def open_data(self):
        # mirflickr25k mirflickr25k_fea MSCOCO_fea nus_wide_tc10_fea IAPR-TC12_fea
        if self.data_name.lower() == 'mirflickr25k':
            data = MIRFlickr25K(self.partition)
        elif self.data_name.lower() == 'nus_wide_tc10':
            data = NUSWIDE(self.partition)
        elif self.data_name.lower() == 'mirflickr25k_fea':
            data = MIRFlickr25K_fea(self.partition)
        elif self.data_name.lower() == 'iapr_fea':
            data = IAPR_fea(self.partition)
        elif self.data_name.lower() == 'nus_wide_tc10_fea':
            data = NUSWIDE_fea(self.partition)
        elif self.data_name.lower() == 'mscoco_fea':
            data = MSCOCO_fea(self.partition)

        if len(data) == 3:
            (self.imgs, self.texts, self.labels) = data
            self.imgs = self.imgs
        else:
            (self.imgs, self.texts, self.labels, root) = data
            self.imgs = Sampler(root, self.imgs)
        self.length = self.labels.shape[0]
        self.text_dim = self.texts.shape[1]

    def __getitem__(self, index):
        image = self.imgs[index]
        text = self.texts[index]
        if isinstance(self.imgs, Sampler):
            multi_crops = list(map(lambda trans: trans(image), self.trans))
            text = list(map(lambda trans: trans(text), [text_transform] * len(self.trans)))
        else:
            multi_crops = [image]
            text = [text]

        label = self.labels[index]

        if self.return_index:
            return index, multi_crops, text, label
        return multi_crops, text, label
        # return multi_crops, text, index

    def __len__(self):
        return self.length

def MIRFlickr25K(partition):
    import h5py
    imgs = h5py.File('./data/MIRFLICKR25K/mirflickr25k-iall.mat', mode='r')['IAll'][()]
    root = './data/MIRFLICKR25K/'
    tags = sio.loadmat('./data/MIRFLICKR25K/mirflickr25k-yall.mat')['YAll']
    labels = sio.loadmat('./data/MIRFLICKR25K/mirflickr25k-lall.mat')['LAll']

    inx = np.arange(imgs.shape[0])
    np.random.shuffle(inx)
    imgs, tags, labels = imgs[inx], tags[inx], labels[inx]
    test_size = 2000
    if 'test' in partition.lower():
        imgs, tags, labels = imgs[-test_size::], tags[-test_size::], labels[-test_size::]
    else:
        imgs, tags, labels = imgs[0: -test_size], tags[0: -test_size], labels[0: -test_size]

    return imgs.transpose([0, 3, 2, 1]), tags, labels, root


def NUSWIDE(partition):
    imgs = h5py.File('./data/NUS-WIDE-TC10/nus-wide-tc10-iall.mat')['IAll'][()]
    root = './data/NUS-WIDE-TC10/'
    tags = sio.loadmat('./data/NUS-WIDE-TC10/nus-wide-tc10-yall.mat')['YAll']
    labels = sio.loadmat('./data/NUS-WIDE-TC10/nus-wide-tc10-lall.mat')['LAll']

    inx = np.arange(imgs.shape[0])
    np.random.shuffle(inx)
    imgs, tags, labels = imgs[inx], tags[inx], labels[inx]
    test_size = 2100
    if 'test' in partition.lower():
        imgs, tags, labels = imgs[-test_size::], tags[-test_size::], labels[-test_size::]
    else:
        imgs, tags, labels = imgs[0: -test_size], tags[0: -test_size], labels[0: -test_size]

    return imgs.transpose([0, 3, 2, 1]), tags, labels, root

def MIRFlickr25K_fea(partition):
    root = '../../DeepMDA/datasets/MIRFLICKR25K/'
    data_img = sio.loadmat(os.path.join(root, 'mirflickr25k-iall-vgg-rand.mat'))['XAll']
    data_txt = sio.loadmat(os.path.join(root, 'mirflickr25k-yall-rand.mat'))['YAll']
    labels = sio.loadmat(os.path.join(root, 'mirflickr25k-lall-rand.mat'))['LAll']

    test_size = 2000
    if 'test' in partition.lower():
        data_img, data_txt, labels = data_img[-test_size::], data_txt[-test_size::], labels[-test_size::]
    else:
        data_img, data_txt, labels = data_img[0: -test_size], data_txt[0: -test_size], labels[0: -test_size]

    return data_img, data_txt, labels

def IAPR_fea(partition):
    root = '../../DeepMDA/datasets/IAPR-TC12/'
    file_path = os.path.join(root, 'iapr-tc12-rand.mat')
    data = sio.loadmat(file_path)

    valid_img = data['VDatabase'].astype('float32')
    valid_txt = data['YDatabase'].astype('float32')
    valid_labels = data['databaseL']

    test_img = data['VTest'].astype('float32')
    test_txt = data['YTest'].astype('float32')
    test_labels = data['testL']

    data_img, data_txt, labels = np.concatenate([valid_img, test_img]), np.concatenate([valid_txt, test_txt]), np.concatenate([valid_labels, test_labels])

    test_size = 2000
    if 'test' in partition.lower():
        data_img, data_txt, labels = data_img[-test_size::], data_txt[-test_size::], labels[-test_size::]
    else:
        data_img, data_txt, labels = data_img[0: -test_size], data_txt[0: -test_size], labels[0: -test_size]

    return data_img, data_txt, labels

def NUSWIDE_fea(partition):
    root = '../../DeepMDA/datasets/NUS-WIDE-TC10/'
    test_size = 2100
    data_img = sio.loadmat(root + 'nus-wide-tc10-xall-vgg.mat')['XAll']
    data_txt = h5py.File(root + 'nus-wide-tc10-yall.mat')['YAll'][()].T
    labels = sio.loadmat(root + 'nus-wide-tc10-lall.mat')['LAll']

    test_size = 2100
    if 'test' in partition.lower():
        data_img, data_txt, labels = data_img[-test_size::], data_txt[-test_size::], labels[-test_size::]
    else:
        data_img, data_txt, labels = data_img[0: -test_size], data_txt[0: -test_size], labels[0: -test_size]
    return data_img, data_txt, labels

def MSCOCO_fea(partition):
    root = '../../DeepMDA/datasets/MSCOCO/'
    path = root + 'MSCOCO_deep_doc2vec_data.h5py'
    data = h5py.File(path)
    data_img = np.concatenate([data['train_imgs_deep'][()], data['test_imgs_deep'][()]], axis=0)
    data_txt = np.concatenate([data['train_text'][()], data['test_text'][()]], axis=0)
    labels = np.concatenate([data['train_imgs_labels'][()], data['test_imgs_labels'][()]], axis=0)

    test_size = 5000
    if 'test' in partition.lower():
        data_img, data_txt, labels = data_img[-test_size::], data_txt[-test_size::], labels[-test_size::]
    else:
        data_img, data_txt, labels = data_img[0: -test_size], data_txt[0: -test_size], labels[0: -test_size]

    return data_img, data_txt, labels