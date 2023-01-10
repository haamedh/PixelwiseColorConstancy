import os,sys
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch
from torchvision.transforms import functional as F

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        # img=img.resize((192, 192))
        return img.convert('RGB')

class TrainDataset(Dataset):
    def __init__(self, path, transform=None):

        self.img_dir = path+"xyz"
        self.gt_dir = path+'gt'
        self.gtsh_dir = path + 'gtsh'
        print(self.gt_dir)
        list1 = []

        illu=np.array([0,5,6,7,8,13,14,15,16,17,18])
        # illu=np.array([0])
        scene=np.array([1,2,6,7])

        for path, subdirs, files in os.walk(self.img_dir):
            for name in files:
                if (int(name[5:7]) in illu) and (int(name[8]) in scene): # and ((os.path.dirname(path).split('/')[-1][-1]) in scene):
                     list1.append(os.path.join(path, name))
        list1 = sorted(list1)              
        listgt = []
        for path, subdirs, files in os.walk(self.gt_dir):
            for name in files:
                if (int(name[5:7]) in illu) and (int(name[8]) in scene): # and ((os.path.dirname(path).split('/')[-1][-1]) in scene):
                    listgt.append(os.path.join(path, name))
        listgt = sorted(listgt)



        self.inputs = list1
        self.targets = listgt


        self.transform = transform
        self.data_loader = pil_loader

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        # image = self.data_loader(self.inputs[index])
        # gt = self.data_loader(self.targets[index])


        image=np.load(self.inputs[index])['xyz']
        gt = np.load(self.targets[index])['colorLabel']
        gtsh = np.load(self.targets[index])['objectLabel']
        material=np.load(self.targets[index])['materialLabel']
        material[:151, :202]=0
        gtsh=np.dstack((gtsh,gt[:,:,0],material))



        value=np.round(gt[:,:,0])
        hue=np.round(gt[:,:,1])
        chroma=np.round(gt[:,:,2])/2


        gt=np.where(gt==-1, np.nan,gt)

        # gt[:, :, 0] = gt[:, :, 0] / 9.5
        # gt[:, :, 1] = gt[:, :, 1] / 80
        # gt[:, :, 2] = (gt[:, :, 2]) / 16

        if self.transform:
            image,gt,gtsh= self.transform(image,gt,gtsh)
            # gt= self.transform(gt)
        return index,self.inputs[index],image, gt,gtsh

class ValDataset(Dataset):
    def __init__(self, path, transform=None):

        self.img_dir = path+"xyz"
        self.gt_dir = path+'gt'
        print(self.gt_dir)
        list1 = []

        illu=np.array([0,5,6,7,8,13,14,15,16,17,18,19,20])
        # illu=np.array([0])
        scene=np.array([1,6,7,8,9])

        for path, subdirs, files in os.walk(self.img_dir):
            for name in files:
                # if (int(name[6:8]) in illu) and (int(name[9]) in scene):
                if (int(name[5:7]) in illu) and (int(name[8]) in scene):
                     list1.append(os.path.join(path, name))
        list1 = sorted(list1)
        listgt = []
        for path, subdirs, files in os.walk(self.gt_dir):
            for name in files:
                # if (int(name[6:8]) in illu) and (int(name[9]) in scene):
                if (int(name[5:7]) in illu) and (int(name[8]) in scene):
                    listgt.append(os.path.join(path, name))
        listgt = sorted(listgt)

        self.inputs = list1
        self.targets = listgt

        self.transform = transform
        self.data_loader = pil_loader

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        # image = self.data_loader(self.inputs[index])
        # gt = self.data_loader(self.targets[index])


        image=np.load(self.inputs[index])['xyz']
        gt = np.load(self.targets[index])['colorLabel']
        gtsh = np.load(self.targets[index])['objectLabel']
        material=np.load(self.targets[index])['materialLabel']
        material[:151, :202]=0
        gtsh=np.dstack((gtsh,gtsh,material))


        gt=np.where(gt==-1, np.nan,gt)

        # gt[:, :, 0] = gt[:, :, 0] / 9.5
        # gt[:, :, 1] = gt[:, :, 1] / 80
        # gt[:, :, 2] = (gt[:, :, 2]) / 16

        if self.transform:
            image,gt,gtsh = self.transform(image,gt,gtsh)
            # gt= self.transform(gt)
        return index,self.inputs[index],image, gt,gtsh



class TestDataset(Dataset):
    def __init__(self, path, transform=None):

        # self.img_dir = r"D:\NewDS\final\train\xyz"
        # self.gt_dir = r'D:\NewDS\final\train\gt'
        # self.img_dir = r"D:\NewDS\tt\xyz"
        # self.gt_dir = r'D:\NewDS\tt\gt1'
        self.img_dir = path+"xyz"
        self.gt_dir = path+'gtfull'
        self.gtl_dir = path+'gtl'
        list1 = []
        print(self.img_dir)
        for path, subdirs, files in os.walk(self.img_dir):
            for name in files:
                list1.append(os.path.join(path, name))
        list1 = sorted(list1)
        print(len(list1))
        listgt = []
        for path, subdirs, files in os.walk(self.gt_dir):
            for name in files:
                listgt.append(os.path.join(path, name))
        listgt = sorted(listgt)

        listgtl=[]
        for path, subdirs, files in os.walk(self.gtl_dir):
            for name in files:
                listgtl.append(os.path.join(path, name))
        listgtl = sorted(listgtl)

        self.inputs = list1
        self.targets = listgt
        self.targetsl = listgtl

        self.transform = transform
        self.data_loader = pil_loader

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        # image = self.data_loader(self.inputs[index])
        # gt = self.data_loader(self.targets[index])

        image=np.load(self.inputs[index])['xyz']
        gt = np.load(self.targets[index])['gtfull']
        colorLabel = np.load(self.targetsl[index])['colorLabel']


        gt[:, :, 0] = gt[:, :, 0] / 9
        gt[:, :, 1] = gt[:, :, 1] / 79
        gt[:, :, 2] = (gt[:, :, 2]) / 8

        # tmp = np.empty_like(gt)
        # tmp[:]=gt
        # gt = np.where(gt < 0, 0, gt)
        # tmp = np.where(tmp < 0, -1, tmp)



        if self.transform:
            image = self.transform(image)
            gt= self.transform(gt)
        #     tmp=self.transform(tmp)
        # gt[tmp==-1]=-1
        return index,self.inputs[index],image, gt,colorLabel