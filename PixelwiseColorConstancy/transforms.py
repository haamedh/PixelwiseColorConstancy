import torch
import random
import numpy as np
from skimage.transform import resize
from torch.nn.functional import one_hot
from torchvision.transforms import functional as F


class ToTensorOneHot(object):
    def __call__(self, image, target, sh):
        image = F.to_tensor(image)
        target = F.to_tensor(target)
        sh = F.to_tensor(sh)

        mat = one_hot(sh[2, :, :].to(torch.int64), 4)

        # V=one_hot(torch.round(target[0, :, :] * 2).to(torch.int64))
        V0 = torch.round(target[0, :, :]).to(torch.int64)
        V0 = torch.where(V0 < 1, 1, V0)
        V0 = torch.where(V0 > 9, 9, V0)
        V = one_hot(V0, 10)[:, :, 1:]

        # H=one_hot(target[1, :, :].to(torch.int64), 81)[:, :, 1:]
        H = one_hot(torch.ceil(target[1, :, :] / 2).to(torch.int64), 41)[:, :, 1:]
        C = one_hot((target[2, :, :] / 2).to(torch.int64), 9)
        target = torch.cat((V, H, C), 2).permute(2, 0, 1).to(torch.float32)

        return image, target, sh


class RandomCrop(object):
    """
    Performs a random crop in a given numpy array using only the first two dimensions (width and height)
    """

    def __init__(self, prob=0.5):

        self.prob = prob

    @staticmethod
    def get_params(pic, output_size):

        # read dimensions (width, height, channels)
        w, h, c = pic.shape

        # read crop size
        th, tw = output_size

        # get crop indexes
        i = random.randint(0, w - tw)
        j = random.randint(0, h - th)

        return i, j, th, tw, h, w

    def __call__(self, image, target, sh):
        """
        :param input: numpy array
        :return: numpy array croped using self.size
        """
        size = random.randint(200, 400)
        if random.random() < self.prob:
            # get crop params: starting pixels and size of the crop
            i, j, th, tw, h, w = self.get_params(image, (size, size))

            image_out = np.zeros((h, w, image.shape[2])).astype('float32')
            target_out = np.zeros((h, w, target.shape[2])).astype('float32')
            sh_out = np.zeros((h, w, sh.shape[2])).astype('float32')

            # perform cropping and return the new image
            image = image[i:i + th, j:j + tw, :]
            target = target[i:i + th, j:j + tw, :]
            sh = sh[i:i + th, j:j + tw, :]

            for i in range(image.shape[2]):
                image_out[:, :, i] = resize(image[:, :, i], (h, w), anti_aliasing=True)
            for i in range(target.shape[2]):
                target_out[:, :, i] = resize(target[:, :, i], (h, w), anti_aliasing=True)
            for i in range(sh.shape[2]):
                sh_out[:, :, i] = resize(sh[:, :, i], (h, w), anti_aliasing=True)
            return image_out, target_out, sh_out
        else:
            return image, target, sh


class RandomHorizontalFlip(object):
    """Horizontally flip the given numpy array randomly with a probability of 0.5."""

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, pic, pic2, pic3):
        """
        Args:
            img (numpy array): Image to be flipped.
        Returns:
            numpy array: Randomly flipped image.
        """
        if random.random() < self.prob:
            pic = np.array(pic[:, ::-1, :])
            pic2 = np.array(pic2[:, ::-1, :])
            pic3 = np.array(pic3[:, ::-1, :])
        return pic, pic2, pic3


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target, sh):
        for t in self.transforms:
            image, target, sh = t(image, target, sh)
        return image, target, sh
