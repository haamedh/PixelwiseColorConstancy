import os
import time
import util
import torch
import random
import logging
import numpy as np
import collections
import dataset58 as dataset
import model.model as model
import torch.optim as optim
from lossOneHot import calc_loss
from collections import defaultdict
from skimage.transform import resize
from torch.optim import lr_scheduler
from torch.nn.functional import one_hot
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import functional as F

class ToTensorOneHot(object):
    def __call__(self, image, target,sh):
        image = F.to_tensor(image)
        target = F.to_tensor(target)
        sh = F.to_tensor(sh)
        
        mat=one_hot(sh[2,:,:].to(torch.int64),4)

        # V=one_hot(torch.round(target[0, :, :] * 2).to(torch.int64))
        V0 = torch.round(target[0, :, :]).to(torch.int64)
        V0=torch.where(V0 < 1, 1, V0)
        V0 = torch.where(V0 > 9, 9, V0)
        V=one_hot(V0,10)[:, :, 1:]

        # H=one_hot(target[1, :, :].to(torch.int64), 81)[:, :, 1:]
        H=one_hot(torch.ceil(target[1, :, :]/2).to(torch.int64), 41)[:, :, 1:]
        C=one_hot((target[2, :, :]/2).to(torch.int64),9)
        target=torch.cat((V, H, C), 2).permute(2, 0, 1).to(torch.float32)
  
        return image, target,sh

class Scale(object):
    """
    Rescale the given numpy image to a specified size.
    """

    def __init__(self, size, interpolation="bilinear"):
        assert isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, pic,target):

        img_out = np.zeros((self.size[0], self.size[1], pic.shape[2]))

        for i in range(pic.shape[2]):
            img_out[:, :, i] = resize(pic[:, :, i], self.size, anti_aliasing=True)

        return img_out

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

        return i, j, th, tw,h,w

    def __call__(self, image,target,sh):
        """
        :param input: numpy array
        :return: numpy array croped using self.size
        """
        size=random.randint(200,400)
        if random.random() < self.prob:
            # get crop params: starting pixels and size of the crop
            i, j, th, tw ,h,w= self.get_params(image, (size,size))

            image_out = np.zeros((h, w, image.shape[2])).astype('float32')
            target_out = np.zeros((h, w, target.shape[2])).astype('float32')
            sh_out=np.zeros((h, w, sh.shape[2])).astype('float32')

            # perform cropping and return the new image
            image=image[i:i + th, j:j + tw, :]
            target = target[i:i + th, j:j + tw, :]
            sh = sh[i:i + th, j:j + tw, :]

            for i in range(image.shape[2]):
                image_out[:, :, i] = resize(image[:, :, i], (h,w), anti_aliasing=True)
            for i in range(target.shape[2]):
                target_out[:, :, i] = resize(target[:, :, i], (h,w), anti_aliasing=True)
            for i in range(sh.shape[2]):
                sh_out[:, :, i] = resize(sh[:, :, i], (h,w), anti_aliasing=True)
            return image_out,target_out,sh_out
        else:
            return image, target,sh

class RandomHorizontalFlip(object):
    """Horizontally flip the given numpy array randomly with a probability of 0.5."""

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, pic,pic2,pic3):
        """
        Args:
            img (numpy array): Image to be flipped.
        Returns:
            numpy array: Randomly flipped image.
        """
        if random.random() < self.prob:
            pic= np.array(pic[:, ::-1, :])
            pic2= np.array(pic2[:, ::-1, :])
            pic3 = np.array(pic3[:, ::-1, :])
        return pic,pic2,pic3

class RandomVerticalFlip(object):
    """Vertically flip the given numpy array randomly with a probability of 0.5 by default."""

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target,sh):
        """
        Args:
            img (numpy array): Image to be flipped.
        Returns:
            numpy array: Randomly flipped image.
        """
        if random.random() < self.prob:
            image = np.array(image[::-1, :, :])
            target = np.array(target[::-1, :, :])
            sh = np.array(sh[::-1, :, :])
        return image,target,sh

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target,sh):
        for t in self.transforms:
            image, target,sh = t(image, target,sh)
        return image, target,sh

def train_model(model, optimizer, scheduler,writer,save_path, num_epochs=25):

    log_interval = 1

    for epoch in range(1,num_epochs):
        print('-' * 10)
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))

        since = time.time()

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            scaler = torch.cuda.amp.GradScaler()

            if phase == 'train':
                for param_group in optimizer.param_groups:
                    print("LR", param_group['lr'])

                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            localsMetrics = defaultdict(float)
            metrics = defaultdict(float)

            epoch_samples = 0
            input_samples = 0
            max_len = len(dataloaders[phase])
            dataLen=len(dataloaders[phase].dataset)

            start_time = time.time()
            for batch_idx, data in enumerate(dataloaders[phase]):
                indexes,filenames,inputs, labels,gtsh=data


                inputs = inputs.to(device)
                labels = labels.to(device)
                gtsh=gtsh.to(device)
                input_samples += inputs.size(0)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    with torch.cuda.amp.autocast():
                        outputs = model(inputs)

                        loss,bcaObjects,bcaBG  = calc_loss(outputs, labels,gtsh)
                        metrics['loss'] += loss.data.cpu().numpy() * labels.size(0)
                        localsMetrics['loss'] += loss.data.cpu().numpy() * labels.size(0)
                        localsMetrics['bcaObjects'] += bcaObjects.data.cpu().numpy() * labels.size(0)
                        localsMetrics['bcaBG'] += bcaBG.data.cpu().numpy() * labels.size(0)

                    if phase == 'train':
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                        # loss.backward()
                        # optimizer.step()
                    if batch_idx % log_interval == log_interval - 1:
                        # loss_string = 'loss={bce:.6f}'.format(bce=localsMetrics['loss'] / log_interval / batch_size)
                        loss_string = 'loss={mse:.4f} Object Loss={mse1:.4f}   Background Loss={mse2:.4f}'\
                            .format(mse=localsMetrics['loss'] / log_interval / batch_size,mse1=localsMetrics['bcaObjects'] / log_interval / batch_size ,mse2=localsMetrics['bcaBG'] / log_interval / batch_size)

                        if phase == 'train':
                            logging.info('Train Epoch: {epoch} [{batch:5d}/{total_batch} ({percent:2d}%)]   time: {time:3.2f}   {loss}   '
                                  .format(epoch=epoch, batch=(batch_idx + 1) * len(inputs), total_batch=max_len * len(inputs), percent=int(100. * batch_idx / max_len), time=time.time() - start_time, loss=loss_string))
                            writer.add_scalars('Loss Train_batch', {'MSE Train': localsMetrics['loss'] / input_samples}, (epoch * dataLen) + ((batch_idx ) * batch_size))
                        else:
                            logging.info('Val Epoch: {epoch} [{batch:5d}/{total_batch} ({percent:2d}%)]   time: {time:3.2f}   {loss}   '
                                  .format(epoch=epoch, batch=(batch_idx + 1) * len(inputs), total_batch=max_len * len(inputs), percent=int(100. * batch_idx / max_len), time=time.time() - start_time, loss=loss_string))
                            writer.add_scalars('Loss Test_batch', {'MSE Val': localsMetrics['loss'] / input_samples}, (epoch * dataLen) + ((batch_idx ) * batch_size))
                        localsMetrics = defaultdict(float)
                        start_time = time.time()
                        input_samples = 0

                epoch_samples += inputs.size(0)
                if (batch_idx==1 or batch_idx==(max_len -1) or batch_idx==50  or batch_idx==100 ):
                    util.PlotHCV(save_path+"/",phase+'_'+str(epoch)+'_'+str(batch_idx)+'_'+os.path.basename(filenames[0])[:-4],inputs[0], labels[0], outputs[0])

            if phase == 'train':
                scheduler.step()
                writer.add_scalars('Loss', {'MSE Train': metrics['loss'] / epoch_samples}, epoch)

                if epoch % 10==0:
                    util.save_checkpoint(model,optimizer,scheduler, epoch, save_path)
            else:
                writer.add_scalars('Loss', {'MSE Val': metrics['loss'] / epoch_samples}, epoch)

            outputs=[]
            for k in metrics.keys():
                outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))
            logging.info("_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_    {}: {}".format(phase, ", ".join(outputs)))

        time_elapsed = time.time() - since
        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    return model
#-----------------------------

if __name__ == '__main__':

    batch_size=1
    num_class = 58
    lr=4e-4
    step_size=40
    gamma = 0.5

    save_path = util.setup_logging_from_args()
    writer = SummaryWriter(save_path+"/logs")


    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('')
    logging.basicConfig(level = logging.INFO)

    # use same transform for train/val for this example
    trans = Compose([
        RandomCrop(prob=0.5),
        RandomHorizontalFlip(prob=0.5),
        ToTensorOneHot(),
    ])

    transVal = Compose([
        ToTensorOneHot(),
    ])


    train_set = dataset.TrainDataset("data/train/", transform=trans)
    val_set = dataset.ValDataset("data/test/", transform=transVal)

    image_datasets = {
        'train': train_set, 'val': val_set
    }



    dataloaders = {
        'train': DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8,pin_memory=True),
        'val': DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=8,pin_memory=True)
    }

    dataset_sizes = {
        x: len(image_datasets[x]) for x in image_datasets.keys()
    }


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)



    model = model.UNetWithResnet50Encoder(num_class).to(device)

    modelname='UNetWithResnet50Encoder'
    logging.info('ModelName:  '+modelname)
    logging.info('First Learning Rate:  ' + str(lr))
    logging.info('step_size:  ' + str(step_size))
    logging.info('Gamma:  ' + str(gamma))
    logging.info('Number of class:  ' + str(num_class))

    # Observe that all parameters are being optimized
    optimizer_ft = optim.Adam(model.parameters(), lr=lr)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=step_size, gamma=gamma)
    model = train_model(model, optimizer_ft, exp_lr_scheduler,writer,save_path, num_epochs=501,)
    print('Finish!')