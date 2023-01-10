import os
import csv
import time
import torch
import shutil
import numpy as np
import logging.config
from skimage import io
from datetime import datetime
from skimage.color import rgb2xyz, xyz2rgb


def setup_logging_from_args():
    """
    Calls setup_logging, exports args and creates a ResultsLog class.
    Can resume training/logging if args.resume is set
    """

    # def set_args_default(field_name, value):
    #     if hasattr(args, field_name):
    #         return eval('args.' + field_name)
    #     else:
    #         return value

    # Set default args in case they don't exist in args
    # resume = set_args_default('resume', False)
    # save_name = set_args_default('save_name', '')
    # results_dir = set_args_default('results_dir', './results')

    resume = False
    save_name = ''
    results_dir =  './results'


    if save_name is '':
        save_name = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    save_path = os.path.join(results_dir, save_name)
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    os.makedirs(save_path, exist_ok=True)
    log_file = os.path.join(save_path, 'log.txt')

    setup_logging(log_file, resume)
    # export_args(args, save_path)
    return save_path

def setup_logging(log_file='log.txt', resume=False):
    """
    Setup logging configuration
    """
    if os.path.isfile(log_file) and resume:
        file_mode = 'a'
    else:
        file_mode = 'w'

    root_logger = logging.getLogger()
    if root_logger.handlers:
        root_logger.removeHandler(root_logger.handlers[0])
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S",
                        filename=log_file,
                        filemode=file_mode)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

def save_checkpoint(model,optimizer,scheduler, epoch, save_path):
    os.makedirs(os.path.join(save_path, 'checkpoints'), exist_ok=True)
    checkpoint_path = os.path.join(save_path, 'checkpoints',
                                   f'model_{epoch}.pth')
    checkpoint = {
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler:': scheduler.state_dict()
    }

    torch.save(checkpoint, checkpoint_path)

def reverse_transformHCV(inp):
    inp = inp.data.cpu().numpy().transpose((1, 2, 0))
    return inp

def reverse_transformRGB(inp):
    inp = inp.data.cpu().numpy().transpose((1, 2, 0))
    inp = np.clip(inp, 0, 1)
    return inp

def getcolor():
    color = []
    with open('color.csv', newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            color.append((row[0],row[8],row[9],row[10],row[17],row[18],row[19]))
    return np.array(color).astype('float32')

def RGB2HCV(img):
    color = getcolor()
    rgbn=np.zeros_like(img)
    hcv=np.zeros_like(img).astype('float32')
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            diffArray=np.sqrt( ((color[:,1]-img[x,y,0])**2) +  ((color[:,2]-img[x,y,1])**2)  +  ((color[:,3]-img[x,y,2])**2) )
            idx=np.array(diffArray).argmin()
            rgbn[x,y,:]=color[idx,1:4]
            hcv[x, y, :] = color[idx, 4:7]
    return hcv,rgbn

def HCV2RGB(img):
    color = getcolor()
    rgbn=np.zeros_like(img).astype('uint8')
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            if (img[x, y, :]==[0,0,0]).all():
                rgbn[x, y, :]=(0,0,0)
            else:
                diffArray=(np.abs(color[:,4]-img[x,y,0])/9.5) +  (np.abs(color[:,5]-img[x,y,1])/80)  +  (np.abs(color[:,6]-img[x,y,2])/16)
                idx=np.array(diffArray).argmin()
                rgbn[x,y,:]=color[idx,1:4]
    return rgbn

def XYZ2RGB(img):
    return np.round(xyz2rgb(img)*255).astype('int')

def Xyz2srgb(XYZ_X, XYZ_Y, XYZ_Z):
    Pow = (1.0 / 2.4)
    Less = 0.0031308

    X = (XYZ_X/ 1)
    Y = (XYZ_Y / 1)
    Z = (XYZ_Z / 1)

    R = ((X * 3.24071) + (Y * -1.53726) + (Z * -0.498571))
    G = ((X * -0.969258) + (Y * 1.87599) + (Z * 0.0415557))
    B = ((X * 0.0556352) + (Y * -0.203996) + (Z * 1.05707))

    if R > Less:
        R = ((1.055 * (R ** Pow)) - 0.055)
    else:
        R *= 12.92

    if G > Less:
        G = ((1.055 * (G ** Pow)) - 0.055)
    else:
        G *= 12.92

    if B > Less:
        B = ((1.055 * (B ** Pow)) - 0.055)
    else:
        B *= 12.92

    return (R, G, B)

def PlotHCV(path,fn,input,label,out):

    v1=torch.argmax(label[:9], dim=0)+1
    h1=(torch.argmax(label[9:49], dim=0)+1)*2
    c1 = torch.argmax(label[49:58], dim=0)*2
    label0=torch.stack((v1, h1, c1))

    v2=torch.argmax(out[:9], dim=0)+1
    h2=(torch.argmax(out[9:49], dim=0)+1)*2
    c2 = torch.argmax(out[49:58], dim=0)*2
    out0=torch.stack((v2, h2, c2))

    input1=reverse_transformRGB(input)
    label1 = reverse_transformHCV(label0)
    out1 = reverse_transformHCV(out0)

    # nrgb=XYZ2RGB(input1).astype('uint8')
    nrgb =np.array([Xyz2srgb(input1[i,j,0],input1[i,j,1],input1[i,j,2]) for i in  range(448) for j in range(448)]).reshape((448,448,3))

    nrgb=np.where(nrgb>1,1,nrgb)
    nrgb = np.where(nrgb < 0, 0, nrgb)
    nrgb=np.round(nrgb * 255).astype(np.uint8)

    label2 = HCV2RGB(label1).astype('uint8')
    out2 = HCV2RGB(out1).astype('uint8')
    together = np.concatenate([nrgb, label2,out2], axis=1)

    io.imsave(path+fn+'_'+str(time.time())+ '.png',together)


    # plt.imshow(together)
    # plt.show()
    # print(1)
