import os
import time
import util
import torch
import transforms as t
import dataset58 as dataset
import model.model as model
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

def test_model(model,save_path ):

    since = time.time()

    model.eval()
    with torch.no_grad():
        start_time = time.time()
        for batch_idx, data in enumerate(dataloader):
            indexes,filenames,inputs, labels,gtsh=data
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            util.PlotHCV(save_path+"/",'val'+'_'+str(batch_idx)+'_'+os.path.basename(filenames[0])[:-4],inputs[0], labels[0], outputs[0])
            print(filenames[0])
    time_elapsed = time.time() - since
    print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

if __name__ == '__main__':
    batch_size=1
    num_class = 58

    save_path = util.setup_logging_from_args()
    writer = SummaryWriter(save_path+"/logs")

    transVal = t.Compose([
        t.ToTensorOneHot(),
    ])

    val_set = dataset.ValDataset("data/test/", transform=transVal)

    dataloader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=8,pin_memory=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    model = model.UNetWithResnet50Encoder(num_class).to(device)

    weights = torch.load('model_500.pth', map_location='cuda:0')
    model.load_state_dict(weights)

    model = test_model(model,save_path)

    print('Finish!')