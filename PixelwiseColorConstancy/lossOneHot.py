import torch
from torch.nn import functional as F

def calc_loss( pred, target,gtsh):

    gtObjId=gtsh[:, 0, :, :]
    gtMaterial = gtsh[:, 1, :, :]

    for i in  range(pred.shape[1]):
        tmp1=pred[:, i, :, :]
        tmp1=tmp1[(gtObjId <= 2)& (gtMaterial != 0)]

        tmp2=target[:, i, :, :]
        tmp2=tmp2[(gtObjId <= 2)& (gtMaterial != 0)]

        tmp3 = pred[:, i, :, :]
        tmp3 = tmp3[(gtObjId >2)& (gtMaterial != 0)]

        tmp4 = target[:, i, :, :]
        tmp4 = tmp4[(gtObjId >2)& (gtMaterial != 0)]

        if i==0:
            predObj=tmp1
            targetObj=tmp2
            predBG=tmp3
            targetBG=tmp4

        else:
            predObj=torch.cat((predObj,tmp1))
            targetObj=torch.cat((targetObj,tmp2))
            predBG=torch.cat((predBG,tmp3))
            targetBG=torch.cat((targetBG,tmp4))

    bceObjects=F.binary_cross_entropy_with_logits(predObj,targetObj,reduction ='mean')
    bceBG=F.binary_cross_entropy_with_logits(predBG,targetBG,reduction ='mean')

    m=0.85
    bca=(m*bceObjects)+((1-m)*bceBG)

    return bca,bceObjects,bceBG




