import torch.optim
import math

import UNet
from DataSet import DataLoaderPNG8
from DataSet import DataLoaderPNG16
import torch.nn as nn
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


PATH = "../tmp/state_dict_model.pt"
################################################################################################################
#########################################Load Data##############################################################
################################################################################################################
def loadData(data,batch_s):
    indices = list(range(len(data)))
    split_index1=int(len(data)*0.6)
    split_index2 = int(len(data) * 0.2)+split_index1
    trainSampler = SubsetRandomSampler(indices[:split_index1])
    testSampler = SubsetRandomSampler(indices[split_index1:split_index2])
    valSampler = SubsetRandomSampler(indices[split_index2:])

    trainDataLoader=DataLoader(dataset=data,
                       batch_size=batch_s,
                       sampler=trainSampler,
                       drop_last=True)

    testDataLoader=DataLoader(dataset=data,
                       batch_size=batch_s,
                       sampler=testSampler,
                       drop_last=True)

    valDataLoader=DataLoader(dataset=data,
                       batch_size=batch_s,
                       sampler=valSampler,
                       drop_last=True)

    #trainIterator = iter(trainDataLoader)
    #data = trainIterator.next()
    #img,seg = data
    #print("Image: ",img,"\n Segmentation: ",seg)


    return trainDataLoader, testDataLoader, valDataLoader

def visualize_images(axs, source, target, result, phase, epoch, every_epoche):
    offset=0
    flair=source[0]
    t1=source[1]
    t1ce=source[2]
    t2=source[3]

    flair=flair[0, 0, ...].cpu()
    t1=t1[0,0,...].cpu()

    target=target[0,0,...].cpu()
    preds=(result.detach()[0,0,...].cpu()).float()

    axs[epoch//every_epoche][0+offset].imshow(flair)
    axs[epoch//every_epoche][0+offset].set_title(str(epoch)+': '+phase+' flair')
    axs[epoch // every_epoche][0 + offset].grid(False)
    axs[epoch//every_epoche][0+offset].imshow(t1)
    axs[epoch//every_epoche][0+offset].set_title(str(epoch)+': '+phase+' t1')
    axs[epoch // every_epoche][0 + offset].grid(False)
    axs[epoch//every_epoche][0+offset].imshow(t1ce)
    axs[epoch//every_epoche][0+offset].set_title(str(epoch)+': '+phase+' t1ce')
    axs[epoch // every_epoche][0 + offset].grid(False)
    axs[epoch//every_epoche][0+offset].imshow(t2)
    axs[epoch//every_epoche][0+offset].set_title(str(epoch)+': '+phase+' t2')
    axs[epoch // every_epoche][0 + offset].grid(False)
    axs[epoch//every_epoche][0+offset].imshow(target)
    axs[epoch//every_epoche][0+offset].set_title(str(epoch)+': '+phase+' prediction')
    axs[epoch // every_epoche][0 + offset].grid(False)
    axs[epoch//every_epoche][0+offset].imshow(preds)
    axs[epoch//every_epoche][0+offset].set_title(str(epoch)+': '+phase+' segmentation')
    axs[epoch // every_epoche][0 + offset].grid(False)
    return axs

################################################################################################################
#############################################Training and Evaluation############################################
################################################################################################################

def train(trainLoader,valLoader,n_epochs,batch_s):
    model=UNet.UNet(4,3)
    model.train()
    l1_loss=nn.L1Loss()
    losses_training=[]
    optimizer=torch.optim.Adam(model.parameters(),lr=0.001)

    progress =tqdm(range(n_epochs),desc='progress')
    for epoch in progress:
        sum_loss = 0
        for i,(patDat,seg) in enumerate(trainLoader):
            predSeg=model.forward(patDat)
            loss=l1_loss(predSeg,seg)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Save
            torch.save(model.state_dict(), PATH)
            evaluate(valLoader,model,batch_s)
            #sum_loss+=loss.item()
           # losses_training.append(sum_loss/len(dataSet))
            #if epoch%5==0:
              #  axs = visualize_images(axs,patDat,seg,predSeg,epoch,every_epoche=5)

def evaluate(valLoader,model):
    with torch.no_grad():
        model.load_state_dict(torch.load(PATH))
        model.eval()
        l1_loss=nn.L1Loss()
        val_loss = 0
        for i,(patDat,seg) in enumerate(valLoader):
            predSeg=model.forward(patDat)
            loss = l1_loss(predSeg, seg)


################################################################################################################
#############################################Test###############################################################
################################################################################################################







################################################################################################################
#############################################Run Code###########################################################
################################################################################################################

if __name__ == '__main__':
    batch_s=3
    epoche=5
    dataSet = DataLoaderPNG8()
    trainDataLoader,testDataloader,valDataLoader=loadData(dataSet,batch_s)
    train(trainDataLoader,valDataLoader,epoche,batch_s)


