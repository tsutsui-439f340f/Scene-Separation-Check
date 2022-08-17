from torchvision import models
import time
import os
import random
import numpy as np
import torch.nn.functional as F
from torchvision import datasets, transforms
from DataSet import BatchDataset
import torch
import torch.nn  as nn
from tqdm import tqdm
import cv2
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from scipy import spatial
import seaborn as sns
import shutil
import sys
import math
import pandas as pd
torch.backends.cudnn.benchmark=True
model_param=None
if len(sys.argv)>1:
    model_param=sys.argv[1]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

epochs=10

model_path = 'model_0814.pth'

image_size1,image_size2=1080//8,1920//8

#メモり20GB
#1シーンのフレーム数
T=60
n_channels=3
def transform(sample,T):
    video=sample["video"]
    label=sample["labels"]
    paths=sample["paths"]
    frames=sample["frames"]
   
    trans_video = torch.empty(n_channels,T,image_size1,image_size2)
    
    trans = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((image_size1,image_size2)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, ), (0.5, )),
    ])
    
    video,s,e=trim(video,T)
    video = torch.Tensor(video)
    
    for i in range(T):
        img = video[:,i]
        img = trans(img)
        img=img.reshape(n_channels,image_size1,image_size2)
        trans_video[:,i] = img
    if sum(label[s:e])>:
        label=1
    else:
        label=0
    sample = {'video': trans_video, 'labels': label,'paths':paths,'frames':frames}
    return sample

def trim(video,T):
    start = np.random.randint(0, video.shape[1] - (T+1))
    end = start + T
    return video[:, start:end, :, :],start,end

class SSdetect_model(nn.Module):
    def __init__(self, ndf=32, ngpu=1,category=2):
        super().__init__()
        self.ngpu = ngpu
        self.ndf = ndf
        self.category = category
        self.image_size = (1080//8,1920//8)
        self.n_channels = 3
        self.encoder = nn.Sequential(
            nn.Conv3d(self.n_channels, ndf, 3, 2, 1, bias=False),
            nn.BatchNorm3d(ndf),
            nn.ReLU(inplace=True),
            nn.Conv3d(ndf, ndf * 2, 3, 2, 1, bias=False),
            nn.BatchNorm3d(ndf * 2),
            nn.ReLU(inplace=True),
            nn.Conv3d(ndf * 2, ndf * 4, 3, 2, 1, bias=False),
            nn.BatchNorm3d(ndf * 4),
            nn.ReLU(inplace=True),
            nn.Conv3d(ndf * 4, ndf * 8, 3, 2, 1, bias=False),
            nn.BatchNorm3d(ndf * 8),
            nn.ReLU(inplace=True),
        )
        self.fc1=nn.Linear(138240,4000)
        self.relu=nn.ReLU(inplace=True)
        self.fc2=nn.Linear(4000,category)
        
    def forward(self,x):
        x=self.encoder(x)
        x = x.view(x.size()[0], -1)
        x=self.relu(self.fc1(x))
        x=self.fc2(x)
        return x

def train(dataloaders,model):
    train_loss=[]
    train_acc=[]
    val_loss=[]
    val_acc=[]
    scaler = torch.cuda.amp.GradScaler()
    print(time.strftime('%Y/%m/%d %H:%M:%S'))
    for epoch in range(epochs):
        for phase in ["train","valid"]:
            if phase=="train":
                model.train()
            else:
                model.eval()
            
            epoch_loss = 0.0
            epoch_correct=0

            for _,batch_data in tqdm(enumerate(dataloaders[phase])):
                    
                optimizer.zero_grad()   
                with torch.set_grad_enabled(phase=="train"):
                    with torch.cuda.amp.autocast():
                        batch_pred=model(batch_data["video"].to(device))
                        loss = criterion(batch_pred,batch_data["labels"].to(device))
                        _, predicted = torch.max(batch_pred, 1)
                    if phase=="train":
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                        
                    epoch_loss+=loss.item()*batch_data["video"].size(0)
                    epoch_correct += torch.sum(predicted==batch_data["labels"].to(device))
            
            epoch_loss=epoch_loss/len(dataloaders[phase].dataset)
            epoch_acc=epoch_correct.double()/len(dataloaders[phase].dataset)
            
            if phase=="train":
                
                train_loss.append(epoch_loss)
                train_acc.append(epoch_acc.item())
                torch.save(model.state_dict(), model_path)
            else:
                val_loss.append(epoch_loss)
                val_acc.append(epoch_acc.item())

            if epoch>0:
                plt.plot(train_loss,label="train")
                plt.plot(val_loss,label="valid")
                plt.xlabel("epochs")
                plt.ylabel("loss")
                plt.legend()
                plt.savefig("loss.png",dpi=200, format="png")

                plt.clf()

                plt.plot(np.array(train_acc),label="train")
                plt.plot(np.array(val_acc),label="valid")
                plt.xlabel("epochs")
                plt.ylabel("ACC")
                plt.legend()
                plt.savefig("ACC.png",dpi=200, format="png")

            print("Time {} | Epoch {}/{}|{:5}  | Loss:{:.4f} Acc:{:.4f}".format(time.strftime('%Y/%m/%d %H:%M:%S'),epoch+1,epochs,phase,epoch_loss,epoch_acc))
    print(time.strftime('END:%Y/%m/%d %H:%M:%S'))

if __name__ == "__main__":
    path="../リサイズ"
    category_key=[i for i in os.listdir(path) ]
    category_id=[i for i in range(len(category_key))]
    category={}
    for i,j in zip(category_key,category_id):
        category[i]=j
    path=[os.path.join(path,i) for i in os.listdir(path) ]
    files=[os.path.join(i,j) for i in path for j in os.listdir(i)]
    data=[]
    for file in files:
        data.append([file,category[file.split("\\")[1]]])

    random.seed(100)
    random.shuffle(data)
    n_train=int(len(data)*0.8)
    n_valid=int(len(data)*0.1)
    n_test=int(len(data)-n_train-n_valid)
    d=[]
    l=[]
    for i in data:
        d.append(i[0])
    
    train_dataset = BatchDataset(
    files=d[:n_train],
    T=T,
    transform=transform
    )
    valid_dataset = BatchDataset(
        files=d[n_train:n_train+n_valid],
        T=T,
        transform=transform
        )
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=64,
        shuffle=True,
        pin_memory=True,
        num_workers=4
        )
    valid_loader = torch.utils.data.DataLoader(
        dataset=valid_dataset,
        batch_size=32,
        pin_memory=True,
        shuffle=True,
        num_workers=4
        )

    model=SSdetect_model(category=2)
    model.to(device)
    lr=1e-5
    criterion= nn.CrossEntropyLoss().to(device)
    optimizer=torch.optim.Adam(model.parameters(), lr=lr)
    dataloaders=dict({"train":train_loader,"valid":valid_loader})

    train(dataloaders,model)

