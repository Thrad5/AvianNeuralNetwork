"""
Created on Wed Oct  4 19:46:35 2023

@author: ram86
"""
# %%
import os
import pandas as pd
from torchvision.io import read_image
import torch
from torch.utils.data import Dataset
from torchvision import datasets
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.utils import make_grid
from torch.utils.data import random_split
from torch.utils.data.dataloader import DataLoader
import torch.nn as nn
from torchsummary import summary

import numpy as np
import sys
sys.path.append("D:/!University/University_year_3/Master_Project/Fully_Self_Made_Code/magnetic_code")
from magnetic_code import error_detect as err

import torch.nn.functional as F

cwd = os.getcwd()
dir_path = os.path.dirname(os.path.realpath(__file__))
if cwd != dir_path:
    os.chdir(dir_path)

# %%
def accuracy(outputs,labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item()/len(preds))

@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

def fit(epochs, lr, model, train_loader, val_loader, opt_func = torch.optim.SGD ):
    #print('in fit')
    history = []
    optimiser = opt_func(model.parameters(),lr)
    #print('out of optimiser function')
    #j= 0
    early_stopper = EarlyStopper(patience=3,min_delta=0.2)
    for epoch in range(epochs):
        #print(f'Starting Epoch {j} out of {epochs}')
        #j+=1
        model.train()
        train_losses = []
        #i = 0
        for batch in train_loader:
            #print(f'Batch No: {i} of {num_batches}')
            loss = model.training_step(batch)
            #print('passed loss')
            train_losses.append(loss)
            #print('appended loss')
            loss.backward()
            #print('done backwards')
            optimiser.step()
            #print('stepped optimiser')
            optimiser.zero_grad()
            #print(f'Finishing Batch no: {i} of {num_batches}')
            #i +=1
        
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        model.epoch_end(epoch, result)
        history.append(result)
        if early_stopper.early_stop(result["val_loss"]):
            print(f"Finished at epoch: {epoch}")
            break
    return history

def plot_accuracies(history):
    """ Plot the history of accuracies"""
    #accuracies = [x['val_acc'] for x in history]
    plt.plot(history['val_acc'], '-x')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Accuracy vs. No. of epochs')
    plt.savefig("accuracies.png")
    plt.show()

def plot_losses(history):
    """ Plot the losses in each epoch"""
    #train_losses = [x.get('train_loss') for x in history]
    #val_losses = [x['val_loss'] for x in history]
    plt.plot(history['train_loss'], '-bx')
    plt.plot(history['val_loss'], '-rx')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['Training', 'Validation'])
    plt.title('Loss vs. No. of epochs')
    
    plt.savefig("losses.png")
    plt.show()

# %%

class EarlyStopper():
    def __init__ (self,patience = 1, min_delta = 0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')
    
    def early_stop(self,validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss+self.min_delta):
            self.counter +=1
            if self.counter > self.patience:
                return True
        return False


class CustomImageDataloader(Dataset):
    def __init__(self,img_dir,transform = None, target_transform=None):
        x = os.scandir(img_dir)
        head = []
        for file in x:
            
            sep = file.name.split("_")
            if sep[0] != "desktop.ini":
                
                try: 
                    h = int(sep[1])
                except IndexError:
                    print(sep)
                
                head.append([file.name,int(h/5)])
        head = pd.DataFrame(head)
        self.img_labels = head
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
    def __len__(self):
        return len(self.img_labels)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[index,0])
        image = read_image(img_path).float()
        image = image[:1,:,:]
        label = self.img_labels.iloc[index,1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image,label

class ImageClassificationBase(nn.Module):

    def training_step(self,batch):
        images, labels = batch
        #print('Seperated image and label')
        out = self(images) #Generate Predictions
        #print('Generated Predictions')
        loss = F.cross_entropy(out,labels) #Calculate loss
        #print('calculated loss')
        return loss
    
    def validation_step(self,batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out,labels)
        acc = accuracy(out,labels)
        return {'val_loss': loss.detach(), 'val_acc': acc}
    
    def validation_epoch_end(self, outputs):
        #print(outputs)
        batch_losses = [x['val_loss'] for x in outputs]
        #print(batch_losses)
        epoch_loss = torch.stack(batch_losses).mean()
        batch_accs = [x['val_acc'] for x in outputs]
        #print(batch_accs)
        epoch_acc = torch.stack(batch_accs).mean()
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self,epoch,result):
        print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['train_loss'], result['val_loss'], result['val_acc']))



# Creates the layers for the neural network
class NaturalSceneClassification(ImageClassificationBase):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential()
    
    def forward(self, xb):
        return self.network(xb)
    




class NeuralNetworkGenerator():

    def __init__(self,input_size):
        self.layers = []
        self.params = 0
        self.curr_dims = input_size
        self.func_str = ("MaxPool2d","Conv2d","ReLu","Flatten","Linear")
        self.head ="{:>9} {:>20}  {:>25} {:>15}".format("Layer Num","Layer (type)", "Output Shape", "Param #")
        self.x = len(self.head)
        self.num_layers = 0

    def __len__(self):
        return len(self.layers)

    def createNN (self):
        
        
        # Input size must be in shape [Channel,Height,Width]
        self.create2dMaxPool(2,2)
        self.create2dConv(32,3,padding=1)
        self.createReLU()
        self.create2dConv(64,3,padding=1)
        self.createReLU()
        self.create2dMaxPool(2,2)
        self.createFlatten()
        self.createLinear(72)
    
    def inputNN(self):
        while True:
            contyn = err.inpTF("Do you wish to add another layer?(y/n)")
            if contyn == False:
                break
            print("What type of layer do you want to add?(see below for number to layer converter)")
            print("""
                  Layer type - number category
                  MaxPool2d  -  0
                  Conv2d     -  1
                  ReLU       -  2
                  Flatten    -  3
                  Linear     -  4""")
            typ = err.inpInt("",0,4)
            if typ == 0:
                self.maxPool2dOpt()
            elif typ == 1:
                self.conv2dOpt()
            elif typ == 2:
                self.createReLU()
            elif typ == 3:
                self.createFlatten()
            elif typ == 4:
                self.linOpt()

    def maxPool2dOpt(self):
        size = err.inpInt("What size do you want the kernel to be?",1)
        stride = err.inpInt("What do you want the stride of the kernel to be?",1)
        pdyn = err.inpTF("Do you wish to change the padding from the default?(0)[y/n]")
        dlyn = err.inpTF("Do you wish to change the dilation from the default?(1)[y/n]")
        if pdyn:
            pd = err.inpInt("What do you want the padding to be?",0)
        else:
            pd =0
        if dlyn:
            dl = err.inpInt("What do you want the dilation to be?",1)
        else:
            dl = 1
        self.create2dMaxPool(size,stride,pd,dl)

    def conv2dOpt(self):
        out_ch = err.inpInt('How many channels do you want the output to be?')
        size = err.inpInt("What size do you want the kernel to be?")
        stryn = err.inpTF("Do yo wish to change the stride from the default?(1)[y/n]")
        pdyn = err.inpTF("Do you wish to change the padding from the default?(0)[y/n]")
        dlyn = err.inpTF("Do you wish to change the dilation from the default?(1)[y/n]")
        if stryn:
            stride = err.inpInt("What do you want the stride to be?",1)
        else:
            stride = 1
        if pdyn:
            pd = err.inpInt("What do you want the padding to be?",0)
        else:
            pd =0
        if dlyn:
            dl = err.inpInt("What do you want the dilation to be?",1)
        else:
            dl = 1
        self.create2dConv(out_ch,size,stride,pd,dl)

    def linOpt(self):
        out = err.inpInt("How many output nodes do you want?",1)
        self.createLinear(out)

    def printNetwork(self):
        print("-"*self.x)
        print(self.head)
        print("="*self.x)
        for i in range(len(self)):
            self.printRow(i)
        print("="*self.x)
        print(f"Total params: {self.params}")
        print("-"*self.x)
        
    def printRow(self,lay_num = -1):
        print("{:>9} {:>20}  {:>25} {:>15}".format( str(self.num_layers) if lay_num == -1 else str(lay_num),str(self.func_str[self.layers[lay_num][0]]),str(self.layers[lay_num][-1]),str(self.layers[lay_num][-2])))

    def create2dMaxPool (self,size,stride,padding = 0,dilation = 1):
        self.num_layers +=1
        self.curr_dims = [self.curr_dims[0],int(np.floor((self.curr_dims[1]+2*padding-(dilation*(size-1))-1)/stride+1)),int(np.floor((self.curr_dims[2]+2*padding-(dilation*(size-1))-1)/stride+1))]
        self.layers.append([0,size,stride,padding,dilation,0,self.curr_dims])
        self.params += self.layers[-1][-2]
        
        
    def create2dConv (self,out_ch,kern_sze,stride = 1, padding = 0,dilation = 1):
        self.num_layers +=1
        a = self.curr_dims[1]
        b = self.curr_dims[2]
        temp = [out_ch,
                          int(np.floor((a+2*padding-(dilation*(kern_sze-1))-1)/stride+1)),
                          int(np.floor((a+2*padding-(dilation*(kern_sze-1))-1)/stride+1))]
        self.layers.append([1,self.curr_dims[0],out_ch,kern_sze,stride,padding,dilation,(self.curr_dims[0]*kern_sze*kern_sze+1)*out_ch,temp])
        self.params += self.layers[-1][-2]
        self.curr_dims = temp
    
    def createReLU (self):
        self.num_layers +=1
        self.layers.append([2,0,self.curr_dims])
    
    def createFlatten (self):
        self.num_layers +=1
        self.layers.append([3,0,self.curr_dims[0]*self.curr_dims[1]*self.curr_dims[2]])
        self.curr_dims = self.curr_dims[0]*self.curr_dims[1]*self.curr_dims[2]
    
    def createLinear(self,out_shape):
        self.num_layers +=1
        self.layers.append([4,self.curr_dims,out_shape,(self.curr_dims+1)*out_shape,out_shape])
        self.params += self.layers[-1][-2]
        self.curr_dims = out_shape
    
    def layersToModel(self,model):
        for i in self.layers:
            if i[0] == 0:
                model.network.append(nn.MaxPool2d(i[1],i[2],i[3],i[4]))
            elif i[0] == 1:
                model.network.append(nn.Conv2d(i[1],i[2],i[3],i[4],i[5],i[6]))
            elif i[0] == 2:
                model.network.append(nn.ReLU())
            elif i[0] == 3:
                model.network.append(nn.Flatten())
            elif i[0] == 4:
                model.network.append(nn.Linear(i[1],i[2]))


# %%
def show_batch(dl):
    for images,labels in dl:
        images = images.type(torch.uint8)
        fig,ax = plt.subplots(figsize = (16,8))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(make_grid(images,nrow=16).permute(1,2,0))
        plt.show()
        break


# %%
if __name__ == "__main__":
    data_dir = r"D:/!University/University_year_3/Master_Project/Fully_Self_Made_Code/!Live_versions/1.1.1/data/png/"
    dataset = CustomImageDataloader(data_dir)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"The device is: {device}")
    batch_size = 128
    val_size = 20
    train_size = len(dataset)-val_size
    train_data, val_data = random_split(dataset,[train_size,val_size])
    train_dl = DataLoader(train_data,batch_size,shuffle=True,num_workers=1,pin_memory=True)
    train_dl.to()
    val_dl = DataLoader(val_data,batch_size*2, num_workers=1,pin_memory=True)

    #for image,label in train_dl:
    #    print(image.dtype)
    #    print(image.size())
    #    print(image[0,0,:,:])
    #    print(image[0,1,:,:])
    #    print(image[0,2,:,:])
    #    print(image[0,3,:,:])
    #    break

    creator = NeuralNetworkGenerator([1,335,335])
    model = NaturalSceneClassification()
    creator.createNN()
    creator.layersToModel(model)
    model.to(device)
    num_epochs = 1
    opt_func = torch.optim.Adam
    lr = 0.01
    print(f'Length of Train : {len(train_data)}')
    print(f'Length of Validation Data : {len(val_data)}')
    summary(model,(1,335,335),128)
    history = fit(num_epochs,lr,model,train_dl,val_dl,opt_func)
    hist = pd.DataFrame.from_dict(history)
    hist.to_csv('history.csv')
    plot_accuracies(hist)
    plot_losses(hist)
    #show_batch(train_dl)