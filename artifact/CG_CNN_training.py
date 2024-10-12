#prereq
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import glob
from natsort import natsorted
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from pathlib import Path
from torch.optim.lr_scheduler import ReduceLROnPlateau,CosineAnnealingLR, StepLR
from sklearn.metrics import f1_score, precision_score, recall_score
import math


# Data Loader
class Load_Data(Dataset):
    def __init__(self, data_files, label_files, transform=None):
        self.data_matrices = np.load(data_files+'/y.npy')
        self.label_matrices = np.load(label_files+'/x.npy')
        self.transform = transform

        for i in range(len(self.label_matrices)):
            self.label_matrices[i][self.label_matrices[i] == 0] = 100  #Add 100 ohm resistance as open circuit
        # Ensure the number of data and label matrices match
        assert len(self.data_matrices) == len(self.label_matrices), "Number of data and label files must match"
        
    def __len__(self):
        return len(self.data_matrices)
    
    def __getitem__(self, idx):
        data = self.data_matrices[idx]
        label = self.label_matrices[idx]

        # print(np.where(label==100,0,1),'\n',label)

        sample = {'features': torch.unsqueeze(torch.tensor(data, dtype=torch.float32),0) / 5.0,
                  'target': torch.tensor(label, dtype=torch.float32) / 100.0,
                 'label': torch.tensor(np.where(label == 100.0, 0, 1), dtype=torch.float32)
                 }
        
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample

# Model

## Position encoding
class PositionalEncoding(nn.Module):
    def __init__(self, num_channels, height, width):
        super(PositionalEncoding, self).__init__()
        self.num_channels = num_channels
        self.height = height
        self.width = width
        self.positional_encoding = self.create_positional_encoding()

    def create_positional_encoding(self):
        pe = torch.zeros(self.num_channels, self.height, self.width)
        # Use a large enough max length for sin/cos calculation
        # max_len = max(self.height, self.width)
        for pos in range(self.width):
            for i in range(0, self.num_channels, 2):
                if i < self.num_channels:
                    pe[i, :, pos] = math.sin(pos / (10 ** (i / self.num_channels)))
                    if i + 1 < self.num_channels:
                        pe[i + 1, :, pos] = math.cos(pos / (10 ** (i / self.num_channels)))
                        
        for pos in range(self.height):
            for i in range(0, self.num_channels, 2):
                if i < self.num_channels:
                    pe[i, pos, : ] += math.sin(pos / (100 ** (i / self.num_channels)))
                    if i + 1 < self.num_channels:
                        pe[i + 1,pos,: ] += math.cos(pos / (100 ** (i / self.num_channels)))
        return pe.unsqueeze(0)  # Add batch dimension

    def forward(self, x):
        return x + self.positional_encoding.to(x.device)

#Model declaration

class Net(nn.Module):
    def __init__(self,num_iteration):
        super().__init__()
        self.num_iteration = num_iteration
        self.dropout_prob = 0.4
        self.Conv1=nn.Conv2d(25,25,(1,11))
        self.Conv2= nn.Conv2d(25,25,(7,1))
        self.Conv3= nn.Conv2d(25,2,(1,1))
        self.Conv_list1 = nn.ModuleList([nn.Conv2d(25, 25, (1, 11)) for _ in range(num_iteration)])
        self.Conv_list2 = nn.ModuleList([nn.Conv2d(25, 25, (7, 1)) for _ in range(num_iteration)]) 
        
        self.Tconv1 = nn.ConvTranspose2d(25,25,(1,11))
        self.Tconv2 = nn.ConvTranspose2d(25,25,(7,1))
        
            
        self.Tconv_list1 = nn.ModuleList([nn.ConvTranspose2d(25,25,(1,11)) for _ in range(num_iteration)])
        self.Tconv_list2 = nn.ModuleList([nn.ConvTranspose2d(25,25,(7,1)) for _ in range(num_iteration)])

        self.positional_encoding = PositionalEncoding(25, 7, 11)

        self.dropout1 = nn.Dropout(p=self.dropout_prob)
        self.dropout2 = nn.Dropout(p=0.4)
    def forward(self, x):
        x=x.repeat(1, 25, 1, 1)
        pos_emb = self.positional_encoding(x)

        pos_emb = 0.2*(pos_emb/2)
        x=torch.add(pos_emb,x)
        
        orgnl=x
        z=x
        for i in range(self.num_iteration):
            conv_layer1 = self.Conv_list1[i]
            conv_layer2 = self.Conv_list2[i]
            tconv_layer1 = self.Tconv_list1[i]
            tconv_layer2 = self.Tconv_list2[i]
            
            a=F.relu(self.Conv1(x))
            a=self.Tconv1(a)
            b=F.relu(self.Conv2(x))
            b=self.Tconv2(b)
            x_middle=torch.sum(torch.stack((a,b),dim=1),1)
            x_middle=self.dropout1(x_middle)
            
            a = F.relu(conv_layer1(x_middle))
            a = tconv_layer1(a)
            b = F.relu(conv_layer2(x_middle))  
            b = tconv_layer2(b)
            x = torch.sum(torch.stack((a,b),dim=1),1)
            x = self.dropout2(x)
            x1 = torch.add(z,x)
            x = F.relu(x1)  
            z=x
        x = self.Conv3(x)
        return x

#Training
def Training(data_path,label_path,batch,scdulr='',lr=1e-3,depth=10,epoch=100):

    dataset = Load_Data(data_path, label_path)
    num_epochs = epoch

    a=np.shape(dataset)

    
    seed = 27 #previous12,27
    # random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    train_ratio = 0.6
    val_ratio = 0.1
    test_ratio = 0.1

    unseen_ratio = 0.2
    train_ratio =0.8


    batch_size=batch
    
    val_error=[]
    tst_error=[]
    f1_list=[]
    res_error=[]

    
    training_size=int(train_ratio*a[0])

    training_dataset = Subset(dataset, range(training_size))
    
    unseen_dataset = torch.utils.data.Subset(dataset, range(training_size, a[0]))

    separate_dataloader = DataLoader(unseen_dataset, batch_size=batch_size, shuffle=False)

    
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("CUDA is available. Using GPU.")
    else:
        device = torch.device("cpu")
        print("CUDA is not available. Using CPU.")

    
    
    
    net = Net(num_iteration = depth)
    net = net.double()
    
    net = net.to(device)
    
    criterion = nn.MSELoss()
    label_criterion=nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(net.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=1e-3)
    # optimizer = optim.Adam(net.parameters(), lr=lr)
    if scdulr=='Platue':
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.01, patience=2)
    if scdulr=='Cosine':
        scheduler = CosineAnnealingLR(optimizer, num_epochs, eta_min=lr * 1e-6)

    train_size = int(train_ratio * len(training_dataset))
    val_size = int(val_ratio * len(training_dataset))
    test_size = len(training_dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(training_dataset, [train_size, val_size, test_size])

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    alpha,beta=0.91,0.09

    epsilon=1e-8
    for epoch in range(num_epochs):
        net.train()
        epoch_loss = 0.0

        if epoch>0:
            alpha = 1/(output_loss.item() +epsilon)
            beta = 1/(label_loss.item() + epsilon)
            weight_sum = alpha + beta
            alpha /= weight_sum
            beta /= weight_sum

	#Dynamic loss attention adaptation
        # if((epoch+1)%15==0):
        #     if alpha<0.89:
        #         alpha,beta=0.91,0.09
        for batch in train_dataloader:
            features, targets, labels = batch['features'], batch['target'], batch['label']

            # Zero the parameter gradients
            optimizer.zero_grad()

            features = features.squeeze(1)
            targets = targets.squeeze(1)
            labels = labels.squeeze(1)

            
            features = features.to(device)
            targets = targets.to(device)
            labels = labels.to(device)
        
            # Forward pass
            outputs = net(features.double())
#             print('output device:',outputs.get_device())
            outputs, predicted_labels = torch.transpose(outputs, 0, 1)
            output_loss = criterion(outputs, targets.double())
            label_loss = label_criterion(predicted_labels, labels.double())
            loss = (alpha * output_loss + beta * label_loss)/2
            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Accumulate the loss
            epoch_loss += loss.item()

        # Calculate and print the average loss for the epoch
        average_loss = epoch_loss / len(train_dataloader)
        if (((epoch+1)%10)==0):
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {average_loss:.4f}')

        # Validation phase
        net.eval()
        val_loss = 0.0
        out_loss = 0.0
        total_f1 = 0.0
        with torch.no_grad():
            for batch in val_dataloader:
                features, targets, labels = batch['features'], batch['target'], batch['label']

                features=features.squeeze(1)
                targets=targets.squeeze(1)
                labels=labels.squeeze(1)
                
                features = features.to(device)
                targets = targets.to(device)
                labels = labels.to(device)
                
                outputs = net(features.double())
                outputs, predicted_labels = torch.transpose(outputs, 0, 1)
                output_loss = criterion(outputs, targets.double())
                label_loss = label_criterion(predicted_labels,labels.double())
                loss=(alpha * output_loss + beta * label_loss)/2
                out_loss += output_loss.item()
                val_loss += loss.item()

                # Calculate F1 score
                outputs_sigmoid = torch.sigmoid(predicted_labels)
                outputs_np = outputs_sigmoid.cpu().numpy().flatten()
                targets_np = labels.cpu().numpy().flatten()
                outputs_bin = (outputs_np > 0.5).astype(int)  # Binarize the outputs
                # print(outputs_bin, targets_np)
                # targets_bin = targets_np.astype(int)
                f1 = f1_score(targets_np, outputs_bin, average='macro')
                total_f1 += f1

        val_loss /= len(val_dataloader)
        out_loss /= len(val_dataloader)
        average_f1 = total_f1 / len(val_dataloader)
        if (((epoch+1) %10)==0):
            print(f'Validation Loss: {output_loss:.4f}, Validation F1 Score: {average_f1:.4f}, Label loss:{label_loss:.4f}, {alpha:.3f},{beta:.3f}')

        save = Path(f'{data_path[:-1]}/ablation/')

        if not save.exists():
            save.mkdir(parents=True)
            
        # torch.save(net.state_dict(),f'{data_path[:-1]}/ablation/model_{depth}.pth')

        if scdulr=='Platue':
            # scheduler.step(val_loss)
            scheduler.step(val_loss)
        if scdulr=='Cosine':
            scheduler.step()
        

    # Final evaluation on test data
    net.eval()
    test_loss = 0.0
    total_f1 = 0.0
    with torch.no_grad():
        for batch in test_dataloader:
            features, targets, labels = batch['features'], batch['target'], batch['label']

            features=features.squeeze(1)
            targets=targets.squeeze(1)
            labels=labels.squeeze(1)
            
            features = features.to(device)
            targets = targets.to(device)
            labels = labels.to(device)
            
            outputs = net(features.double())
            
            outputs, predicted_labels = torch.transpose(outputs, 0, 1)
            output_loss = criterion(outputs, targets.double())
            label_loss = label_criterion(predicted_labels,labels.double())
            loss=(output_loss + label_loss *0.5)/2
            # loss=(output_loss + label_loss)/2
            test_loss += loss.item()

            # Calculate F1 score
            outputs_sigmoid = torch.sigmoid(predicted_labels)
            outputs_np = outputs_sigmoid.cpu().numpy().flatten()
            targets_np = labels.cpu().numpy().flatten()
            outputs_bin = (outputs_np > 0.5).astype(int)  # Binarize the outputs
            # print(outputs_bin, targets_np)
            # targets_bin = targets_np.astype(int)
            f1 = f1_score(targets_np, outputs_bin, average='macro')
            total_f1 += f1

    test_loss /= len(test_dataloader)
    average_f1 = total_f1 / len(test_dataloader)
    
    val_error.append(val_loss)
    tst_error.append(test_loss)
    
    print(f'Final Test Loss: {test_loss:.4f}, Final Test F1 Score: {average_f1:.4f}, Label loss:{label_loss}')

    print("Training complete")

    return average_f1


data_path='../figure_7x11/r_top=3.0,r_bottom=0.02/(7, 11)/'
F1=Training(data_path,data_path,batch=16,scdulr='Cosine',lr=1e-3,depth=5,epoch=200)#
