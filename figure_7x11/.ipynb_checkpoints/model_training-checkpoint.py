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

from torch.optim.lr_scheduler import ReduceLROnPlateau,CosineAnnealingLR
from sklearn.metrics import f1_score

def plot_images_in_rows(images_list, row_labels=None):
    rows = len(images_list)
    columns = max(len(images) for images in images_list)  # Determine the number of columns needed
    
    fig, axes = plt.subplots(rows, columns, figsize=(20, 5), gridspec_kw={'hspace': 0.5})
    
    for row, (images, label) in enumerate(zip(images_list, row_labels)):
        for col in range(columns):
            if col < len(images):
                axes[row, col].imshow(images[col])
                axes[row, col].axis('off')
            else:
                axes[row, col].axis('off')  # Hide any extra subplots
        if label is not None:
            axes[row, 0].text(-0.25, 0.5, label, transform=axes[row, 0].transAxes, fontsize=12, va='center',rotation=90)

    plt.tight_layout()
    plt.show()
    
class Load_Data(Dataset):
    def __init__(self, data_files, label_files, transform=None):
        self.data_matrices = np.load(data_files+'/y.npy')
        self.label_matrices = np.load(label_files+'/x.npy')
        self.transform = transform

        for i in range(len(self.label_matrices)):
            self.label_matrices[i][self.label_matrices[i] == 0] = 100
        # Ensure the number of data and label matrices match
        assert len(self.data_matrices) == len(self.label_matrices), "Number of data and label files must match"
        
    def __len__(self):
        return len(self.data_matrices)
    
    def __getitem__(self, idx):
        data = self.data_matrices[idx]
        label = self.label_matrices[idx]

        sample = {'features': torch.unsqueeze(torch.tensor(data, dtype=torch.float32),0) / 5.0,
                  'target': torch.tensor(label, dtype=torch.float32) / 100.0,
                 'label': torch.tensor(np.where(label == 0, 0, 1), dtype=torch.float32)
                 }
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.Conv1=nn.Conv2d(2,25,(1,5))
        self.Conv2= nn.Conv2d(2,25,(5,1))
        self.Conv_list1 = nn.ModuleList([nn.Conv2d(2, 25, (1, 5)) for _ in range(10)])
        self.Conv_list2 = nn.ModuleList([nn.Conv2d(2, 25, (5, 1)) for _ in range(10)]) 
        
        self.Tconv1 = nn.ConvTranspose2d(25,2,(1,5))
        self.Tconv2 = nn.ConvTranspose2d(25,2,(5,1))
        
            
        self.Tconv_list1 = nn.ModuleList([nn.ConvTranspose2d(25,2,(1,5)) for _ in range(10)])
        self.Tconv_list2 = nn.ModuleList([nn.ConvTranspose2d(25,2,(5,1)) for _ in range(10)])
    def forward(self, x):
        x=x.repeat(1, 2, 1, 1)
        z=x
        for i in range(10):
            conv_layer1 = self.Conv_list1[i]
            conv_layer2 = self.Conv_list2[i]
            tconv_layer1 = self.Tconv_list1[i]
            tconv_layer2 = self.Tconv_list2[i]
            
            a=F.relu(self.Conv1(x))
            # print('a',a.shape)
            a=self.Tconv1(a)
            # print('a',a.shape)
            b=F.relu(self.Conv2(x))
            # print('b',b.shape)
            b=self.Tconv2(b)
            # print('b',b.shape)
            x=torch.sum(torch.stack((a,b),dim=1),1)
            
            a = F.relu(conv_layer1(x))
            a = tconv_layer1(a)
            b = F.relu(conv_layer2(x))  
            b = tconv_layer2(b)
            x = torch.sum(torch.stack((a,b),dim=1),1)
            x = F.relu(x)
            
            x=torch.add(z,x)
            x = F.relu(x)  
            z=x
        return x

def Training(data_path,label_path,scdulr=''):
    torch.manual_seed(0)

    dataset = Load_Data(data_path, label_path)
    num_epochs = 100

    a=np.shape(dataset)

    
    train_ratio = 0.8
    val_ratio = 0.1
    test_ratio = 0.1

    unseen_ratio = 0.2
    train_ratio =0.8


    batch_size=8
    
    val_error=[]
    tst_error=[]
    f1_list=[]
    res_error=[]

    
    training_size=int(train_ratio*a[0])

    training_dataset = Subset(dataset, range(training_size))
    
    unseen_dataset = torch.utils.data.Subset(dataset, range(training_size, a[0]))

    separate_dataloader = DataLoader(unseen_dataset, batch_size=batch_size, shuffle=False)

    
    
    
    
    
    net = Net()
    net = net.double()
    
    criterion = nn.MSELoss()
    label_criterion=nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(net.parameters(), lr=1e-5, betas=(0.8, 0.9), weight_decay=0.1)
    # optimizer = optim.Adam(net.parameters(), lr=1e-4)
    if scdulr=='Platue':
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.10, patience=4, verbose=True)

    if scdulr=='Cosine':
        scheduler = CosineAnnealingLR(optimizer, num_epochs * 2)

    train_size = int(train_ratio * len(training_dataset))
    val_size = int(val_ratio * len(training_dataset))
    test_size = len(training_dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(training_dataset, [train_size, val_size, test_size])

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


    for epoch in range(num_epochs):
        net.train()
        epoch_loss = 0.0

        for batch in train_dataloader:
            features, targets, labels = batch['features'], batch['target'], batch['label']

            # Zero the parameter gradients
            optimizer.zero_grad()

            features=features.squeeze(1)
            targets=targets.squeeze(1)
            labels=targets.squeeze(1)

            
            # print(targets.shape,features.shape)
            # tar=targets.squeeze(1)
            # fea=features.squeeze(1)
            # print(tar.shape,fea.shape)


            # tar=tar.detach().numpy()
            # fea=fea.detach().numpy()

            # print(tar[0],fea[0])
            # Forward pass
            outputs = net(features.double())
            outputs, predicted_labels = torch.transpose(outputs, 0, 1)
            output_loss = criterion(outputs, targets.double())
            label_loss = label_criterion(predicted_labels, labels.double())
            loss = (output_loss + label_loss * 0.2) / 2
            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Accumulate the loss
            epoch_loss += loss.item()

        # Calculate and print the average loss for the epoch
        average_loss = epoch_loss / len(train_dataloader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {average_loss:.4f}')

        # Validation phase
        net.eval()
        val_loss = 0.0
        total_f1 = 0.0
        with torch.no_grad():
            for batch in val_dataloader:
                features, targets, labels = batch['features'], batch['target'], batch['label']

                features=features.squeeze(1)
                targets=targets.squeeze(1)
                labels=labels.squeeze(1)
                
                outputs = net(features.double())
                outputs, predicted_labels = torch.transpose(outputs, 0, 1)
                # print(predicted_labels.shape, labels.shape)
                output_loss = criterion(outputs, targets.double())
                label_loss = label_criterion(predicted_labels,labels.double())
                loss=(output_loss + label_loss * 0.2)/2
                val_loss += loss.item()

                # Calculate F1 score
                outputs_np = outputs.cpu().numpy().flatten()
                targets_np = targets.cpu().numpy().flatten()
                outputs_bin = (outputs_np > 0.5).astype(int)  # Binarize the outputs
                targets_bin = targets_np.astype(int)
                f1 = f1_score(targets_bin, outputs_bin, average='macro')
                total_f1 += f1

        val_loss /= len(val_dataloader)
        average_f1 = total_f1 / len(val_dataloader)
        print(f'Validation Loss: {val_loss:.4f}, Validation F1 Score: {average_f1:.4f}')

        # if(((epoch+1)%50)==0):
        #     torch.save(net.state_dict(),f'model_scheduler{scdulr}.pth')
        # print(data_path[:-1])
        torch.save(net.state_dict(),f'{data_path[:-1]}/model_scheduler{scdulr}.pth')

        if scdulr=='Platue':
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
            
            outputs = net(features.double())
            outputs, predicted_labels = torch.transpose(outputs, 0, 1)
            output_loss = criterion(outputs, targets.double())
            label_loss = label_criterion(predicted_labels,labels.double())
            loss=(output_loss + label_loss * 0.2)/2
            test_loss += loss.item()

            # Calculate F1 score
            outputs_np = outputs.cpu().numpy().flatten()
            targets_np = targets.cpu().numpy().flatten()
            outputs_bin = (outputs_np > 0).astype(int)  # Binarize the outputs
            targets_bin = (targets_np>0).astype(int)
            f1 = f1_score(targets_bin, outputs_bin, average='binary')
            total_f1 += f1

    test_loss /= len(test_dataloader)
    average_f1 = total_f1 / len(test_dataloader)
    
    val_error.append(val_loss)
    tst_error.append(test_loss)
    
    print(f'Final Test Loss: {test_loss:.4f}, Final Test F1 Score: {average_f1:.4f}')

    print("Training complete")
    
    
    # all_f1=[]
    # all_error=[]
    # a=80
    # for batch in separate_dataloader:
    #     features, targets = batch['features'], batch['target']
    #     out = net(features.double())

    #     targets = targets.double().detach().numpy()
    #     targets=np.squeeze(targets,0)

    #     out = out.double().detach().numpy()
    #     out = np.squeeze(out,0)

    #     # print(out,targets)
    #     diff=np.abs(out-targets).flatten()
    #     all_error.append(diff)
    #     out[out < a] = 1
    #     out[out >= a] = 0




    #     targets[targets<99]=1
    #     targets[targets>99]=0

    #     # print(out,targets)

    #     targets=targets.flatten()
    #     out=out.flatten()

    #     f1 = f1_score(targets, out, average='macro')

    #     all_f1.append(f1)
    # all_f1=np.array(all_f1)
    # all_error=np.array(all_error).flatten()
    
    # f1_list.append(np.mean(all_f1))
    # res_error.append(np.mean(all_error))
    # print(np.mean(all_f1),np.mean(all_error))

    # return net


fol=glob.glob('/origami-sensor/figure_2/r_top*')
for i in fol:
    print(i)
    data_path=i+'/(5, 5)/'
    model = Training(data_path,data_path,scdulr='Platue')# scheduler Platue, Cosine