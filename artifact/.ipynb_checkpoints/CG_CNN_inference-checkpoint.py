import torch
import torch.nn as nn
import torch.nn.functional as F
import glob
from natsort import natsorted
import math
from sklearn.metrics import f1_score, precision_score, recall_score
import numpy as np
import matplotlib.pyplot as plt

# Position Encoding
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


# Model Declaration
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

#convert adc scanned frame to voltage 
def adc_to_vol(arr):
    out=np.array(arr)*(5/1024)
    return out
    
    
path=glob.glob('../figure_7x11/realworld_data2/*')#need to change path

patterns = [ i for i in path if i[-3:] !='jpg']
patterns = [ i for i in patterns if i[-3:] !='csv']

gt_list = [ i for i in path if i[-3:] =='csv']
patterns=natsorted(patterns)
patterns=patterns[:-1]

depth=5
all_thresh_val,all_thresh_std=[],[]
thresh=97

model = Net(num_iteration = depth)
model = model.double()
model.load_state_dict(torch.load('model_5.pth',map_location=torch.device('cpu')))
model=model.eval()


f1_pattern,all_pattern_f1,all_pattern_std = [],[],[]
f1_before,all_pattern_f1_before,all_pattern_std_before = [],[],[]


temp=[]
for i in range(len(patterns)):
    if((i)%5==0):
        a=patterns[i].split('/')
        ground_t = np.loadtxt('../figure_7x11/realworld_data2/'+a[-1]+'.csv',delimiter=',')


    files=glob.glob(patterns[i]+'/*.csv')

    input_arr=np.loadtxt(files[1000],delimiter=',')

    input_arr=adc_to_vol(input_arr)
    input_arr = np.transpose(input_arr)/5.0
    feature = torch.unsqueeze(torch.tensor(input_arr, dtype=torch.float32),0) 
    output = model(feature.double())
    input_arr = np.transpose(input_arr)

    output, predicted_label = torch.transpose(output, 0, 1)
    output = output.squeeze(0)
    output = output.double().detach().numpy()

    output=np.transpose(output)
    output = output*100
    output = np.where(output > thresh, 0, 1)

    predicted_label = torch.sigmoid(predicted_label)
    predicted_label = predicted_label.squeeze(0)
    predicted_label = predicted_label.double().detach().numpy()
    predicted_label = (predicted_label > 0.5).astype(int)
    predicted_label = np.transpose(predicted_label)
    orgnl = input_arr
    input_arr = np.where(input_arr>0.1,1,0)

    test = output*input_arr

    flat_test=test.flatten()
    flat_gt=ground_t.flatten()
    flat_in=input_arr.flatten()

    f1_input = f1_score(flat_gt, flat_in, average='binary')
    f1_after = f1_score(flat_gt, flat_test, average='binary')

    f1_pattern.append(f1_after)
    f1_before.append(f1_input)


    if((i+1)%5==0):
        temp.append(np.mean(f1_pattern))

        all_pattern_f1.append(np.mean(f1_pattern))
        all_pattern_std.append(np.std(f1_pattern))

        all_pattern_f1_before.append(np.mean(f1_before))
        all_pattern_std_before.append(np.std(f1_before))


        f1_pattern=[]
        f1_before=[]
all_thresh_val.append(np.mean(temp))
all_thresh_std.append(np.std(temp))
# print(all_thresh_val)

oms_sensor = all_pattern_f1_before
oms_system = all_pattern_f1





#velostat inference
path=glob.glob('../figure_7x11/realworld_data2/velostat/*')

patterns = [ i for i in path if i[-3:] !='jpg']
patterns = [ i for i in patterns if i[-3:] !='csv']

gt_list = [ i for i in path if i[-3:] =='csv']
patterns = natsorted(patterns)

noise_path = patterns[0]

# calculate the threshold value for noise
noise_csv = glob.glob(noise_path+'/*.csv')
noise_csv = natsorted(noise_csv)

noise_list=[]
for i in range(1,len(noise_csv)):
    noise_list.append(np.loadtxt(noise_csv[i],delimiter=','))

noise_stack = np.stack(noise_list)
noise_mean = np.mean(noise_stack,axis=0)
noise_std = np.std(noise_stack,axis=0)

nosie_thresh = noise_mean+ (3 * noise_std)


patterns=patterns[1:]

depth=5
all_thresh_val,all_thresh_std=[],[]
thresh=3880

model = Net(num_iteration = depth)
model = model.double()

model.load_state_dict(torch.load(f'velostat_model_{depth}.pth',map_location=torch.device('cpu')))
model=model.eval()

def adc_to_vol(arr):
    out=np.array(arr)*(5/1024)
    return out


f1_pattern,all_pattern_f1,all_pattern_std = [],[],[]
f1_before,all_pattern_f1_before,all_pattern_std_before = [],[],[]


temp=[]
for i in range(len(patterns)):
    if((i)%5==0):
        a=patterns[i].split('/')
        ground_t = np.loadtxt('../figure_7x11/realworld_data2/'+a[-1]+'.csv',delimiter=',')

    # print(patterns[i])
    files=glob.glob(patterns[i]+'/*.csv')

    input_arr=np.loadtxt(files[40],delimiter=',')

    input_arr=adc_to_vol(input_arr)
    input_arr = np.transpose(input_arr)/5.0
    feature = torch.unsqueeze(torch.tensor(input_arr, dtype=torch.float32),0) 
    output = model(feature.double())
    input_arr = np.transpose(input_arr)
    
    output, predicted_label = torch.transpose(output, 0, 1)
    output = output.squeeze(0)
    output = output.double().detach().numpy()

    output=np.transpose(output)
    output = output*4000
    output = np.where(output > thresh, 0, 1)

    predicted_label = torch.sigmoid(predicted_label)
    predicted_label = predicted_label.squeeze(0)
    predicted_label = predicted_label.double().detach().numpy()
    predicted_label = (predicted_label > 0.5).astype(int)
    predicted_label = np.transpose(predicted_label)
    orgnl = input_arr
    input_arr = np.where(input_arr>0.1,1,0)

    test = output*input_arr
    


    flat_test=test.flatten()
    flat_gt=ground_t.flatten()
    flat_in=input_arr.flatten()

    f1_input = f1_score(flat_gt, flat_in, average='binary')
    f1_after = f1_score(flat_gt, flat_test, average='binary')

    f1_pattern.append(f1_after)
    f1_before.append(f1_input)


    if((i+1)%5==0):


        temp.append(np.mean(f1_pattern))

        all_pattern_f1.append(np.mean(f1_pattern))
        all_pattern_std.append(np.std(f1_pattern))

        all_pattern_f1_before.append(np.mean(f1_before))
        all_pattern_std_before.append(np.std(f1_before))


        f1_pattern=[]
        f1_before=[]
all_thresh_val.append(np.mean(temp))
all_thresh_std.append(np.std(temp))

vel_sensor = all_pattern_f1_before
vel_system = all_pattern_f1

fig,ax=plt.subplots()
labels=['1','2','3','4']
x = np.arange(len(labels))

width=0.2

bar1 = ax.bar(x-width, vel_sensor,width,label='Velostat w/o CG-CNN')
bar2 = ax.bar(x, vel_system,width,label='Velostat ')
bar3 = ax.bar(x+width, oms_sensor,width,label='OMSense w/o CG-CNN')
bar4 = ax.bar(x+2*width, oms_system,width,label='OMSense ')



ax.set_ylabel('F1 Score')
ax.set_xlabel('Pattern')
ax.set_title('Scores by group and gender')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
plt.show()
