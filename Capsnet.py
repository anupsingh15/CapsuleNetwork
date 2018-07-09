
# coding: utf-8

# In[1]:

import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.optim as optim 
import torch.utils.data as data
from torch.autograd import Variable


# In[2]:

class LoadData:
    def __init__(self, batch_size):
        
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        self.trainset = datasets.MNIST(root = './data', train = True, transform = transform, download = True)
        self.testset = datasets.MNIST(root = './data', train = False, transform = transform, download = True)
    
        self.trainset_loader = data.DataLoader(self.trainset, shuffle = True, batch_size = batch_size)
        self.testset_loader = data.DataLoader(self.testset, shuffle = False, batch_size = batch_size)


# In[3]:

batch_size = 100
data = LoadData(batch_size)


# In[4]:

class ConvLayer(nn.Module):
    def __init__(self, in_channels=1, out_channels=256, kernel_size=9):
        super(ConvLayer, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size,
                               stride=1
                             )

    def forward(self, x):
        return F.relu(self.conv(x)) #return an output of shape: [batch_size, channels, H, W]


# In[5]:

a = ConvLayer()
inp = torch.rand((10, 1, 28, 28))
out = a.forward(inp)
out.size()


# In[6]:

class PrimeLayerCaps(nn.Module):
    def __init__(self, num_capsules = 32, input_dim = 256, output_dim = 8, kernel_size = 9):
        super(PrimeLayerCaps, self).__init__()
        #returns a list of conv layers, each giving an output of shape[batch_size, channels, H, W]
        self.capsules = nn.ModuleList([nn.Conv2d(in_channels = input_dim, out_channels = output_dim, kernel_size = kernel_size, stride = 2) for _ in range(num_capsules)])
        self.num_capsules = num_capsules
        
    def forward(self, X):
        u = [capsule(X) for capsule in self.capsules]#[batch_size, channels, H, W]
        u = torch.stack(u, dim=1)#all capsules stacked above each other [batch_size, num_capsule_layers, channels, H, W]
        u = u.view(X.size(0), self.num_capsules*6*6, -1)# for each batch sample, creating total no. of capsule vectors, #[batch_size, total_caps, dim_of_capsules]
        return self.squash(u)
    
    def squash(self, tensor):
        norm_squared = torch.sum(tensor ** 2, dim = -1, keepdim=True)
        squashed_tensor = (norm_squared/(1+norm_squared)) * (tensor/(0.001 + torch.sqrt(norm_squared)))
        return squashed_tensor        


# In[7]:

a = PrimeLayerCaps()
inp = torch.rand((10, 256, 20, 20))
out = a(inp)
out.size()


# In[8]:

class DigitLayerCaps:
    def __init__(self, num_capsules2=10, input_dim = 8, output_dim = 16, num_capsules1 = 32*6*6):
        super(DigitLayerCaps, self).__init__()
        self.W = nn.Parameter(torch.randn(num_capsules1, num_capsules2, output_dim, input_dim))
        self.num_capsules2 = num_capsules2
        self.num_capsules1 = num_capsules1
        
    def squash(self, tensor):
        norm_squared = torch.sum(tensor ** 2, dim = -1, keepdim=True)
        squashed_tensor = (norm_squared/(1+norm_squared)) * (tensor/(0.001 + torch.sqrt(norm_squared)))
        return squashed_tensor
        
    def forward(self, X):
        batch_size = X.size(0)
        repeat_W = torch.unsqueeze(self.W, 0).repeat(batch_size, 1, 1, 1, 1)
        shapemodified_X = torch.unsqueeze(torch.unsqueeze(X, 3), 2).repeat(1,1,10,1,1)
        shapemodified_X = shapemodified_X.cuda()
        repeat_W = repeat_W.cuda()
        u_hat = torch.matmul(repeat_W, shapemodified_X) #predicted_outputs from each capsules of lower layer 
        # [batch_size, num_caps1, num_caps2, caps2_dim, 1]
        u_hat = u_hat.cuda()
        
        #print (shapemodified_X.type(), repeat_W.type())
        
        
        
        b_ij = Variable(torch.zeros(batch_size, self.num_capsules1, self.num_capsules2, 1, 1))
        if torch.cuda.is_available():
            b_ij = b_ij.cuda()

        iterations = 3
        for iteration in range(iterations):
            softmax_layer = nn.Softmax(dim=2)
            c_ij = softmax_layer(b_ij) #same shape as b_ij
            weighted_prediction = c_ij * u_hat #element wise multiplication [batch_size, num_caps1, num_caps2, caps2_dim, 1]
            s_j = torch.sum(weighted_prediction, dim=1, keepdim = True) #average prediction for each caps2 [batch_size, 1, 10, caps2_dim, 1]
            v_j = self.squash(s_j)
           # print (weighted_prediction.type(), s_j.type(), v_j.type()) 
            if iteration < iterations - 1: #??????
                agreement =  torch.matmul(s_j.transpose(3,4), u_hat)#last two dims transposed [batch_size, num_caps1, num_caps2, 1, 1]
                b_ij += agreement #updating routing weights
            
            v_j = v_j.squeeze(1)
            return v_j.squeeze(3)
            
        
                


# In[9]:

a = DigitLayerCaps()
inp = torch.rand((10, 1152, 8))
out = a.forward(inp)
out.size()


# In[10]:

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.reconstruction = nn.Sequential(nn.Linear(16 * 10, 512), nn.ReLU(inplace=True),nn.Linear(512, 1024),
            nn.ReLU(inplace=True), nn.Linear(1024, 784), nn.Sigmoid()) #sequence of layer to reconstruct image of predicted output
        
    def forward(self, caps2_output, data_targets, use_training = True):
        caps_length = torch.sqrt((caps2_output ** 2).sum(dim=2))
        soft = nn.Softmax(1)
        caps_length = soft(caps_length)
        _, index = caps_length.max(dim=1) #index of the capsule with maximum length
        mask = torch.zeros(caps2_output.size())
        
        mask = mask.cuda()
        if use_training: #while training will mask using target
            for batch_sample in range(caps2_output.size(0)):
                mask[batch_sample, data_targets[batch_sample], :] = 1
        else: #while testing will mask using index of capsule with highest prob. 
            for batch_sample in range(caps2_output.size(0)):
                mask[batch_sample, index[batch_sample], :] = 1
                
        masked_caps2_output = mask * caps2_output #mask applied 
        masked_caps2_output = masked_caps2_output.view(-1, 16*10)
        output = self.reconstruction(masked_caps2_output)
        reconstruct = output.view(-1, 1, 28, 28)#decoding an image
        return reconstruct, index


# In[11]:

import random
a = Decoder()
a = a.cuda()
inp = torch.rand(2, 10, 16)
inp = inp.cuda()
targets = np.array([random.sample(range(0,10),5)])
targets = torch.from_numpy(targets)
targets = targets.squeeze()
targets = targets.cuda()
out, re = a.forward(inp, targets)
out.size()


# In[12]:

class CapsuleNetwork(nn.Module):
    def __init__(self):
        super(CapsuleNetwork, self).__init__()
        self.Conv_layer = ConvLayer()
        self.PrimeCaps_layer = PrimeLayerCaps()
        self.DigitCaps_layer = DigitLayerCaps()
        self.Decoder = Decoder()
        #self.MSE_loss = nn.MSELoss()
        
    def mse_loss(self, inp, target):
        return torch.sum((inp - target)**2) / inp.data.nelement()    
        
        
    def forward(self, batch_samples, targets, use_training=True):
        primecaps_out = self.PrimeCaps_layer(self.Conv_layer(batch_samples))
        digitcaps_out = self.DigitCaps_layer.forward(primecaps_out)
        reconstructions, index = self.Decoder(digitcaps_out, targets, use_training)
        return digitcaps_out, reconstructions, index
    
    def reconstruction_loss(self, batch_samples, reconstructions):
        loss = self.mse_loss(batch_samples.view(batch_samples.size(0), -1), reconstructions.view(batch_samples.size(0), 784))
        return 0.0005 * loss
    
    def margin_loss(self, targets, digitcaps_out):
        batch_size, classes = digitcaps_out.size(0), digitcaps_out.size(1)
        onehot_target = (torch.eye(classes).cuda()).index_select(dim=0, index = targets)
        v_k = torch.sqrt((digitcaps_out ** 2).sum(dim=2))
        loss = (onehot_target * F.relu(0.9 - v_k)**2) + (0.5 * (1-onehot_target) * F.relu(v_k - 0.1)**2)
        loss = (loss.sum(dim=1)).mean()
        return loss
    
    def loss(self, batch_samples, reconstruction, targets, digitcaps_out):
        loss = self.reconstruction_loss(batch_samples, reconstructions) + self.margin_loss(targets, digitcaps_out)
        return loss


# In[18]:

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
CapsNet = CapsuleNetwork()
CapsNet = CapsNet.to(device)
optimizer = optim.Adam(CapsNet.parameters())


# In[19]:

#training
epochs = 5
for epoch in range(epochs):
    train_loss = 0
    test_loss = 0
    CapsNet.train()
    for batch_id, (batch, target) in enumerate(data.trainset_loader):
        batch, target = Variable(batch), Variable(target)
        if torch.cuda.is_available():
            batch, target = batch.cuda(), target.cuda()
        optimizer.zero_grad()
        digitcaps_out, reconstructions, index = CapsNet.forward(batch, target, use_training = True)
        loss = CapsNet.loss(batch, reconstructions, target, digitcaps_out)
        loss.backward()
        optimizer.step()
    
        train_loss += loss
        
        if batch_id % 100 == 0:
            print ("epoch: %f  batch: %d  Training accuracy: %f" %((epoch+1), batch_id+1, ((index==target).float()).sum()/batch.size(0) ))
    
    print ("Train Loss: %f" %(train_loss/len(data.trainset_loader)))
        
    test_accuracy = 0     
    for batch_id, (batch, target) in enumerate(data.testset_loader):
        batch, target = Variable(batch), Variable(target)
        if torch.cuda.is_available():
            batch, target = batch.cuda(), target.cuda()
        digitcaps_out, reconstructions, index = CapsNet.forward(batch, target, use_training = False)
        test_accuracy +=  ((index==target).float()).sum()/batch.size(0)
        #loss = CapsNet.loss(batch, reconstructions, target, digitcaps_out)
        #test_loss += loss
        #if batch_id % 100 == 0:
        print ("epoch: %f  batch: %d  Test accuracy: %f"  %(epoch+1, batch_id+1, ((index==target).float()).sum()/batch.size(0) ))
            
    print ("Test Accuracy: %f" %(test_accuracy/len(data.testset_loader)))    


# In[ ]:




# In[20]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



