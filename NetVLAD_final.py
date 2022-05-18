#!/usr/bin/env python
# coding: utf-8

# In[1]:


cd /workspace/vl


# In[2]:


get_ipython().run_line_magic('pwd', '')


# In[3]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision

from torchvision.models import resnet18,vgg16

torch.manual_seed(777)


# In[4]:


class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# In[5]:


class NetVLAD(nn.Module):
    """NetVLAD layer implementation"""

    def __init__(self, num_clusters=12, dim=128, alpha=100.0,normalize_input=True):
        """
        Args:
            num_clusters : int
                The number of clusters
            dim : int
                Dimension of descriptors
            alpha : float
                Parameter of initialization. Larger value is harder assignment.
            normalize_input : bool
                If true, descriptor-wise L2 normalization is applied to input.
        """
        super(NetVLAD, self).__init__()
        self.num_clusters = num_clusters
        self.dim = dim
        self.alpha = alpha
        self.normalize_input = normalize_input
        self.conv = nn.Conv2d(dim, num_clusters, kernel_size=(1, 1), bias=False)
        self.centroids = nn.Parameter(torch.rand(num_clusters, dim))
        self._init_params()

    def _init_params(self):
        self.conv.weight = nn.Parameter(
            (2.0 * self.alpha * self.centroids).unsqueeze(-1).unsqueeze(-1)
        )

    def forward(self, x):

        N, C = x.shape[:2]

        if self.normalize_input:
            x = F.normalize(x, p=2, dim=1)  # across descriptor dim

        # soft-assignment
        soft_assign = self.conv(x).view(N, self.num_clusters, -1)
        soft_assign = F.softmax(soft_assign, dim=1)

        x_flatten = x.view(N, C, -1)
        
        # calculate residuals to each clusters
        vlad = torch.zeros([N, self.num_clusters, C], dtype=x.dtype, layout=x.layout, device=x.device)
        for C in range(self.num_clusters): # slower than non-looped, but lower memory usage 
            residual = x_flatten.unsqueeze(0).permute(1, 0, 2, 3) -                     self.centroids[C:C+1, :].expand(x_flatten.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0)
            residual *= soft_assign[:,C:C+1,:].unsqueeze(2)
            vlad[:,C:C+1,:] = residual.sum(dim=-1)

        vlad = F.normalize(vlad, p=2, dim=2)  # intra-normalization
        vlad = vlad.view(x.size(0), -1)  # flatten
        vlad = F.normalize(vlad, p=2, dim=1)  # L2 normalize

        return vlad


# In[6]:


# VGG 16
encoder = vgg16(pretrained=True) # pretrained
layers = list(encoder.features.children())[:-2]

for l in layers[:-5]: 
    for p in l.parameters():
        p.requires_grad = False

model = nn.Module() 

encoder = nn.Sequential(*layers)
model.add_module('encoder', encoder)

dim = list(encoder.parameters())[-1].shape[0]  # last channels (512)

# Define model for embedding
net_vlad = NetVLAD(num_clusters=16, dim=dim)
model.add_module('pool', net_vlad)

model = model.cuda()


# In[7]:


# Pretrained Model Load(Pittsburgh 30k)
# NetVLAD는 피츠버그 데이터셋을 이용해 학습하였음, 하지만 데이터셋의 용량이 크고 학습에 필요한 하드웨어 리소스 문제로 본 챌린지에서는 사전학습된 체크포인트를 제공함

load_model = torch.load('./pittsburgh_checkpoint.pth.tar')
model.load_state_dict(load_model['state_dict'],strict = False)


# In[8]:


import torch.utils.data as data
import torchvision.transforms as transforms

from random import choice
from os.path import join, exists
from collections import namedtuple
from scipy.io import loadmat
from PIL import Image
from sklearn.neighbors import NearestNeighbors
from IPython.display import display

root_dir = '/workspace/PyTorch/localdata/NaverLabs_Dataset' # 필요시 경로수정
dataset_dir = '/workspace/PyTorch/localdata/NaverLabs_Dataset/pose.mat'

def parse_dbStruct(path):
    mat = loadmat(path)

    matStruct = mat['pose'][0]

    dataset = 'dataset'

    whichSet = 'VPR'

    dbImage = matStruct[0]
    locDb = matStruct[1]

    qImage = matStruct[2]
    locQ = matStruct[3]

    numDb = matStruct[4].item()
    numQ = matStruct[5].item()

    posDistThr = matStruct[6].item()
    posDistSqThr = matStruct[7].item()

    return dbStruct(whichSet, dataset, dbImage, locDb, qImage, 
            locQ, numDb, numQ, posDistThr, 
            posDistSqThr)
  
dbStruct = namedtuple('pose', ['whichSet', 'dataset',
  'dbImage', 'locDb', 'qImage', 'locQ', 'numDb', 'numQ',
  'posDistThr', 'posDistSqThr'])
  
class BerlinDataset(data.Dataset) :
  
  def __init__(self,condition='train') :
    self.dbStruct = parse_dbStruct(dataset_dir) # 필요시 경로수정
    self.input_transform = transforms.Compose([
        transforms.Resize([640,480]),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
      ])
    
    self.condition = condition

    if self.condition == 'train' :
      self.images = [join(root_dir, dbIm.replace(' ','')) for dbIm in self.dbStruct.dbImage]
    elif self.condition == 'test' :
      self.images = [join(root_dir, qIm.replace(' ','')) for qIm in self.dbStruct.qImage]
    else :
      self.images = [join(root_dir, dbIm.replace(' ','')) for dbIm in self.dbStruct.dbImage]
    
    self.positives = None
    self.distances = None

    self.getPositives()
  
  def __getitem__(self, index):

      if self.condition == 'train' :
        img = Image.open(self.images[index])
        img = self.input_transform(img)

        pos_list = self.positives[index].tolist()
        pos_list.remove(index)
        pos_index = self.positives[index][np.random.randint(0,len(self.positives[index]))]
        pos_img = Image.open(self.images[pos_index])
        pos_img = self.input_transform(pos_img)

        pos_list = pos_list + [index]
        neg_index = choice([i for i in range(len(self.images)) if i not in pos_list])
        neg_img = Image.open(self.images[neg_index])
        neg_img = self.input_transform(neg_img)
        
        img = torch.stack([img,pos_img,neg_img],dim=0)
        label = torch.Tensor([0, 0, 1])

        return img, label

      elif self.condition == 'test' :
        img = Image.open(self.images[index])
        img = self.input_transform(img)

        return img
      
      else :
        img = Image.open(self.images[index])
        img = self.input_transform(img)

        return img


  def __len__(self):
      return len(self.images)

  def getPositives(self):
      # positives for evaluation are those within trivial threshold range
      #fit NN to find them, search by radius
      if  self.condition == 'train' :
          knn = NearestNeighbors(n_jobs=1)
          knn.fit(self.dbStruct.locDb)

          self.distances, self.positives = knn.radius_neighbors(self.dbStruct.locDb,radius=self.dbStruct.posDistThr)
      else :
          knn = NearestNeighbors(n_jobs=1)
          knn.fit(self.dbStruct.locDb)

          self.distances, self.positives = knn.radius_neighbors(self.dbStruct.locQ,
                  radius=self.dbStruct.posDistThr)
            
      return self.positives


# In[9]:


train_dataset = BerlinDataset(condition="train")
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1,shuffle=True,num_workers=0)   


# In[10]:


epochs = 5
global_batch_size = 4
lr = 0.00001
momentum = 0.9
weightDecay = 0.001
losses = AverageMeter()
best_loss = 100.0
margin = 0.1 

criterion = nn.TripletMarginLoss(margin=margin**0.5, p=2, reduction='sum').cuda()
optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, momentum=momentum, weight_decay=weightDecay)

model.train()

count = 0

for epoch in range(epochs):
  for batch_idx, (train_image,train_label) in enumerate(train_loader) :
    output_train = model.encoder(train_image.squeeze().cuda()) # vgg16 통과
    output_train = model.pool(output_train) # netvlad 통과
    triplet_loss = criterion(output_train[0].reshape(1,-1),output_train[1].reshape(1,-1),output_train[2].reshape(1,-1))
        
    #이미지를 순서대로 뽑아 세개씩 연속적으로 비교
    #writer.add_scalar('Loss/train', losses.avg, batch_idx)

    if batch_idx == 0 :
      optimizer.zero_grad() # gradient 초기화

    triplet_loss.backward(retain_graph=True) # 역전파
    losses.update(triplet_loss.item()) # loss 업데이트
    
    if (batch_idx +1) % global_batch_size == 0 :# 모르겟
      for name, p in model.named_parameters():
        if p.requires_grad:
          p.grad /= global_batch_size
    
        optimizer.step()
        optimizer.zero_grad()

    if batch_idx % 20 == 0 and batch_idx != 0:
        print('epoch : {}, batch_idx  : {}, triplet_loss : {}'.format(epoch,batch_idx,losses.avg))
    
    
    if best_loss > losses.avg : # 최적의 model 발견시 best_model 업데이트
        best_save_name = 'best_model.pt'
        best_path = F"./ckpt/{best_save_name}" 
        torch.save(model.state_dict(), best_path)
        count = 0
    else:
        count += 1
        if count >= 1000 :
            break
            
  model_save_name = 'model_{:02d}.pt'.format(epoch) # epoch마다 model 저장
  path = F"./ckpt/{model_save_name}" 
  torch.save(model.state_dict(), path)
#writer.close()

