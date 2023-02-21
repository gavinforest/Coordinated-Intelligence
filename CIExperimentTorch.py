#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 16:56:29 2023

@author: gavin
"""
import torch
import time
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

# ---------------------------------
#   Local Functions and Classes
# ---------------------------------
class Classifier(torch.nn.Module):
     def __init__(self, input_dim, output_dim):
         super(Classifier, self).__init__()
         self.flatten = torch.nn.Flatten()
         self.linear_relu_stack = torch.nn.Sequential(
             torch.nn.Linear(input_dim, 512),
             torch.nn.ReLU(),
             torch.nn.Linear(512, 512),
             torch.nn.ReLU(),
             torch.nn.Linear(512, output_dim))
     def forward(self, x):
         outputs = torch.sigmoid(self.linear_relu_stack(self.flatten(x))) * 0.98 + 0.01
         return outputs

def codinginefficiency(joint, probabilityTensor):
    (M,N) = probabilityTensor.shape
    
    (AB, ABc) = joint
    
    unconditionals = torch.diagonal(AB)
    
    conditional = AB / unconditionals #unconditionals is a row vector
    conditionalComp = ABc / (1-unconditionals)
    
    expandedProbTensor = probabilityTensor.t().view(1,N,M).expand(N,N,M)
    sampleConditionals = conditional.view(N,N,1).expand(N,N,M) * expandedProbTensor + \
                         conditionalComp.view(N,N,1).expand(N,N,M) * (1-expandedProbTensor)
    
    # sampleConditionals(i,j,k) = P(logit i | logit j_k)
    # expandedProbTensor(i,j,k) = logit j_k
    
    transProbTensor = expandedProbTensor.transpose(0,1)
    # transProbTensor(i,j,k) = logit i_k
    pairwiseIneffs = L(transProbTensor, sampleConditionals) - H(transProbTensor)
    avgPairwiseIneffs = torch.mean(pairwiseIneffs, dim=2)
    avgPairwiseIneffs.fill_diagonal_(0) # no self pairwise inefficiences
    avgExternalIneffs = torch.mean(avgPairwiseIneffs, dim=1) * (N/(N-1)) #account for 0 diagonals
    
    selfIneffs = L(probabilityTensor, unconditionals) - H(probabilityTensor)
    avgSelfIneffs = torch.mean(selfIneffs, dim=0).t()
    
    ineffs = avgSelfIneffs + ineffBalance * avgExternalIneffs
    
    return ineffs.t()
    

def jointDistribution(probabilityTensor):
    # rows are samples and columns are logits
    M = probabilityTensor.shape[0]
    AB = probabilityTensor.t() @ probabilityTensor / M
    ABc = probabilityTensor.t() @ (1- probabilityTensor) / M
    
    # Correct for self-flips being perfectly correlated because they're the same
    # coin.
    AB.fill_diagonal_(0)
    ABc.fill_diagonal_(0)
    
    unconditionalProbabilities = torch.mean(probabilityTensor, dim=0)
    AB += torch.diag(unconditionalProbabilities)


    joint = (AB, ABc)
    return joint
    
def H(p):
    return -(p * torch.log2(p) + (1-p) * torch.log2(1-p))

def L(a,p):
    return -(a * torch.log2(p) + (1-a) * torch.log2(1-p))

# ----------------------------------------------------
#               Datasets and Data Loaders
# ----------------------------------------------------

# Download training data from open datasets.
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

# Download test data from open datasets.
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)
batch_size = 128
# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

# ----------------------------------------------------
#               Models and Hyperparameters
# ----------------------------------------------------

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using {device} device")

output_dim = 20
aliceModel = Classifier(28 * 28, output_dim).to(device)
bobModel = Classifier(28*28,output_dim).to(device)
max_bits_possible = (output_dim * 2) ** 2

ineffBalance = output_dim * 2 - 1
epochs = 5
lr = 0.01

# ----------------------------------------------------
#               Encoding Training Loop
# ----------------------------------------------------

payoff_tracking = torch.zeros(epochs, output_dim * 2, requires_grad=False)
start_time = time.time()
for epoch in range(epochs):
    numBatches = len(train_dataloader) #assuming of equal size!
    totalSize = len(train_dataloader.dataset)
    batch_payoffs = torch.zeros(numBatches, output_dim * 2)
    epoch_joint_AB = torch.zeros(output_dim * 2, output_dim * 2)
    epoch_joint_ABc = torch.zeros_like(epoch_joint_AB)
    epoch_outputs = torch.zeros(totalSize, output_dim * 2)
    for batchInd, (batch, _) in enumerate(train_dataloader):
        alice = aliceModel(batch)
        bob = bobModel(batch)
        
        outputs = torch.cat([alice, bob], dim=1)
        epoch_outputs[(batchInd * batch_size):((batchInd + 1) * batch_size), :] = outputs.detach()
        
        joint = jointDistribution(outputs)
        (AB, ABc) = joint
        epoch_joint_AB += AB.detach() / numBatches
        epoch_joint_ABc += ABc.detach() / numBatches
        
        payoffs = codinginefficiency(joint, outputs)
        batch_payoffs[batchInd,:] = payoffs.detach()
        socialgood = sum(payoffs)
        
        # Calculate social payoff gradients
        socialgood.backward()
        
        # Gradient ascend Alice and Bob
        with torch.no_grad():
            for model in [aliceModel, bobModel]:
                for p in model.parameters():
                    p += p.grad * lr
                model.zero_grad()
        
        if batchInd % 100 == 99:
            sampleCount = (batchInd + 1) * len(batch)
            print(f"socialgood: {socialgood.item():>7f}  [{sampleCount:>5d}/{totalSize:>5d}]")
    
    epoch_payoffs = codinginefficiency((epoch_joint_AB, epoch_joint_ABc), epoch_outputs)
    epoch_social_good = torch.sum(epoch_payoffs)
    payoff_tracking[epoch, :]= epoch_payoffs
    
    print(f"\n EPOCH: {epoch}    \n\
          \t average social good: {epoch_social_good} bits    \n\
          \t percent of capacity: {epoch_social_good / max_bits_possible * 100} %\n")

print("Took ", time.time() - start_time)

        
    

