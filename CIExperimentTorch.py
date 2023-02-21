#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 16:56:29 2023

@author: gavin
"""
import torch
import time

def synthesizeData(means, stdDevs, samps):
    dists = [torch.randn(samps, 1) * stdDev + mean for (mean, stdDev) in zip(means, stdDevs)]
    return torch.cat(dists, dim=0)
dataset = torch.utils.data.TensorDataset(synthesizeData([-2, 2, 6, -6], [0.5, 0.5, 0.5, 0.5], 128))
dataloader = torch.utils.data.DataLoader(dataset,  batch_size=32, shuffle=True)
class Classifier(torch.nn.Module):
     def __init__(self, input_dim, output_dim):
         super(Classifier, self).__init__()
         self.linear_relu_stack = torch.nn.Sequential(
             torch.nn.Linear(1, 20),
             torch.nn.ReLU(),
             torch.nn.Linear(20, 20),
             torch.nn.ReLU(),
             torch.nn.Linear(20, 1))
     def forward(self, x):
         outputs = torch.sigmoid(self.linear_relu_stack(x)) * 0.98 + 0.01
         return outputs
     
     
aliceModel = Classifier(1, 1)
bobModel = Classifier(1,1)
ineffBalance = 1.0
epochs = 400
lr = 0.01

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
    return -(p * torch.log(p) + (1-p) * torch.log(1-p))

def L(a,p):
    return -(a * torch.log(p) + (1-a) * torch.log(1-p))

payoff_tracking = torch.zeros(epochs, 2, requires_grad=False)
start_time = time.time()
for epoch in range(epochs):
    numBatches = len(dataloader) #assuming of equal size!
    batch_payoffs = torch.zeros(numBatches, 2)
    for (batchInd, batchedTensors) in enumerate(dataloader):
        batch = batchedTensors[0]
        alice = aliceModel(batch)
        bob = bobModel(batch)
        
        outputs = torch.cat([alice, bob], dim=1)
        
        joint = jointDistribution(outputs)
        
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
    
    payoff_tracking[epoch, :]= torch.mean(batch_payoffs, dim=0)

print("Took ", time.time() - start_time)

        
    

