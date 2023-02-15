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
syntheticData = synthesizeData([-2, 2, 6, -6], [0.5, 0.5, 0.5, 0.5], 1)
class LogisticRegression(torch.nn.Module):
     def __init__(self, input_dim, output_dim):
         super(LogisticRegression, self).__init__()
         self.linear_relu_stack = torch.nn.Sequential(
             torch.nn.Linear(1, 20),
             torch.nn.ReLU(),
             torch.nn.Linear(20, 20),
             torch.nn.ReLU(),
             torch.nn.Linear(20, 1))
     def forward(self, x):
         outputs = torch.sigmoid(self.linear_relu_stack(x)) * 0.98 + 0.01
         return outputs
     
     
aliceModel = LogisticRegression(1, 1)
bobModel = LogisticRegression(1,1)
epochs = 1000
lr = 0.01

def codinginefficiency(joint, a, b):
    M = len(a)
    pA = torch.sum(joint, 1)[0]
    pB = torch.sum(joint, 0)[0]
    
    # Conditional probabilities
    pACondB = joint[0, 0] / pB
    pACondBc = joint[0,1] / (1-pB)
    pBCondA = joint[0, 0] / pA
    pBCondAc = joint[1,0] / (1-pA)
    
    pACondb_k = pACondB * b + pACondBc * (1-b)
    pBConda_k = pBCondA * a + pBCondAc * (1-a)
    
    aIneffs = 1/M * torch.sum(L(a, pA) + L(a, pACondb_k) - 2 * H(a))
    bIneffs = 1/M * torch.sum(L(b, pB) + L(b, pBConda_k) - 2 * H(b))
    
    # Torch autograd can't handle things like torch.tensor([aIneffs, bIneffs])
    # so have to get a bit elaborite
    return aIneffs * torch.tensor([1,0]) +  bIneffs * torch.tensor([0,1])
    

def jointDistribution(a,b):
    M = len(a)
    AB = a.t() @ b / M
    ABc = a.t() @ (1-b) / M
    AcB = (1-a).t() @ b / M
    AcBc = (1-a).t() @ (1-b) / M
    
    joint = torch.tensor([[1, 0], [0, 0]]) * AB + \
           torch.tensor([[0, 1], [0, 0]]) * ABc + \
           torch.tensor([[0, 0], [1, 0]]) * AcB + \
           torch.tensor([[0, 0], [0, 1]]) * AcBc
    return joint
    
def H(p):
    return -(p * torch.log(p) + (1-p) * torch.log(1-p))

def L(a,p):
    return -(a * torch.log(p) + (1-a) * torch.log(1-p))

payoff_tracking = torch.zeros(epochs, 2, requires_grad=False)
start_time = time.time()
for epoch in range(epochs):
    alice = aliceModel(syntheticData)
    bob = bobModel(syntheticData)
    alice.retain_grad()
    bob.retain_grad()
    
    joint = jointDistribution(alice, bob)
    
    losses = codinginefficiency(joint, alice, bob)
    payoff_tracking[epoch,:] = losses.detach()
    aliceLoss = losses[0]
    bobLoss = losses[1]
    
    # Calculate Alice grads
    bobModel.requires_grad_(False)
    aliceLoss.backward(retain_graph = True)
    
    # Calculate Bob grads
    bobModel.requires_grad_(True)
    aliceModel.requires_grad_(False)
    bobLoss.backward()
    
    # Restore requires_grad to defauts
    aliceModel.requires_grad_(True)
    bobModel.requires_grad_(True)
    
    # Gradient ascend Alice and Bob
    with torch.no_grad():
        for p in aliceModel.parameters():
            scale = 1 / torch.norm(p.grad)
            if torch.any(torch.isnan(scale)):
                scale = 1
            p += p.grad * scale * lr + (torch.rand_like(p) -0.5) * lr**2
            
        for p in bobModel.parameters():
            scale = 1 / torch.norm(p.grad)
            if torch.any(torch.isnan(scale)):
                scale = 1
            p += p.grad * scale * lr + (torch.rand_like(p) -0.5) * lr**2
            
        aliceModel.zero_grad()
        bobModel.zero_grad()

print("Took ", time.time() - start_time)

        
    

