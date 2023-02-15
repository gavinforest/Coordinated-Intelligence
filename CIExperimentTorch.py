#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 16:56:29 2023

@author: gavin
"""
import torch
import time

M = 4
# alice = torch.tensor([0.01, 0.01, 0.99, 0.99], dtype = torch.float64).t()
# bob = torch.tensor([0.01, 0.01, 0.99, 0.99], dtype = torch.float64).t()
alice = torch.rand(M,1)
bob = torch.rand(M,1)
epochs = 5000
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
    alice.requires_grad_(True)
    bob.requires_grad_(True)

    joint = jointDistribution(alice, bob)
    
    payoffs = codinginefficiency(joint, alice, bob)
    payoff_tracking[epoch,:] = payoffs.detach()
    alicePayoff = payoffs[0]
    bobPayoff = payoffs[1]
    
    
    # Calculate Alice grads
    # bob.requires_grad_(False)
    alicePayoff.backward(retain_graph = True)
    
    # Calculate Bob grads
    # bob.requires_grad_(True)
    # alice.requires_grad_(False)
    bobPayoff.backward()
    
    # Restore requires_grad to defauts
    alice.requires_grad_(True)
    bob.requires_grad_(True)
    
    # Gradient ascend Alice and Bob
    with torch.no_grad():
        alice = alice + alice.grad * lr + (torch.rand_like(alice) - 0.5) * lr
        alice = torch.minimum(torch.maximum(alice, torch.tensor(0.001)), torch.tensor(0.999))
        bob = bob + bob.grad * lr + (torch.rand_like(bob) -0.5) * lr
        bob = torch.minimum(torch.maximum(bob, torch.tensor(0.001)), torch.tensor(0.999))

print("Took ", time.time() - start_time)

        
    

