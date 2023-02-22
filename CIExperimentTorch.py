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
class Embedder(torch.nn.Module):
     def __init__(self, input_dim, output_dim):
         super(Embedder, self).__init__()
         self.flatten = torch.nn.Flatten()
         self.linear_relu_stack = torch.nn.Sequential(
             torch.nn.Linear(input_dim, 512),
             torch.nn.ReLU(),
             torch.nn.Linear(512, 1024),
             torch.nn.ReLU(),
             torch.nn.Linear(1024, 512),
             torch.nn.ReLU(),
             torch.nn.Linear(512, output_dim))
     def forward(self, x):
         outputs = torch.sigmoid(self.linear_relu_stack(self.flatten(x))) * torch.tensor(0.98).to(device) + torch.tensor(0.01).to(device)
         return outputs

class Classifier(torch.nn.Module):
     def __init__(self, input_dim, output_dim):
         super(Classifier, self).__init__()
         self.linear_relu_stack = torch.nn.Sequential(
             torch.nn.Linear(input_dim, 10 * input_dim),
             torch.nn.ReLU(),
             torch.nn.Linear(10 * input_dim, 10 * input_dim),
             torch.nn.ReLU(),
             torch.nn.Linear(10 * input_dim, output_dim))
     def forward(self, x):
         outputs = torch.softmax(self.linear_relu_stack(x), dim=1) * 0.98 + 0.01
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
train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

# ----------------------------------------------------
#               Models and Hyperparameters
# ----------------------------------------------------

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using {device} device")

output_dim = 50
aliceModel = Embedder(28 * 28, output_dim).to(device)
bobModel = Embedder(28*28,output_dim).to(device)
max_bits_possible = (output_dim * 2) ** 2

eveModel = Classifier(output_dim * 2, 10).to(device)
classifier_loss_fn = torch.nn.CrossEntropyLoss()
classifier_optimizer = torch.optim.Adam(eveModel.parameters(), lr = 0.01)

ineffBalance = output_dim * 2 - 1
epochs = 50
lr = 0.005

def test(dataloader, alice, bob, eve, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    alice.eval()
    bob.eval()
    eve.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            embeddings = torch.cat([alice(X), bob(X)], dim = 1)
            pred = eve(embeddings)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f}")



# ----------------------------------------------------
#               Encoding Training Loop
# ----------------------------------------------------

payoff_tracking = torch.zeros(epochs, output_dim * 2, requires_grad=False)
start_time = time.time()
for epoch in range(epochs):
    numBatches = torch.tensor(len(train_dataloader)).to(device) #assuming of equal size!
    totalSize = torch.tensor(len(train_dataloader.dataset)).to(device)
    batch_payoffs = torch.zeros(numBatches, output_dim * 2).to(device)
    epoch_joint_AB = torch.zeros(output_dim * 2, output_dim * 2).to(device)
    epoch_joint_ABc = torch.zeros_like(epoch_joint_AB).to(device)
    epoch_outputs = torch.zeros(totalSize, output_dim * 2).to(device)
    for batchInd, (X, Y ) in enumerate(train_dataloader):
        alice = aliceModel(X.to(device))
        bob = bobModel(X.to(device))
        
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
        torch.nn.utils.clip_grad_norm_(aliceModel.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(bobModel.parameters(), 1.0)


        
        # Gradient ascend Alice and Bob
        with torch.no_grad():
            for model in [aliceModel, bobModel]:
                for p in model.parameters():
                    p += p.grad * lr
                model.zero_grad()
        
        ### Train classifier
        classifier_optimizer.zero_grad()
        classifier_outputs = eveModel(outputs.detach())
        classifier_loss = classifier_loss_fn(classifier_outputs, Y.to(device))
        
        classifier_loss.backward()
        classifier_optimizer.step()
        
        
        if batchInd % 100 == 99:
            sampleCount = (batchInd + 1) * batch_size
            print(f"socialgood: {socialgood.item():>7f}, loss: {classifier_loss:>7f}  [{sampleCount:>5d}/{totalSize:>5d}]")
    
    epoch_payoffs = codinginefficiency((epoch_joint_AB, epoch_joint_ABc), epoch_outputs)
    epoch_social_good = torch.sum(epoch_payoffs)
    payoff_tracking[epoch, :]= epoch_payoffs
    
    print(f"\n EPOCH: {epoch}    \n\
          \t average social good: {epoch_social_good} bits    \n\
          \t percent of capacity: {epoch_social_good / max_bits_possible * 100} %\n")

    test(test_dataloader, aliceModel, bobModel, eveModel, classifier_loss_fn)
print("Took ", time.time() - start_time)
# ----------------------------------------------------
#         Compute Embeddings from Alice and Bob
# ----------------------------------------------------

# # Precompute embeddings
# no_shuffe_train_data_loader = DataLoader(training_data, batch_size=batch_size)
# no_shuffle_test_data_loader = DataLoader(test_data, batch_size = batch_size)
# embedded_training_data = torch.zeros(len(training_data), output_dim * 2)
# embedded_test_data = torch.zeros(len(test_data), output_dim * 2)
# for batchInd, (X,_) in enumerate(no_shuffe_train_data_loader):
#     embedding =  torch.cat([aliceModel(X), bobModel(X)], dim=1)
#     embedded_training_data[(batchInd * batch_size):((batchInd + 1) * batch_size),:] = embedding

# for batchInd, (X,_) in enumerate(no_shuffle_test_data_loader):
#     embedding =  torch.cat([aliceModel(X), bobModel(X)], dim=1)
#     embedded_test_data[(batchInd * batch_size):((batchInd+ 1) * batch_size),:] = embedding

# embedding_data_loader = 



        
    

