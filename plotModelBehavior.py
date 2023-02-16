#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 13:46:04 2023

@author: gavin
"""
import matplotlib.pyplot as plt
plt.plot(torch.linspace(1, epochs, epochs), payoff_tracking[:,0], 
          torch.linspace(1,epochs, epochs), payoff_tracking[:,1])
fig, ax = plt.subplots()
ax.scatter(dataset.tensors[0], aliceModel(dataset.tensors[0]).detach(), label="Alice")
ax.scatter(dataset.tensors[0], bobModel(dataset.tensors[0]).detach(), label="Bob")
ax.legend()