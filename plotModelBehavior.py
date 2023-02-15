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
ax.scatter(torch.linspace(1, M, M), alice.detach(), label="Alice")
ax.scatter(torch.linspace(1, M, M), bob.detach(), label="Bob")
ax.legend()