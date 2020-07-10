# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 22:09:16 2020

@author: Gold
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns

fig=plt.figure()
ax=fig.add_subplot()
a=np.random.normal(0,0.1,100000)
b=np.random.chisquare(2,100000)
s=0

def plot(data):
    global ax, s
    ax.set_xlim(-1.1)
    ax.set_ylim(0,8)
    ax=sns.kdeplot(a[s:s+1000],b[s:s+1000], bw=0.1, shade=True, n_levels=30, shade_lowest=False, cmap='hot',vmin=-0.5)
    s+=100
    print("/b"*10,(s*100)/10000-1,"%", end=" ")
ani=animation.FuncAnimation(fig, plot, interval=100, frames=100)
ani.save("plot.gif", writer="imagemagic")

