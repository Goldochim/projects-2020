# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 13:44:46 2020

@author: Gold
"""
from mpl_toolkits.mplot3d.axes3d import Axes3D
import matplotlib.pyplot as plt
#imports specific to the plots in this example
import numpy as np
from matplotlib import cm
from mpl_toolkits.mplot3d.axes3d import get_test_data

#Twce as wide as it is tall
fig=plt.figure(figsize=plt.figaspect(0.5))

#first subplot
ax=fig.add_subplot(1,2,1,projection='3d')
x=np.arange(-5,5,0.25)
y=np.arange(-5,5,0.25)
x,y=np.meshgrid(x,y)
R=np.sqrt(x**2+y**2)
z=np.sin(R)
surf=ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
ax.set_zlim3d(-1.01,1.01)

fig.colorbar(surf,shrink=0.5, aspect=10)

#second subplot
ax=fig.add_subplot(1,2,2,projection='3d')
x,y,z=get_test_data(0.05)
ax.plot_wireframe(x,y,z, rstride=10, cstride=10)
plt.show()

