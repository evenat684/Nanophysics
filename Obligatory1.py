import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider
#Expression we want to plot
#E(kx,ky) = pm sqrt(M^2 + gamma^2(1+4cos(3kxa/2)cos(sqrt(3)kya/2) 4cos^2(sqrt(3)kya/2)))
#Switching variables to x and y where x = kxa and y = kya
#Meshgrid
N = 10000

X = np.linspace(-2.45,2.45, N)
Y = X.copy()
XX,YY = np.meshgrid(X, Y)
def getE(XX,YY, pos, gamma, M):
    if pos:
        E = np.sqrt(M**2 + (gamma**2)*(1+4*np.cos(3*XX/2)*np.cos(np.sqrt(3)*YY/2) + 4*np.cos(np.sqrt(3)*YY/2)*np.cos(np.sqrt(3)*YY/2)))
    else:
        E = -np.sqrt(M**2 + (gamma**2)*(1+ 4*np.cos(3*XX/2)*np.cos(np.sqrt(3)*YY/2) + 4*np.cos(np.sqrt(3)*YY/2)*np.cos(np.sqrt(3)*YY/2)))
    return E

def updateM(val):
    global M
    M = M_sl.val
    E_pos = getE(XX,YY,True, gamma,M)
    E_neg = getE(XX,YY,False, gamma,M)
    ax.clear()
    
    #u.set_3d_properties(E_pos[:])
    u = ax.plot_surface(XX,YY,E_pos,alpha = 1, cmap = "rainbow")
    #u2.set_3d_properties(E_neg[:])
    u2 = ax.plot_surface(XX,YY,E_neg,alpha = 1, cmap = "rainbow")

def updateGamma(val):
    global gamma
    gamma = gam_sl.val
    E_pos = getE(XX,YY,True, gamma,M)
    E_neg = getE(XX,YY,False, gamma,M)
    ax.clear()
    
    #u.set_3d_properties(E_pos[:])
    u = ax.plot_surface(XX,YY,E_pos,alpha = 1, cmap = "rainbow")
    #u2.set_3d_properties(E_neg[:])
    u2 = ax.plot_surface(XX,YY,E_neg,alpha = 1, cmap = "rainbow")
gamma = 0.5
M = 0.2
E_pos = getE(XX,YY,True, gamma, M)
E_neg = getE(XX,YY,False, gamma, M)
fig = plt.figure()
ax = fig.gca(projection = "3d")
u = ax.plot_surface(XX,YY,E_pos,alpha = 1, cmap = "rainbow")
u2 = ax.plot_surface(XX,YY,E_neg,alpha = 1, cmap = "rainbow")
ax.set_xlabel('$k_x a$')
ax.set_ylabel('$k_y a$')
ax.set_zlabel('$E$')
ax.set_title('Dispersion relation $E(k_x a, k_y a)$ for $\gamma$ %.2f and $M$ %.2f' %(gamma,M))
#M_sl = Slider(fig.add_subplot(50,1,50),  valmin = 0, valmax = 1, valinit = 0, label = "M")
#M_sl.on_changed(updateM)
#gam_sl = Slider(fig.add_subplot(1,50,50),  valmin = -1, valmax = 0, valinit = -0.5, label = '$\gamma$', orientation="vertical")
#gam_sl.on_changed(updateGamma)
plt.show()