import numpy as np 
import scipy.linalg
import numba as nb
from tqdm import tqdm
import matplotlib.pyplot as plt

@nb.jit
def fastInverseMatrix(M):
    return np.linalg.inv(M)

@nb.jit
def scatteringMatrixCombo(s_1, s_2):
    #Get N from the shape of s_1(2Nx2N) 
    N = int(s_1.shape[0]/2)

    #Set all elements first
    #S_1
    t_1 = s_1[:N,:N]
    r_1p = s_1[:N,N:]
    r_1 = s_1[N:,:N]
    t_1p = s_1[N:,N:]
    
    #S_2
    t_2 = s_2[:N,:N]
    r_2p = s_2[:N,N:]
    r_2 = s_2[N:,:N]
    t_2p = s_2[N:,N:]

    #Find the inverse matrix that is used in all expressions
    inverse_mat = fastInverseMatrix(np.identity(N) + r_1p*r_2)

    #Define result (the out array)
    result = np.empty_like(s_1)
    
    #Apply formula from my pdf
    result[:N,:N] = t_1 * t_2 * inverse_mat
    result[:N,N:] = r_2p + t_2p*t_2*r_1p*inverse_mat
    result[N:,:N] = r_1 + t_1*t_1p*r_2*inverse_mat
    result[N:,N:] = t_1p * t_2p*inverse_mat

    
    return result

def getTransmission(s):
    #Extract upper left block
    t_s = s[:int(s.shape[0]/2),:int(s.shape[0]/2)]
    
    #Since alpha = 0 we only have diagonals.
    T = np.sum(np.abs(t_s)**2, axis = 0) 
    return T

def getConductance(T):
    #Conductance is G = G_Q*sum(T_n), G_q = 2e^2/h
    G_q = 2*(1.6*10e-19)**2/(6.62607015*10e-34)
    #This is what we got from the balistic wire as well when
    #we used fermi distributions and stuff.
    G = G_q*np.sum(T)
    return G

def ScatteringCalculator(P,s):
    scattering_result = np.empty_like(s)
    for i in range(P.shape[0]-1,-1,-1):
        if i== P.shape[0]-1:
            scattering_result = scatteringMatrixCombo(s,P[i])
        elif i == 0:
            scattering_result = scatteringMatrixCombo(P[i],scattering_result)
        else:
            scattering_result = scatteringMatrixCombo(P[i],scattering_result)
            scattering_result = scatteringMatrixCombo(s,scattering_result)

    return scattering_result


def km(m):
    return np.sqrt(30.5**2 -m**2)

def getPMatrix(dists, channels = 30):
    #get the m values to create ks
    m = np.arange(1,channels+1, dtype = np.complex128)
    
    #Find a list of k_ms
    k_m = km(m)

    #Generate a N + 1,channels matrix for P
    p_n = np.zeros((dists.shape[0], 2*channels, 2*channels), dtype = np.complex64)
    
    #Loop through channels and set each element along the array
    for i in range(channels):
       p_n[:,i,i] = np.exp(1j*dists[:]*k_m[i])
       p_n[:,i + channels,i + channels]  = np.exp(1j*dists[:]*k_m[i])
    
    return p_n


def main(N, alpha = 0.0, channels = 30, incoherent = False):
    
    #Generate random points
    x_n = np.random.uniform(0, 60000, N)
    x_n.sort()

    #Generate N+1 dists, set dtyp to complex64
    dists = np.zeros(N+1, dtype = np.complex128)

    #Find dists from x_array
    dists[1:-1] = x_n[1:] - x_n[0:-1]
    
    #Set first and last element(not included in the above)
    dists[0], dists[-1] = x_n[0], 60000-x_n[-1]

    #Generate all the Ps in a list P[0] is 1st, P[1] second and so on
    P = getPMatrix(dists, channels)
    
    #Set s with alpha and convert to regular matrix
    s = scipy.linalg.expm(1j*alpha*np.ones((2*channels,2*channels)))
    
    #square them if incoherent
    if incoherent:
        s = np.abs(s)**2
        P = np.abs(P)**2

    s_tot = ScatteringCalculator(P,s)
    
    return s_tot

def taskf():
    N = 600
    channels = 30
    
    #Find the total scattering matrix for given params
    s_tot = main(N, channels = channels)

    T = getTransmission(s_tot)

    #Conductance is G = G_Q*sum(T_n), G_q = 2e^2/h
    G_q = 2*(1.6*10e-19)**2/(6.62607015*10e-34)
    
    #This is what we got from the balistic wire as well when
    #we used fermi distributions and stuff.
    G = G_q*np.sum(T)
    return G.real

def taskg(make = True, fname_conductance = ""):
    #Set alpha
    alpha = 0.035
    #This is just an option if you want to make the array each time
    #or just load it from file
    if make:
        #Array to contain conductances
        cond_list = np.zeros(200)
        
        #Loop through all the 200 times
        for i in tqdm(range(200)):
            #Get conductance from each and add to cond_list
            s = main(600,alpha = alpha)
            T = getTransmission(s)
            G = getConductance(T)
            cond_list[i] = G.real
    
        #Save it here so you don't need to run evry time
        np.savetxt(fname_conductance, cond_list)


    #Read in numbers
    conductances = np.loadtxt(fname_conductance)

    #Getting variance and mean
    variance = np.var(conductances)
    mean = np.average(conductances)

    X = np.arange(1,201,1)

    #Plotting the values
    plt.figure()
    plt.plot(X,mean*np.ones(200), color = "r", label = 'Mean: $\mu$')
    plt.title('Conductances for 200 different configurations with \n $\mu = {:.4e}$ and $\sigma^2 = {:.4e}$' .format(mean,variance))
    plt.scatter(X,conductances, label = "Conductances")
    plt.ylabel("Resulting conductance")
    plt.xlabel("Simulation")
    plt.legend()
    plt.show()
    
def taskh(make = True, fname_conductance = ""):
    #Set alpha
    alpha = 0.035
    #This is just an option if you want to make the array each time
    #or just load it from file
    
    if make:
        #Array to contain conductances
        cond_list = np.zeros(200)
        for i in tqdm(range(200)):
            #Get conductance from each and add to cond_list
            s = main(600,alpha = alpha, incoherent=True)
            T = np.sum(s[:int(s.shape[0]/2),:int(s.shape[0]/2)], axis = 0)
            G = getConductance(T)           
            cond_list[i] = G.real
    
        #Save it here so you don't need to run every time
        np.savetxt(fname_conductance, cond_list)

    #Read in numbers
    conductances = np.loadtxt(fname_conductance)


    #Getting variance and mean
    variance = np.var(conductances)
    mean = np.average(conductances)
    X = np.arange(1,201,1)

    #Plotting the values
    plt.figure()
    plt.plot(X,mean*np.ones(200), color = "r", label = 'Mean: $\mu$')
    plt.title('Incoherent conductances for 200 different configurations with \n $\mu = {:.4e}$ and $\sigma^2 = {:.4e}$' .format(mean,variance))
    plt.scatter(X,conductances, label = "Conductances")
    plt.ylabel("Resulting conductance")
    plt.xlabel("Simulation")
    plt.legend()
    plt.show()

    return mean

taskg(make = True, fname_conductance="conductances_g.txt")
#mean = taskh(make = True, fname_conductance = "conductances_g_inch.txt")
#print(mean)