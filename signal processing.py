#%%
import numpy as np
import pandas as pd
import math
from scipy.stats import bernoulli
import scipy as sp
import os
import pygsp as g
import networkx as nx
import matplotlib.pyplot as plt
import time
import itertools as it
import pickle
#%%
#These are functions for generating graphs:
# Erdos-Renyi graphs
def gen_graph_er(n: int, p: float):
    if p<0.3:
        return nx.adjacency_matrix(nx.fast_gnp_random_graph(n,p))
    else:
        return nx.adjacency_matrix(nx.gnp_random_graph(n,p))
# Block graphs
def gen_graph_block(n:int,p:float,q:float):
    return nx.adjacency_matrix(nx.stochastic_block_model([n,n],[[p,q],[q,p]]))
# Small-world graphs
def gen_graph_sw(n:int,k:int,p:float):
    return nx.adjacency_matrix(nx.watts_strogatz_graph(n,k,p))
# Preferential attachment graphs
def gen_graph_pa(n:int,m:int):
    #n is the number of vertices
    #m is the number of edges each node is allowed
    return nx.adjacency_matrix(nx.barabasi_albert_graph(n,m))

# This function checks a graph for connectivity:
def checkcon(gp): # returns True if connected, False if disconnected
    c=np.linalg.matrix_power(gp,len(gp))+np.linalg.matrix_power(gp,len(gp)+1)
    for i in range(len(gp)**2):
        if c[i//len(gp),i%len(gp)]==0:
            return False
    return True

#%%
#We'd like to save and retrieve our matrices
os.chdir('C:\\Users\\gerke\\Documents\\VSCode\\Python')
def graphsave(gp, name: str):
    with open(name, "wb") as fp:
        pickle.dump(gp, fp)

def graphload(name: str):
    with open(name, "rb") as fp:
        ld=pickle.load(fp)
    return  ld

graph=graphload("test1")
Graph=g.graphs.Graph(graph)
#%%
# We'd like to generate smooth signals, lets start with the bandlimited cases:
def BL_sig_L(G, bandlim: int):
    G.compute_fourier_basis()
    eigenvectors=G.U.T
    #Select the n smallest eigenvectors, multiply with random value in [-1,1]
    return sum([(2*np.random.rand()-1)/2*eigenvectors[i,:] for i in range(min(bandlim,G.N))])

# Defining cost function for sample recovery, the cost is the relative MSE, called the loss function in the report.
def cost(x,recovered_x):
    return np.linalg.norm(x-recovered_x)**2/(np.linalg.norm(x)**2)

# These functions will help reconstruction by finding the sample operator and U_RV,U_RS:
def sample_nodes(sample_list): 
    #A sample list is an array of 1 when sampled, 0 if not
    #Makes I_TS from the provided set of sample nodes
    sample_op = np.array([sample_list for _ in range(sum(sample_list))])
    count=0
    for i in range(len(sample_op[0])):
        if sum(sample_op[:,i]):
            sample_op[:,i]=np.zeros(len(sample_op))
            sample_op[count,i]=1
            count+=1
    return sample_op

def recovery_matrices(G, bandlim, sample_op):
    VR=[1]*bandlim+[0]*(G.N-bandlim)
    U_VR=G.U@np.diag(VR)[:,0:bandlim]
    U_SR=sample_op@U_VR
    return U_VR,U_SR

# This will plot the sample selection for a graph G:
def plot_sample(G,sample,title="Plot of sample"):
    try:
        _=G.coords
    except:
        G.set_coordinates()
    highlight=[i for i in pd.DataFrame(sample).index if sample[i]!=0]
    G.plot(vertex_color=[.72]*G.N, limits=[0,1], highlight=highlight,title=title,backend="matplotlib");

# This will generate and plot a graph of a given type, size and parameters:
def plot_graph(type: str, size: int, p: float, **kwargs):
    # generate a beautiful graph:
    if type=="er":
        A=g.graphs.Graph(gen_graph_er(size,p))
        title="Example Erdos-Renyi graph"
    elif type=="block":
        q=kwargs.get("q", 0.01) #default value for q is 0.01
        A=g.graphs.Graph(gen_graph_block(size,p,q))
        title="Example block graph"
    elif type=="sw":
        k=kwargs.get("k", 4) #default value for k is 4
        A=g.graphs.Graph(gen_graph_sw(size,k,p))
        title="Example small-world graph"
    elif type=="pa":
        m=kwargs.get("m", 3) #default value for m is 3
        A=g.graphs.Graph(gen_graph_pa(size,m))
        title="Example preferential attachment graph"
    else:
        print("Invalid graph-type, try again :)")
        return None
    A.set_coordinates()
    if type=="sw":
        A.coords=np.array([[math.cos(2*t*math.pi/A.N),math.sin(2*t*math.pi/A.N)] for t in range(A.N)])
    fig, ax1=plt.subplots(1,1)
    A.plot(vertex_color="b", ax=ax1)
    ax1.set_xticks([])
    ax1.set_title(title, fontsize=15)
    ax1.set_yticks([]);

# The exact recovery from the BLUE: 
def sample_and_recover(x,sample,G,bandlim, verbose=False):
    assert len(x)==len(sample), "sample set and signal are incompatible due to size difference."

    sample_op=sample_nodes(sample)
    if verbose:
        print("Sampling the signal & computing recovery matrices")
    s=sample_op@x
    U_VR,U_SR=recovery_matrices(G,bandlim,sample_op)
    
    sig_recovered=U_VR@np.linalg.pinv(np.transpose(U_SR)@U_SR)@np.transpose(U_SR)@s
    if verbose:
        if np.linalg.matrix_rank(U_SR)>=bandlim:
            print("Sample size sufficient for recovery")
        else:
            print("Sample size insufficient, used pseudo-inverse instead of inverse")
    return sig_recovered

#%%
#Defining sampling methods:
def uniform_sampling(G,n):
    """samples uniformly n nodes from graph G"""
    s = np.random.uniform(0,1,G.N) #this could also be gaussian, as long as it is continuous and random
    t = pd.DataFrame(s)
    ind = [i for i in t.nlargest(n,0).index]
    sample = [0]*G.N
    for i in ind:
        sample[i]=1
    return sample

# Optimal designs. Note that noise_op=E(eps*eps^T):
def A_optimal_sampling(G,n,bandlim, noise_op=None):
    options=np.array(list(set(list(it.permutations([1]*n+[0]*(G.N-n))))))
    min=np.inf
    if noise_op==None:
        noise_op=np.identity(G.N)
    for i in options:
        op=sample_nodes(i)
        U_VR,U_SR=recovery_matrices(G,bandlim,op)
        R=np.linalg.pinv(noise_op)
        MSE=np.trace(np.linalg.pinv(U_VR.T@op.T@op@R@U_VR))
        if MSE<min:
            min=MSE
            candidate=i
    return candidate

def A_optimal_greed(G,n,bandlim, noise_op=None):
    sample=[0]*G.N
    if noise_op==None:
        R=np.identity(G.N)
    else:
        R=np.linalg.pinv(noise_op)
    for _ in range(n):
        min=np.inf
        for i in range(G.N):
            if sample[i]==0:
                sample[i]=1
                op=sample_nodes(sample)
                U_VR,U_SR=recovery_matrices(G,bandlim,op)
                #R=jnp.pinv(noise_op)
                MSE=np.trace(np.linalg.pinv(U_VR.T@op.T@op@R@U_VR))
                if MSE<min:
                    min=MSE
                    candidate=i
                sample[i]=0
        sample[candidate]=1
    return sample

def E_optimal_sampling(G,n,bandlim):
    options=np.array(list(set(list(it.permutations([1]*n+[0]*(G.N-n))))))
    max=-np.inf
    for i in options:
        op=sample_nodes(i)
        U_VR,U_SR=recovery_matrices(G,bandlim,op)
        sigma_min=min(np.linalg.svd(op.T@U_SR)[1])
        if sigma_min>max:
            max=sigma_min
            candidate=i
    return candidate

def E_optimal_greed(G,n,bandlim):
    sample=[0]*G.N
    for _ in range(n):
        max=-np.inf
        for i in range(G.N):
            if sample[i]==0:
                sample[i]=1
                op=sample_nodes(sample)
                U_VR,U_SR=recovery_matrices(G,bandlim,op)
                sigma_min=np.linalg.svd(op.T@U_SR)[1][-1]
                if sigma_min>max:
                    max=sigma_min
                    candidate=i
                sample[i]=0
        sample[candidate]=1
    return sample

def D_optimal_sampling(G,n,bandlim,noise_op=None):
    options=np.array(list(set(list(it.permutations([1]*n+[0]*(G.N-n))))))
    max=-np.inf
    if noise_op==None:
        noise_op=np.identity(G.N)
    for i in options:
        op=sample_nodes(i)
        U_VR,U_SR=recovery_matrices(G,bandlim,op)
        R=np.linalg.pinv(noise_op)
        log_det=np.linalg.det(U_VR.T@op.T@op@R@U_VR)
        if log_det>max:
            max=log_det
            candidate=i
    return candidate

def D_optimal_greed(G,n,bandlim,noise_op=None):
    sample=[0]*G.N
    if noise_op==None:
        noise_op=np.identity(G.N)
    for _ in range(n):
        max=-np.inf
        for i in range(G.N):
            if sample[i]==0:
                sample[i]=1
                op=sample_nodes(sample)
                U_VR,U_SR=recovery_matrices(G,bandlim,op)
                R=np.linalg.pinv(noise_op)
                log_det=np.linalg.det(U_VR.T@op.T@op@R@U_VR)
                if log_det>max:
                    max=log_det
                    candidate=i
                sample[i]=0
        sample[candidate]=1
    return sample

def Optimal_sampling(x,G,n,bandlim):
    options=np.array(list(set(list(it.permutations([1]*n+[0]*(G.N-n))))))
    min=np.inf
    for i in options:
        MSE=cost(sample_and_recover(x,i,G,bandlim),x)
        if MSE<min:
            min=MSE
            candidate=i
    return candidate

# The experimental exponential sampling strategies:
# Using the adjacency matrix
def Exponential_sampling(G,n,bandlim): #Here bandlim is used as exponent 
    A_k=(G.W**bandlim).toarray()
    Prob=A_k@np.ones(G.N)
    
    # I chose to leave it deterministic for now
    t = pd.DataFrame(Prob)
    ind = [i for i in t.nsmallest(n,0).index]
    sample = [0]*G.N
    for i in ind:
        sample[i]=1
    return sample
# Using left normalized adjacency matrix
def NExponential_sampling(G,n,bandlim,noise_op=None): #Here bandlim is not used 
    # for the signal bandwidth, instead for the power of $A^bandlim$:
    A_k=((G.W@np.linalg.inv(np.diag(G.d)))**bandlim)
    Prob=A_k@np.ones(G.N)
    
    # I chose to leave it deterministic for now
    t = pd.DataFrame(Prob)
    ind = [i for i in t.nsmallest(n,0).index]
    sample = [0]*G.N
    for i in ind:
        sample[i]=1
    return sample
# Using a polynomial of the adjacency matrix with c_i=1 coefficients
def Polynomial_sampling(G,n,bandlim,noise_op=None): #Here bandlim is not used 
    # for the signal bandwidth, instead for the power of $A^bandlim$:
    A_k=sum([(G.W**i)for i in range(bandlim)]).toarray()/bandlim
    Prob=A_k@np.ones(G.N)
    
    # I chose to leave it deterministic for now
    t = pd.DataFrame(Prob)
    ind = [i for i in t.nsmallest(n,0).index]
    sample = [0]*G.N
    for i in ind:
        sample[i]=1
    return sample

# Sample vertices of minimal degree
def Degree_sampling(G,n,bandlim=None,noise_op=None): #Here bandlim is not used 
    t = pd.DataFrame(G.d)
    ind = [i for i in t.nsmallest(n,0).index]
    sample = [0]*G.N
    for i in ind:
        sample[i]=1
    return sample

# An experiment using the root of the adjacency matrix. Can be ignored
def Root_A_sampling(G,n,bandlim=None,noise_op=None): #Here bandlim is not used 
    t = pd.DataFrame(abs(sp.linalg.sqrtm(G.W.toarray()))@np.ones(G.N))
    ind = [i for i in t.nsmallest(n,0).index]
    sample = [0]*G.N
    for i in ind:
        sample[i]=1
    return sample

# The exponential method for k=1/2
def Root_L_sampling(G,n,bandlim=None,noise_op=None): #Here bandlim is not used 
    t = pd.DataFrame(abs(sp.linalg.sqrtm(G.L.toarray()))@np.ones(G.N))
    ind = [i for i in t.nsmallest(n,0).index]
    sample = [0]*G.N
    for i in ind:
        sample[i]=1
    return sample

# The exponential method for k=1/4
def DRoot_L_sampling(G,n,bandlim=None,noise_op=None): #Here bandlim is not used 
    t = pd.DataFrame(abs(sp.linalg.sqrtm(sp.linalg.sqrtm(G.L.toarray())))@np.ones(G.N))
    ind = [i for i in t.nsmallest(n,0).index]
    sample = [0]*G.N
    for i in ind:
        sample[i]=1
    return sample

# The exponential method for k=1/8
def DDRoot_L_sampling(G,n,bandlim=None,noise_op=None): #Here bandlim is not used 
    t = pd.DataFrame(abs(sp.linalg.sqrtm(sp.linalg.sqrtm(sp.linalg.sqrtm(G.L.toarray()))))@np.ones(G.N))
    ind = [i for i in t.nsmallest(n,0).index]
    sample = [0]*G.N
    for i in ind:
        sample[i]=1
    return sample

# Exponential method for |L^k|
def absL_sampling(G,n,bandlim,noise_op=None): #Here bandlim is not used 
    t = pd.DataFrame((abs(G.L**bandlim)).toarray()@np.ones(G.N))
    ind = [i for i in t.nsmallest(n,0).index]
    sample = [0]*G.N
    for i in ind:
        sample[i]=1
    return sample

# Exponential method for |L|^k
def Labs_sampling(G,n,bandlim,noise_op=None): #Here bandlim is not used 
    t = pd.DataFrame((abs(G.L)**bandlim).toarray()@np.ones(G.N))
    ind = [i for i in t.nsmallest(n,0).index]
    sample = [0]*G.N
    for i in ind:
        sample[i]=1
    return sample

# Experimental method using D+gamma*A, where gamma is a parameter to be chosen
def D_plus_gamma_A_sampling(G,n,gamma:float):
    Abs_Weighted_L=np.diag(G.d)+gamma*G.W
    t = pd.DataFrame(((Abs_Weighted_L**2)@np.ones(G.N)).T)
    ind = [i for i in t.nsmallest(n,0).index]
    sample = [0]*G.N
    for i in ind:
        sample[i]=1
    return sample

# These are the main probabilistic sampling strategies.
# Note that we immediately generate a sampling set from the given probability distribution.

# Conversion of exponential sampling to probabilistic sampling:
def Root_L_prob(G,n,bandlim,noise_op=None): #Here bandlim is not used 
    ar=abs(sp.linalg.sqrtm(G.L.toarray()))@np.ones(G.N)
    prob_ar=(1/ar)/np.linalg.norm((1/ar),ord=1)
    prob_ar=(prob_ar**bandlim)/np.linalg.norm((prob_ar**bandlim),ord=1)
    t = pd.DataFrame([i for i in range(G.N)])
    ind=t.sample(n,weights=prob_ar)
    sample = [0]*G.N
    for i in ind.values:
        sample[i[0]]=1
    return sample

# The exact solution for graph coherence optimization.
def Paper_prob_exact(G,n,bandlim):
    U_VR=G.U[:,0:bandlim]
    distr=np.array([np.linalg.norm(U_VR[i])**2/bandlim for i in range(G.N)])
    t = pd.DataFrame([i for i in range(G.N)])
    ind=t.sample(n,weights=distr)
    sample = [0]*G.N
    for i in ind.values:
        sample[i[0]]=1
    return sample

#%%
# We test a sampling strategy (in the string) on a single graph G.
def test_strategy(G,bandlim,ntrials,strategy:str,noise_op=None, verbose=False):
    t=time.time()
    total=np.zeros((ntrials,G.N-1))
    for j in range(ntrials):
        sig=BL_sig_L(G,bandlim)
        sig=sig/np.linalg.norm(sig)
        for i in range(1,G.N):
            if strategy=="uniform":
                result=sample_and_recover(sig,uniform_sampling(G,i+1),G,bandlim)
            elif strategy=="A_optimal":
                result=sample_and_recover(sig,A_optimal_sampling(G,i+1,bandlim,noise_op),G,bandlim)
            elif strategy=="A_greed":
                result=sample_and_recover(sig,A_optimal_greed(G,i+1,bandlim,noise_op),G,bandlim)
            elif strategy=="E_optimal":
                result=sample_and_recover(sig,E_optimal_sampling(G,i+1,bandlim),G,bandlim)
            elif strategy=="E_greed":
                result=sample_and_recover(sig,E_optimal_greed(G,i+1,bandlim),G,bandlim)
            elif strategy=="D_optimal":
                result=sample_and_recover(sig,D_optimal_sampling(G,i+1,bandlim,noise_op),G,bandlim)
            elif strategy=="D_greed":
                result=sample_and_recover(sig,D_optimal_greed(G,i+1,bandlim,noise_op),G,bandlim)
            elif strategy=="Optimal":
                result=sample_and_recover(sig,Optimal_sampling(sig,G,i+1,bandlim),G,bandlim)
            elif strategy=="Exp_sampling":
                result=sample_and_recover(sig,Exponential_sampling(G,i+1,10),G,bandlim) #I took 10 as a test value
            elif strategy=="Nexp_sampling":
                result=sample_and_recover(sig,NExponential_sampling(G,i+1,bandlim),G,bandlim)
            total[j,i-1]+=cost(result,sig)
    if verbose:
        print(strategy+" design took "+str(time.time()-t)+" seconds")
    mean=sum(np.log(total))/ntrials
    std=(sum((np.log(total)-mean)**2)/(ntrials-1))**(1/2)
    #return total/ntrials
    return mean, mean-std, mean+std

# A new symulation is introduced using Gaussian or Bernoulli-type noise.
def test_strategy_gaus(G,bandlim,ntrials,strategy:str,sigma=0.01,noise_op=None, verbose=False):
    t=time.time()
    total=np.zeros((ntrials,G.N-1))
    for j in range(ntrials):
        sig=BL_sig_L(G,bandlim)
        sig=sig/np.linalg.norm(sig)
        noise=np.random.multivariate_normal(np.zeros(G.N),np.identity(G.N)*(sigma**2))
        sig+=noise
        for i in range(1,G.N):
            if strategy=="uniform":
                result=sample_and_recover(sig,uniform_sampling(G,i+1),G,bandlim)
            elif strategy=="A_optimal":
                result=sample_and_recover(sig,A_optimal_sampling(G,i+1,bandlim,noise_op),G,bandlim)
            elif strategy=="A_greed":
                result=sample_and_recover(sig,A_optimal_greed(G,i+1,bandlim,noise_op),G,bandlim)
            elif strategy=="E_optimal":
                result=sample_and_recover(sig,E_optimal_sampling(G,i+1,bandlim),G,bandlim)
            elif strategy=="E_greed":
                result=sample_and_recover(sig,E_optimal_greed(G,i+1,bandlim),G,bandlim)
            elif strategy=="D_optimal":
                result=sample_and_recover(sig,D_optimal_sampling(G,i+1,bandlim,noise_op),G,bandlim)
            elif strategy=="D_greed":
                result=sample_and_recover(sig,D_optimal_greed(G,i+1,bandlim,noise_op),G,bandlim)
            elif strategy=="Optimal":
                result=sample_and_recover(sig,Optimal_sampling(sig,G,i+1,bandlim),G,bandlim)
            elif strategy=="Exp_sampling":
                result=sample_and_recover(sig,Exponential_sampling(G,i+1,bandlim),G,bandlim)
            elif strategy=="Nexp_sampling":
                result=sample_and_recover(sig,NExponential_sampling(G,i+1,bandlim),G,bandlim)
            total[j,i-1]+=cost(result,sig-noise)
    if verbose:
        print(strategy+" design took "+str(time.time()-t)+" seconds")
    mean=sum(np.log(total))/ntrials
    std=(sum((np.log(total)-mean)**2)/(ntrials-1))**(1/2)
    return mean, mean-std, mean+std

def test_strategy_bern(G,bandlim,ntrials,strategy:str, sigma=0.01,p=0.2,noise_op=None, verbose=False):
    t=time.time()
    total=np.zeros((ntrials,G.N-1))
    for j in range(ntrials):
        sig=BL_sig_L(G,bandlim)
        sig=sig/np.linalg.norm(sig)
        noise=np.random.choice([-1,1],size=G.N)*bernoulli.rvs(p,size=G.N)*((sigma**2/p)**(1/2))
        sig+=noise
        for i in range(1,G.N):
            if strategy=="uniform":
                result=sample_and_recover(sig,uniform_sampling(G,i+1),G,bandlim)
            elif strategy=="A_optimal":
                result=sample_and_recover(sig,A_optimal_sampling(G,i+1,bandlim,noise_op),G,bandlim)
            elif strategy=="A_greed":
                result=sample_and_recover(sig,A_optimal_greed(G,i+1,bandlim,noise_op),G,bandlim)
            elif strategy=="E_optimal":
                result=sample_and_recover(sig,E_optimal_sampling(G,i+1,bandlim),G,bandlim)
            elif strategy=="E_greed":
                result=sample_and_recover(sig,E_optimal_greed(G,i+1,bandlim),G,bandlim)
            elif strategy=="D_optimal":
                result=sample_and_recover(sig,D_optimal_sampling(G,i+1,bandlim,noise_op),G,bandlim)
            elif strategy=="D_greed":
                result=sample_and_recover(sig,D_optimal_greed(G,i+1,bandlim,noise_op),G,bandlim)
            elif strategy=="Optimal":
                result=sample_and_recover(sig,Optimal_sampling(sig,G,i+1,bandlim),G,bandlim)
            elif strategy=="Exp_sampling":
                result=sample_and_recover(sig,Exponential_sampling(G,i+1,bandlim),G,bandlim)
            elif strategy=="Nexp_sampling":
                result=sample_and_recover(sig,NExponential_sampling(G,i+1,bandlim),G,bandlim)
            total[j,i-1]+=cost(result,sig-noise)
    if verbose:
        print(strategy+" design took "+str(time.time()-t)+" seconds")
    mean=sum(np.log(total))/ntrials
    std=(sum((np.log(total)-mean)**2)/(ntrials-1))**(1/2)
    return mean, mean-std, mean+std



#%%
# This initial data is needed for the figures 7-11.
bl=4
graphs=[]
count=1
size=9
for i in range(30,100):
    graph=gen_graph_er(size,i/100)
    exec("Graph"+str(count)+"=g.graphs.Graph(graph)")
    exec("Graph"+str(count)+".set_coordinates()")
    exec("graphs+=[Graph"+str(count)+"]")
    count+=1
nruns=10
optimum_wanted=False #Change to True if you want to test optimal design with an exhaustive search
tot_uni=np.array([test_strategy(j,bl,nruns,"uniform") for j in graphs]).mean(axis=0)
tot_Agr=np.array([test_strategy(j,bl,nruns,"A_greed") for j in graphs]).mean(axis=0)
tot_Egr=np.array([test_strategy(j,bl,nruns,"E_greed") for j in graphs]).mean(axis=0)
tot_Dgr=np.array([test_strategy(j,bl,nruns,"D_greed") for j in graphs]).mean(axis=0)
if optimum_wanted:
    tot_Aop=np.array([test_strategy(j,bl,nruns,"A_optimal") for j in graphs]).mean(axis=0)
    tot_Eop=np.array([test_strategy(j,bl,nruns,"E_optimal") for j in graphs]).mean(axis=0)
    tot_Dop=np.array([test_strategy(j,bl,nruns,"D_optimal") for j in graphs]).mean(axis=0)
tot_Opt=np.array([test_strategy(j,bl,nruns,"Optimal") for j in graphs]).mean(axis=0)

#These are only for figure 11
# tot_uni_g=np.array([test_strategy_gaus(j,bl,nruns,"uniform") for j in graphs]).mean(axis=0)
# tot_Agr_g=np.array([test_strategy_gaus(j,bl,nruns,"A_greed") for j in graphs]).mean(axis=0)
# tot_Egr_g=np.array([test_strategy_gaus(j,bl,nruns,"E_greed") for j in graphs]).mean(axis=0)
# tot_Dgr_g=np.array([test_strategy_gaus(j,bl,nruns,"D_greed") for j in graphs]).mean(axis=0)
# tot_uni_b=np.array([test_strategy_bern(j,bl,nruns,"uniform") for j in graphs]).mean(axis=0)
# tot_Agr_b=np.array([test_strategy_bern(j,bl,nruns,"A_greed") for j in graphs]).mean(axis=0)
# tot_Egr_b=np.array([test_strategy_bern(j,bl,nruns,"E_greed") for j in graphs]).mean(axis=0)
# tot_Dgr_b=np.array([test_strategy_bern(j,bl,nruns,"D_greed") for j in graphs]).mean(axis=0)

#%%
# #%%
# # Fig 7
# plt.plot([i+1 for i in range(1,size)],tot_uni[0])
# plt.plot([i+1 for i in range(1,size)],tot_Aop[0])
# plt.plot([i+1 for i in range(1,size)],tot_Eop[0])
# plt.plot([i+1 for i in range(1,size)],tot_Dop[0])
# plt.xlabel("Sample size")
# plt.ylabel("log MSE")
# plt.legend(["uniform","A-design", "E-design","D-design"])
# plt.title("Comparison of four sampling strategies");

# #%%
# # Fig 8a
# plt.plot([i+1 for i in range(1,size)],tot_Aop[0])
# plt.plot([i+1 for i in range(1,size)],tot_Agr[0])
# plt.xlabel("Sample size")
# plt.ylabel("log MSE")
# plt.legend(["optimal sample","greedy sample"])
# plt.title("Comparison of A-optimal design with and without greedy algorithm");

# #%%
# # Fig 8b
# plt.plot([i+1 for i in range(1,size)],tot_Eop[0])
# plt.plot([i+1 for i in range(1,size)],tot_Egr[0])
# plt.xlabel("Sample size")
# plt.ylabel("log MSE")
# plt.legend(["optimal sample","greedy sample"])
# plt.title("Comparison of E-optimal design with and without greedy algorithm");

# #%%
# # Fig 8c
# plt.plot([i+1 for i in range(1,size)],tot_Dop[0])
# plt.plot([i+1 for i in range(1,size)],tot_Dgr[0])
# plt.xlabel("Sample size")
# plt.ylabel("log MSE")
# plt.legend(["optimal sample","greedy sample"])
# plt.title("Comparison of D-optimal design with and without greedy algorithm");

# #%%
# Fig 9abc
# a:
#plt.plot([i+1 for i in range(1,size)],tot_Agr[0])
# b:
#plt.plot([i+1 for i in range(1,size)],tot_Egr[0])
# c:
plt.plot([i+1 for i in range(1,size)],tot_Dgr[0])

# all:
plt.plot([i+1 for i in range(1,size)],tot_Opt[0])
plt.fill_between([i+1 for i in range(1,size)],tot_Dgr[1],tot_Dgr[2],alpha=.3)
plt.xlabel("Sample size")
plt.ylabel("log MSE")
plt.legend(["D-design", "optimal sample"])
plt.title("Comparison of D optimal design sampling versus the true optimum");

# #%%
# # Fig 10 (a for the full plot range, b for the zoomed in range between 5 and 9)
# Grid1=g.graphs.Grid2d(3)
# uni_gd=test_strategy(Grid1,4,50,"uniform")
# Agr_gd=test_strategy(Grid1,4,50,"A_greed", verbose=False)
# Egr_gd=test_strategy(Grid1,4,50,"E_greed", verbose=False)
# Dgr_gd=test_strategy(Grid1,4,50,"D_greed", verbose=False)
# plt.plot([i+1 for i in range(5,Grid1.N)],uni_gd[0][4:])
# plt.plot([i+1 for i in range(5,Grid1.N)],Agr_gd[0][4:])
# plt.plot([i+1 for i in range(5,Grid1.N)],Egr_gd[0][4:])
# plt.plot([i+1 for i in range(5,Grid1.N)],Dgr_gd[0][4:])
# plt.xticks([6,7,8,9])
# plt.xlabel("Sample size")
# plt.ylabel("log MSE")
# plt.title("Comparison of optimal designs on a 3x3 grid")
# plt.legend(["uniform","A-design", "E-design", "D-design"]);

# #%%
# # Fig 11a
# plt.plot([i+1 for i in range(1,size)],tot_uni_g[0])
# plt.plot([i+1 for i in range(1,size)],tot_Agr_g[0])
# plt.plot([i+1 for i in range(1,size)],tot_Egr_g[0])
# plt.plot([i+1 for i in range(1,size)],tot_Dgr_g[0])
# plt.xlabel("Sample size")
# plt.ylabel("log MSE")
# plt.legend(["uniform","A-design", "E-design","D-design"])
# plt.title("Comparison methods for Gaussian noise");
# #%%
# # Fig 11b
# fig,ax=plt.subplots(2,2,figsize=(5,5))
# ax[0,0].plot([i+1 for i in range(1,size)],tot_uni_g[0])
# ax[0,1].plot([i+1 for i in range(1,size)],tot_Agr_g[0])
# ax[1,0].plot([i+1 for i in range(1,size)],tot_Egr_g[0])
# ax[1,1].plot([i+1 for i in range(1,size)],tot_Dgr_g[0])
# ax[0,0].fill_between([i+1 for i in range(1,size)],tot_uni_g[1],tot_uni_g[2],alpha=.3)
# ax[0,1].fill_between([i+1 for i in range(1,size)],tot_Agr_g[1],tot_Agr_g[2],alpha=.3)
# ax[1,0].fill_between([i+1 for i in range(1,size)],tot_Egr_g[1],tot_Egr_g[2],alpha=.3)
# ax[1,1].fill_between([i+1 for i in range(1,size)],tot_Dgr_g[1],tot_Dgr_g[2],alpha=.3)
# ax[0,0].set_title("Uniform")
# ax[0,1].set_title("A design")
# ax[1,0].set_title("E design")
# ax[1,1].set_title("D design")
# ax[0,0].set_ylim(-9,2)
# ax[0,1].set_ylim(-9,2)
# ax[1,0].set_ylim(-9,2)
# ax[1,1].set_ylim(-9,2)
# fig.supxlabel("Sample size")
# fig.supylabel("log MSE")
# plt.tight_layout()

# #%%
# # Fig 11c
# plt.plot([i+1 for i in range(1,size)],tot_uni_b[0])
# plt.plot([i+1 for i in range(1,size)],tot_Agr_b[0])
# plt.plot([i+1 for i in range(1,size)],tot_Egr_b[0])
# plt.plot([i+1 for i in range(1,size)],tot_Dgr_b[0])
# plt.xlabel("Sample size")
# plt.ylabel("log MSE")
# plt.legend(["uniform","A-design", "E-design","D-design"])
# plt.title("Comparison methods for Bernoulli noise");
# #%%
# # Fig 11d
# fig,ax=plt.subplots(2,2,figsize=(5,5))
# ax[0,0].plot([i+1 for i in range(1,size)],tot_uni_b[0])
# ax[0,1].plot([i+1 for i in range(1,size)],tot_Agr_b[0])
# ax[1,0].plot([i+1 for i in range(1,size)],tot_Egr_b[0])
# ax[1,1].plot([i+1 for i in range(1,size)],tot_Dgr_b[0])
# ax[0,0].fill_between([i+1 for i in range(1,size)],tot_uni_b[1],tot_uni_b[2],alpha=.3)
# ax[0,1].fill_between([i+1 for i in range(1,size)],tot_Agr_b[1],tot_Agr_b[2],alpha=.3)
# ax[1,0].fill_between([i+1 for i in range(1,size)],tot_Egr_b[1],tot_Egr_b[2],alpha=.3)
# ax[1,1].fill_between([i+1 for i in range(1,size)],tot_Dgr_b[1],tot_Dgr_b[2],alpha=.3)
# ax[0,0].set_title("Uniform")
# ax[0,1].set_title("A design")
# ax[1,0].set_title("E design")
# ax[1,1].set_title("D design")
# ax[0,0].set_ylim(-60,7)
# ax[0,1].set_ylim(-60,7)
# ax[1,0].set_ylim(-60,7)
# ax[1,1].set_ylim(-60,7)
# fig.supxlabel("Sample size")
# fig.supylabel("log MSE")
# plt.tight_layout()

#%%
# We repeat the experiment in Fig 7, except for larger graphs and bandlimit, namely 50/9 times bigger.
bl=22
graphs=[]
count=1
size=50
for i in range(10):
    graph=gen_graph_er(size,0.2)
    exec("Graph"+str(count)+"=g.graphs.Graph(graph)")
    exec("Graph"+str(count)+".set_coordinates()")
    exec("graphs+=[Graph"+str(count)+"]")
    count+=1
nruns=10

# Be warned, this operation has a long runtime (> 1 hour for my pc)

# tot_uni=np.array([test_strategy(j,bl,nruns,"uniform") for j in graphs]).mean(axis=0)
# tot_Agr=np.array([test_strategy(j,bl,nruns,"A_greed") for j in graphs]).mean(axis=0)
# tot_Egr=np.array([test_strategy(j,bl,nruns,"E_greed") for j in graphs]).mean(axis=0)
# tot_Dgr=np.array([test_strategy(j,bl,nruns,"D_greed") for j in graphs]).mean(axis=0)

# # Fig 12a
# plt.plot([i+1 for i in range(1,size)],tot_uni[0])
# plt.plot([i+1 for i in range(1,size)],tot_Agr[0])
# plt.plot([i+1 for i in range(1,size)],tot_Egr[0])
# plt.plot([i+1 for i in range(1,size)],tot_Dgr[0])
# plt.xlabel("Sample size")
# plt.ylabel("log MSE")
# plt.legend(["uniform","A-design", "E-design","D-design"])
# plt.title("Comparison of four sampling strategies");

# # Fig 12b
# plt.plot([i+1 for i in range(25,size)],tot_uni[0][24:size])
# plt.plot([i+1 for i in range(25,size)],tot_Agr[0][24:size])
# plt.plot([i+1 for i in range(25,size)],tot_Egr[0][24:size])
# plt.plot([i+1 for i in range(25,size)],tot_Dgr[0][24:size])
# plt.xlabel("Sample size")
# plt.ylabel("log MSE")
# plt.legend(["uniform","A-design", "E-design","D-design"])
# plt.title("Comparison of four sampling strategies");


#%%
# These functions immediately plot the comparison of a few sampling strategies.
# The range is chosen between 10 (or sometimes 20) and 200. 
def uni_vs_exp1(Gtype: str, ngraphs, ntrials, bandlim_ratio, minsize=10, maxsize=210, stepsize=10, expsize=10, ntype=None, **kwargs):
    #Defining the neessary extra parameters:
    if Gtype=="sw":
        k=kwargs["k"]
    if Gtype=="pa":
        m=kwargs["m"]
    #Generating graph list:
    G_lst=[]    
    for i in range(minsize,maxsize,stepsize):
        if Gtype=="er":
            p=2/math.sqrt(i)
            G_lst+=[[g.graphs.Graph(gen_graph_er(i,p)) for _ in range(ngraphs)]]
        elif Gtype=="sw":
            p=1/k
            G_lst+=[[g.graphs.Graph(gen_graph_sw(i,k,p)) for _ in range(ngraphs)]]
        elif Gtype=="block":
            p=2/math.sqrt(i/2)
            q=0.2/math.sqrt(i/2)
            G_lst+=[[g.graphs.Graph(gen_graph_block(round(i/2),p,q)) for _ in range(ngraphs)]]
        elif Gtype=="pa":
            G_lst+=[[g.graphs.Graph(gen_graph_pa(i,m)) for _ in range(ngraphs)]]
    #The actual sampling problem + reconstruction
    total=np.zeros((int((maxsize-minsize)/stepsize),ngraphs*ntrials,2))
    count=0
    for j in G_lst:
        tot=np.zeros((ntrials,2))
        c=0
        for i in j:
            for k in range(ntrials):
                bandlim=round(bandlim_ratio*i.N)
                sig=BL_sig_L(i,bandlim)
                sig=sig/np.linalg.norm(sig)
                if ntype!=None:
                    sigma=kwargs["sigma"]
                    prob=kwargs["prob"]
                    if ntype=="gaus":
                        noise=np.random.multivariate_normal(np.zeros(i.N),np.identity(i.N)*(sigma**2))
                    elif ntype=="bern":
                        noise=np.random.choice([-1,1],size=i.N)*bernoulli.rvs(prob,size=i.N)*((sigma**2/prob)**(1/2))
                    elif ntype=="bega":
                        bern=bernoulli.rvs(prob,size=i.N)
                        while sum(bern)==0:
                            bern=bernoulli.rvs(prob,size=i.N)
                        noise=bern*np.random.multivariate_normal(np.zeros(i.N),np.identity(i.N)*(sigma**2*i.N/sum(bern)))
                    sig+=noise
                res_uni=sample_and_recover(sig,uniform_sampling(i,min(i.N,2*bandlim+1)),i,min(bandlim,i.N))
                res_exp=sample_and_recover(sig,Exponential_sampling(i,min(i.N,2*bandlim+1),expsize),i,min(bandlim,i.N))
                total[count,c*ntrials+k,0]=cost(res_uni,sig-noise)
                total[count,c*ntrials+k,1]=cost(res_exp,sig-noise)
            c+=1
        count+=1
    #The statistical analysis: 
    tot=total.mean(axis=1)
    std=total.std(axis=1)
    #Finishing with the plot:
    plt.plot([i for i in range(minsize,maxsize,stepsize)],tot)
    plt.fill_between([i for i in range(minsize,maxsize,stepsize)],(tot-std)[:,0],(tot+std)[:,0], alpha=0.3)
    plt.fill_between([i for i in range(minsize,maxsize,stepsize)],(tot-std)[:,1],(tot+std)[:,1], alpha=0.3)
    plt.xlabel("graph size")
    plt.ylabel("MSE")
    if Gtype=="er":
        plt.title("Comparison uniform and exponential sampling \nfor Erdos-Renyi graphs")
    elif Gtype=="sw":
        plt.title("Comparison uniform and exponential sampling \nfor Small World graphs")
    elif Gtype=="block":
        plt.title("Comparison uniform and exponential sampling \nfor Block model graphs")
    elif Gtype=="pa":
        plt.title("Comparison uniform and exponential sampling \nfor Preferential Attachment graphs")
    plt.legend(["Uniform","Exponential"])
    lbnd,ubnd=kwargs["lbnd"],kwargs["ubnd"]
    plt.ylim(lbnd,ubnd)
    plt.show();
    print(str(ngraphs)+" graphs, "+str(ntrials)+" test signals, bandlimit "+ str(bandlim_ratio)+"N\n"
        "probability 1/sqrt(n), and sample size 2*bandlimit+1\n"
        "exponentiation of A to power "+str(expsize)+".\n"
        "sigma = "+str(sigma)+" and p = "+ str(prob)+" on "+str(ntype)+" noise.")

def uni_vs_Nexp1(Gtype: str, ngraphs, ntrials, bandlim_ratio, minsize=10, maxsize=210, stepsize=10, expsize=10, ntype=None, **kwargs):
    #Defining the neessary extra parameters:
    if Gtype=="sw":
        k=kwargs["k"]
    if Gtype=="pa":
        m=kwargs["m"]
    #Generating graph list:
    G_lst=[]    
    for i in range(minsize,maxsize,stepsize):
        if Gtype=="er":
            p=2/math.sqrt(i)
            G_lst+=[[g.graphs.Graph(gen_graph_er(i,p)) for _ in range(ngraphs)]]
        elif Gtype=="sw":
            p=1/k
            G_lst+=[[g.graphs.Graph(gen_graph_sw(i,k,p)) for _ in range(ngraphs)]]
        elif Gtype=="block":
            p=2/math.sqrt(i/2)
            q=0.2/math.sqrt(i/2)
            G_lst+=[[g.graphs.Graph(gen_graph_block(round(i/2),p,q)) for _ in range(ngraphs)]]
        elif Gtype=="pa":
            G_lst+=[[g.graphs.Graph(gen_graph_pa(i,m)) for _ in range(ngraphs)]]
    #The actual sampling problem + reconstruction
    total=np.zeros((int((maxsize-minsize)/stepsize),ngraphs*ntrials,2))
    count=0
    for j in G_lst:
        tot=np.zeros((ntrials,2))
        c=0
        for i in j:
            for k in range(ntrials):
                bandlim=round(bandlim_ratio*i.N)
                sig=BL_sig_L(i,bandlim)
                sig=sig/np.linalg.norm(sig)
                if ntype!=None:
                    sigma=kwargs["sigma"]
                    prob=kwargs["prob"]
                    if ntype=="gaus":
                        noise=np.random.multivariate_normal(np.zeros(i.N),np.identity(i.N)*(sigma**2))
                    elif ntype=="bern":
                        noise=np.random.choice([-1,1],size=i.N)*bernoulli.rvs(p,size=i.N)*((sigma**2/prob)**(1/2))
                    elif ntype=="bega":
                        bern=bernoulli.rvs(prob,size=i.N)
                        while sum(bern)==0:
                            bern=bernoulli.rvs(prob,size=i.N)
                        noise=bern*np.random.multivariate_normal(np.zeros(i.N),np.identity(i.N)*(sigma**2*i.N/sum(bern)))
                    sig+=noise
                res_uni=sample_and_recover(sig,uniform_sampling(i,min(i.N,2*bandlim+1)),i,min(bandlim,i.N))
                res_exp=sample_and_recover(sig,NExponential_sampling(i,min(i.N,2*bandlim+1),expsize),i,min(bandlim,i.N))
                total[count,c*ntrials+k,0]=cost(res_uni,sig-noise)
                total[count,c*ntrials+k,1]=cost(res_exp,sig-noise)
            c+=1
        count+=1
    #The statistical analysis: 
    tot=total.mean(axis=1)
    std=total.std(axis=1)
    #Finishing with the plot:
    plt.plot([i for i in range(minsize,maxsize,stepsize)],tot)
    plt.fill_between([i for i in range(minsize,maxsize,stepsize)],(tot-std)[:,0],(tot+std)[:,0], alpha=0.3)
    plt.fill_between([i for i in range(minsize,maxsize,stepsize)],(tot-std)[:,1],(tot+std)[:,1], alpha=0.3)
    plt.xlabel("graph size")
    plt.ylabel("MSE")
    if Gtype=="er":
        plt.title("Comparison uniform and exponential sampling \nfor Erdos-Renyi graphs")
    elif Gtype=="sw":
        plt.title("Comparison uniform and exponential sampling \nfor Small World graphs")
    elif Gtype=="block":
        plt.title("Comparison uniform and exponential sampling \nfor Block model graphs")
    elif Gtype=="pa":
        plt.title("Comparison uniform and exponential sampling \nfor Preferential Attachment graphs")
    plt.legend(["Uniform","Normalized Exponential"])
    lbnd,ubnd=kwargs["lbnd"],kwargs["ubnd"]
    plt.ylim(lbnd,ubnd)
    plt.show();
    print(str(ngraphs)+" graphs, "+str(ntrials)+" test signals, bandlimit "+ str(bandlim_ratio)+"N\n"
        "probability 1/sqrt(n), and sample size 2*bandlimit+1\n"
        "exponentiation of A to power "+str(expsize)+".\n"
        "sigma = "+str(sigma)+" and p = "+ str(prob)+" on "+str(ntype)+" noise.")

def uni_vs_deg1(Gtype: str, ngraphs, p, ntrials, bandlim_ratio, minsize=10, maxsize=100, stepsize=5, expsize=10, ntype=None, **kwargs):
        #Defining the necessary extra parameters:
        if Gtype=="sw":
            k=kwargs["k"]
        if Gtype=="block":
            q=kwargs["q"]
        if Gtype=="pa":
            m=kwargs["m"]
        #Generating graph list:
        G_lst=[]
        for i in range(minsize,maxsize,stepsize):
            if Gtype=="er":
                p=2/math.sqrt(i)
                G_lst+=[[g.graphs.Graph(gen_graph_er(i,p)) for _ in range(ngraphs)]]
            elif Gtype=="sw":
                p=1/k
                G_lst+=[[g.graphs.Graph(gen_graph_sw(i,k,p)) for _ in range(ngraphs)]]
            elif Gtype=="block":
                p=2/math.sqrt(i/2)
                q=0.2/math.sqrt(i/2)
                G_lst+=[[g.graphs.Graph(gen_graph_block(round(i/2),p,q)) for _ in range(ngraphs)]]
            elif Gtype=="pa":
                G_lst+=[[g.graphs.Graph(gen_graph_pa(i,m)) for _ in range(ngraphs)]]
        #The actual sampling problem +reconstruction
        total=np.zeros((int((maxsize-minsize)/stepsize),ngraphs*ntrials,2))
        count=0
        for j in G_lst:
            tot=np.zeros((ntrials,2))
            c=0
            for i in j:
                for k in range(ntrials):
                    bandlim=round(bandlim_ratio*i.N)
                    sig=BL_sig_L(i,bandlim)
                    sig=sig/np.linalg.norm(sig)
                    if ntype!=None:
                        sigma=kwargs["sigma"]
                        prob=kwargs["prob"]
                        if ntype=="gaus":
                            noise=np.random.multivariate_normal(np.zeros(i.N),np.identity(i.N)*(sigma**2))
                        elif ntype=="bern":
                            noise=np.random.choice([-1,1],size=i.N)*bernoulli.rvs(p,size=i.N)*((sigma**2/prob)**(1/2))
                        elif ntype=="bega":
                            bern=bernoulli.rvs(prob,size=i.N)
                            while sum(bern)==0:
                                bern=bernoulli.rvs(prob,size=i.N)
                            noise=bern*np.random.multivariate_normal(np.zeros(i.N),np.identity(i.N)*(sigma**2*i.N/sum(bern)))
                        sig+=noise
                    res_uni=sample_and_recover(sig,uniform_sampling(i,min(i.N,2*bandlim+1)),i,min(bandlim,i.N))
                    res_deg=sample_and_recover(sig,Degree_sampling(i,min(i.N,2*bandlim+1)),i,min(bandlim,i.N))
                    total[count,c*ntrials+k,0]=cost(res_uni,sig-noise)
                    total[count,c*ntrials+k,1]=cost(res_deg,sig-noise)
                c+=1
            count+=1
        #The statistical analysis:
        tot=total.mean(axis=1)
        std=total.std(axis=1)
        #Finishing with the plot:
        plt.plot([i for i in range(minsize,maxsize,stepsize)],tot)
        plt.fill_between([i for i in range(minsize,maxsize,stepsize)],(tot-std)[:,0],(tot+std)[:,0], alpha=0.3)
        plt.fill_between([i for i in range(minsize,maxsize,stepsize)],(tot-std)[:,1],(tot+std)[:,1], alpha=0.3)
        plt.xlabel("graph size")
        plt.ylabel("MSE")
        if Gtype=="er":
            plt.title("Comparison uniform and degree sampling \nfor Erdos-Renyi graphs")
        elif Gtype=="sw":
            plt.title("Comparison uniform and degree sampling \nfor Small World graphs")
        elif Gtype=="block":
            plt.title("Comparison uniform and degree sampling \nfor Block model graphs")
        elif Gtype=="pa":
            plt.title("Comparison uniform and degree sampling \nfor Preferential Attachment graphs")
        plt.legend(["Uniform","Degree"])
        lbnd,ubnd=kwargs["lbnd"],kwargs["ubnd"]
        plt.ylim(lbnd,ubnd)
        plt.show();
        print(str(ngraphs)+" graphs, "+str(ntrials)+" test signals, bandlimit "+ str(bandlim_ratio)+"N\n"
            "probability 1/sqrt(n), and sample size 2*bandlimit+1\n"
            "exponentiation of A to power "+str(expsize)+".\n"
            "sigma = "+str(sigma)+" and p = "+ str(prob)+" on "+str(ntype)+" noise.")

def uni_vs_absL(Gtype: str, ngraphs, p, ntrials, bandlim_ratio, minsize=10, maxsize=100, stepsize=5, expsize=10, ntype=None, **kwargs):
    #Defining the necessary extra parameters:
    if Gtype=="sw":
        k=kwargs["k"]
    if Gtype=="block":
        q=kwargs["q"]
    if Gtype=="pa":
        m=kwargs["m"]
    #Generating graph list:
    G_lst=[]
    for i in range(minsize,maxsize,stepsize):
        if Gtype=="er":
            p=2/math.sqrt(i)
            G_lst+=[[g.graphs.Graph(gen_graph_er(i,p)) for _ in range(ngraphs)]]
        elif Gtype=="sw":
            p=1/k
            G_lst+=[[g.graphs.Graph(gen_graph_sw(i,k,p)) for _ in range(ngraphs)]]
        elif Gtype=="block":
            p=2/math.sqrt(i/2)
            q=0.2/math.sqrt(i/2)
            G_lst+=[[g.graphs.Graph(gen_graph_block(round(i/2),p,q)) for _ in range(ngraphs)]]
        elif Gtype=="pa":
            G_lst+=[[g.graphs.Graph(gen_graph_pa(i,m)) for _ in range(ngraphs)]]
    #The actual sampling problem +reconstruction
    total=np.zeros((int((maxsize-minsize)/stepsize),ngraphs*ntrials,2))
    count=0
    for j in G_lst:
        tot=np.zeros((ntrials,2))
        c=0
        for i in j:
            for k in range(ntrials):
                bandlim=round(bandlim_ratio*i.N)
                sig=BL_sig_L(i,bandlim)
                sig=sig/np.linalg.norm(sig)
                if ntype!=None:
                    sigma=kwargs["sigma"]
                    prob=kwargs["prob"]
                    if ntype=="gaus":
                        noise=np.random.multivariate_normal(np.zeros(i.N),np.identity(i.N)*(sigma**2))
                    elif ntype=="bern":
                        noise=np.random.choice([-1,1],size=i.N)*bernoulli.rvs(p,size=i.N)*((sigma**2/prob)**(1/2))
                    elif ntype=="bega":
                        bern=bernoulli.rvs(prob,size=i.N)
                        while sum(bern)==0:
                            bern=bernoulli.rvs(prob,size=i.N)
                        noise=bern*np.random.multivariate_normal(np.zeros(i.N),np.identity(i.N)*(sigma**2*i.N/sum(bern)))
                    sig+=noise
                res_uni=sample_and_recover(sig,uniform_sampling(i,min(i.N,2*bandlim+1)),i,min(bandlim,i.N))
                res_exp=sample_and_recover(sig,absL_sampling(i,min(i.N,2*bandlim+1),expsize),i,min(bandlim,i.N))
                total[count,c*ntrials+k,0]=cost(res_uni,sig-noise)
                total[count,c*ntrials+k,1]=cost(res_exp,sig-noise)
            c+=1
        count+=1
    #The statistical analysis:
    tot=total.mean(axis=1)
    std=total.std(axis=1)
    #Finishing with the plot:
    plt.plot([i for i in range(minsize,maxsize,stepsize)],tot)
    plt.fill_between([i for i in range(minsize,maxsize,stepsize)],(tot-std)[:,0],(tot+std)[:,0], alpha=0.3)
    plt.fill_between([i for i in range(minsize,maxsize,stepsize)],(tot-std)[:,1],(tot+std)[:,1], alpha=0.3)
    plt.xlabel("graph size")
    plt.ylabel("MSE")
    if Gtype=="er":
        plt.title("Comparison uniform and exponential sampling \nfor "+r"Erdos-Renyi graphs using $|L^k|$")
    elif Gtype=="sw":
        plt.title("Comparison uniform and exponential sampling \nfor "+r"Small World graphs using $|L^k|$")
    elif Gtype=="block":
        plt.title("Comparison uniform and exponential sampling \nfor "+r"Block model graphs using $|L^k|$")
    elif Gtype=="pa":
        plt.title("Comparison uniform and exponential sampling \nfor "+r"Preferential Attachment graphs using $|L^k|$")
    plt.legend(["Uniform","Exponential"])
    lbnd,ubnd=kwargs["lbnd"],kwargs["ubnd"]
    plt.ylim(lbnd,ubnd)
    plt.show();
    print(str(ngraphs)+" graphs, "+str(ntrials)+" test signals, bandlimit "+ str(bandlim_ratio)+"N\n"
        "probability 1/sqrt(n), and sample size 2*bandlimit+1\n"
        "exponentiation of A to power "+str(expsize)+".\n"
        "sigma = "+str(sigma)+" and p = "+ str(prob)+" on "+str(ntype)+" noise.")

def uni_vs_Labs(Gtype: str, ngraphs, p, ntrials, bandlim_ratio, minsize=10, maxsize=100, stepsize=5, expsize=10, ntype=None, **kwargs):
    #Defining the necessary extra parameters:
    if Gtype=="sw":
        k=kwargs["k"]
    if Gtype=="block":
        q=kwargs["q"]
    if Gtype=="pa":
        m=kwargs["m"]
    #Generating graph list:
    G_lst=[]
    for i in range(minsize,maxsize,stepsize):
        if Gtype=="er":
            p=2/math.sqrt(i)
            G_lst+=[[g.graphs.Graph(gen_graph_er(i,p)) for _ in range(ngraphs)]]
        elif Gtype=="sw":
            p=1/k
            G_lst+=[[g.graphs.Graph(gen_graph_sw(i,k,p)) for _ in range(ngraphs)]]
        elif Gtype=="block":
            p=2/math.sqrt(i/2)
            q=0.2/math.sqrt(i/2)
            G_lst+=[[g.graphs.Graph(gen_graph_block(round(i/2),p,q)) for _ in range(ngraphs)]]
        elif Gtype=="pa":
            G_lst+=[[g.graphs.Graph(gen_graph_pa(i,m)) for _ in range(ngraphs)]]
    #The actual sampling problem +reconstruction
    total=np.zeros((int((maxsize-minsize)/stepsize),ngraphs*ntrials,2))
    count=0
    for j in G_lst:
        tot=np.zeros((ntrials,2))
        c=0
        for i in j:
            for k in range(ntrials):
                bandlim=round(bandlim_ratio*i.N)
                sig=BL_sig_L(i,bandlim)
                sig=sig/np.linalg.norm(sig)
                if ntype!=None:
                    sigma=kwargs["sigma"]
                    prob=kwargs["prob"]
                    if ntype=="gaus":
                        noise=np.random.multivariate_normal(np.zeros(i.N),np.identity(i.N)*(sigma**2))
                    elif ntype=="bern":
                        noise=np.random.choice([-1,1],size=i.N)*bernoulli.rvs(p,size=i.N)*((sigma**2/prob)**(1/2))
                    elif ntype=="bega":
                        bern=bernoulli.rvs(prob,size=i.N)
                        while sum(bern)==0:
                            bern=bernoulli.rvs(prob,size=i.N)
                        noise=bern*np.random.multivariate_normal(np.zeros(i.N),np.identity(i.N)*(sigma**2*i.N/sum(bern)))
                    sig+=noise
                res_uni=sample_and_recover(sig,uniform_sampling(i,min(i.N,2*bandlim+1)),i,min(bandlim,i.N))
                res_exp=sample_and_recover(sig,Labs_sampling(i,min(i.N,2*bandlim+1),expsize),i,min(bandlim,i.N))
                total[count,c*ntrials+k,0]=cost(res_uni,sig-noise)
                total[count,c*ntrials+k,1]=cost(res_exp,sig-noise)
            c+=1
        count+=1
    #The statistical analysis:
    tot=total.mean(axis=1)
    std=total.std(axis=1)
    #Finishing with the plot:
    plt.plot([i for i in range(minsize,maxsize,stepsize)],tot)
    plt.fill_between([i for i in range(minsize,maxsize,stepsize)],(tot-std)[:,0],(tot+std)[:,0], alpha=0.3)
    plt.fill_between([i for i in range(minsize,maxsize,stepsize)],(tot-std)[:,1],(tot+std)[:,1], alpha=0.3)
    plt.xlabel("graph size")
    plt.ylabel("MSE")
    if Gtype=="er":
        plt.title("Comparison uniform and exponential sampling \nfor "+r"Erdos-Renyi graphs using $|L|^k$")
    elif Gtype=="sw":
        plt.title("Comparison uniform and exponential sampling \nfor "+r"Small World graphs using $|L|^k$")
    elif Gtype=="block":
        plt.title("Comparison uniform and exponential sampling \nfor "+r"Block model graphs using $|L|^k$")
    elif Gtype=="pa":
        plt.title("Comparison uniform and exponential sampling \nfor "+r"Preferential Attachment graphs using $|L|^k$")
    plt.legend(["Uniform","Exponential"])
    lbnd,ubnd=kwargs["lbnd"],kwargs["ubnd"]
    plt.ylim(lbnd,ubnd)
    plt.show();
    print(str(ngraphs)+" graphs, "+str(ntrials)+" test signals, bandlimit "+ str(bandlim_ratio)+"N\n"
        "probability 1/sqrt(n), and sample size 2*bandlimit+1\n"
        "exponentiation of A to power "+str(expsize)+".\n"
        "sigma = "+str(sigma)+" and p = "+ str(prob)+" on "+str(ntype)+" noise.")

def uni_vs_RL1(Gtype: str, ngraphs, p, ntrials, bandlim_ratio, minsize=10, maxsize=100, stepsize=5, expsize=10, ntype=None, **kwargs):
    #Defining the necessary extra parameters:
    if Gtype=="sw":
        k=kwargs["k"]
    if Gtype=="block":
        q=kwargs["q"]
    if Gtype=="pa":
        m=kwargs["m"]
    #Generating graph list:
    G_lst=[]
    for i in range(minsize,maxsize,stepsize):
        if Gtype=="er":
            p=2/math.sqrt(i)
            G_lst+=[[g.graphs.Graph(gen_graph_er(i,p)) for _ in range(ngraphs)]]
        elif Gtype=="sw":
            p=1/k
            G_lst+=[[g.graphs.Graph(gen_graph_sw(i,k,p)) for _ in range(ngraphs)]]
        elif Gtype=="block":
            p=2/math.sqrt(i/2)
            q=0.2/math.sqrt(i/2)
            G_lst+=[[g.graphs.Graph(gen_graph_block(round(i/2),p,q)) for _ in range(ngraphs)]]
        elif Gtype=="pa":
            G_lst+=[[g.graphs.Graph(gen_graph_pa(i,m)) for _ in range(ngraphs)]]
    #The actual sampling problem +reconstruction
    total=np.zeros((int((maxsize-minsize)/stepsize),ngraphs*ntrials,2))
    count=0
    for j in G_lst:
        tot=np.zeros((ntrials,2))
        c=0
        for i in j:
            for k in range(ntrials):
                bandlim=round(bandlim_ratio*i.N)
                sig=BL_sig_L(i,bandlim)
                sig=sig/np.linalg.norm(sig)
                if ntype!=None:
                    sigma=kwargs["sigma"]
                    prob=kwargs["prob"]
                    if ntype=="gaus":
                        noise=np.random.multivariate_normal(np.zeros(i.N),np.identity(i.N)*(sigma**2))
                    elif ntype=="bern":
                        noise=np.random.choice([-1,1],size=i.N)*bernoulli.rvs(p,size=i.N)*((sigma**2/prob)**(1/2))
                    elif ntype=="bega":
                        bern=bernoulli.rvs(prob,size=i.N)
                        while sum(bern)==0:
                            bern=bernoulli.rvs(prob,size=i.N)
                        noise=bern*np.random.multivariate_normal(np.zeros(i.N),np.identity(i.N)*(sigma**2*i.N/sum(bern)))
                    sig+=noise
                res_uni=sample_and_recover(sig,uniform_sampling(i,min(i.N,2*bandlim+1)),i,min(bandlim,i.N))
                res_RL=sample_and_recover(sig,Root_L_sampling(i,min(i.N,2*bandlim+1)),i,min(bandlim,i.N))
                total[count,c*ntrials+k,0]=cost(res_uni,sig-noise)
                total[count,c*ntrials+k,1]=cost(res_RL,sig-noise)
            c+=1
        count+=1
    #The statistical analysis:
    tot=total.mean(axis=1)
    std=total.std(axis=1)
    #Finishing with the plot:
    plt.plot([i for i in range(minsize,maxsize,stepsize)],tot)
    plt.fill_between([i for i in range(minsize,maxsize,stepsize)],(tot-std)[:,0],(tot+std)[:,0], alpha=0.3)
    plt.fill_between([i for i in range(minsize,maxsize,stepsize)],(tot-std)[:,1],(tot+std)[:,1], alpha=0.3)
    plt.xlabel("graph size")
    plt.ylabel("MSE")
    if Gtype=="er":
        plt.title("Comparison uniform and exponential sampling \nfor "+r"Erdos-Renyi graphs using $|L^k|$, $k=\frac{1}{2}$", fontsize=14)
    elif Gtype=="sw":
        plt.title("Comparison uniform and exponential sampling \nfor "+r"Small World graphs using $|L^k|$, $k=\frac{1}{2}$", fontsize=14)
    elif Gtype=="block":
        plt.title("Comparison uniform and exponential sampling \nfor "+r"Block model graphs using $|L^k|$, $k=\frac{1}{2}$", fontsize=14)
    elif Gtype=="pa":
        plt.title("Comparison uniform and exponential sampling \nfor "+r"Preferential Attachment graphs using $|L^k|$, $k=\frac{1}{2}$", fontsize=14)
    plt.legend(["Uniform","Exponential"])
    lbnd,ubnd=kwargs["lbnd"],kwargs["ubnd"]
    plt.ylim(lbnd,ubnd)
    plt.show();
    print(str(ngraphs)+" graphs, "+str(ntrials)+" test signals, bandlimit "+ str(bandlim_ratio)+"N\n"
        "probability 1/sqrt(n), and sample size 2*bandlimit+1\n"
        "exponentiation of A to power "+str(expsize)+".\n"
        "sigma = "+str(sigma)+" and p = "+ str(prob)+" on "+str(ntype)+" noise.")

def uni_vs_DRL1(Gtype: str, ngraphs, p, ntrials, bandlim_ratio, minsize=10, maxsize=100, stepsize=5, expsize=10, ntype=None, **kwargs):
    #Defining the necessary extra parameters:
    if Gtype=="sw":
        k=kwargs["k"]
    if Gtype=="block":
        q=kwargs["q"]
    if Gtype=="pa":
        m=kwargs["m"]
    #Generating graph list:
    G_lst=[]
    for i in range(minsize,maxsize,stepsize):
        if Gtype=="er":
            p=2/math.sqrt(i)
            G_lst+=[[g.graphs.Graph(gen_graph_er(i,p)) for _ in range(ngraphs)]]
        elif Gtype=="sw":
            p=1/k
            G_lst+=[[g.graphs.Graph(gen_graph_sw(i,k,p)) for _ in range(ngraphs)]]
        elif Gtype=="block":
            p=2/math.sqrt(i/2)
            q=0.2/math.sqrt(i/2)
            G_lst+=[[g.graphs.Graph(gen_graph_block(round(i/2),p,q)) for _ in range(ngraphs)]]
        elif Gtype=="pa":
            G_lst+=[[g.graphs.Graph(gen_graph_pa(i,m)) for _ in range(ngraphs)]]
    #The actual sampling problem +reconstruction
    total=np.zeros((int((maxsize-minsize)/stepsize),ngraphs*ntrials,2))
    count=0
    for j in G_lst:
        tot=np.zeros((ntrials,2))
        c=0
        for i in j:
            for k in range(ntrials):
                bandlim=round(bandlim_ratio*i.N)
                sig=BL_sig_L(i,bandlim)
                sig=sig/np.linalg.norm(sig)
                if ntype!=None:
                    sigma=kwargs["sigma"]
                    prob=kwargs["prob"]
                    if ntype=="gaus":
                        noise=np.random.multivariate_normal(np.zeros(i.N),np.identity(i.N)*(sigma**2))
                    elif ntype=="bern":
                        noise=np.random.choice([-1,1],size=i.N)*bernoulli.rvs(p,size=i.N)*((sigma**2/prob)**(1/2))
                    elif ntype=="bega":
                        bern=bernoulli.rvs(prob,size=i.N)
                        while sum(bern)==0:
                            bern=bernoulli.rvs(prob,size=i.N)
                        noise=bern*np.random.multivariate_normal(np.zeros(i.N),np.identity(i.N)*(sigma**2*i.N/sum(bern)))
                    sig+=noise
                res_uni=sample_and_recover(sig,uniform_sampling(i,min(i.N,2*bandlim+1)),i,min(bandlim,i.N))
                res_RRL=sample_and_recover(sig,DRoot_L_sampling(i,min(i.N,2*bandlim+1)),i,min(bandlim,i.N))
                total[count,c*ntrials+k,0]=cost(res_uni,sig-noise)
                total[count,c*ntrials+k,1]=cost(res_RRL,sig-noise)
            c+=1
        count+=1
    #The statistical analysis:
    tot=total.mean(axis=1)
    std=total.std(axis=1)
    #Finishing with the plot:
    plt.plot([i for i in range(minsize,maxsize,stepsize)],tot)
    plt.fill_between([i for i in range(minsize,maxsize,stepsize)],(tot-std)[:,0],(tot+std)[:,0], alpha=0.3)
    plt.fill_between([i for i in range(minsize,maxsize,stepsize)],(tot-std)[:,1],(tot+std)[:,1], alpha=0.3)
    plt.xlabel("graph size")
    plt.ylabel("MSE")
    if Gtype=="er":
        plt.title("Comparison uniform and exponential sampling \nfor "+r"Erdos-Renyi graphs using $|L^k|$, $k=\frac{1}{4}$", fontsize=14)
    elif Gtype=="sw":
        plt.title("Comparison uniform and exponential sampling \nfor "+r"Small World graphs using $|L^k|$, $k=\frac{1}{4}$", fontsize=14)
    elif Gtype=="block":
        plt.title("Comparison uniform and exponential sampling \nfor "+r"Block model graphs using $|L^k|$, $k=\frac{1}{4}$", fontsize=14)
    elif Gtype=="pa":
        plt.title("Comparison uniform and exponential sampling \nfor "+r"Preferential Attachment graphs using $|L^k|$, $k=\frac{1}{4}$", fontsize=14)
    plt.legend(["Uniform","Exponential"])
    lbnd,ubnd=kwargs["lbnd"],kwargs["ubnd"]
    plt.ylim(lbnd,ubnd)
    plt.show();
    print(str(ngraphs)+" graphs, "+str(ntrials)+" test signals, bandlimit "+ str(bandlim_ratio)+"N\n"
        "probability 1/sqrt(n), and sample size 2*bandlimit+1\n"
        "exponentiation of A to power "+str(expsize)+".\n"
        "sigma = "+str(sigma)+" and p = "+ str(prob)+" on "+str(ntype)+" noise.")

def uni_vs_DDRL1(Gtype: str, ngraphs, p, ntrials, bandlim_ratio, minsize=10, maxsize=100, stepsize=5, expsize=10, ntype=None, **kwargs):
    #Defining the necessary extra parameters:
    if Gtype=="sw":
        k=kwargs["k"]
    if Gtype=="block":
        q=kwargs["q"]
    if Gtype=="pa":
        m=kwargs["m"]
    #Generating graph list:
    G_lst=[]
    for i in range(minsize,maxsize,stepsize):
        if Gtype=="er":
            p=2/math.sqrt(i)
            G_lst+=[[g.graphs.Graph(gen_graph_er(i,p)) for _ in range(ngraphs)]]
        elif Gtype=="sw":
            p=1/k
            G_lst+=[[g.graphs.Graph(gen_graph_sw(i,k,p)) for _ in range(ngraphs)]]
        elif Gtype=="block":
            p=2/math.sqrt(i/2)
            q=0.2/math.sqrt(i/2)
            G_lst+=[[g.graphs.Graph(gen_graph_block(round(i/2),p,q)) for _ in range(ngraphs)]]
        elif Gtype=="pa":
            G_lst+=[[g.graphs.Graph(gen_graph_pa(i,m)) for _ in range(ngraphs)]]
    #The actual sampling problem +reconstruction
    total=np.zeros((int((maxsize-minsize)/stepsize),ngraphs*ntrials,2))
    count=0
    for j in G_lst:
        tot=np.zeros((ntrials,2))
        c=0
        for i in j:
            for k in range(ntrials):
                bandlim=round(bandlim_ratio*i.N)
                sig=BL_sig_L(i,bandlim)
                sig=sig/np.linalg.norm(sig)
                if ntype!=None:
                    sigma=kwargs["sigma"]
                    prob=kwargs["prob"]
                    if ntype=="gaus":
                        noise=np.random.multivariate_normal(np.zeros(i.N),np.identity(i.N)*(sigma**2))
                    elif ntype=="bern":
                        noise=np.random.choice([-1,1],size=i.N)*bernoulli.rvs(p,size=i.N)*((sigma**2/prob)**(1/2))
                    elif ntype=="bega":
                        bern=bernoulli.rvs(prob,size=i.N)
                        while sum(bern)==0:
                            bern=bernoulli.rvs(prob,size=i.N)
                        noise=bern*np.random.multivariate_normal(np.zeros(i.N),np.identity(i.N)*(sigma**2*i.N/sum(bern)))
                    sig+=noise
                res_uni=sample_and_recover(sig,uniform_sampling(i,min(i.N,2*bandlim+1)),i,min(bandlim,i.N))
                res_RRRL=sample_and_recover(sig,DDRoot_L_sampling(i,min(i.N,2*bandlim+1)),i,min(bandlim,i.N))
                total[count,c*ntrials+k,0]=cost(res_uni,sig-noise)
                total[count,c*ntrials+k,1]=cost(res_RRRL,sig-noise)
            c+=1
        count+=1
    #The statistical analysis:
    tot=total.mean(axis=1)
    std=total.std(axis=1)
    #Finishing with the plot:
    plt.plot([i for i in range(minsize,maxsize,stepsize)],tot)
    plt.fill_between([i for i in range(minsize,maxsize,stepsize)],(tot-std)[:,0],(tot+std)[:,0], alpha=0.3)
    plt.fill_between([i for i in range(minsize,maxsize,stepsize)],(tot-std)[:,1],(tot+std)[:,1], alpha=0.3)
    plt.xlabel("graph size")
    plt.ylabel("MSE")
    if Gtype=="er":
        plt.title("Comparison uniform and exponential sampling \nfor "+r"Erdos-Renyi graphs using $|L^k|$, $k=\frac{1}{8}$", fontsize=14)
    elif Gtype=="sw":
        plt.title("Comparison uniform and exponential sampling \nfor "+r"Small World graphs using $|L^k|$, $k=\frac{1}{8}$", fontsize=14)
    elif Gtype=="block":
        plt.title("Comparison uniform and exponential sampling \nfor "+r"Block model graphs using $|L^k|$, $k=\frac{1}{8}$", fontsize=14)
    elif Gtype=="pa":
        plt.title("Comparison uniform and exponential sampling \nfor "+r"Preferential Attachment graphs using $|L^k|$, $k=\frac{1}{8}$", fontsize=14)
    plt.legend(["Uniform","Exponential"])
    lbnd,ubnd=kwargs["lbnd"],kwargs["ubnd"]
    plt.ylim(lbnd,ubnd)
    plt.show();
    print(str(ngraphs)+" graphs, "+str(ntrials)+" test signals, bandlimit "+ str(bandlim_ratio)+"N\n"
        "probability 1/sqrt(n), and sample size 2*bandlimit+1\n"
        "exponentiation of A to power "+str(expsize)+".\n"
        "sigma = "+str(sigma)+" and p = "+ str(prob)+" on "+str(ntype)+" noise.")

def uni_vs_coh(Gtype: str, ngraphs, p, ntrials, bandlim_ratio, minsize=10, maxsize=100, stepsize=5, expsize=10, ntype=None, **kwargs):
    #Defining the necessary extra parameters:
    if Gtype=="sw":
        k=kwargs["k"]
    if Gtype=="block":
        q=kwargs["q"]
    if Gtype=="pa":
        m=kwargs["m"]
    #Generating graph list:
    G_lst=[]
    for i in range(minsize,maxsize,stepsize):
        if Gtype=="er":
            p=2/math.sqrt(i)
            G_lst+=[[g.graphs.Graph(gen_graph_er(i,p)) for _ in range(ngraphs)]]
        elif Gtype=="sw":
            p=1/k
            G_lst+=[[g.graphs.Graph(gen_graph_sw(i,k,p)) for _ in range(ngraphs)]]
        elif Gtype=="block":
            p=2/math.sqrt(i/2)
            q=0.2/math.sqrt(i/2)
            G_lst+=[[g.graphs.Graph(gen_graph_block(round(i/2),p,q)) for _ in range(ngraphs)]]
        elif Gtype=="pa":
            G_lst+=[[g.graphs.Graph(gen_graph_pa(i,m)) for _ in range(ngraphs)]]
    #The actual sampling problem +reconstruction
    total=np.zeros((int((maxsize-minsize)/stepsize),ngraphs*ntrials,2))
    count=0
    for j in G_lst:
        tot=np.zeros((ntrials,2))
        c=0
        for i in j:
            for k in range(ntrials):
                bandlim=round(bandlim_ratio*i.N)
                sig=BL_sig_L(i,bandlim)
                sig=sig/np.linalg.norm(sig)
                if ntype!=None:
                    sigma=kwargs["sigma"]
                    prob=kwargs["prob"]
                    if ntype=="gaus":
                        noise=np.random.multivariate_normal(np.zeros(i.N),np.identity(i.N)*(sigma**2))
                    elif ntype=="bern":
                        noise=np.random.choice([-1,1],size=i.N)*bernoulli.rvs(p,size=i.N)*((sigma**2/prob)**(1/2))
                    elif ntype=="bega":
                        bern=bernoulli.rvs(prob,size=i.N)
                        while sum(bern)==0:
                            bern=bernoulli.rvs(prob,size=i.N)
                        noise=bern*np.random.multivariate_normal(np.zeros(i.N),np.identity(i.N)*(sigma**2*i.N/sum(bern)))
                    sig+=noise
                res_uni=sample_and_recover(sig,uniform_sampling(i,min(i.N,2*bandlim+1)),i,min(bandlim,i.N))
                res_coh=sample_and_recover(sig,Paper_prob_exact(i,min(i.N,2*bandlim+1), bandlim),i,min(bandlim,i.N))
                total[count,c*ntrials+k,0]=cost(res_uni,sig-noise)
                total[count,c*ntrials+k,1]=cost(res_coh,sig-noise)
            c+=1
        count+=1
    #The statistical analysis:
    tot=total.mean(axis=1)
    std=total.std(axis=1)
    #Finishing with the plot:
    plt.plot([i for i in range(minsize,maxsize,stepsize)],tot)
    plt.fill_between([i for i in range(minsize,maxsize,stepsize)],(tot-std)[:,0],(tot+std)[:,0], alpha=0.3)
    plt.fill_between([i for i in range(minsize,maxsize,stepsize)],(tot-std)[:,1],(tot+std)[:,1], alpha=0.3)
    plt.xlabel("graph size")
    plt.ylabel("MSE")
    if Gtype=="er":
        plt.title("Comparison uniform and optimal coherence sampling \nfor "+r"Erdos-Renyi graphs")
    elif Gtype=="sw":
        plt.title("Comparison uniform and optimal coherence sampling \nfor "+r"Small World graphs")
    elif Gtype=="block":
        plt.title("Comparison uniform and optimal coherence sampling \nfor "+r"Block model graphs")
    elif Gtype=="pa":
        plt.title("Comparison uniform and optimal coherence sampling \nfor "+r"Preferential Attachment graphs")
    plt.legend(["Uniform","Optimal coherence"])
    lbnd,ubnd=kwargs["lbnd"],kwargs["ubnd"]
    plt.ylim(lbnd,ubnd)
    plt.show();
    print(str(ngraphs)+" graphs, "+str(ntrials)+" test signals, bandlimit "+ str(bandlim_ratio)+"N\n"
        "probability 1/sqrt(n), and sample size 2*bandlimit+1\n"
        "exponentiation of A to power "+str(expsize)+".\n"
        "sigma = "+str(sigma)+" and p = "+ str(prob)+" on "+str(ntype)+" noise.")

def exp_vs_coh(Gtype: str, ngraphs, p, ntrials, bandlim_ratio, minsize=10, maxsize=100, stepsize=5, expsize=10, ntype=None, **kwargs):
    #Defining the necessary extra parameters:
    if Gtype=="sw":
        k=kwargs["k"]
    if Gtype=="block":
        q=kwargs["q"]
    if Gtype=="pa":
        m=kwargs["m"]
    #Generating graph list:
    G_lst=[]
    for i in range(minsize,maxsize,stepsize):
        if Gtype=="er":
            p=2/math.sqrt(i)
            G_lst+=[[g.graphs.Graph(gen_graph_er(i,p)) for _ in range(ngraphs)]]
        elif Gtype=="sw":
            p=1/k
            G_lst+=[[g.graphs.Graph(gen_graph_sw(i,k,p)) for _ in range(ngraphs)]]
        elif Gtype=="block":
            p=2/math.sqrt(i/2)
            q=0.2/math.sqrt(i/2)
            G_lst+=[[g.graphs.Graph(gen_graph_block(round(i/2),p,q)) for _ in range(ngraphs)]]
        elif Gtype=="pa":
            G_lst+=[[g.graphs.Graph(gen_graph_pa(i,m)) for _ in range(ngraphs)]]
    #The actual sampling problem +reconstruction
    total=np.zeros((int((maxsize-minsize)/stepsize),ngraphs*ntrials,2))
    count=0
    for j in G_lst:
        tot=np.zeros((ntrials,2))
        c=0
        for i in j:
            for k in range(ntrials):
                bandlim=round(bandlim_ratio*i.N)
                sig=BL_sig_L(i,bandlim)
                sig=sig/np.linalg.norm(sig)
                if ntype!=None:
                    sigma=kwargs["sigma"]
                    prob=kwargs["prob"]
                    if ntype=="gaus":
                        noise=np.random.multivariate_normal(np.zeros(i.N),np.identity(i.N)*(sigma**2))
                    elif ntype=="bern":
                        noise=np.random.choice([-1,1],size=i.N)*bernoulli.rvs(p,size=i.N)*((sigma**2/prob)**(1/2))
                    elif ntype=="bega":
                        bern=bernoulli.rvs(prob,size=i.N)
                        while sum(bern)==0:
                            bern=bernoulli.rvs(prob,size=i.N)
                        noise=bern*np.random.multivariate_normal(np.zeros(i.N),np.identity(i.N)*(sigma**2*i.N/sum(bern)))
                    sig+=noise
                res_exp=sample_and_recover(sig,Root_L_sampling(i,min(i.N,2*bandlim+1)),i,min(bandlim,i.N))
                res_coh=sample_and_recover(sig,Paper_prob_exact(i,min(i.N,2*bandlim+1), bandlim),i,min(bandlim,i.N))
                total[count,c*ntrials+k,0]=cost(res_exp,sig-noise)
                total[count,c*ntrials+k,1]=cost(res_coh,sig-noise)
            c+=1
        count+=1
    #The statistical analysis:
    tot=total.mean(axis=1)
    std=total.std(axis=1)
    #Finishing with the plot:
    plt.plot([i for i in range(minsize,maxsize,stepsize)],tot)
    plt.fill_between([i for i in range(minsize,maxsize,stepsize)],(tot-std)[:,0],(tot+std)[:,0], alpha=0.3)
    plt.fill_between([i for i in range(minsize,maxsize,stepsize)],(tot-std)[:,1],(tot+std)[:,1], alpha=0.3)
    plt.xlabel("graph size")
    plt.ylabel("MSE")
    if Gtype=="er":
        plt.title("Comparison exponential and optimal coherence sampling \nfor "+r"Erdos-Renyi graphs")
    elif Gtype=="sw":
        plt.title("Comparison exponential and optimal coherence sampling \nfor "+r"Small World graphs")
    elif Gtype=="block":
        plt.title("Comparison exponential and optimal coherence sampling \nfor "+r"Block model graphs")
    elif Gtype=="pa":
        plt.title("Comparison exponential and optimal coherence sampling \nfor "+r"Preferential Attachment graphs")
    plt.legend(["Exponential","Optimal coherence"])
    lbnd,ubnd=kwargs["lbnd"],kwargs["ubnd"]
    plt.ylim(lbnd,ubnd)
    plt.show();
    print(str(ngraphs)+" graphs, "+str(ntrials)+" test signals, bandlimit "+ str(bandlim_ratio)+"N\n"
        "probability 1/sqrt(n), and sample size 2*bandlimit+1\n"
        "exponentiation of A to power "+str(expsize)+".\n"
        "sigma = "+str(sigma)+" and p = "+ str(prob)+" on "+str(ntype)+" noise.")

def exprob_vs_coh(Gtype: str, ngraphs, p, ntrials, bandlim_ratio, minsize=10, maxsize=100, stepsize=5, expsize=10, ntype=None, **kwargs):
    #Defining the necessary extra parameters:
    if Gtype=="sw":
        k=kwargs["k"]
    if Gtype=="block":
        q=kwargs["q"]
    if Gtype=="pa":
        m=kwargs["m"]
    #Generating graph list:
    G_lst=[]
    for i in range(minsize,maxsize,stepsize):
        if Gtype=="er":
            p=2/math.sqrt(i)
            G_lst+=[[g.graphs.Graph(gen_graph_er(i,p)) for _ in range(ngraphs)]]
        elif Gtype=="sw":
            p=1/k
            G_lst+=[[g.graphs.Graph(gen_graph_sw(i,k,p)) for _ in range(ngraphs)]]
        elif Gtype=="block":
            p=2/math.sqrt(i/2)
            q=0.2/math.sqrt(i/2)
            G_lst+=[[g.graphs.Graph(gen_graph_block(round(i/2),p,q)) for _ in range(ngraphs)]]
        elif Gtype=="pa":
            G_lst+=[[g.graphs.Graph(gen_graph_pa(i,m)) for _ in range(ngraphs)]]
    #The actual sampling problem +reconstruction
    total=np.zeros((int((maxsize-minsize)/stepsize),ngraphs*ntrials,2))
    count=0
    for j in G_lst:
        tot=np.zeros((ntrials,2))
        c=0
        for i in j:
            for k in range(ntrials):
                bandlim=round(bandlim_ratio*i.N)
                sig=BL_sig_L(i,bandlim)
                sig=sig/np.linalg.norm(sig)
                if ntype!=None:
                    sigma=kwargs["sigma"]
                    prob=kwargs["prob"]
                    if ntype=="gaus":
                        noise=np.random.multivariate_normal(np.zeros(i.N),np.identity(i.N)*(sigma**2))
                    elif ntype=="bern":
                        noise=np.random.choice([-1,1],size=i.N)*bernoulli.rvs(p,size=i.N)*((sigma**2/prob)**(1/2))
                    elif ntype=="bega":
                        bern=bernoulli.rvs(prob,size=i.N)
                        while sum(bern)==0:
                            bern=bernoulli.rvs(prob,size=i.N)
                        noise=bern*np.random.multivariate_normal(np.zeros(i.N),np.identity(i.N)*(sigma**2*i.N/sum(bern)))
                    sig+=noise
                res_exp=sample_and_recover(sig,Root_L_prob(i,min(i.N,2*bandlim+1),expsize),i,min(bandlim,i.N))
                res_coh=sample_and_recover(sig,Paper_prob_exact(i,min(i.N,2*bandlim+1), bandlim),i,min(bandlim,i.N))
                total[count,c*ntrials+k,0]=cost(res_exp,sig-noise)
                total[count,c*ntrials+k,1]=cost(res_coh,sig-noise)
            c+=1
        count+=1
    #The statistical analysis:
    tot=total.mean(axis=1)
    std=total.std(axis=1)
    #Finishing with the plot:
    plt.plot([i for i in range(minsize,maxsize,stepsize)],tot)
    plt.fill_between([i for i in range(minsize,maxsize,stepsize)],(tot-std)[:,0],(tot+std)[:,0], alpha=0.3)
    plt.fill_between([i for i in range(minsize,maxsize,stepsize)],(tot-std)[:,1],(tot+std)[:,1], alpha=0.3)
    plt.xlabel("graph size")
    plt.ylabel("MSE")
    if Gtype=="er":
        plt.title("Comparison probabilistic exponential and optimal coherence sampling \nfor "+r"Erdos-Renyi graphs")
    elif Gtype=="sw":
        plt.title("Comparison probabilistic exponential and optimal coherence sampling \nfor "+r"Small World graphs")
    elif Gtype=="block":
        plt.title("Comparison probabilistic exponential and optimal coherence sampling \nfor "+r"Block model graphs")
    elif Gtype=="pa":
        plt.title("Comparison exponential and optimal coherence sampling \nfor "+r"Preferential Attachment graphs")
    plt.legend(["Exponential","Optimal coherence"])
    lbnd,ubnd=kwargs["lbnd"],kwargs["ubnd"]
    plt.ylim(lbnd,ubnd)
    plt.show();
    print(str(ngraphs)+" graphs, "+str(ntrials)+" test signals, bandlimit "+ str(bandlim_ratio)+"N\n"
        "probability 1/sqrt(n), and sample size 2*bandlimit+1\n"
        "exponentiation of A to power "+str(expsize)+".\n"
        "sigma = "+str(sigma)+" and p = "+ str(prob)+" on "+str(ntype)+" noise.")

#Request by Palina (This is used in Figure 26?)
def uni_vs_exp_vs_coh(Gtype: str, ngraphs, p, ntrials, bandlim_ratio, minsize=10, maxsize=100, stepsize=5, expsize=10, ntype=None, **kwargs):
    #Defining the necessary extra parameters:
    if Gtype=="sw":
        k=kwargs["k"]
    if Gtype=="block":
        q=kwargs["q"]
    if Gtype=="pa":
        m=kwargs["m"]
    #Generating graph list:
    G_lst=[]
    for i in range(minsize,maxsize,stepsize):
        if Gtype=="er":
            p=2/math.sqrt(i)
            G_lst+=[[g.graphs.Graph(gen_graph_er(i,p)) for _ in range(ngraphs)]]
        elif Gtype=="sw":
            p=1/k
            G_lst+=[[g.graphs.Graph(gen_graph_sw(i,k,p)) for _ in range(ngraphs)]]
        elif Gtype=="block":
            p=2/math.sqrt(i/2)
            q=0.2/math.sqrt(i/2)
            G_lst+=[[g.graphs.Graph(gen_graph_block(round(i/2),p,q)) for _ in range(ngraphs)]]
        elif Gtype=="pa":
            G_lst+=[[g.graphs.Graph(gen_graph_pa(i,m)) for _ in range(ngraphs)]]
    #The actual sampling problem +reconstruction
    total=np.zeros((int((maxsize-minsize)/stepsize),ngraphs*ntrials,3))
    count=0
    for j in G_lst:
        tot=np.zeros((ntrials,2))
        c=0
        for i in j:
            for k in range(ntrials):
                bandlim=round(bandlim_ratio*i.N)
                sig=BL_sig_L(i,bandlim)
                sig=sig/np.linalg.norm(sig)
                if ntype!=None:
                    sigma=kwargs["sigma"]
                    prob=kwargs["prob"]
                    if ntype=="gaus":
                        noise=np.random.multivariate_normal(np.zeros(i.N),np.identity(i.N)*(sigma**2))
                    elif ntype=="bern":
                        noise=np.random.choice([-1,1],size=i.N)*bernoulli.rvs(p,size=i.N)*((sigma**2/prob)**(1/2))
                    elif ntype=="bega":
                        bern=bernoulli.rvs(prob,size=i.N)
                        while sum(bern)==0:
                            bern=bernoulli.rvs(prob,size=i.N)
                        noise=bern*np.random.multivariate_normal(np.zeros(i.N),np.identity(i.N)*(sigma**2*i.N/sum(bern)))
                    sig+=noise
                res_uni=sample_and_recover(sig,uniform_sampling(i,min(i.N,2*bandlim+1)),i,min(bandlim,i.N))
                res_exp=sample_and_recover(sig,Root_L_sampling(i,min(i.N,2*bandlim+1)),i,min(bandlim,i.N))
                res_coh=sample_and_recover(sig,Paper_prob_exact(i,min(i.N,2*bandlim+1), bandlim),i,min(bandlim,i.N))
                total[count,c*ntrials+k,0]=cost(res_uni,sig-noise)
                total[count,c*ntrials+k,1]=cost(res_exp,sig-noise)
                total[count,c*ntrials+k,2]=cost(res_coh,sig-noise)
            c+=1
        count+=1
    #The statistical analysis:
    tot=total.mean(axis=1)
    std=total.std(axis=1)
    #Finishing with the plot:
    plt.plot([i for i in range(minsize,maxsize,stepsize)],tot)
    plt.fill_between([i for i in range(minsize,maxsize,stepsize)],(tot-std)[:,0],(tot+std)[:,0], alpha=0.3)
    plt.fill_between([i for i in range(minsize,maxsize,stepsize)],(tot-std)[:,1],(tot+std)[:,1], alpha=0.3)
    plt.fill_between([i for i in range(minsize,maxsize,stepsize)],(tot-std)[:,2],(tot+std)[:,2], alpha=0.3)
    plt.xlabel("graph size")
    plt.ylabel("MSE")
    if Gtype=="er":
        plt.title("Comparison uniform, exponential, and optimal coherence sampling \nfor "+r"Erdos-Renyi graphs")
    elif Gtype=="sw":
        plt.title("Comparison uniform, exponential, and optimal coherence sampling \nfor "+r"Small World graphs")
    elif Gtype=="block":
        plt.title("Comparison uniform, exponential, and optimal coherence sampling \nfor "+r"Block model graphs")
    elif Gtype=="pa":
        plt.title("Comparison uniform, exponential, and optimal coherence sampling \nfor "+r"Preferential Attachment graphs")
    plt.legend(["Uniform","Exponential","Optimal coherence"])
    lbnd,ubnd=kwargs["lbnd"],kwargs["ubnd"]
    plt.ylim(lbnd,ubnd)
    plt.show();

#Additionally, we would like to plot some of the very non-interesting influence
# (it is basically constant) of the parameter gamma:
def Gamma_parameter_plot_size(Gtype: str, ngraphs, sz, ntrials, bandlim_ratio, minsize=10, maxsize=100, stepsize=5, expsize=10, ntype=None, **kwargs):
    #Defining the necessary extra parameters:
    if Gtype=="sw":
        k=kwargs["k"]
    if Gtype=="block":
        q=kwargs["q"]
    if Gtype=="pa":
        m=kwargs["m"]
    #Generating graph list:
    if Gtype=="er":
        p=2/math.sqrt(sz)
        G_lst=[g.graphs.Graph(gen_graph_er(sz,p)) for _ in range(ngraphs)]
    elif Gtype=="sw":
        p=1/k
        G_lst=[g.graphs.Graph(gen_graph_sw(sz,k,p)) for _ in range(ngraphs)]
    elif Gtype=="block":
        p=2/math.sqrt(sz/2)
        q=0.2/math.sqrt(sz/2)
        G_lst=[g.graphs.Graph(gen_graph_block(round(sz/2),p,q)) for _ in range(ngraphs)]
    elif Gtype=="pa":
        G_lst=[g.graphs.Graph(gen_graph_pa(sz,m)) for _ in range(ngraphs)]
    #The actual sampling problem +reconstruction
    total=np.zeros((ngraphs*ntrials,20))
    count=0
    problematic_graphs=[]
    c=0
    gms=[]
    for i in range(10):
        gms+=[0.0001*10**i]
        gms+=[0.0005*10**i]
    for j in G_lst:
        count=0
        for gmma in gms:
            for k in range(ntrials):
                bandlim=round(bandlim_ratio*j.N)
                sig=BL_sig_L(j,bandlim)
                sig=sig/np.linalg.norm(sig)
                if ntype!=None:
                    sigma=kwargs["sigma"]
                    prob=kwargs["prob"]
                    if ntype=="gaus":
                        noise=np.random.multivariate_normal(np.zeros(j.N),np.identity(j.N)*(sigma**2))
                    elif ntype=="bern":
                        noise=np.random.choice([-1,1],size=j.N)*bernoulli.rvs(p,size=j.N)*((sigma**2/prob)**(1/2))
                    elif ntype=="bega":
                        bern=bernoulli.rvs(prob,size=j.N)
                        while sum(bern)==0:
                            bern=bernoulli.rvs(prob,size=j.N)
                        noise=bern*np.random.multivariate_normal(np.zeros(j.N),np.identity(j.N)*(sigma**2*j.N/sum(bern)))
                    sig+=noise
                res_GL2=sample_and_recover(sig,D_plus_gamma_A_sampling(j,min(j.N,2*bandlim+1),gmma),j,min(bandlim,j.N))
                total[c*ntrials+k, count]=cost(res_GL2,sig-noise)
                #if total[count,c*ntrials+k,0]>1 or total[count,c*ntrials+k,1]>1:
                #    problematic_graphs+=[j]
            count+=1
        c+=1
    #The statistical analysis:
    tot=total.mean(axis=0)
    std=total.std(axis=0)
    #Finishing with the plot:
    plt.plot(gms,tot)
    plt.fill_between(gms,(tot-std)[:],(tot+std)[:], alpha=0.3)
    plt.xlabel(r"$\gamma$")
    plt.ylabel("MSE")
    plt.xscale("log")
    if Gtype=="er":
        plt.title("Comparison of exponential sampling \nfor "+r"Erdos-Renyi graphs using various $\gamma$ values")
    elif Gtype=="sw":
        plt.title("Comparison of exponential sampling \nfor "+r"Small World graphs using various $\gamma$ values")
    elif Gtype=="block":
        plt.title("Comparison of exponential sampling \nfor "+r"Block model graphs using various $\gamma$ values")
    elif Gtype=="pa":
        plt.title("Comparison of exponential sampling \nfor "+r"Preferential Attachment graphs using various $\gamma$ values")
    plt.show() 

#%%
# # Fig 13
# uni_vs_exp1("er", 10, 10, 0.2, minsize=10, maxsize=210, stepsize=10, expsize=1, ntype="gaus", sigma=0.1, prob=1,lbnd=-0.4, ubnd=1.2)
# uni_vs_exp1("er", 10, 10, 0.2, minsize=10, maxsize=210, stepsize=10, expsize=5, ntype="gaus", sigma=0.1, prob=1,lbnd=-0.4, ubnd=1.2)
# uni_vs_exp1("er", 10, 10, 0.2, minsize=10, maxsize=210, stepsize=10, expsize=10, ntype="gaus", sigma=0.1, prob=1,lbnd=-0.4, ubnd=1.2)
# uni_vs_exp1("er", 10, 10, 0.2, minsize=10, maxsize=210, stepsize=10, expsize=50, ntype="gaus", sigma=0.1, prob=1,lbnd=-0.4, ubnd=1.2)
# uni_vs_exp1("er", 10, 10, 0.2, minsize=10, maxsize=210, stepsize=10, expsize=1, ntype="gaus", sigma=0.01, prob=1,lbnd=-0.02, ubnd=0.1)
# uni_vs_exp1("er", 10, 10, 0.2, minsize=10, maxsize=210, stepsize=10, expsize=5, ntype="gaus", sigma=0.01, prob=1,lbnd=-0.02, ubnd=0.1)
# uni_vs_exp1("er", 10, 10, 0.2, minsize=10, maxsize=210, stepsize=10, expsize=10, ntype="gaus", sigma=0.01, prob=1,lbnd=-0.02, ubnd=0.1)
# uni_vs_exp1("er", 10, 10, 0.2, minsize=10, maxsize=210, stepsize=10, expsize=50, ntype="gaus", sigma=0.01, prob=1,lbnd=-0.02, ubnd=0.1)
# uni_vs_exp1("er", 10, 10, 0.2, minsize=10, maxsize=210, stepsize=10, expsize=1, ntype="gaus", sigma=0.001, prob=1,lbnd=-0.0005, ubnd=0.001)
# uni_vs_exp1("er", 10, 10, 0.2, minsize=10, maxsize=210, stepsize=10, expsize=5, ntype="gaus", sigma=0.001, prob=1,lbnd=-0.0005, ubnd=0.001)
# uni_vs_exp1("er", 10, 10, 0.2, minsize=10, maxsize=210, stepsize=10, expsize=10, ntype="gaus", sigma=0.001, prob=1,lbnd=-0.0005, ubnd=0.001)
# uni_vs_exp1("er", 10, 10, 0.2, minsize=10, maxsize=210, stepsize=10, expsize=50, ntype="gaus", sigma=0.001, prob=1,lbnd=-0.0005, ubnd=0.001)

# # Fig 14
# uni_vs_exp1("block", 10, 10, 0.2, minsize=10, maxsize=210, stepsize=10, expsize=5, ntype="gaus", sigma=0.01, prob=1,lbnd=-0.02, ubnd=0.1)
# uni_vs_exp1("block", 10, 10, 0.2, minsize=10, maxsize=210, stepsize=10, expsize=15, ntype="gaus", sigma=0.01, prob=1,lbnd=-0.02, ubnd=0.1)
# uni_vs_exp1("sw", 10, 10, 0.2, minsize=10, maxsize=210, stepsize=10, expsize=5, ntype="gaus", sigma=0.01, prob=0.1,lbnd=-0.04, ubnd=0.2, k=4)
# uni_vs_exp1("pa", 10, 10, 0.2, minsize=10, maxsize=210, stepsize=10, expsize=5, ntype="gaus", sigma=0.01, prob=0.1,lbnd=-0.04, ubnd=0.2, m=2)

# # Fig 15
# uni_vs_Nexp1("er", 10, 10, 0.2, minsize=10, maxsize=210, stepsize=10, expsize=2, ntype="gaus", sigma=0.01, prob=1,lbnd=-0.4, ubnd=1.2)
# uni_vs_Nexp1("sw", 10, 10, 0.2, minsize=10, maxsize=210, stepsize=10, expsize=2, ntype="gaus", sigma=0.01, prob=1,lbnd=-0.4, ubnd=1.2, K=4)
# uni_vs_Nexp1("pa", 10, 10, 0.2, minsize=10, maxsize=210, stepsize=10, expsize=2, ntype="gaus", sigma=0.01, prob=1,lbnd=-0.4, ubnd=1.2, m=2)

# # Fig 16
# uni_vs_exp1("er", 10, 10, 0.2, minsize=10, maxsize=210, stepsize=10, expsize=5, ntype="bern", sigma=0.01, prob=0.2,lbnd=-0.02, ubnd=0.1)
# uni_vs_exp1("block", 10, 10, 0.2, minsize=10, maxsize=210, stepsize=10, expsize=5, ntype="bern", sigma=0.01, prob=0.2,lbnd=-0.02, ubnd=0.1)
# uni_vs_exp1("sw", 10, 10, 0.2, minsize=10, maxsize=210, stepsize=10, expsize=5, ntype="bern", sigma=0.01, prob=0.2,lbnd=-0.04, ubnd=0.2, k=4)
# uni_vs_exp1("pa", 10, 10, 0.2, minsize=10, maxsize=210, stepsize=10, expsize=5, ntype="bern", sigma=0.01, prob=0.2,lbnd=-0.04, ubnd=0.2, m=2)
        
# # Fig 17
# PA=g.graphs.Graph(gen_graph_pa(50,1))
# plot_sample(PA, Degree_sampling(PA,21), title="Degree sampling on a preferential attachment graph");
# plt.show();
# uni_vs_deg1("pa", 10, 0.1, 10, 0.2, minsize=10, maxsize=210, stepsize=10, expsize=5, ntype="gaus", sigma=0.01, prob=1,lbnd=-.3, ubnd=1.5, m=1);
# uni_vs_deg1("er", 10, 0.3, 10, 0.2, minsize=10, maxsize=210, stepsize=10, expsize=5, ntype="gaus", sigma=0.01, prob=1,lbnd=-.02, ubnd=.1, m=1);

# # Fig 18 abc
# uni_vs_absL("sw", 10, 0.1, 10, 0.2, minsize=10, maxsize=210, stepsize=10, expsize=2, ntype="gaus", sigma=0.01, prob=1,lbnd=-0.02, ubnd=0.08, k=4);
# uni_vs_Labs("sw", 10, 0.1, 10, 0.2, minsize=10, maxsize=210, stepsize=10, expsize=2, ntype="gaus", sigma=0.01, prob=1,lbnd=-0.02, ubnd=0.08, k=4);
# uni_vs_exp1("sw", 10, 10, 0.2, minsize=10, maxsize=210, stepsize=10, expsize=2, ntype="gaus", sigma=0.01, prob=1,lbnd=-0.02, ubnd=0.08, k=4);

# # Fig 18 de
# uni_vs_absL("pa", 10, 0.1, 10, 0.2, minsize=10, maxsize=210, stepsize=10, expsize=2, ntype="gaus", sigma=0.01, prob=1,lbnd=-0.02, ubnd=0.08, m=2);
# uni_vs_Labs("pa", 10, 0.1, 10, 0.2, minsize=10, maxsize=210, stepsize=10, expsize=2, ntype="gaus", sigma=0.01, prob=1,lbnd=-0.02, ubnd=0.08, m=2);

# Fig 19 ab
# uni_vs_absL("pa", 10, 0.1, 10, 0.2, minsize=10, maxsize=210, stepsize=10, expsize=20, ntype="gaus", sigma=0.01, prob=1,lbnd=-0.02, ubnd=0.08, m=2);
# uni_vs_Labs("pa", 10, 0.1, 10, 0.2, minsize=10, maxsize=210, stepsize=10, expsize=20, ntype="gaus", sigma=0.01, prob=1,lbnd=-0.02, ubnd=0.08, m=2);

#%%
# # Fig 20 abc
# Gamma_parameter_plot_size("er", 10, 50, ntrials=10, bandlim_ratio=0.2, ntype="gaus", sigma=0.01, prob=1,lbnd=-0.02, ubnd=0.08);
# Gamma_parameter_plot_size("sw", 10, 50, ntrials=10, bandlim_ratio=0.2, ntype="gaus", sigma=0.01, prob=1,lbnd=-0.02, ubnd=0.08, k=4);
# Gamma_parameter_plot_size("pa", 10, 50, ntrials=10, bandlim_ratio=0.2, ntype="gaus", sigma=0.01, prob=1,lbnd=-0.02, ubnd=0.08, m=2);

# Gamma_parameter_plot_size("er", 10, 100, ntrials=10, bandlim_ratio=0.2, ntype="gaus", sigma=0.01, prob=1,lbnd=-0.02, ubnd=0.08);
# Gamma_parameter_plot_size("sw", 10, 100, ntrials=10, bandlim_ratio=0.2, ntype="gaus", sigma=0.01, prob=1,lbnd=-0.02, ubnd=0.08, k=4);
# Gamma_parameter_plot_size("pa", 10, 100, ntrials=10, bandlim_ratio=0.2, ntype="gaus", sigma=0.01, prob=1,lbnd=-0.02, ubnd=0.08, m=2);

# # Fig 21 abc
# uni_vs_RL1("er", 10, 0.1, 10, 0.2, minsize=10, maxsize=210, stepsize=10, expsize=2, ntype="gaus", sigma=0.01, prob=1,lbnd=-0.02, ubnd=0.08);
# uni_vs_RL1("sw", 10, 0.1, 10, 0.2, minsize=10, maxsize=210, stepsize=10, expsize=2, ntype="gaus", sigma=0.01, prob=1,lbnd=-0.02, ubnd=0.08, k=4);
# uni_vs_RL1("pa", 10, 0.1, 10, 0.2, minsize=10, maxsize=210, stepsize=10, expsize=2, ntype="gaus", sigma=0.01, prob=1,lbnd=-0.02, ubnd=0.08, m=2);

# # Fig 21 def
# uni_vs_DRL1("er", 10, 0.1, 10, 0.2, minsize=10, maxsize=210, stepsize=10, expsize=2, ntype="gaus", sigma=0.01, prob=1,lbnd=-0.02, ubnd=0.08);
# uni_vs_DRL1("sw", 10, 0.1, 10, 0.2, minsize=10, maxsize=210, stepsize=10, expsize=2, ntype="gaus", sigma=0.01, prob=1,lbnd=-0.02, ubnd=0.08, k=4);
# uni_vs_DRL1("pa", 10, 0.1, 10, 0.2, minsize=10, maxsize=210, stepsize=10, expsize=2, ntype="gaus", sigma=0.01, prob=1,lbnd=-0.02, ubnd=0.08, m=2);

# # Fig 21 ghi
# uni_vs_DDRL1("er", 10, 0.1, 10, 0.2, minsize=10, maxsize=210, stepsize=10, expsize=2, ntype="gaus", sigma=0.01, prob=1,lbnd=-0.02, ubnd=0.08);
# uni_vs_DDRL1("sw", 10, 0.1, 10, 0.2, minsize=10, maxsize=210, stepsize=10, expsize=2, ntype="gaus", sigma=0.01, prob=1,lbnd=-0.02, ubnd=0.08, k=4);
# uni_vs_DDRL1("pa", 10, 0.1, 10, 0.2, minsize=10, maxsize=210, stepsize=10, expsize=2, ntype="gaus", sigma=0.01, prob=1,lbnd=-0.02, ubnd=0.08, m=2);

# # Fig 22
# PA=g.graphs.Graph(gen_graph_pa(50,1))
# plot_sample(PA, Exponential_sampling(PA,21,2), title=r"Exponential sampling on a preferential attachment graph, using $k=2$");
# plt.show();
# plot_sample(PA, Root_L_sampling(PA,21), title=r"Exponential sampling on a preferential attachment graph using $k=\frac{1}{2}$");
# plt.show();

# # Fig 23
# uni_vs_coh("er", 10, 0.1, 10, 0.2, minsize=10, maxsize=210, stepsize=10, expsize=2, ntype="gaus", sigma=0.01, prob=1,lbnd=-0.02, ubnd=0.08);

# # Fig 24
# exp_vs_coh("er", 10, 0.1, 10, 0.2, minsize=10, maxsize=210, stepsize=10, expsize=2, ntype="gaus", sigma=0.01, prob=1,lbnd=-0.002, ubnd=0.015);
# exp_vs_coh("sw", 10, 0.1, 10, 0.2, minsize=10, maxsize=210, stepsize=10, expsize=2, ntype="gaus", sigma=0.01, prob=1,lbnd=-0.02, ubnd=0.08, k=4);
# exp_vs_coh("pa", 10, 0.1, 10, 0.2, minsize=10, maxsize=210, stepsize=10, expsize=2, ntype="gaus", sigma=0.01, prob=1,lbnd=-0.015, ubnd=0.045, m=2);

# # Fig 25
# exprob_vs_coh("er", 10, 0.1, 10, 0.2, minsize=10, maxsize=210, stepsize=10, expsize=1, ntype="gaus", sigma=0.01, prob=1,lbnd=-0.01, ubnd=0.06);
# exprob_vs_coh("sw", 10, 0.1, 10, 0.2, minsize=10, maxsize=210, stepsize=10, expsize=1, ntype="gaus", sigma=0.01, prob=1,lbnd=-0.01, ubnd=0.06, k=4);
# exprob_vs_coh("pa", 10, 0.1, 10, 0.2, minsize=10, maxsize=210, stepsize=10, expsize=1, ntype="gaus", sigma=0.01, prob=1,lbnd=-0.01, ubnd=0.06, m=2);

# # Fig 26
# exprob_vs_coh("er", 10, 0.1, 10, 0.2, minsize=10, maxsize=210, stepsize=10, expsize=10000, ntype="gaus", sigma=0.01, prob=1,lbnd=-0.01, ubnd=0.06);
# exprob_vs_coh("sw", 10, 0.1, 10, 0.2, minsize=10, maxsize=210, stepsize=10, expsize=10000, ntype="gaus", sigma=0.01, prob=1,lbnd=-0.01, ubnd=0.06, k=4);
# exprob_vs_coh("pa", 10, 0.1, 10, 0.2, minsize=10, maxsize=210, stepsize=10, expsize=10000, ntype="gaus", sigma=0.01, prob=1,lbnd=-0.01, ubnd=0.06, m=2);

#Work for Palina (probably a figure that I would use in my report):

# This is Fig 24 in the report, using ER, SW and PA.
# uni_vs_exp_vs_coh("er", 10, 0.1, 10, 0.2, minsize=10, maxsize=210, stepsize=10, expsize=2, ntype="gaus", sigma=0.01, prob=1,lbnd=-0.02, ubnd=0.08);
# uni_vs_exp_vs_coh("sw", 10, 0.1, 10, 0.2, minsize=10, maxsize=210, stepsize=10, expsize=2, ntype="gaus", sigma=0.01, prob=1,lbnd=-0.02, ubnd=0.08, k=4);
# uni_vs_exp_vs_coh("pa", 10, 0.1, 10, 0.2, minsize=10, maxsize=210, stepsize=10, expsize=2, ntype="gaus", sigma=0.01, prob=1,lbnd=-0.02, ubnd=0.08, m=2);
# uni_vs_exp_vs_coh("block", 10, 0.1, 10, 0.2, minsize=20, maxsize=210, stepsize=10, expsize=10000, ntype="gaus", sigma=0.01, prob=1,lbnd=-0.02, ubnd=0.08, q=0.2);

#%%
# Plotting example graphs
# size=20
# pb=0.2
# q=0.01
# k=4
# m=2

# Erdos Renyi
# ER=g.graphs.Graph(gen_graph_er(size,pb))
# ER.set_coordinates()
# fig, ax1=plt.subplots(1,1)
# ER.plot(vertex_color="b", ax=ax1)
# ax1.set_xticks([])
# ax1.set_title("Erdos-Renyi")
# ax1.set_yticks([]);

# Block model
# BM=g.graphs.Graph(gen_graph_block(int(2*size),pb,q))
# BM.set_coordinates()
# fig, ax1=plt.subplots(1,1)
# BM.plot(vertex_color="b", ax=ax1)
# ax1.set_xticks([])
# ax1.set_title("Block model")
# ax1.set_yticks([]);

# Small world
# SW=g.graphs.Graph(gen_graph_sw(size,k,1/k))
# SW.set_coordinates()
# SW.coords=np.array([[math.cos(2*t*math.pi/SW.N),math.sin(2*t*math.pi/SW.N)] for t in range(SW.N)]) 
# fig, ax1=plt.subplots(1,1)
# SW.plot(vertex_color="b", ax=ax1)
# ax1.set_xticks([])
# ax1.set_title("Small world")
# ax1.set_yticks([]);

# Preferential attachment
# PA=g.graphs.Graph(gen_graph_pa(size,m))
# PA.set_coordinates()
# fig, ax1=plt.subplots(1,1)
# PA.plot(vertex_color="b", ax=ax1)
# ax1.set_xticks([])
# ax1.set_title("Preferential attachment")
# ax1.set_yticks([]);

#%%
# Fig 9abc
# a:
plt.plot([i+1 for i in range(1,size)],tot_Agr[0])

# all:
plt.plot([i+1 for i in range(1,size)],tot_Opt[0])
plt.fill_between([i+1 for i in range(1,size)],tot_Agr[1],tot_Agr[2],alpha=.3)
plt.xlabel("Sample size")
plt.ylabel("log MSE")
plt.legend(["A-design", "optimal sample"])
plt.title("Comparison of A optimal design sampling versus the true optimum");
plt.show()

# b:
plt.plot([i+1 for i in range(1,size)],tot_Egr[0])

# all:
plt.plot([i+1 for i in range(1,size)],tot_Opt[0])
plt.fill_between([i+1 for i in range(1,size)],tot_Egr[1],tot_Egr[2],alpha=.3)
plt.xlabel("Sample size")
plt.ylabel("log MSE")
plt.legend(["E-design", "optimal sample"])
plt.title("Comparison of E optimal design sampling versus the true optimum");
plt.show()

# c:
plt.plot([i+1 for i in range(1,size)],tot_Dgr[0])

# all:
plt.plot([i+1 for i in range(1,size)],tot_Opt[0])
plt.fill_between([i+1 for i in range(1,size)],tot_Dgr[1],tot_Dgr[2],alpha=.3)
plt.xlabel("Sample size")
plt.ylabel("log MSE")
plt.legend(["D-design", "optimal sample"])
plt.title("Comparison of D optimal design sampling versus the true optimum");
plt.show()