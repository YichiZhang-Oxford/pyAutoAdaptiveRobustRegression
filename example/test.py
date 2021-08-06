import numpy as np
import pyAutoAdaptiveRobustRegression as arr
from tqdm import tqdm
import matplotlib.pyplot as plt
from sstudentt import SST

# ---------------
# Mean Estimation
# ---------------

n = 1000
X=np.random.lognormal(0,1.5,n)-np.exp(1.5**2/2)
huber_mean_result = arr.huber_mean(X)
print("Huber Mean:", huber_mean_result)

agd_bb_result = arr.agd_bb(X)
print("AGD BB Mean:", agd_bb_result)

agd_result = arr.agd(X)
print("AGD Mean:", agd_result)

# ----------------------------------
# Huber Covariance Matrix Estimation
# ----------------------------------

n = 100
d = 50
X = np.random.standard_t(n*d,(n,d))/np.sqrt(3)
hubCov = arr.huber_cov(X)
print("Huber Covariance:", hubCov)

# ---------------------------
# Huber Regression Estimation
# ---------------------------

n = 500
d = 5
thetaStar = np.repeat(3, d + 1)
X = np.random.normal( size=(n , d))
error=np.random.lognormal(0,1.5,n)-np.exp(1.5**2/2)
Y = np.squeeze(np.dot(np.hstack((np.repeat(1,n).reshape((n,1)),X)),thetaStar.reshape(d+1,1)))+error
huber_reg = arr.huber_reg(X, Y)
print("Huber Regression:", huber_reg)

# ===========================================================================================================
# Figure 1
# ===========================================================================================================

# -------------
# Figure 1 Plot
# -------------

alpha_list = [round(x,2) for x in np.arange(0.50, 1.01, 0.01)]

def plot_data(alpha_list, error_dict):
    sample_mean_error_list = []
    huber_mean_error_list = []
    agdBB_mean_error_list = []
    agd_mean_error_list = []
    for alpha in alpha_list:
        sample_mean_error_list.append(np.quantile(error_dict["sample_mean_error_array"],alpha))
        huber_mean_error_list.append(np.quantile(error_dict["huber_mean_error_array"],alpha))
        agdBB_mean_error_list.append(np.quantile(error_dict["agdBB_mean_error_array"],alpha))
        agd_mean_error_list.append(np.quantile(error_dict["agd_mean_error_array"],alpha))
    return {"sample_mean_error_list":sample_mean_error_list,
            "huber_mean_error_list":huber_mean_error_list,
            "agdBB_mean_error_list":agdBB_mean_error_list,
            "agd_mean_error_list":agd_mean_error_list}

def plot_fig1(alpha_list,data_dict,title):
    f = plt.figure()
    plt.plot(alpha_list, data_dict["sample_mean_error_list"], color = "red", label = "Sample Mean", linestyle="-")
    plt.plot(alpha_list, data_dict["huber_mean_error_list"], color = "blue", label = "DA-Huber", linestyle="--")
    plt.plot(alpha_list, data_dict["agdBB_mean_error_list"], color = "yellow", label = "AGD_BB", linestyle="-.")
    plt.plot(alpha_list, data_dict["agd_mean_error_list"], color = "green", label = "AGD", linestyle=":")
    plt.xlabel('Confidence level')
    plt.ylabel('Estimation error')
    plt.legend()
    plt.title(title)
    #plt.show()
    #f.savefig(title+"_figure_1.pdf", dpi=300)
    f.savefig(title+"_figure_1.png", dpi=300)

# -------------------
# Normal Distribution
# -------------------

def normal_simulation(mu,sigma,n=100,simulation_number=2000):
    sample_mean_error_array = []
    huber_mean_error_array = []
    agdBB_mean_error_array = []
    agd_mean_error_array = []
    for i in tqdm(range(0,simulation_number)):
        sample = np.random.normal(mu,sigma,n)
        sample_mean_error = abs(np.mean(sample)-mu)
        huber_mean_error = abs(arr.huber_mean(sample)-mu)
        agdBB_mean_error = abs(arr.agd_bb(sample)-mu)
        agd_mean_error = abs(arr.agd(sample)-mu)
        sample_mean_error_array=np.append(sample_mean_error_array,sample_mean_error)
        huber_mean_error_array=np.append(huber_mean_error_array,huber_mean_error)
        agdBB_mean_error_array=np.append(agdBB_mean_error_array,agdBB_mean_error)
        agd_mean_error_array=np.append(agd_mean_error_array,agd_mean_error)
    return {"sample_mean_error_array":sample_mean_error_array,
            "huber_mean_error_array":huber_mean_error_array,
            "agdBB_mean_error_array":agdBB_mean_error_array,
            "agd_mean_error_array":agd_mean_error_array}

mu = 0
sigma =1
normal_error_dict = normal_simulation(mu,sigma)
normal_data_dict = plot_data(alpha_list,normal_error_dict)
title = "N(0, 1)"
plot_fig1(alpha_list,normal_data_dict,title)

# ---------------------------------
# Skewed Generalized t Distribution
# ---------------------------------

def sgt_simulation(mu,sigma,lam,q,n=100,simulation_number=2000):

    sample_mean_error_array = []
    huber_mean_error_array = []
    agdBB_mean_error_array = []
    agd_mean_error_array = []
    for i in tqdm(range(0,simulation_number)):
        sample = SST(mu, sigma, lam, q).r(n)
        sample_mean_error = abs(np.mean(sample)-mu)
        huber_mean_error = abs(arr.huber_mean(sample)-mu)
        agdBB_mean_error = abs(arr.agd_bb(sample)-mu)
        agd_mean_error = abs(arr.agd(sample)-mu)
        sample_mean_error_array=np.append(sample_mean_error_array,sample_mean_error)
        huber_mean_error_array=np.append(huber_mean_error_array,huber_mean_error)
        agdBB_mean_error_array=np.append(agdBB_mean_error_array,agdBB_mean_error)
        agd_mean_error_array=np.append(agd_mean_error_array,agd_mean_error)
    return {"sample_mean_error_array":sample_mean_error_array,
            "huber_mean_error_array":huber_mean_error_array,
            "agdBB_mean_error_array":agdBB_mean_error_array,
            "agd_mean_error_array":agd_mean_error_array}

mu = 0
sigma = np.sqrt(5)
lam = 0.75
q = 2.5
sgt_error_dict = sgt_simulation(mu,sigma,lam,q)
sgt_data_dict = plot_data(alpha_list,sgt_error_dict)
title = "SGT(0, 5, 0.75, 2, 2.5)"
plot_fig1(alpha_list,sgt_data_dict,title)

# ----------------------
# Lognormal Distribution
# ----------------------

def lognormal_simulation(mu,sigma,n=100,simulation_number=2000):
    sample_mean_error_array = []
    huber_mean_error_array = []
    agdBB_mean_error_array = []
    agd_mean_error_array = []
    for i in tqdm(range(0,simulation_number)):
        sample = np.random.lognormal(mu,sigma,n)
        sample_mean_error = abs(np.mean(sample)-mu)
        huber_mean_error = abs(arr.huber_mean(sample)-mu)
        agdBB_mean_error = abs(arr.agd_bb(sample)-mu)
        agd_mean_error = abs(arr.agd(sample)-mu)
        sample_mean_error_array=np.append(sample_mean_error_array,sample_mean_error)
        huber_mean_error_array=np.append(huber_mean_error_array,huber_mean_error)
        agdBB_mean_error_array=np.append(agdBB_mean_error_array,agdBB_mean_error)
        agd_mean_error_array=np.append(agd_mean_error_array,agd_mean_error)
    return {"sample_mean_error_array":sample_mean_error_array,
            "huber_mean_error_array":huber_mean_error_array,
            "agdBB_mean_error_array":agdBB_mean_error_array,
            "agd_mean_error_array":agd_mean_error_array}

mu = 0
sigma = 1.5
lognormal_error_dict = lognormal_simulation(mu, sigma)
lognormal_data_dict = plot_data(alpha_list,lognormal_error_dict)
title = "LN(0, 1.5)"
plot_fig1(alpha_list,lognormal_data_dict,title)

# -------------------
# Pareto Distribution
# -------------------

def pareto_simulation(scale,shape,n=100,simulation_number=2000):
    sample_mean_error_array = []
    huber_mean_error_array = []
    agdBB_mean_error_array = []
    agd_mean_error_array = []
    mu = (shape*scale)/(shape-1)
    for i in tqdm(range(0,simulation_number)):
        sample = (np.random.pareto(shape, n) + 1) * scale
        sample_mean_error = abs(np.mean(sample)-mu)
        huber_mean_error = abs(arr.huber_mean(sample)-mu)
        agdBB_mean_error = abs(arr.agd_bb(sample)-mu)
        agd_mean_error = abs(arr.agd(sample)-mu)
        sample_mean_error_array=np.append(sample_mean_error_array,sample_mean_error)
        huber_mean_error_array=np.append(huber_mean_error_array,huber_mean_error)
        agdBB_mean_error_array=np.append(agdBB_mean_error_array,agdBB_mean_error)
        agd_mean_error_array=np.append(agd_mean_error_array,agd_mean_error)
    return {"sample_mean_error_array":sample_mean_error_array,
            "huber_mean_error_array":huber_mean_error_array,
            "agdBB_mean_error_array":agdBB_mean_error_array,
            "agd_mean_error_array":agd_mean_error_array}

scale = 1
shape = 2
pareto_error_dict = pareto_simulation(scale,shape)
pareto_data_dict = plot_data(alpha_list,pareto_error_dict)
title = "Par(1, 2)"
plot_fig1(alpha_list,pareto_data_dict,title)

# ===========================================================================================================
# Figure 2
# ===========================================================================================================

# -------------
# Figure 2 Plot
# -------------

def plot_fig2(par_list,data_dict,title):
    f = plt.figure()
    plt.plot(par_list, data_dict["sample_mean_error_99quantile_array"], color = "red", label = "Sample Mean", linestyle="-")
    plt.plot(par_list, data_dict["huber_mean_error_99quantile_array"], color = "blue", label = "DA-Huber", linestyle="--")
    plt.plot(par_list, data_dict["agdBB_mean_error_99quantile_array"], color = "yellow", label = "AGD_BB", linestyle="-.")
    plt.plot(par_list, data_dict["agd_mean_error_99quantile_array"], color = "green", label = "AGD", linestyle=":")
    plt.xlabel('Parameter')
    plt.ylabel('Estimation error')
    plt.legend()
    plt.title(title)
    #plt.show()
    f.savefig(title+"_figure_2.png", dpi=300)

# -------------------
# Normal Distribution
# -------------------

def normal_par_simulation(par_list,mu,n=100,simulation_number=2000):

    sample_mean_error_99quantile_array = []
    huber_mean_error_99quantile_array = []
    agdBB_mean_error_99quantile_array = []
    agd_mean_error_99quantile_array = []
    for par in tqdm(par_list):
        sample_mean_error_array = []
        huber_mean_error_array = []
        agdBB_mean_error_array = []
        agd_mean_error_array = []
        for i in range(0,simulation_number):
            sample = np.random.normal(mu,par,n)
            sample_mean_error = abs(np.mean(sample)-mu)
            huber_mean_error = abs(arr.huber_mean(sample)-mu)
            agdBB_mean_error = abs(arr.agd_bb(sample)-mu)
            agd_mean_error = abs(arr.agd(sample)-mu)
            sample_mean_error_array=np.append(sample_mean_error_array,sample_mean_error)
            huber_mean_error_array=np.append(huber_mean_error_array,huber_mean_error)
            agdBB_mean_error_array=np.append(agdBB_mean_error_array,agdBB_mean_error)
            agd_mean_error_array=np.append(agd_mean_error_array,agd_mean_error)
        sample_mean_error_99quantile_array = np.append(sample_mean_error_99quantile_array,np.quantile(sample_mean_error_array,0.99))
        huber_mean_error_99quantile_array = np.append(huber_mean_error_99quantile_array,np.quantile(huber_mean_error_array,0.99))
        agdBB_mean_error_99quantile_array = np.append(agdBB_mean_error_99quantile_array,np.quantile(agdBB_mean_error_array,0.99))
        agd_mean_error_99quantile_array = np.append(agd_mean_error_99quantile_array,np.quantile(agd_mean_error_array,0.99))
    return {"sample_mean_error_99quantile_array":sample_mean_error_99quantile_array,
            "huber_mean_error_99quantile_array":huber_mean_error_99quantile_array,
            "agdBB_mean_error_99quantile_array":agdBB_mean_error_99quantile_array,
            "agd_mean_error_99quantile_array":agd_mean_error_99quantile_array}

mu = 0
normal_sigma_list = [round(x,2) for x in np.arange(1, 4.1, 0.05)]
normal_quantile_dict = normal_par_simulation(normal_sigma_list,mu)
title = "Normal distribution"
plot_fig2(normal_sigma_list,normal_quantile_dict,title)

# ---------------------------------
# Skewed Generalized t Distribution
# ---------------------------------

def sgt_par_simulation(par_list,mu,lam,n=100,simulation_number=2000):

    sample_mean_error_99quantile_array = []
    huber_mean_error_99quantile_array = []
    agdBB_mean_error_99quantile_array = []
    agd_mean_error_99quantile_array = []
    for par in tqdm(par_list):
        sample_mean_error_array = []
        huber_mean_error_array = []
        agdBB_mean_error_array = []
        agd_mean_error_array = []

        for i in range(0,simulation_number):
            sample = SST(mu, np.sqrt(par/(par-2)), lam, par).r(n)
            sample_mean_error = abs(np.mean(sample)-mu)
            huber_mean_error = abs(arr.huber_mean(sample)-mu)
            agdBB_mean_error = abs(arr.agd_bb(sample)-mu)
            agd_mean_error = abs(arr.agd(sample)-mu)
            sample_mean_error_array=np.append(sample_mean_error_array,sample_mean_error)
            huber_mean_error_array=np.append(huber_mean_error_array,huber_mean_error)
            agdBB_mean_error_array=np.append(agdBB_mean_error_array,agdBB_mean_error)
            agd_mean_error_array=np.append(agd_mean_error_array,agd_mean_error)
        sample_mean_error_99quantile_array = np.append(sample_mean_error_99quantile_array,np.quantile(sample_mean_error_array,0.99))
        huber_mean_error_99quantile_array = np.append(huber_mean_error_99quantile_array,np.quantile(huber_mean_error_array,0.99))
        agdBB_mean_error_99quantile_array = np.append(agdBB_mean_error_99quantile_array,np.quantile(agdBB_mean_error_array,0.99))
        agd_mean_error_99quantile_array = np.append(agd_mean_error_99quantile_array,np.quantile(agd_mean_error_array,0.99))
    return {"sample_mean_error_99quantile_array":sample_mean_error_99quantile_array,
            "huber_mean_error_99quantile_array":huber_mean_error_99quantile_array,
            "agdBB_mean_error_99quantile_array":agdBB_mean_error_99quantile_array,
            "agd_mean_error_99quantile_array":agd_mean_error_99quantile_array}

mu = 0
lam = 0.75
q_list = [round(x,2) for x in np.arange(2.5, 4.1, 0.05)]
sgt_quantile_dict = sgt_par_simulation(q_list,mu,lam)
title = "Skewed generalized t distribution"
plot_fig2(q_list,sgt_quantile_dict,title)

# ----------------------
# Lognormal Distribution
# ----------------------

def lognormal_par_simulation(par_list,mu,n=100,simulation_number=2000):

    sample_mean_error_99quantile_array = []
    huber_mean_error_99quantile_array = []
    agdBB_mean_error_99quantile_array = []
    agd_mean_error_99quantile_array = []
    for par in tqdm(par_list):
        sample_mean_error_array = []
        huber_mean_error_array = []
        agdBB_mean_error_array = []
        agd_mean_error_array = []
        for i in range(0,simulation_number):
            sample = np.random.lognormal(mu,par,n)
            sample_mean_error = abs(np.mean(sample)-mu)
            huber_mean_error = abs(arr.huber_mean(sample)-mu)
            agdBB_mean_error = abs(arr.agd_bb(sample)-mu)
            agd_mean_error = abs(arr.agd(sample)-mu)
            sample_mean_error_array=np.append(sample_mean_error_array,sample_mean_error)
            huber_mean_error_array=np.append(huber_mean_error_array,huber_mean_error)
            agdBB_mean_error_array=np.append(agdBB_mean_error_array,agdBB_mean_error)
            agd_mean_error_array=np.append(agd_mean_error_array,agd_mean_error)
        sample_mean_error_99quantile_array = np.append(sample_mean_error_99quantile_array,np.quantile(sample_mean_error_array,0.99))
        huber_mean_error_99quantile_array = np.append(huber_mean_error_99quantile_array,np.quantile(huber_mean_error_array,0.99))
        agdBB_mean_error_99quantile_array = np.append(agdBB_mean_error_99quantile_array,np.quantile(agdBB_mean_error_array,0.99))
        agd_mean_error_99quantile_array = np.append(agd_mean_error_99quantile_array,np.quantile(agd_mean_error_array,0.99))
    return {"sample_mean_error_99quantile_array":sample_mean_error_99quantile_array,
            "huber_mean_error_99quantile_array":huber_mean_error_99quantile_array,
            "agdBB_mean_error_99quantile_array":agdBB_mean_error_99quantile_array,
            "agd_mean_error_99quantile_array":agd_mean_error_99quantile_array}

mu = 0
lognormal_sigma_list = [round(x,3) for x in np.arange(0.25, 2.01, 0.05)]
lognormal_quantile_dict = lognormal_par_simulation(lognormal_sigma_list,mu)
title = "Lognormal distribution"
plot_fig2(lognormal_sigma_list,lognormal_quantile_dict,title)

# -------------------
# Pareto Distribution
# -------------------

def pareto_par_simulation(par_list,scale,n=100,simulation_number=2000):

    sample_mean_error_99quantile_array = []
    huber_mean_error_99quantile_array = []
    agdBB_mean_error_99quantile_array = []
    agd_mean_error_99quantile_array = []
    for par in tqdm(par_list):
        sample_mean_error_array = []
        huber_mean_error_array = []
        agdBB_mean_error_array = []
        agd_mean_error_array = []
        mu = (par*scale)/(par-1)
        for i in range(0,simulation_number):
            sample = (np.random.pareto(par, n) + 1) * scale
            sample_mean_error = abs(np.mean(sample)-mu)
            huber_mean_error = abs(arr.huber_mean(sample)-mu)
            agdBB_mean_error = abs(arr.agd_bb(sample)-mu)
            agd_mean_error = abs(arr.agd(sample)-mu)
            sample_mean_error_array=np.append(sample_mean_error_array,sample_mean_error)
            huber_mean_error_array=np.append(huber_mean_error_array,huber_mean_error)
            agdBB_mean_error_array=np.append(agdBB_mean_error_array,agdBB_mean_error)
            agd_mean_error_array=np.append(agd_mean_error_array,agd_mean_error)
        sample_mean_error_99quantile_array = np.append(sample_mean_error_99quantile_array,np.quantile(sample_mean_error_array,0.99))
        huber_mean_error_99quantile_array = np.append(huber_mean_error_99quantile_array,np.quantile(huber_mean_error_array,0.99))
        agdBB_mean_error_99quantile_array = np.append(agdBB_mean_error_99quantile_array,np.quantile(agdBB_mean_error_array,0.99))
        agd_mean_error_99quantile_array = np.append(agd_mean_error_99quantile_array,np.quantile(agd_mean_error_array,0.99))
    return {"sample_mean_error_99quantile_array":sample_mean_error_99quantile_array,
            "huber_mean_error_99quantile_array":huber_mean_error_99quantile_array,
            "agdBB_mean_error_99quantile_array":agdBB_mean_error_99quantile_array,
            "agd_mean_error_99quantile_array":agd_mean_error_99quantile_array}

scale = 1
shape_list = [round(x,2) for x in np.arange(1.5, 3.1, 0.05)]
pareto_quantile_dict = pareto_par_simulation(shape_list, scale)
title = "Pareto distribution"
plot_fig2(shape_list,pareto_quantile_dict,title)