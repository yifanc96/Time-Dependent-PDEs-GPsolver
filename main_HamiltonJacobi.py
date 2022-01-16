# author: Yifan Chen
# yifanc96@gmail.com
# 10/10/2021

import os

import jax.numpy as jnp
from jax import vmap, jit
from jax.config import config; 
config.update("jax_enable_x64", True)

import numpy as onp
import argparse
import logging
import datetime
import random
from time import time

def get_parser():
    parser = argparse.ArgumentParser(description='HJ equation GP solver')
    parser.add_argument("--dim", type = int, default = 100)
    # sigma = args.sigma-scale*sqrt(dim)
    parser.add_argument("--kernel", type=str, default="inv_quadratics", choices=["gaussian","inv_quadratics"])
    parser.add_argument("--sigma-scale", type = float, default = 100)
    parser.add_argument("--N_domain", type = int, default = 1000)
    parser.add_argument("--dt", type = float, default = 1e-2)
    parser.add_argument("--nugget", type = float, default = 1e-3)
    parser.add_argument("--GNsteps", type = int, default = 4)
    parser.add_argument("--logroot", type=str, default='./logs/')
    parser.add_argument("--randomseed", type=int, default=9999)
    args = parser.parse_args()    
    return args
@jit
def get_GNkernel_train(x,y,wx0,wx1,wy0,wy1,d,sigma):
    return wx0*wy0*kappa(x,y,d,sigma) + wx0*D_wy_kappa(x,y,d, sigma,wy1) + wy0* D_wx_kappa(x,y,d, sigma,wx1) + D_wx_D_wy_kappa(x,y,d,sigma,wx1,wy1)
@jit
def get_GNkernel_val_predict(x,y,wy0,wy1,d,sigma):
    return wy0*kappa(x,y,d,sigma) + D_wy_kappa(x,y,d, sigma,wy1)
@jit
def get_GNkernel_grad_predict(x,y,wy0,wy1,d,sigma):
    return wy0*D_x_kappa(x,y,d, sigma) + D_x_D_wy_kappa(x,y,d,sigma,wy1)


def assembly_Theta(X_domain, w0, w1, sigma):
    # X_domain, dim: N_domain*d; 
    # w0 col vec: coefs of Diracs, dim: N_domain; 
    # w1 coefs of gradients, dim: N_domain*d
    
    N_domain,d = onp.shape(X_domain)
    Theta = onp.zeros((N_domain,N_domain))
    
    XdXd0 = onp.reshape(onp.tile(X_domain,(1,N_domain)),(-1,d))
    XdXd1 = onp.tile(X_domain,(N_domain,1))
    
    arr_wx0 = onp.reshape(onp.tile(w0,(1,N_domain)),(-1,1))
    arr_wx1 = onp.reshape(onp.tile(w1,(1,N_domain)),(-1,d))
    arr_wy0 = onp.tile(w0,(N_domain,1))
    arr_wy1 = onp.tile(w1,(N_domain,1))
    
    val = vmap(lambda x,y,wx0,wx1,wy0,wy1: get_GNkernel_train(x,y,wx0,wx1,wy0,wy1,d,sigma))(XdXd0,XdXd1,arr_wx0,arr_wx1,arr_wy0,arr_wy1)
    Theta[:N_domain,:N_domain] = onp.reshape(val, (N_domain,N_domain))
    return Theta
    
def assembly_Theta_value_and_grad_predict(X_infer, X_domain, w0, w1, sigma):
    N_infer, d = onp.shape(X_infer)
    N_domain, _ = onp.shape(X_domain)
    Theta = onp.zeros((N_infer*(d+1),N_domain))
    
    XiXd0 = onp.reshape(onp.tile(X_infer,(1,N_domain)),(-1,d))
    XiXd1 = onp.tile(X_domain,(N_infer,1))
    
    arr_wy0 = onp.tile(w0,(N_infer,1))
    arr_wy1 = onp.tile(w1,(N_infer,1))
    
    val = vmap(lambda x,y,wy0,wy1: get_GNkernel_val_predict(x,y,wy0,wy1,d,sigma))(XiXd0,XiXd1,arr_wy0,arr_wy1)
    Theta[:N_infer,:N_domain] = onp.reshape(val, (N_domain,N_domain))

    val = vmap(lambda x,y,wy0,wy1: get_GNkernel_grad_predict(x,y,wy0,wy1,d,sigma))(XiXd0,XiXd1,arr_wy0,arr_wy1)
    Theta[N_infer:,:N_domain] = onp.reshape(val,(N_infer*d,N_domain))
    return Theta
    
def assembly_Theta_stanGP(X_domain,sigma):
    N_domain,d = onp.shape(X_domain)
    Theta = onp.zeros((N_domain,N_domain))
    
    XdXd0 = onp.reshape(onp.tile(X_domain,(1,N_domain)),(-1,d))
    XdXd1 = onp.tile(X_domain,(N_domain,1))
    
    val = vmap(lambda x,y: kappa(x,y,d,sigma))(XdXd0,XdXd1)
    Theta[:N_domain,:N_domain] = onp.reshape(val, (N_domain,N_domain))
    return Theta
    
def assembly_Theta_predict_value_and_grad_stanGP(X_infer, X_domain,sigma):
    N_infer,d = onp.shape(X_infer)
    N_domain = onp.shape(X_domain)[0]
    Theta = onp.zeros((N_infer*(d+1),N_domain))
    
    XdXd0 = onp.reshape(onp.tile(X_infer,(1,N_domain)),(-1,d))
    XdXd1 = onp.tile(X_domain,(N_infer,1))
    
    val = vmap(lambda x,y: kappa(x,y,d,sigma))(XdXd0,XdXd1)
    Theta[:N_infer,:N_domain] = onp.reshape(val, (N_infer,N_domain))
    val = vmap(lambda x,y: D_x_kappa(x,y,d,sigma))(XdXd0,XdXd1)
    Theta[N_infer:,:N_domain] = onp.reshape(val,(N_infer*d,N_domain))
    return Theta

def generate_path(X_init, N_domain, dt, T):
    if onp.ndim(X_init)==1: X_init = X_init[onp.newaxis,:]
    _,d = onp.shape(X_init)
    Nt = int(T/dt)+1
    arr_X = onp.zeros((Nt,N_domain,d))
    arr_X[0,:,:] = X_init
    rdn = onp.random.normal(0, 1, (Nt-1, N_domain,d))
    for i in range(Nt-1):
        arr_X[i+1,:,:] = arr_X[i,:,:] + onp.sqrt(2*dt)*rdn[i,:,:]
    return arr_X

def one_step_iteration(V_future, X_future, X_now, dt, sigma, nugget, GN_step):
    N_domain = onp.shape(X_now)[0]
    Theta_train = assembly_Theta_stanGP(X_future,sigma)
    Theta_infer = assembly_Theta_predict_value_and_grad_stanGP(X_now, X_future,sigma)
    
    V_val_n_grad = Theta_infer @ (onp.linalg.solve(Theta_train + nugget*onp.diag(onp.diag(Theta_train)),V_future))
    w0 = onp.ones((N_domain,1))
    for i in range(GN_step):
        # get grad V_{old}
        V_old = V_val_n_grad[:N_domain]
        logging.info(f'  [logs] GN step: {i}, and sol val at the 1st point {V_old[0]}')
        V_old_grad = onp.reshape(V_val_n_grad[N_domain:],(N_domain,d))
        
        w1 = 2*V_old_grad+(X_future-X_now)
        Theta_train = assembly_Theta(X_now, w0, w1, sigma)
        Theta_infer = assembly_Theta_value_and_grad_predict(X_now, X_now, w0, w1, sigma)
        rhs = V_future + onp.sum(V_old_grad**2,axis=1)*dt
        V_val_n_grad = Theta_infer @ (onp.linalg.solve(Theta_train + nugget*onp.diag(onp.diag(Theta_train)),rhs))
    
    return V_val_n_grad[:N_domain]


def g(x):
    return jnp.log(1/2+1/2*sum(x**2))

def GPsolver(X_init, N_domain, dt, T, sigma, nugget, GN_step = 4):
    if onp.ndim(X_init)==1: X_init = X_init[onp.newaxis,:]
    _,d = onp.shape(X_init)
    Nt = int(T/dt)+1
    arr_X = generate_path(X_init, N_domain, dt, T)
    V = onp.zeros((Nt,N_domain))
    V[-1,:] = vmap(g)(arr_X[-1,:,:])
    
    time_begin = time()
    # solve V[-i-1,:] from V[-i,:]
    for i in onp.arange(1,Nt):
        t = (Nt-i-1)*dt
        logging.info(f'[Time marching] at iteration {i}/{Nt-1}, solving eqn at time t = {t:.2f}')
        V[-i-1,:] = one_step_iteration(V[-i,:], arr_X[-i,:,:], arr_X[-i-1,:,:], dt, sigma, nugget, GN_step)
        
        total_mins = (time() - time_begin) / 60
        logging.info(f'[Timer] finished in {total_mins:.2f} minutes')
    return V

def logger(args, level = 'INFO'):
    log_root = args.logroot + 'HJ'
    log_name = 'dim' + str(args.dim) + '_kernel' + str(args.kernel)
    logdir = os.path.join(log_root, log_name)
    os.makedirs(logdir, exist_ok=True)
    log_para = 'sigma-scale' + str(args.sigma_scale) + '_Ndomain' + str(args.N_domain) + '_dt' + str(args.dt).replace(".","") + '_diag-nugget' + str(args.nugget).replace(".","")
    date = str(datetime.datetime.now())
    log_base = date[date.find("-"):date.rfind(".")].replace("-", "").replace(":", "").replace(" ", "_")
    filename = log_para + '_' + log_base + '.log'
    logging.basicConfig(level=logging.__dict__[level],
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(logdir+'/'+filename),
        logging.StreamHandler()]
    )

def set_random_seeds(args):
    random_seed = args.randomseed
    onp.random.seed(random_seed)
    random.seed(random_seed)

if __name__ == '__main__':
    ## get argument parser
    args = get_parser()
    
    set_random_seeds(args)
    logger(args, level = 'INFO')
    logging.info(f"[Seeds] random seeds: {args.randomseed}")

    if args.kernel == "gaussian":
        from kernels.Gaussian_kernel import *
    elif args.kernel == "inv_quadratics":
        from kernels.inv_quadratics import *

    d = args.dim
    X_init = onp.zeros((1,d))
    N_domain = args.N_domain
    dt = args.dt
    T = 1
    ratio = args.sigma_scale
    sigma = ratio*onp.sqrt(d)
    nugget = args.nugget
    GN_step = args.GNsteps
    
    
    logging.info(f'argument is {args}')
    logging.info(f'GN step: {GN_step}, d: {d}, sigma: {sigma}, number of points: {N_domain}, dt: {dt}, T: {T}, kernel: {args.kernel}')
    
    V = GPsolver(X_init, N_domain, dt, T, sigma, nugget, GN_step)

