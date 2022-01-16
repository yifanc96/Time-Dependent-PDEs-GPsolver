# author: Yifan Chen
# yifanc96@gmail.com
# 1/15/2022

import jax.numpy as jnp
from jax import vmap, jit, hessian
from jax.config import config; 
config.update("jax_enable_x64", True)

# numpy
import numpy as onp
from numpy import random 

import argparse
import logging
import datetime
from time import time
import os

# solving Allen-Cahn: u_t - 0.0001u_xx + 5u^3-5u = 0, x in [-1,1] 
# periodic b.c.
def get_parser():
    parser = argparse.ArgumentParser(description='Allen Cahn GP solver')
    
    # equation parameters
    parser.add_argument("--nu", type=float, default = 1e-4)

    # kernel setting
    parser.add_argument("--kernel", type=str, default="periodic")
    parser.add_argument("--sigma", type = float, default = 0.02)
    
     # sampling points
    parser.add_argument("--dt", type = float, default = 0.04)
    parser.add_argument("--T", type = float, default = 1.0)
    parser.add_argument("--N_domain", type = int, default = 512)
    
    # GN iterations
    parser.add_argument("--nugget", type = float, default = 1e-9)
    parser.add_argument("--GNsteps", type = int, default = 2)
    
    parser.add_argument("--logroot", type=str, default='./logs/')
    parser.add_argument("--randomseed", type=int, default=9999)
    parser.add_argument("--show_figure", type=bool, default=True)
    args = parser.parse_args()    
    return args

# sample points according to a grid
def sample_points(num_pts, dt, T, option = 'grid'):
    Nt = int(T/dt)+1
    X_domain = onp.zeros((Nt,num_pts,2))
    if option == 'grid':
        for i in range(Nt):
            X_domain[i,:,0] = i*dt
            X_domain[i,:,1] = onp.linspace(-1.0,1.0, num_pts+1)[:-1]
    return X_domain


@jit
def get_GNkernel_train(x,y,wx0,wx1,wy0,wy1,d,sigma):
    return wx0*wy0*kappa(x,y,d,sigma) + wx0*wy1*Delta_y_kappa(x,y,d,sigma) + wy0*wx1*Delta_x_kappa(x,y,d,sigma) + wx1*wy1*Delta_x_Delta_y_kappa(x,y,d,sigma)
@jit
def get_GNkernel_train_boundary(x,y,wy0,wy1,d,sigma):
    return wy0*kappa(x,y,d,sigma) + wy1*Delta_y_kappa(x,y,d,sigma)
@jit
def get_GNkernel_val_predict(x,y,wy0,wy1,d,sigma):
    return wy0*kappa(x,y,d,sigma) + wy1*Delta_y_kappa(x,y,d,sigma)

@jit
def get_GNkernel_uxx_predict(x,y,wy0,wy1,d,sigma):
    return wy0*Delta_x_kappa(x,y,d,sigma) + wy1*Delta_x_Delta_y_kappa(x,y,d,sigma)


def assembly_Theta(X_domain, w0, w1, sigma):
    # X_domain, dim: N_domain*d; 
    # w0 col vec: coefs of Diracs, dim: N_domain; 
    # w1 coefs of Laplacians, dim: N_domain
    
    N_domain,d = onp.shape(X_domain)
    Theta = onp.zeros((N_domain,N_domain))
    
    XdXd0 = onp.reshape(onp.tile(X_domain,(1,N_domain)),(-1,d))
    XdXd1 = onp.tile(X_domain,(N_domain,1))
    
    arr_wx0 = onp.reshape(onp.tile(w0,(1,N_domain)),(-1,1))
    arr_wx1 = onp.reshape(onp.tile(w1,(1,N_domain)),(-1,1))
    arr_wy0 = onp.tile(w0,(N_domain,1))
    arr_wy1 = onp.tile(w1,(N_domain,1))
    
    val = vmap(lambda x,y,wx0,wx1,wy0,wy1: get_GNkernel_train(x,y,wx0,wx1,wy0,wy1,d,sigma))(XdXd0,XdXd1,arr_wx0,arr_wx1,arr_wy0,arr_wy1)
    Theta[:N_domain,:N_domain] = onp.reshape(val, (N_domain,N_domain))
    
    return Theta

def assembly_Theta_value_predict(X_infer, X_domain, w0, w1, sigma):
    N_infer, d = onp.shape(X_infer)
    N_domain, _ = onp.shape(X_domain)
    Theta = onp.zeros((2*N_infer,N_domain))
    
    XiXd0 = onp.reshape(onp.tile(X_infer,(1,N_domain)),(-1,d))
    XiXd1 = onp.tile(X_domain,(N_infer,1))
    
    arr_wy0 = onp.tile(w0,(N_infer,1))
    arr_wy1 = onp.tile(w1,(N_infer,1))
    
    val = vmap(lambda x,y,wy0,wy1: get_GNkernel_val_predict(x,y,wy0,wy1,d,sigma))(XiXd0,XiXd1,arr_wy0,arr_wy1)
    Theta[:N_infer,:N_domain] = onp.reshape(val, (N_infer,N_domain))
    
    
    val = vmap(lambda x,y,wy0,wy1: get_GNkernel_uxx_predict(x,y,wy0,wy1,d,sigma))(XiXd0,XiXd1,arr_wy0,arr_wy1)
    Theta[N_infer:,:N_domain] = onp.reshape(val, (N_infer,N_domain))
    return Theta

def time_steping_GPsolver(X_domain, dt, sigma=0.2, nugget = 1e-10, GN_step = 4):

    Nt,num_pts,_ = onp.shape(X_domain)
    
    sol_u = onp.zeros((Nt,num_pts))
    sol_u_xx = onp.zeros((Nt,num_pts))
    
    sol_u[0,:]=vmap(u)(X_domain[0,:,0],X_domain[0,:,1])
    sol_u_xx[0,:]=vmap(u_xx)(X_domain[0,:,0],X_domain[0,:,1])
    
    N_domain = num_pts
    
    time_begin = time()
    for iter_i in range(Nt-1):
        # solve at t = (iter_i+1)*dt
        
        # get rhs_f and bdy_g
        temp = sol_u[iter_i,:]
        temp_xx = sol_u_xx[iter_i,:]
        
        w1 = - nu * onp.ones(N_domain)
        
        cur_X_domain = X_domain[iter_i+1,:,1]
        cur_X_domain = cur_X_domain[:, onp.newaxis]
        
        rhs_CN = (2/dt+5)*temp - 5*temp**3 + nu*temp_xx # RHS given by Crank–Nicolson
        for iter_step in range(GN_step):
            rhs= rhs_CN + 10*temp**3
            w0 = 2/dt - 5 + 15*temp**2

            Theta_train = assembly_Theta(cur_X_domain, w0[:, onp.newaxis], w1[:, onp.newaxis],sigma)
            Theta_test = assembly_Theta_value_predict(cur_X_domain, cur_X_domain, w0[:, onp.newaxis], w1[:, onp.newaxis], sigma)
            sol = Theta_test @ (onp.linalg.solve(Theta_train + nugget*onp.diag(onp.diag(Theta_train)),rhs))
            
            temp = sol[:N_domain]
            temp_xx = sol[N_domain:]
            
            total_mins = (time() - time_begin) / 60
            logging.info(f'[Timer] GP iteration {iter_step+1}/{GN_step}, finished in {total_mins:.2f} minutes')
            
        sol_u[iter_i+1,:] = temp
        sol_u_xx[iter_i+1,:] = temp_xx
        
        t = (iter_i+1)*dt
        logging.info(f'[Time marching] at iteration {iter_i+1}/{Nt-1}, solving eqn at time t = {t:.2f}') 
    return sol_u 

# log the results
def logger(args, level = 'INFO'):
    log_root = args.logroot + 'AllenCahn'
    log_name = 'kernel' + str(args.kernel)
    logdir = os.path.join(log_root, log_name)
    os.makedirs(logdir, exist_ok=True)
    log_para = 'nu' + f'{args.nu:.3f}' + 'sigma' + str(args.sigma) + '_Ndomain' + str(args.N_domain) + '_nugget' + str(args.nugget).replace(".","")
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
    random.seed(random_seed)
    
# boundary condition
# x1 = t
def u(x1, x2):
    return (x2**2)*jnp.cos(jnp.pi*x2)

def u_x(x1, x2):
    return grad(u,1)(x1,x2)
@jit
def u_xx(x1, x2):
    return grad(u_x,1)(x1,x2)

# right hand side
def f(x1, x2):
    return 0

# get the truth
# to add


if __name__ == '__main__':
    ## get argument parser
    args = get_parser()
    logger(args, level = 'INFO')
    logging.info(f'argument is {args}')
    
    ######## get the equation parameters
    nu = args.nu
    logging.info(f"[Equation] nu: {nu}")

    set_random_seeds(args)
    logging.info(f"[Seeds] random seeds: {args.randomseed}")

    ######## get kernels
    if args.kernel == "periodic":
        from kernels.periodic_kernel import *
        
    N_domain = args.N_domain
    dt = args.dt
    T = args.T
    X_domain = sample_points(N_domain, dt, T, option = 'grid')
    sigma = args.sigma
    nugget = args.nugget
    GN_step = args.GNsteps
    
    logging.info(f'GN step: {GN_step}, sigma: {sigma}, number of points: N_domain {N_domain}, kernel: {args.kernel}, nugget: {args.nugget}')
    
    # solve the equation
    sol_domain = time_steping_GPsolver(X_domain, dt, nugget = nugget, sigma=sigma, GN_step = GN_step)
    
    # logging.info('[Calculating errs at collocation points ...]')
    # ######### get the solution error
    # # obtain the ground truth solution via the Cole-Hopf transformation
    # # we use numerical integration to get the true solution
    # [Gauss_pts, weights] = onp.polynomial.hermite.hermgauss(80)
    # y_true = vmap(u_true)(X_domain[:,:,0].flatten(),X_domain[:,:,1].flatten())
    # Nt = int(T/dt)+1
    # y_true = onp.reshape(y_true, (Nt,N_domain))
    
    # # compute errors
    # err = abs(sol_domain[:,:]-y_true) 
    # L2err = onp.sqrt(onp.sum(err**2)/onp.size(err))
    # Maxerr = onp.max(err)
    # logging.info(f'[Errors] L2 err in the whole domain is {L2err} and Max err is {Maxerr}')
    # err_t_1 = err[-1,:]
    # L2err_t_1 = onp.sqrt(onp.sum(err_t_1**2)/N_domain)
    # Maxerr_t_1 = onp.max(err[-1,:])
    # logging.info(f'[Errors] L2 err at t=1 is {L2err_t_1} and Max err is {Maxerr_t_1}')
    
    
    if args.show_figure:
        # plot figure
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
        
        # plot parameters
        def set_plot():
            fsize = 15
            tsize = 15
            tdir = 'in'
            major = 5.0
            minor = 3.0
            lwidth = 0.8
            lhandle = 2.0
            plt.style.use('default')
            plt.rcParams['text.usetex'] = True
            plt.rcParams['font.size'] = fsize
            plt.rcParams['legend.fontsize'] = tsize
            plt.rcParams['xtick.direction'] = tdir
            plt.rcParams['ytick.direction'] = tdir
            plt.rcParams['xtick.major.size'] = major
            plt.rcParams['xtick.minor.size'] = minor
            plt.rcParams['ytick.major.size'] = 5.0
            plt.rcParams['ytick.minor.size'] = 3.0
            plt.rcParams['axes.linewidth'] = lwidth
            plt.rcParams['legend.handlelength'] = lhandle
            fmt = ticker.ScalarFormatter(useMathText=True)
            fmt.set_powerlimits((0, 0))
            return fmt

        fmt = set_plot()
        
        fig = plt.figure(figsize=(15,5))
        ax = fig.add_subplot(121)
        # err = abs(sol_domain[:,:]-y_true)
        # u_contourf=ax.contourf(X_domain[:,:,1], X_domain[:,:,0], err, 50, cmap=plt.cm.coolwarm)
        # plt.xlabel('$x$')
        # plt.ylabel('$t$')
        # plt.title('Contour of errors')
        # fig.colorbar(u_contourf, format=fmt)


        ax = fig.add_subplot(122)
        # ax.plot(X_domain[0,:,1], y_true[-1,:], linewidth=2.5, label='true sol')
        ax.plot(X_domain[0,:,1], sol_domain[0,:], linewidth=2.5, label='initial sol')
        ax.plot(X_domain[0,:,1], sol_domain[-1,:], linestyle='dashed', linewidth=2.5, color='red', label='numeric sol')
        ax.set_xlabel('$x$')
        ax.legend(loc="upper right")
        plt.title('At time $t = 1$')

        plt.show()
        fig.tight_layout()