import jax.numpy as jnp
from jax import grad

def kappa(x1,y1,sigma):
    return jnp.exp(-(x1-y1)**2/(2*sigma**2))

def D_y1_kappa(x1,y1, sigma):
    val = grad(kappa,1)(x1, y1, sigma)
    return val

def Delta_y1_kappa(x1,y1, sigma):
    val = grad(D_y1_kappa,1)(x1, y1, sigma)
    return val


def D_x1_kappa(x1,y1, sigma):
    val = grad(kappa,0)(x1, y1, sigma)
    return val

def D_x1_D_y1_kappa(x1,y1, sigma):
    val = grad(D_x1_kappa,1)(x1, y1, sigma)
    return val

def D_x1_Delta_y1_kappa(x1,y1, sigma):
    val = grad(Delta_y1_kappa,0)(x1, y1, sigma)
    return val


def Delta_x1_kappa(x1,y1, sigma):
    val = grad(D_x1_kappa,0)(x1, y1, sigma)
    return val

def Delta_x1_D_y1_kappa(x1,y1, sigma):
    val = grad(Delta_x1_kappa,1)(x1, y1, sigma)
    return val

def Delta_x1_Delta_y1_kappa(x1,y1, sigma):
    val = grad(Delta_x1_D_y1_kappa,1)(x1, y1, sigma)
    return val