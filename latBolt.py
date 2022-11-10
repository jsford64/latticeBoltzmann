#!/usr/bin/env python3
# Import python libraries for math and plotting
import numpy as np
import matplotlib.pyplot as plt
# plot sreamlines
import plotly.figure_factory as ff
import sys
import time

## Problem Setup

# physical parameters
H = 1 #length of x grid, y grid
U0 = 1 #physical characteristic velocity (lid)
Re = 300#5000 #Reynolds number

# lattice parameters
D,Q = (2,9)

dx = 1 #x step size
dy = 1 #y step size
dt = 1 #t step size
cs = 1/np.sqrt(3)   #lattice speed of sound
rhoo = 5.00         #lattice density initial
Ma = 0.1           #lattice mach number
tau = 0.75        #lattice relaxation time

tolerance = 1e-8    # solver final residual

## Constants

# lattice parameters
uo = Ma*cs #lattice characteristic velocity
ur = U0/uo #reference velocity

omega = 1/tau               #collision frequency
alpha = cs**2*(tau-0.5)     #lattice kinematic viscosity
N = Re*alpha/uo             #lattice Re matching physical Re
nx = int(2*np.floor(N/2))   #number of nodes in x-direction
ny = nx                     #number of nodes in y-direction
print(f'grid size: {nx}x{ny}')
np.seterr(over='raise') # raise exception on numerical overrun
# physical parameters
x = np.linspace(0,H,nx) #x nodes
y = np.linspace(0,H,ny) #y nodes

# D2Q9 model parameters
w  = np.array([4/9.0, 1/9.0, 1/9.0, 1/9.0, 1/9.0, 1/36.0, 1/36.0, 1/36.0, 1/36.0]) # weight in equilibrium distribution function
cx = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1]) # discrete velocity x component
cy = np.array([0, 0, 1, 0, -1, 1,  1, -1, -1]) # discrete velocity y component

## Initialization

f = np.zeros([Q,ny,nx])
feq = np.zeros([Q,ny,nx])
rho = np.ones([ny,nx])*rhoo
u = np.zeros([ny,nx])
v = np.zeros([ny,nx])
ut = np.copy(u)
vt = np.copy(v)

error = 1.0
iterations = 0

# for i in range(1,nx-1):
#     u[ny-1,i] = uo
#     v[ny-1,i] = 0.0   # redundant

# top row (y=ny-1) gets initial velocity (excluding l/r sides, x[0] and x[-1])
u[-1,1:-1] = uo
## Solving Governing Equations

def chk(name,f):
    n = f.size
    matlab = []
    for c in range(n):
        matlab += [np.float32(sys.stdin.readline()[0:-2])]

    matlab = np.array(matlab).reshape(f.shape)
    try:
        assert(np.allclose(f,matlab))
        print(name,'matches!')
    except AssertionError:
        print(f'Mismatch Error in {name}:')
        print(f'python:\n{f.shape}\n{f}\n')
        print(f'matlab:\n{matlab.shape}\n\n{matlab}')
        print(np.abs(f-matlab) > 1e-5)
        exit()

# chk('u',u)
# chk('v',v)

start = time.time()

while error > tolerance: # and iterations<1000:

    # collision
    t1 = u**2 + v**2
    # chk('t1',t1)

    # multiply every u[j,i] * cx; same for v[j,i] * cy; add together,
    # then reshape t2 back to an array with shape [Q,ny,nx]

    t2 =  (np.outer(cx,u) + np.outer(cy,v)).reshape([Q,ny,nx])
    t2t1 = 1.0 + t2*(3.0 + 4.50*t2) - 1.50*t1
    feq = np.outer(w,rho).reshape(Q,ny,nx) * t2t1
    f = omega*feq + (1-omega)*f

    # chk('t2',t2)
    # chk('feq',feq)
    # chk('f',f)

    # streaming

    # for j in range(ny):
    #     # for i in range(nx-1,0,-1): # right to left
    #     #     f[1,j,i] = f[1,j,i-1]
    #
    #     for i in range(nx-1): # left to right
    #         f[3,j,i] = f[3,j,i+1]

    f[1,:,1:] = f[1,:,:-1]
    f[3,:,0:-1] = f[3,:,1:]
    # chk('f (r->l,l->r)',f)

    # for j in range(ny-1,0,-1): # top to bottom
        # for i in range(nx):
        #     f[2,j,i] = f[2,j-1,i]

        # for i in range(nx-1,0,-1):
        #     f[5,j,i] = f[5,j-1,i-1]
        #
        # for i in range(nx-1):
        #     f[6,j,i] = f[6,j-1,i+1]

    f[2, 1:] = f[2, :-1]
    f[5, 1:, 1:] = f[5, :-1, :-1 ]
    f[6, 1:, :-1] = f[6, :-1, 1:]
    #chk('f (t->b)',f)

    # for j in range(ny-1): # bottom to top
    #     for i in range(nx):
    #         f[4,j,i] = f[4,j+1,i]
    #
    #     for i in range(nx-1):
    #         f[7,j,i] = f[7,j+1,i+1]
    #
    #     for i in range(nx-1,0,-1):
    #         f[8,j,i] = f[8,j+1,i-1]

    f[4, :-1]      = f[4, 1:]
    f[7, :-1, :-1] = f[7, 1:, 1: ]
    f[8, :-1,  1:] = f[8, 1:, :-1 ]
    #chk('f (b->t)',f)

# boundary conditions
    #for j in range(ny):
    # bounce back on west boundary
    f[1,:,0] = f[3,:,0]
    f[5,:,0] = f[7,:,0]
    f[8,:,0] = f[6,:,0]

    # bounce back on east boundary
    f[3,:,nx-1] = f[1,:,nx-1]
    f[7,:,nx-1] = f[5,:,nx-1]
    f[6,:,nx-1] = f[8,:,nx-1]

    # chk('f (ew)',f)

    # bounce back on south boundary
    #for i in range(nx):
    f[2,0] = f[4,0]
    f[5,0] = f[7,0]
    f[6,0] = f[8,0]

    #chk('f (s)',f)

    # moving lid, north boundary
    #for i in range(1,nx-1):
    rhon =        f[0, ny-1, 1:-1]+f[1, ny-1, 1:-1]+f[3, ny-1, 1:-1] \
            +2.0*(f[2, ny-1, 1:-1]+f[6, ny-1, 1:-1]+f[5, ny-1, 1:-1])
    #print(f'rhon = {rhon:.4f}')
    rhom = rhon*uo/6.0
    f[4, ny-1, 1:-1] = f[2, ny-1, 1:-1]
    f[8, ny-1, 1:-1] = f[6, ny-1, 1:-1]+rhom
    f[7, ny-1, 1:-1] = f[5, ny-1, 1:-1]-rhom

    #chk('f (n)',f)

    # rho, u, v
    # for j in range(ny):
    #     for i in range(nx):
    #         ssum = np.float64(0.0)
    #         for k in range(Q):
    #             ssum = ssum+f[k,j,i]
    #         rho[j,i] = ssum
    rho = np.sum(f,axis=0)
    #chk('rho 0',rho)

    #for i in range(nx):
    rho[ny-1] =       f[0,ny-1]+f[1,ny-1]+f[3,ny-1] \
                +2.0*(f[2,ny-1]+f[6,ny-1]+f[5,ny-1])

    #chk('rho 1',rho)

    # for i in range(1,nx):
    #     for j in range(1,ny-1):
    #         usum = 0.0
    #         vsum = 0.0
    #         for k in range(Q):
    #             usum = usum+f[k,j,i]*cx[k]
    #             vsum = vsum+f[k,j,i]*cy[k]

    usum = np.tensordot( cx, f[:, 1:-1, 1:], axes=1)
    vsum = np.tensordot( cy, f[:, 1:-1, 1:], axes=1)

    # shape = [j,i]
    u[1:-1, 1:] = usum/rho[1:-1, 1:]
    v[1:-1, 1:] = vsum/rho[1:-1, 1:]

    # chk('u',u)
    # chk('v',v)

    # error monitoring
    error = np.linalg.norm(u-ut)/(nx*ny)+np.linalg.norm(v-vt)/(nx*ny)
    ut = np.copy(u)
    vt = np.copy(v)
    iterations = iterations+1
    # if iterations % 100 == 0:
    #     print(f'iterations = {iterations}')

# Results

# convert to physical parameters
u = ur*u+1e-6
v = ur*v+1e-6
# p = cs**2*rho
print(f'Elapsed Time: {time.time()-start}')
# Correct indexing
# uu = np.flipud(np.rot90(u))
# vv = np.flipud(np.rot90(v))
# pp = flipud(rot90(p))

## Plotting
ff.create_streamline(x,y,u,v,arrow_scale=.01,density=2).show()

# plt.show()
    # set(streams,'LineWidth',1,'color',[0 0 0])
    # daspect([1 1 1])
    # xlim([0 nx])
    # xlabel('x','fontweight','bold')
    # ylim([0 ny])
    # ylabel('y','fontweight','bold')
    # axis tight
    # box on
