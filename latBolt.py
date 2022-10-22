#!/usr/bin/env python3
# Import python libraries for math and plotting
import numpy as np
import matplotlib.pyplot as plt
# plot sreamlines
import plotly.figure_factory as ff

## Problem Setup

# physical parameters
H = 1 #length of x grid, y grid
U0 = 1 #physical characteristic velocity (lid)
Re = 5000 #Reynolds number

# lattice parameters
dx = 1 #x step size
dy = 1 #y step size
dt = 1 #t step size
cs = 1/np.sqrt(3)   #lattice speed of sound
rhoo = 5.00         #lattice density initial
Ma = 0.1           #lattice mach number
tau = 0.5355        #lattice relaxation time

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

print(f'Re:{Re:4.1f} nx:{nx}')
np.seterr(over='raise')
# physical parameters
x = np.linspace(0,H,nx) #x nodes
y = np.linspace(0,H,ny) #y nodes

# D2Q9 model parameters
w  = np.array([4/9.0, 1/9.0, 1/9.0, 1/9.0, 1/9.0, 1/36.0, 1/36.0, 1/36.0, 1/36.0],dtype=np.float64) # weight in equilibrium distribution function
cx = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1],dtype=np.float64) # discrete velocity x component
cy = np.array([0, 0, 1, 0, -1, 1,  1, -1, -1],dtype=np.float64) # discrete velocity y component

## Initialization

f = np.zeros([9,nx,ny])
feq = np.zeros([9,nx,ny])
rho = np.ones([nx,ny])*rhoo
u = np.zeros([nx,ny])
v = np.zeros([nx,ny])
ut = np.copy(u)
vt = np.copy(v)

error = 1.0
iterations = 0

for i in range(1,nx):
    u[i,ny-1] = uo
    v[i,ny-1] = 0.0   # redundant

# u[:,-1] = uo

## Solving Governing Equations
print(type(omega),feq.dtype,f.dtype)
while error > tolerance:
    print(f'Iteration: {iterations}')
    # collision
    for i in range(nx):
        for j in range(ny):
            t1 = u[i,j]*u[i,j]+v[i,j]*v[i,j]
            for k in range(9):
                t2 = u[i,j]*cx[k]+v[i,j]*cy[k]
                feq[k,i,j] = rho[i,j]*w[k]*(1.0+3.0*t2+4.50*t2*t2-1.50*t1)
                f[k,i,j] = omega*feq[k,i,j]+(1.0-omega)*f[k,i,j]




    # streaming
    for j in range(ny):
        for i in range(nx-1,1,-1): # right to left
            f[1,i,j] = f[1,i-1,j]

        for i in range(nx-1): # left to right
            f[3,i,j] = f[3,i+1,j]



    for j in range(ny-1,1,-1): # top to bottom
        for i in range(nx):
            f[2,i,j] = f[2,i,j-1]

        for i in range(nx-1,1,-1):
            f[5,i,j] = f[5,i-1,j-1]

        for i in range(nx-1):
            f[6,i,j] = f[6,i+1,j-1]



    for j in range(ny-1): # bottom to top
        for i in range(nx):
            f[4,i,j] = f[4,i,j+1]

        for i in range(nx-1):
            f[7,i,j] = f[7,i+1,j+1]

        for i in range(nx-1,1,-1):
            f[8,i,j] = f[8,i-1,j+1]



# boundary conditions
    for j in range(ny):
        # bounce back on west boundary
        f[1,0,j] = f[3,0,j]
        f[5,0,j] = f[7,0,j]
        f[8,0,j] = f[6,0,j]

        # bounce back on east boundary
        f[3,nx-1,j] = f[1,nx-1,j]
        f[7,nx-1,j] = f[5,nx-1,j]
        f[6,nx-1,j] = f[8,nx-1,j]


    # bounce back on south boundary
    for i in range(nx):
        f[2,i,0] = f[4,i,0]
        f[5,i,0] = f[7,i,0]
        f[6,i,0] = f[8,i,0]


    # moving lid, north boundary
    for i in range(1,nx):
        rhon = f[0,i,ny-1]+f[1,i,ny-1]+f[3,i,ny-1]+2.0*(f[2,i,ny-1]+f[6,i,ny-1]+f[5,i,ny-1])
        f[4,i,ny-1] = f[2,i,ny-1]
        f[8,i,ny-1] = f[6,i,ny-1]+rhon*uo/6.0
        f[7,i,ny-1] = f[5,i,ny-1]-rhon*uo/6.0

    # rho, u, v
    for j in range(ny):
        for i in range(nx):
            ssum = np.float64(0.0)
            for k in range(9):
                ssum = ssum+f[k,i,j]
            rho[i,j] = ssum

    for i in range(nx):
        rho[i,ny-1] = f[0,i,ny-1]+f[1,i,ny-1]+f[3,i,ny-1]+2.0*(f[2,i,ny-1]+f[6,i,ny-1]+f[5,i,ny-1])


    for i in range(1,nx):
        for j in range(1,ny-1):
            usum = 0.0
            vsum = 0.0
            for k in range(9):
                usum = usum+f[k,i,j]*cx[k]
                vsum = vsum+f[k,i,j]*cy[k]

            u[i,j] = usum/rho[i,j]
            v[i,j] = vsum/rho[i,j]


    # error monitoring
    error = np.linalg.norm(u-ut)/(nx*ny)+np.linalg.norm(v-vt)/(nx*ny)
    ut = np.copy(u)
    vt = np.copy(v)
    iterations = iterations+1



## Results
print(f'u:\n{u}')
print(f'v:\n{v}')

# convert to physical parameters
u = ur*u
v = ur*v
p = cs**2*rho

# Correct indexing
# uu = flipud(rot90(u))
# vv = flipud(rot90(v))
# pp = flipud(rot90(p))

## Plotting
ff.create_streamline(x,y,u,v)
plt.show()
    # set(streams,'LineWidth',1,'color',[0 0 0])
    # daspect([1 1 1])
    # xlim([0 nx])
    # xlabel('x','fontweight','bold')
    # ylim([0 ny])
    # ylabel('y','fontweight','bold')
    # axis tight
    # box on