clear; close all; clc;






%% Problem Setup

% physical parameters
H = 1; %length of x grid, y grid
U0 = 1; %physical characteristic velocity (lid)
Re = 100; %Reynolds number

% lattice parameters
dx = 1; %x step size
dy = 1; %y step size
dt = 1; %t step size
cs = 1/sqrt(3); %lattice speed of sound
rhoo = 5.00; %lattice density initial
Ma = 0.10; %lattice mach number
tau = 0.5355; %lattice relaxation time

tolerance = 1e-8; % solver final residual

%% Constants

% lattice parameters
uo = Ma*cs; %lattice characteristic velocity
ur = U0/uo; %reference velocity

omega = 1/tau; %collision frequency
alpha = cs^2*(tau-0.5); %lattice kinematic viscosity
N = Re*alpha/uo; %lattice Re matching physical Re
nx = 2*floor(N/2); %number of nodes in x-direction
ny = nx; %number of nodes in y-direction



% physical parameters
x = linspace(0,H,nx); %x nodes
y = linspace(0,H,ny); %y nodes

% D2Q9 model parameters
w = [4/9 1/9 1/9 1/9 1/9 1/36 1/36 1/36 1/36]; % weight in equilibrium distribution function
cx = [0  1  0 -1  0  1 -1 -1  1]; % discrete velocity x component
cy = [0  0  1  0 -1  1  1 -1 -1 ]; % discrete velocity y component

%% Initialization

f = zeros(9,nx,ny);
feq = zeros(9,nx,ny);
rho = ones(nx,ny)*rhoo;
u = zeros(nx,ny);
v = zeros(nx,ny);
ut = u;
vt = v;

error = 1.0;
iterations = 0;

for i = 2:nx-1
    u(i,ny) = uo;
    v(i,ny) = 0.0;
end

function _ = chk2(n,f)
    for j = 1:n
        for i = 1:n
            printf("%0.9f\n",f(i,j))
        end
    end
end

function _ = chk3(n,f)
    for k = 1:9
        for j = 1:n
            for i = 1:n
                printf("%0.9f\n",f(k,i,j))
            end
        end
    end
end
t2 = zeros(9,nx,ny);

%% Solving Governing Equations

% chk2(nx,u);
% chk2(nx,v);


while error > tolerance %& iterations<1000

% collision
    for j = 1:ny
        for i = 1:nx
            t1 = u(i,j)*u(i,j)+v(i,j)*v(i,j);
            %printf("%0.9f\n",t1);
            for k = 1:9
                t2(k,i,j) = u(i,j)*cx(k)+v(i,j)*cy(k);
                feq(k,i,j) = rho(i,j)*w(k)*(1.0+3.0*t2(k,i,j)+4.50*t2(k,i,j)*t2(k,i,j)-1.50*t1);
                f(k,i,j) = omega*feq(k,i,j)+(1.0-omega)*f(k,i,j);
            end
        end
    end

    % chk3(nx,t2);
    % chk3(nx,feq);
    % chk3(nx,f);

% streaming
    for j = 1:ny
        for i = nx:-1:2 % right to left
            f(2,i,j) = f(2,i-1,j);
        end
        for i = 1:nx-1 % left to right
            f(4,i,j) = f(4,i+1,j);
        end
    end

    %chk3(nx,f);

    for j = ny:-1:2 % top to bottom
        for i = 1:nx
            f(3,i,j) = f(3,i,j-1);
        end
        for i = nx:-1:2
            f(6,i,j) = f(6,i-1,j-1);
        end
        for i = 1:nx-1
            f(7,i,j) = f(7,i+1,j-1);
        end
    end

    %chk3(nx,f);

    for j = 1:ny-1 % bottom to top
        for i = 1:nx
            f(5,i,j) = f(5,i,j+1);
        end
        for i = 1:nx-1
            f(8,i,j) = f(8,i+1,j+1);
        end
        for i = nx:-1:2
            f(9,i,j) = f(9,i-1,j+1);
        end
    end

    %chk3(nx,f);

% boundary conditions
    for j = 1:ny
        % bounce back on west boundary
        f(2,1,j) = f(4,1,j);
        f(6,1,j) = f(8,1,j);
        f(9,1,j) = f(7,1,j);

        % bounce back on east boundary
        f(4,nx,j) = f(2,nx,j);
        f(8,nx,j) = f(6,nx,j);
        f(7,nx,j) = f(9,nx,j);
    end

    % chk3(nx,f);

    % bounce back on south boundary
    for i = 1:nx
        f(3,i,1) = f(5,i,1);
        f(6,i,1) = f(8,i,1);
        f(7,i,1) = f(9,i,1);
    end

    %chk3(nx,f);

    % moving lid, north boundary
    for i = 2:nx-1
        rhon = f(1,i,ny)+f(2,i,ny)+f(4,i,ny)+2.0*(f(3,i,ny)+f(7,i,ny)+f(6,i,ny));
        f(5,i,ny) = f(3,i,ny);
        f(9,i,ny) = f(7,i,ny)+rhon*uo/6.0;
        f(8,i,ny) = f(6,i,ny)-rhon*uo/6.0;
    end

    %chk3(nx,f);

    % rho, u, v
    for j = 1:ny
        for i = 1:nx
            ssum = 0.0;
            for k = 1:9
                ssum = ssum+f(k,i,j);
            end
            rho(i,j) = ssum;
        end
    end

    %chk2(nx,rho);

    for i = 1:nx
        rho(i,ny) = f(1,i,ny)+f(2,i,ny)+f(4,i,ny)+2.0*(f(3,i,ny)+f(7,i,ny)+f(6,i,ny));
    end

    %chk2(nx,rho);

    for i = 2:nx
        for j = 2:ny-1
            usum = 0.0;
            vsum = 0.0;
            for k = 1:9
                usum = usum+f(k,i,j)*cx(k);
                vsum = vsum+f(k,i,j)*cy(k);
            end
            u(i,j) = usum/rho(i,j);
            v(i,j) = vsum/rho(i,j);
        end
    end

    chk2(nx,u);
    chk2(nx,v);

    % error monitoring
    error = norm(u-ut)/(nx*ny)+norm(v-vt)/(nx*ny);
    ut = u;
    vt = v;
    iterations = iterations+1;

end

%% Results

% convert to physical parameters
u = ur*u;
v = ur*v;
p = cs^2*rho;

% Correct indexing
uu = flipud(rot90(u));
vv = flipud(rot90(v));
pp = flipud(rot90(p));

%% Plotting

% figure;
%     streams = streamslice(x,y,uu,vv,2);
%     set(streams,'LineWidth',1,'color',[0 0 0])
%     daspect([1 1 1])
%     xlim([0 nx])
%     xlabel('x','fontweight','bold')
%     ylim([0 ny])
%     ylabel('y','fontweight','bold')
%     axis tight
%     box on
