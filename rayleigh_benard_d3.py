"""
The problem is non-dimensionalized using the box height and freefall time, so
the resulting thermal diffusivity and viscosity are related to the Prandtl
and Rayleigh numbers as:

    kappa = (Rayleigh * Prandtl)**(-1/2)
    nu = (Rayleigh / Prandtl)**(-1/2)

For incompressible hydro with two boundaries, we need two tau terms for each the
velocity and buoyancy. Here we choose to use a first-order formulation, putting
one tau term each on auxiliary first-order gradient variables and the others in
the PDE, and lifting them all to the first derivative basis. This formulation puts
a tau term in the divergence constraint, as required for this geometry.

To run and plot using e.g. 4 processes:
    $ mpiexec -n 4 python3 rayleigh_benard.py
"""

# Prevent multi-threading upon initialising mpi4py
import os,mpi4py
os.environ["OMP_NUM_THREADS"] = "1";
mpi4py.rc.thread_level = 'single';

import numpy as np
import dedalus.public as d3
import logging
logger = logging.getLogger(__name__)

# Parameters
Lx, Lz = 4, 1
Nx, Nz = 256,64
Rayleigh = 10**6
Prandtl = 1
dealias = 3/2
stop_sim_time = 250
timestepper = d3.MCNAB2
max_timestep = 0.1
dtype = np.float64

type = 'RBC'; filename = None; #"/data/pmannix/PDF_DNS_Data/RBC8_1e10/checkpoints/checkpoints_s1.h5" 
#type = 'IC'; filename = "/data/pmannix/PDF_DNS_Data/IC8_1e11/checkpoints/checkpoints_s1.h5"
#type = 'HC'; filename = "/home/pmannix/Dstratify/DNS_RBC/HC_Ra1e10_T2e04_old/checkpoints/checkpoints_s1.h5"
#type = 'PLUME'; filename = "/home/pmannix/Dstratify/DNS_RBC/PLUME_Ra1e9_T4e03/checkpoints/checkpoints_s1.h5"
#type = 'STEP'; filename = "/home/pmannix/Dstratify/DNS_RBC/STEP_Ra1e9_T4e03/checkpoints/checkpoints_s1.h5"
#type = 'IC_Random'; filename = "/home/pmannix/Dstratify/DNS_RBC/ICR_Ra1e11_T4e04/checkpoints/checkpoints_s1.h5"


# Bases
coords = d3.CartesianCoordinates('x', 'z')
dist   = d3.Distributor(coords, dtype=dtype)
xbasis = d3.RealFourier(coords['x'], size=Nx, bounds=(0, Lx), dealias=dealias)
zbasis = d3.ChebyshevT( coords['z'], size=Nz, bounds=(0, Lz), dealias=dealias)

# Fields
p = dist.Field(name='p', bases=(xbasis,zbasis))
b = dist.Field(name='b', bases=(xbasis,zbasis))
u = dist.VectorField(coords, name='u', bases=(xbasis,zbasis))
tau_p  = dist.Field(name='tau_p')
tau_b1 = dist.Field(name='tau_b1', bases=xbasis)
tau_b2 = dist.Field(name='tau_b2', bases=xbasis)
tau_u1 = dist.VectorField(coords, name='tau_u1', bases=xbasis)
tau_u2 = dist.VectorField(coords, name='tau_u2', bases=xbasis)

# Substitutions
kappa= (Rayleigh * Prandtl)**(-1/2) # 1/Pe
nu   = (Rayleigh / Prandtl)**(-1/2) # 1/Re
x, z = dist.local_grids(xbasis, zbasis)
ex,ez= coords.unit_vector_fields(dist)
lift_basis = zbasis.derivative_basis(1)
lift = lambda A: d3.Lift(A, lift_basis, -1)
grad_u = d3.grad(u) + ez*lift(tau_u1) # First-order reduction
grad_b = d3.grad(b) + ez*lift(tau_b1) # First-order reduction

def boundary_data(type='RBC'):
    
    # Costant-Flux
    #problem.add_equation("db_z(z=0)  = -1")
    #problem.add_equation("db_z(z=Lz) = -1")

    if type == 'RBC':

        # Rayleigh Benard Convection

        g_1 = dist.Field(name='g_1')
        g_0 = dist.Field(name='g_0')

        g_1['g'] = 0.;
        g_0['g'] = 1.;
    
        # Initial conditions
        b.fill_random('g', seed=42, distribution='normal', scale=1e-3) # Random noise
        b['g'] *= z * (Lz - z) # Damp noise at walls
        b['g'] += Lz - z # Add linear background 
        
    elif (type == 'IC') or (type == 'IC_Random'):

        # Internally heated Convection

        g_1 = dist.Field(name='g_1')
        g_0 = dist.Field(name='g_0')

        g_1['g'] = 0.;
        g_0['g'] = 0.;
    
        # Initial conditions
        b.fill_random('g', seed=42, distribution='normal', scale=1e-3) # Random noise
        b['g'] *= z * (Lz - z) # Damp noise at walls
        b['g'] += z*(1- z)/2. # Add background

    elif type == 'HC':

        # Horizontal Convection
        
        g_1      = dist.Field(name='g_1')
        g_1['g'] = 0.

        g_0      = dist.Field(name='g_0',bases=xbasis)
        g_0['g'] = np.sin((2.*np.pi/Lx)*x)

        # Initial conditions
        b.fill_random('g', seed=42, distribution='normal', scale=1e-3) # Random noise
        b['g'] *= z * (Lz - z) # Damp noise at walls

    elif type == 'PLUME':
        
        # Plume convection
        lim_n = 10.

        g_1      = dist.Field(name='g_1',bases=xbasis)
        g_1['g'] =-np.sqrt(lim_n**2/(.5*np.pi))*np.exp(-0.5*(lim_n**2)*(x - 0.75*Lx)**2)

        g_0      = dist.Field(name='g_0',bases=xbasis)
        g_0['g'] = np.sqrt(lim_n**2/(.5*np.pi))*np.exp(-0.5*(lim_n**2)*(x - 0.25*Lx)**2)

        # Initial conditions
        b.fill_random('g', seed=42, distribution='normal', scale=1e-3) # Random noise
        b['g'] *= z * (Lz - z) # Damp noise at walls

    elif type == 'STEP':

        # Step convection
        lim_n = 10.

        g_1      = dist.Field(name='g_1')
        g_1['g'] = 0.
        
        g_0      = dist.Field(name='g_0',bases=xbasis)
        I = np.ones(x.shape)
        g_0['g'] = I + np.tanh(lim_n*(x - (Lx/2)*I))

        # Initial conditions
        b.fill_random('g', seed=42, distribution='normal', scale=1e-3) # Random noise
        b['g'] *= z * (Lz - z) # Damp noise at walls

    return g_1,g_0;

# Problem
# First-order form: "div(f)" becomes "trace(grad_f)"
# First-order form: "lap(f)" becomes "div(grad_f)"
problem = d3.IVP([p, b, u, tau_p, tau_b1, tau_b2, tau_u1, tau_u2], namespace=locals())
problem.add_equation("trace(grad_u) + tau_p = 0")
problem.add_equation("dt(u) - nu*div(grad_u) + grad(p) - b*ez + lift(tau_u2) = - u@grad(u)")
problem.add_equation("u(z=0 )  = 0")
problem.add_equation("u(z=Lz)  = 0") # no-slip
problem.add_equation("integ(p) = 0") # Pressure gauge

if type == 'IC':
    problem.add_equation("dt(b) - kappa*div(grad_b) + lift(tau_b2) = -u@grad(b) + kappa")
elif type == 'IC_Random':

    # Generate a Random field, filter it & normalise
    q = dist.Field(name='q', bases=(xbasis,zbasis))
    q.fill_random('g', seed=42, distribution='normal',loc=1.,scale=1.)
    q.low_pass_filter(shape=(5,5),scales=None)
    q.normalize(normalize_volume=True);

    problem.add_equation("dt(b) - kappa*div(grad_b) + lift(tau_b2) = -u@grad(b) + kappa*q")
else:
    problem.add_equation("dt(b) - kappa*div(grad_b) + lift(tau_b2) = -u@grad(b)")

g_1, g_0 = boundary_data(type=type)
problem.add_equation("b(z=Lz) = g_1")
problem.add_equation("b(z=0 ) = g_0") 

# Solver
solver = problem.build_solver(timestepper)
solver.stop_sim_time = stop_sim_time

if filename != None:
    write,initial_timestep = solver.load_state(filename);
#solver.iteration=0
#solver.stop_iteration=1000; #int(stop_sim_time/max_timestep)

# Analysis
checkpoints = solver.evaluator.add_file_handler('checkpoints', sim_dt=5.)
checkpoints.add_tasks(solver.state)

# Snapshots
snapshots = solver.evaluator.add_file_handler('snapshots', sim_dt=.1)

snapshots.add_task(-d3.div(d3.skew(u)), name='vorticity',scales=1)
snapshots.add_task(b,    name='buoyancy',scales=1)
snapshots.add_task(u@ex, name='u',scales=1)
snapshots.add_task(u@ez, name='w',scales=1)
snapshots.add_task(grad_b       , name='grad_b',scales=1)
snapshots.add_task(d3.grad(u@ez), name='grad_w',scales=1)
snapshots.add_task(d3.grad(p)   , name='grad_p',scales=1)

# Time Series and spectra
scalar = solver.evaluator.add_file_handler('scalar_data',iter=10)

scalar.add_task(d3.Integrate(u@u ),  layout='g', name='Eu(t)')
scalar.add_task(d3.Integrate(b**2),  layout='g', name='Eb(t)')

scalar.add_task(nu*d3.Integrate(d3.grad(u@ez)@d3.grad(u@ez) + d3.grad(u@ex)@d3.grad(u@ex)),  layout='g', name='dU^2(t)/Re')
scalar.add_task(   d3.Integrate(d3.grad(b)@d3.grad(b)                                    ),  layout='g', name='dB^2(t)'   )


scalar.add_task(d3.Integrate(u@u ,'z'), layout='c', name='Eu(k)' )
scalar.add_task(d3.Integrate(u@u ,'x'), layout='c', name='Eu(Tz)')
scalar.add_task(d3.Integrate(b**2,'z'), layout='c', name='Eb(k)' )
scalar.add_task(d3.Integrate(b**2,'x'), layout='c', name='Eb(Tz)')

Z = dist.Field(name='Z', bases=zbasis); Z['g'] = z[:];
scalar.add_task(d3.Integrate((u@ez)*b),  layout='g', name='<wB>')
scalar.add_task(d3.Integrate(b)       ,  layout='g', name='<B>' )
scalar.add_task(d3.Integrate(Z*b)     ,  layout='g', name='<zB>')

# CFL
CFL = d3.CFL(solver, initial_dt=max_timestep, cadence=10, safety=0.35, threshold=0.05, max_change=1.5, min_change=0.5, max_dt=max_timestep)
CFL.add_velocity(u)

# Flow properties
flow = d3.GlobalFlowProperty(solver, cadence=100)

flow.add_property(d3.Integrate((u@ez)*b)     , name='<wB>')
flow.add_property(nu*d3.Integrate(d3.grad(u@ez)@d3.grad(u@ez) + d3.grad(u@ex)@d3.grad(u@ex)), name='dU^2/Re')
flow.add_property(d3.Integrate(b)            , name='<B>' )
flow.add_property(d3.Integrate(grad_b@grad_b), name='dB^2')


# Main loop
startup_iter = 10
try:
    logger.info('Starting main loop')
    while solver.proceed:
        timestep = CFL.compute_timestep()
        solver.step(timestep)
        if (solver.iteration-1) % 100 == 0:

            wB_avg = flow.grid_average('<wB>')
            dU_avg = flow.grid_average('dU^2/Re' )

            B_avg  = flow.grid_average('<B>')
            dB_avg = flow.grid_average('dB^2' )
            
            logger.info('Iteration=%i, Time=%e, dt=%e'%(solver.iteration, solver.sim_time, timestep))
            logger.info('<wB>=%f, <dU^2>/Re =%f'%(wB_avg          ,dU_avg))
            logger.info('< B>=%f, <dB^2>    =%f'%( B_avg          ,dB_avg))
            
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    solver.log_stats()
