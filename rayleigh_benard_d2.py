"""
Dedalus script for 3D Rayleigh-Benard convection.

This script uses parity-bases in the x and y directions to mimick stress-free,
insulating sidewalls.  The equations are scaled in units of the thermal
diffusion time (Pe = 1).

This script should be ran in parallel, and would be most efficient using a
2D process mesh.  It uses the built-in analysis framework to save 2D data slices
in HDF5 files.  The `merge_procs` command can be used to merge distributed analysis
sets from parallel runs, and the `plot_slices.py` script can be used to plot
the slices.

To run, merge, and plot using 4 processes, for instance, you could use:
    $ mpiexec -n 4 python3 rayleigh_benard.py
    $ mpiexec -n 4 python3 -m dedalus merge_procs snapshots
    $ mpiexec -n 4 python3 plot_slices.py snapshots/*.h5

The simulation should take roughly 400 process-minutes to run, but will
automatically stop after an hour.

"""

import sys,os,mpi4py,time

# Prevent multi-threading upon initialising mpi4py
os.environ["OMP_NUM_THREADS"] = "1";
mpi4py.rc.thread_level = 'single';

import numpy as np
from mpi4py import MPI
from dedalus import public as de
from dedalus.extras import flow_tools


import logging
logger = logging.getLogger(__name__)


# Parameters
Lx, Lz = 4, 1
Nx, Nz = 512,64
Rayleigh = 10**7
Prandtl  = 1

filename = None; #"/data/pmannix/PDF_DNS_Data/RBC8_1e10/checkpoints/checkpoints_s1.h5" 

# Create bases and domain
start_init_time = time.time()
x_basis = de.SinCos(   'x', Nx, interval=(0, Lx), dealias=3/2)
z_basis = de.Chebyshev('z', Nz, interval=(0, Lz), dealias=3/2)
domain  = de.Domain([x_basis, z_basis], grid_dtype=np.float64)

# 2D Boussinesq hydrodynamics
problem = de.IVP(domain, variables=['p','b','u','w','bz','uz','wz'], time='t')
problem.meta['p','b','w','bz','wz']['x']['parity'] = 1 # cos(kx)
problem.meta['u','uz']['x']['parity'] = -1 # sin(kx)
problem.parameters['kappa']= (Rayleigh * Prandtl)**(-1/2) # 1/Pe
problem.parameters['nu']   = (Rayleigh / Prandtl)**(-1/2) # 1/Re
problem.parameters['inv_Vol'] = 1./domain.hypervolume;

# Plume convection
x = domain.grid(0)
f = lambda x,μ,s: ( 1./(s*np.sqrt(2*np.pi)) )*np.exp(-0.5*( (x - μ)/s )**2)

g_0      = domain.new_field()
g_0.meta['x']['parity'] = 1
g_0['g'] =-f(x,μ=0.25*Lx,s=0.1)

g_1      = domain.new_field()
g_1.meta['x']['parity'] = 1
g_1['g'] =-f(x,μ=0.75*Lx,s=0.1)

problem.parameters['g_0'] = g_0;
problem.parameters['g_1'] = g_1;

problem.add_equation("dx(u) + wz = 0")
problem.add_equation("dt(b) - kappa*(dx(dx(b)) + dz(bz))             = - u*dx(b) - w*bz")
problem.add_equation("dt(u) - nu*(   dx(dx(u)) + dz(uz)) + dx(p)     = - u*dx(u) - w*uz")
problem.add_equation("dt(w) - nu*(   dx(dx(w)) + dz(wz)) + dz(p) - b = - u*dx(w) - w*wz")
problem.add_equation("bz - dz(b) = 0")
problem.add_equation("uz - dz(u) = 0")
problem.add_equation("wz - dz(w) = 0")

# Vertical bcs
problem.add_bc("left(bz) = left(g_0)")
problem.add_bc("left(u)  = 0")
problem.add_bc("left(w)  = 0")

problem.add_bc("right(bz) = right(g_1)")
problem.add_bc("right(u)  = 0")
problem.add_bc("right(w)  = 0", condition="(nx != 0)")

problem.add_bc("integ_z(p) = 0", condition="(nx == 0)")

# Build solver
solver = problem.build_solver(de.timesteppers.SBDF2)
logger.info('Solver built')

# Integration parameters
solver.stop_sim_time  = 500
solver.stop_wall_time = 60 * 60.
solver.stop_iteration = np.inf

if filename != None:
    write,initial_timestep = solver.load_state(filename);

# Analysis
checkpoints = solver.evaluator.add_file_handler('checkpoints', sim_dt=5.)
checkpoints.add_system(solver.state)

# Snapshots
snapshots = solver.evaluator.add_file_handler('snapshots', sim_dt=1)

snapshots.add_task('b', name='buoyancy',scales=1)
snapshots.add_task('u', name='u',scales=1)
snapshots.add_task('w', name='w',scales=1)

snapshots.add_task('uz - dx(w)', name='vorticity',scales=1)
snapshots.add_task('dx(b)', name='grad_bx',scales=1)
snapshots.add_task('bz'   , name='grad_bz',scales=1)
snapshots.add_task('dx(w)', name='grad_wx',scales=1)
snapshots.add_task('wz'   , name='grad_wz',scales=1)
snapshots.add_task('dz(p)', name='grad_pz',scales=1)

# Time Series and spectra
scalar = solver.evaluator.add_file_handler('scalar_data',iter=10, mode='overwrite')

scalar.add_task("integ(u**2 + w**2)",  name='Eu(t)')
scalar.add_task("integ(b**2)"       ,  name='Eb(t)')

scalar.add_task("nu*integ(dx(u)**2 + uz**2 + dx(w)**2 + wz**2 )",  name='dU^2(t)_div_Re')
scalar.add_task("integ(dx(b)**2 + bz**2)"                    ,  name='dB^2(t)'   )

scalar.add_task("inv_Vol*integ(u**2 + w**2,'z')", layout='c', name='Eu(k)' )
scalar.add_task("inv_Vol*integ(u**2 + w**2,'x')", layout='c', name='Eu(Tz)')
scalar.add_task("inv_Vol*integ(b**2,'z')", layout='c', name='Eb(k)' )
scalar.add_task("inv_Vol*integ(b**2,'x')", layout='c', name='Eb(Tz)')

scalar.add_task("integ(w*b)",  name='<wB>')
scalar.add_task("integ(b)"  ,  name='<B>' )
scalar.add_task("integ(z*b)",  name='<zB>')


# CFL
CFL = flow_tools.CFL(solver, initial_dt=1e-4, cadence=5, safety=0.5,max_change=1.5, min_change=0.5, max_dt=0.05)
CFL.add_velocities(('u', 'w'))

# Flow properties
flow = flow_tools.GlobalFlowProperty(solver, cadence=10)

flow.add_property("inv_Vol*integ(w*b)", name='<wB>')
flow.add_property("nu*inv_Vol*integ( dx(u)**2 + uz**2 + dx(w)**2 + wz**2)", name='dU^2/Re')
flow.add_property("inv_Vol*integ(b)"  , name='<B>' )
flow.add_property("inv_Vol*integ(dx(b)**2 + bz**2)", name='dB^2')

# Main loop
end_init_time = time.time()
logger.info('Initialization time: %f' %(end_init_time-start_init_time))
try:
    logger.info('Starting loop')
    while solver.ok:
        dt = CFL.compute_dt()
        solver.step(dt)
        if (solver.iteration-1) % 100 == 0:
            
            wB_avg = flow.volume_average('<wB>')
            dU_avg = flow.volume_average('dU^2/Re' )

            B_avg  = flow.volume_average('<B>')
            dB_avg = flow.volume_average('dB^2' )
            
            logger.info('Iteration=%i, Time=%e, dt=%e'%(solver.iteration, solver.sim_time, dt))
            logger.info('<wB>=%f, <dU^2>/Re =%f'%(wB_avg          ,dU_avg))
            logger.info('< B>=%f, <dB^2>    =%f'%( B_avg          ,dB_avg))
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    solver.log_stats()

from dedalus.tools import post
post.merge_process_files("checkpoints", cleanup=True, comm=MPI.COMM_WORLD);
post.merge_process_files("snapshots",   cleanup=True, comm=MPI.COMM_WORLD);
post.merge_process_files("scalar_data", cleanup=True, comm=MPI.COMM_WORLD);