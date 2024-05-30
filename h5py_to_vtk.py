import numpy as np
import os,h5py,glob,
from pyevtk.hl import gridToVTK # 

def Make_VTK(DIR_ABS_PATH,OUT_FILE,time_index=-1):

    """ 
    # Prepares data in paraview readable format by Changing from numpy to VTS filetype

    -- Options
    time_index, int
    time_index = 0 or -1; # take first or -1 last
    
    DIR_ABS_PATH, string
    # Where to find the dedalus simulation data
    DIR_ABS_PATH = "/home/pmannix/MinSeeds_DAL_Paper/Test_DAL_Re100_Pm9_M0.20";  
    
    OUT_FILE, string
    # Name of vts file created
    OUT_FILE = "./paul_TF_Pm9_M0.20";

    """

    try:
        F1   = glob.glob(DIR_ABS_PATH + '/snapshots/*');
        print(F1)
        file = h5py.File(F1[0], 'r'); 
    except:
        pass;
    
    Iterations= file['tasks/buoyancy/'].dims[0]['write_number']
    sim_time  = file['tasks/buoyancy/'].dims[0]['sim_time'    ]

    x_coords  = file['tasks/buoyancy'].dims[1][0][:]
    z_coords  = file['tasks/buoyancy'].dims[2][0][:]

    x = np.array(x_coords)
    y = np.ones(2);
    z = np.array(z_coords)

    nx = len(x)
    ny = len(y)
    nz = len(z)

    print("nx=", nx, "ny=", ny, "nz=", nz)

    s_shape = (nx, ny, nz);
    U = np.zeros(s_shape); 
    W = np.zeros(s_shape);
    B = np.zeros(s_shape); 
    
    for i in range(ny):
        U[:,i,:] = file['tasks/u'       ][time_index,:,:]; 
        W[:,i,:] = file['tasks/w'       ][time_index,:,:];
        B[:,i,:] = file['tasks/buoyancy'][time_index,:,:];


    os.chdir(DIR_ABS_PATH)
    gridToVTK(OUT_FILE, x,y,z, pointData = {"U":U,"W":W,  "B":B})

    return None;

# Execute as main
if __name__ == "__main__":

    PATH         = "/data/pmannix/PDF_DNS_Data/";
    
    #IN_FILE      = "RBC9_1e10";
    #IN_FILE      = "HC_Ra1e10_T2e04";
    #IN_FILE      = "STEP_Ra1e9_T3e03";
    #IN_FILE      = "IC_Ra1e11_T3e04";
    IN_FILE      = "ICR_Ra1e11_T8e04";
    #IN_FILE      = "PLUME_Ra1e9_T2e03";


    DIR_ABS_PATH = PATH + IN_FILE; 
    OUT_FILE     = PATH + "para_visualisation"

    Make_VTK(DIR_ABS_PATH, OUT_FILE);