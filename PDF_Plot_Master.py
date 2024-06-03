"""
Given the input files from a 2D Dedalus Rayleigh - Benard simulation
construct the following plots:

- joint PDF(s) f_Y(y)
- conditional expectations E[ (ð_iY_j)^T(ð_jY_k) |Y=y]

which [if Y = (B,Z) dim = 2D] or [if Y=(W,B,Z) dim = 3D]
"""

import h5py,glob,pickle,os
import numpy as np

from matplotlib import rc
#rc('text', usetex=True)
#rc('font', family='serif')
rc('font', size=25.0)
import matplotlib.pyplot as plt

from   scipy.interpolate import interp1d,LinearNDInterpolator
from   scipy.ndimage     import gaussian_filter


class PDF_Gen_Master(object):

    """
    A simple base class calculating PDFs and expectations from DNS_Data
    - Needs modification in order to correctly set the bandwidth
    """

    def __init__(self,file_dir,N_pts=2**10,frames=10, down_scale = 8, method = "HIST"):

        """
        Initialise the class to hold the grid, PDFs and Expectations
        """

        # Parameters
        self.file   = file_dir + '/';
        self.frames = frames
        self.N      = N_pts
        self.method = method;
        
        self.pts    = int(self.N/down_scale) # Grid for doing 3D computations
        self.Sbw    = down_scale;            # Bandwidth scaling

        self.domain = {'w':(-1,1),'b':(-1,1),'z':(0,1)} 

        # Statistics ----------------
        self.stats = None;

        # Data ----------------------
        self.data    = self.Data()
        self.weights = self.Weights()

        # Grid ----------------------
        self.b = np.zeros(self.N)
        self.w = np.zeros(self.N)
        self.z = np.zeros(self.N)

        # PDFs ----------------------

        # x3 1D
        self.fB = np.zeros(self.N)
        self.fW = np.zeros(self.N)
        self.fZ = np.zeros(self.N)

        # x3 2D
        self.fWB = np.zeros((self.N,self.N))
        self.fWZ = np.zeros((self.N,self.N))
        self.fBZ = np.zeros((self.N,self.N))

        self.scale_1D_pdf = {'fW':1.,'fB':1.}
        self.scale_2D_pdf = {'fWB':[1.,1.],'fWZ':0.0075,'fBZ':0.0075}

        # Expectations -----------------
        self.Plot_Titles  = [r'$E\{∂z P |Y = y\}$',r'E\{ |∂xB|^2 + |∂zB|^2 |Y = y}',r'$E\{ (∂xB)(∂xW ) + (∂z B)(∂z W ) | Y = y \}$',r'$E\{ |∂xW|^2 + |∂zW|^2 |Y = y\}$']
        self.Save_Handles = ['dz Pressure','Grad buoyancy squared','Cross Grad Buoyancy Velocity','Grad velocity squared']
        self.Expectations = {'\partial_z P':{},'\|dB\|^2':{},' dW^T dB':{},'\| dW \|^2':{}}

        self.scale_bw_2D = {'\partial_z P':np.ones(2),'\|dB\|^2':np.ones(2),' dW^T dB':np.ones(2),'\| dW \|^2':np.ones(2)}
        self.scale_bw_3D = {'\partial_z P':np.ones(3),'\|dB\|^2':np.ones(3),' dW^T dB':np.ones(3),'\| dW \|^2':np.ones(3)}

        for key,value in self.Expectations.items():
    
            self.Expectations[key] = {'1D':{},'2D':{}}

            # x3 1D 
            self.Expectations[key]['1D'] = {'w':np.zeros(self.N),'b':np.zeros(self.N),'z':np.zeros(self.N)}

            # x3 2D
            self.Expectations[key]['2D'] = {'wb':np.zeros((self.N,self.N)),'wz':np.zeros((self.N,self.N)),'bz':np.zeros((self.N,self.N))}

        return None;

    # Load data-----------------------

    def Data(self):

        """
        Processes the data from the dedalus format for KDEpy
        """    
        
        print('------  Loading Data ------- \n ')

        file  = h5py.File(self.file + 'snapshots/snapshots_s1.h5', mode='r')
       
        # Y = [W,B,Z], y = [w,b,z] 
        w_split = []; 
        b_split = []; 
        z_data  = file['tasks/buoyancy'].dims[2][0][:]
        
        dpGrad_split = [];
        b2Grad_split = []; 
        wbGrad_split = []; 
        w2Grad_split = []; 

        for i in range(1,self.frames+1,1):
        
            # PDF variables ---------------------
            w_cheb = file['tasks/w'       ][-i,:,:]
            b_cheb = file['tasks/buoyancy'][-i,:,:]
            w_split.append(w_cheb)
            b_split.append(b_cheb)

            # Expectation variables -------------
            try:
                dB_x = file['tasks/grad_b'][-i,0,:,:]# d/dx
                dB_z = file['tasks/grad_b'][-i,1,:,:]# d/dz
                dW_x = file['tasks/grad_w'][-i,0,:,:]# d/dx
                dW_z = file['tasks/grad_w'][-i,1,:,:]# d/dz
                dP_z = file['tasks/grad_p'][-i,1,:,:]# d/dz
            except:
                dB_x = file['tasks/grad_bx'][-i,:,:]# d/dx
                dB_z = file['tasks/grad_bz'][-i,:,:]# d/dz
                dW_x = file['tasks/grad_wx'][-i,:,:]# d/dx
                dW_z = file['tasks/grad_wz'][-i,:,:]# d/dz
                dP_z = file['tasks/grad_pz'][-i,:,:]# d/dz

            R_00 = (dW_x**2   + dW_z**2  )
            R_01 = (dB_x*dW_x + dB_z*dW_z)
            R_11 = (dB_x**2   + dB_z**2  )
            
            b2Grad_split.append( R_00)
            wbGrad_split.append( R_01)
            w2Grad_split.append( R_11)
            dpGrad_split.append( dP_z)

        b2Grad_data = np.concatenate(b2Grad_split)
        wbGrad_data = np.concatenate(wbGrad_split)
        w2Grad_data = np.concatenate(w2Grad_split)
        dpGrad_data = np.concatenate(dpGrad_split)
        
        w_data      = np.concatenate(     w_split)
        b_data      = np.concatenate(     b_split)

        return w_data,b_data,z_data,    dpGrad_data,b2Grad_data,wbGrad_data,w2Grad_data

    def Weights(self):

        """
        Generates the weights vector to correct for the Chebyshev sampling
        """

        file   = h5py.File(self.file + 'snapshots/snapshots_s1.h5', mode='r')
        
        z_cheb = file['tasks/buoyancy'].dims[2][0][:]
        x1D    = file['tasks/buoyancy'].dims[1][0][:]

        Nz  = len(z_cheb)
        x   = [ 0.5*(1. -np.cos((np.pi/Nz)*(i + 0.5)) ) for i in range(Nz) ];
        x_l = (3.*x[ 0] - x[ 1])/2.0;
        x_r = (3.*x[-1] - x[-2])/2.0;
        x_mod = np.concatenate( ([x_l],x,[x_r]));
        x_bin = np.asarray([ 0.5*(x_mod[i]+x_mod[i+1]) for i in range(len(x_mod)-1) ]);
        we_i  = abs(x_bin[0:-1]-x_bin[1:])/abs(x_bin[-1] - x_bin[0]);

        I    = np.ones(len(x1D))
        WE_i = np.outer(I,we_i).flatten() 

        weights_split = [];
        for i in range(1,self.frames+1,1):
            weights_split.append(WE_i);

        return np.concatenate(weights_split);

    def Scalings(self,name):

        if name == 'RBC': 
            # frames 200
            pdf.domain = {'w':(-1,1),'b':(0.4,0.6),'z':(0.,1.)} 
            self.scale_1D_pdf = {'fW':2.,'fB':4.}
            self.scale_2D_pdf = {'fWB':[2.,4.],'fWZ':0.0075,'fBZ':0.0075}
            
            pdf.scale_bw_2D = {'\partial_z P':[4.,12.],'\|dB\|^2':[4.,320.],' dW^T dB':[4.,840.],'\| dW \|^2':[4.,640.]}
            pdf.scale_bw_3D = {'\partial_z P':[8.,8.,24.],'\|dB\|^2':[8.,8.,640.],' dW^T dB':[8.,8.,1680.],'\| dW \|^2':[8.,8.,1280.]}
            
            #pdf.scale_bz_3D = {'\partial_z P':[8.,8.,6.],'\|dB\|^2':[2.,2.,160.],' dW^T dB':[8.,8.,420.],'\| dW \|^2':[4.,4.,640.]}

        elif name == 'STEP':
            # frames 1000
            pdf.domain = {'w':(-.75,.75),'b':(0.2,.8),'z':(0.,1.)}
            self.scale_1D_pdf = {'fW':2.,'fB':2.}
            self.scale_2D_pdf = {'fWB':[2.,2.],'fWZ':0.0075,'fBZ':0.0075}
            
            pdf.scale_bw_2D = {'\partial_z P':[2.,6.],'\|dB\|^2':[2.,40.],' dW^T dB':[4.,120.],'\| dW \|^2':[4.,640.]}
            pdf.scale_bw_3D = {'\partial_z P':[4.,4.,12.],'\|dB\|^2':[4.,4.,80.],' dW^T dB':[8.,8.,240.],'\| dW \|^2':[8.,8.,1280.]}

        elif name == 'RBC_Sine': 
            
            #frames 800
            pdf.domain = {'w':(-.3,.3),'b':(0.05,.25),'z':(0.,1.)}
            self.scale_1D_pdf = {'fW':2.,'fB':8.}
            self.scale_2D_pdf = {'fWB':[2.,8.],'fWZ':0.0075,'fBZ':0.0075}
            
            pdf.scale_bw_2D = {'\partial_z P':[8.,6.],'\|dB\|^2':[8.,640.],' dW^T dB':[8.,640.],'\| dW \|^2':[10.,1500.]}
            pdf.scale_bw_3D = {'\partial_z P':[8.,8.,6.],'\|dB\|^2':[8.,8.,640.],' dW^T dB':[8.,8.,640.],'\| dW \|^2':[15.,15.,3000.]}

        elif name == 'PLUME':
            # frames 400
            pdf.domain = {'w':(-.025,.025),'b':(-.015,.015),'z':(0.,1.)}
            self.scale_1D_pdf = {'fW':2.,'fB':8.}
            self.scale_2D_pdf = {'fWB':[2.,8.],'fWZ':0.0075,'fBZ':0.015}
            
            pdf.scale_bw_2D = {'\partial_z P':[8.,4.],'\|dB\|^2':[8.,60.],' dW^T dB':[8.,1000.],'\| dW \|^2':[8.,20000.]}
            pdf.scale_bw_3D = {'\partial_z P':[8.,32.,16.],'\|dB\|^2':[8.,32.,180.],' dW^T dB':[8.,32.,2000.],'\| dW \|^2':[8.,32.,40000.]}

        elif name == 'IC' or name == 'IC_Noise': 
            #frames 750
            pdf.domain = {'w':(-0.05,0.05),'b':(0.004,0.00725),'z':(0.,1.)} 
            self.scale_1D_pdf = {'fW':2.,'fB':1.}
            self.scale_2D_pdf = {'fWB':[2.,1.],'fWZ':0.0002,'fBZ':0.0001}
            
            pdf.scale_bw_2D = {'\partial_z P':[2.,4.],'\|dB\|^2':[4.,40.],' dW^T dB':[4.,160.],'\| dW \|^2':[4.,320.]}
            pdf.scale_bw_3D = {'\partial_z P':[2.,2.,4.],'\|dB\|^2':[4.,4.,40.],' dW^T dB':[4.,4.,160.],'\| dW \|^2':[4.,4.,320.]}

        return None;
     
    # Calculation -----------------------

    def Fit(self,data,weights=None,scale_bw=None):

        """
        Fit the data using the KDEpy module
        - Note whilst we use the silverman rule to select the bandwidth i.e h
          this doesn't technically support weighted data so we must handle this 
          manually. We use the identity bw matrix, which requires some rescaling.
        """
        
        #bws = FFTKDE._bw_methods.keys(); print('bandwidths',bws)
        #'silverman', 'scott', 'ISJ'

        # ----- Rescale forwards
        bw_array = [];
        for i in range(data.shape[1]):
            bw_i  = bw_selection.silvermans_rule(data[:, [i]])
            bw_array.append(bw_i)
        bw_array = np.asarray(bw_array)
        
        if   isinstance(scale_bw,float):
            bw_array *= scale_bw;
        elif isinstance(scale_bw,list):
            bw_array *= np.asarray(scale_bw);
        
        data_scaled = data/bw_array              

        # ----- KDE Fit -----------     
        kde = FFTKDE(kernel='gaussian', bw=1.0,norm=2).fit(data=data_scaled,weights=weights)
        
        if   data.shape[1] == 1:
            grid, points = kde.evaluate(grid_points=self.N)
        elif data.shape[1] == 2:
            grid, points = kde.evaluate(grid_points=(self.N,self.N))
        elif data.shape[1] == 3:
            grid, points = kde.evaluate(grid_points=(self.pts,self.pts,self.pts))
    
        # ~~~~~ Rescale backwards ~~~~~~~~~~~
        grid   = grid * bw_array
        points = points / np.prod(bw_array)

        #print('bw_array = ',bw_array)

        return grid, points

    def Generate_PDF(self):

        self.Generate_PDF_HIST(); 

        return None;

    def Generate_Expectations(self):

        
        self.Generate_Expectations_HIST(); 

        # Interpolate both in z onto a finer grid
         
        return None;

    def Generate_PDF_HIST(self):

        """
        Generates all enumerations of 1D & 2D PDFs
        """
        
        print('------  Generating PDFs ------- \n ')

        # Get Data & weights
        W,B,Z,  dPGrad,B2Grad,WBGrad,W2Grad = self.data
        weights = self.weights

        # 1D PDFs
        pdfs_1D = ['fW','fB']
        grid_1D = ['w',  'b']
        data_1D = [W.flatten(),B.flatten()]
        for pdf_name,grid_name,X in zip(pdfs_1D,grid_1D,data_1D):
            
            # Histogram
            points,bin_edges = np.histogram(X,bins=self.N,weights=weights,density=True);
            grid = 0.5*(bin_edges[1:] + bin_edges[:-1]);
            
            setattr(self,pdf_name,points)
            setattr(self,grid_name,grid)

        setattr(self,'fZ',np.ones(len(Z)))
        setattr(self,'z',Z)

        # HIST
        fWB = np.histogram2d(x=W.flatten(),y=B.flatten(),bins=self.N,weights=weights,density=True)[0]
        setattr(self,'fWB',fWB)

        fWZ = np.zeros((len(self.w),len(self.z)));
        fBZ = np.zeros((len(self.b),len(self.z)));
        for j in range(len(self.z)):
            
            # HIST
            fWZ[:,j] = np.histogram(W[:,j],bins=self.N, range = (min(self.w),max(self.w)),density=True)[0]
            fBZ[:,j] = np.histogram(B[:,j],bins=self.N, range = (min(self.b),max(self.b)),density=True)[0]

        setattr(self,'fWZ',fWZ)
        setattr(self,'fBZ',fBZ)

        return None;

    def Generate_Expectations_HIST(self):

        """
        Compute the 1D & 2D conditional expectations by creating the 2D pdf
        
            E[Φ|B = b] = int φfΦ|B (φ|b)dφ = int φ fBΦ(b,φ)/fB(b) dφ,

        and integrating out the independent variable φ. The same
        approach is used to compute 2D conditional expectations.
        """

        print('------  Generating Conditional Expectations ------- \n ')

        W,B,Z,  dPGrad,B2Grad,WBGrad,W2Grad = self.data
        weights = self.weights
        Term_E  = [dPGrad,B2Grad,WBGrad,W2Grad]
        Nz      = len(self.z)

        # x2 1D E{Φ|W=w},E{Φ|B=b}
        data_1D = [W,B]
        grid_1D = ['w','b']
        for key,value, Φ in zip(self.Expectations.keys(),self.Expectations.values(),Term_E):
            
            for key_1D,X_i,grid_name in zip(value['1D'].keys(),data_1D,grid_1D):

                f_XΦ,x,φ = np.histogram2d(X_i.flatten(),Φ.flatten(),bins=self.N,density=True,weights=weights)
                φ = .5*(φ[1:]+φ[:-1]); dφ = φ[1] - φ[0];

                # E{Φ|X} = int_φ f_Φ|X(φ|x)*φ dφ
                f_X                 =  np.sum(  f_XΦ,axis=1)*dφ     # f_X(x)
                value['1D'][key_1D] = (np.sum(φ*f_XΦ,axis=1)*dφ)/f_X;

        #'''
        # 2D E{Φ|W=w,B=b}
        key_2D  = 'wb'
        for key,value, Φ in zip(self.Expectations.keys(),self.Expectations.values(),Term_E):
    
            data = [W.flatten(),B.flatten(),Φ.flatten()]
            f_XΦ,Edges = np.histogramdd(data,bins=self.N,density=True,weights=weights) # f_XΦ(x_1,x_2,φ)
            φ = .5*(Edges[2][1:]+ Edges[2][:-1]); dφ = φ[1] - φ[0];

            # E{Φ|X} = int_φ f_Φ|X(φ|x)*φ dφ
            f_X                 =  np.sum(  f_XΦ,axis=2)*dφ     # f_X(x)
            value['2D'][key_2D] = (np.sum(φ*f_XΦ,axis=2)*dφ)/f_X;
        #'''

        # 2D E{Φ|Z=z,B=b}
        key_2D  = 'bz'
        for key,value, Φ in zip(self.Expectations.keys(),self.Expectations.values(),Term_E):
            
            E_X     = np.zeros((self.N,Nz)) # f_BZΦ
            err     = []
            Sum_pij = []
            for j in range(Nz):
                
                # HIST
                f_XΦ,b,φ = np.histogram2d(x=B[:,j].flatten(),y=Φ[:,j].flatten(),bins=self.N,density=True)
                b = .5*(b[1:]+b[:-1]); db = b[1] - b[0];
                φ = .5*(φ[1:]+φ[:-1]); dφ = φ[1] - φ[0];
                f_X      =  np.sum(  f_XΦ,axis=1)*dφ
                E        = (np.sum(φ*f_XΦ,axis=1)*dφ)/f_X;
                E_X[:,j] = interp1d(b,E,bounds_error=False)(self.b)

                #Check smoothness
                err.append( np.linalg.norm(self.fB - f_X,2)/np.linalg.norm(self.fB,2) )
                Sum_pij.append( np.sum(f_X)*db )

            value['2D'][key_2D] = E_X
            #print('\n ~~~~~~~~ E{%s|B=b,Z=z} ~~~~~~~~~~~~ '%key)
            
        key_2D  = 'wz'
        for key,value, Φ in zip(self.Expectations.keys(),self.Expectations.values(),Term_E):
            
            
            E_X     = np.zeros((self.N,Nz)) # f_WZΦ
            err     = []
            Sum_pij = []
            for j in range(Nz):
                
                f_XΦ,w,φ = np.histogram2d(x=W[:,j].flatten(),y=Φ[:,j].flatten(),bins=self.N,density=True)
                w = .5*(w[1:]+w[:-1]); dw = w[1] - w[0];
                φ = .5*(φ[1:]+φ[:-1]); dφ = φ[1] - φ[0];

                # E{Φ|W,Z=zj} = int_φ f_Φ|W(φ|w,z_j)*φ dφ
                f_X      =  np.sum(  f_XΦ,axis=1)*dφ
                E        = (np.sum(φ*f_XΦ,axis=1)*dφ)/f_X;
                E_X[:,j] = interp1d(w,E,bounds_error=False)(self.w)

                #Check smoothness
                err.append( np.linalg.norm(self.fW - f_X,2)/np.linalg.norm(self.fW,2) )
                Sum_pij.append(  np.sum(f_X)*dw )
            
            value['2D'][key_2D] = E_X
            #print('\n ~~~~~~~~ E{%s|W=w,Z=z} ~~~~~~~~~~~~ '%key)
            
        return None;
    

    def Spectra(self):

        """
        Plot the time-averaged spectra of the Kinetic energy and buoyancy variance
        to verify the spatial convergence of the simulations.
        """


        f  = h5py.File(self.file + 'scalar_data/scalar_data_s1.h5', mode='r')
        
        fig, (ax1,ax2) = plt.subplots(ncols=2,figsize=(8,4))
        ax1.semilogy(f['tasks/Eu(k)'][-1,:,0] ,'r.')
        ax1.set_ylabel(r'Kinetic Energy')
        ax1.set_xlabel(r'Fourier mode k')

        ax2.semilogy(f['tasks/Eu(Tz)'][-1,0,:],'b.')
        ax2.set_xlabel(r'Chebyshev polynomial Tz')
        fig.savefig('Kinetic Energy Spectra', dpi=200)
        plt.close(fig)
        
        fig, (ax1,ax2) = plt.subplots(ncols=2,figsize=(8,4))
        ax1.semilogy(f['tasks/Eb(k)'][-1,:,0] ,'r.')
        ax1.set_ylabel(r'Buoyancy Energy')
        ax1.set_xlabel(r'Fourier mode k')

        ax2.semilogy(f['tasks/Eb(Tz)'][-1,0,:],'b.')
        ax2.set_xlabel(r'Chebyshev polynomial Tz')
        fig.savefig('Buoyancy Energy Spectra', dpi=200)
        plt.close(fig)


        # Shape time,x,z
        Eu     = f['tasks/Eu(t)'][:,0,0]
        Eb     = f['tasks/Eb(t)'][:,0,0]
        wB_avg = f['tasks/<wB>'][:,0,0]
        B_avg  = f['tasks/<B>' ][:,0,0]
        t      = f['scales/sim_time'][()]
        
        fig, axs = plt.subplots(2, 2)
        axs[0, 0].plot(t,Eu,'b-',label=r'$E_u$')
        axs[0, 0].set_title(r'$E_u$')
        axs[0, 1].plot(t,Eb,'r-',label=r'$E_b$')
        axs[0, 1].set_title(r'$E_b$')
        axs[1, 0].plot(t,wB_avg,'b:',label=r'$\langle wB \rangle$')
        axs[1, 0].set_title(r'$\langle wB \rangle$')
        axs[1, 1].plot(t, B_avg,'r:',label=r'$\langle B  \rangle$')
        axs[1, 1].set_title(r'$\langle B  \rangle$')
        plt.tight_layout()
        fig.savefig('EnergyTimeSeries.png',dpi=200)
        plt.close(fig)

        return None;

    def Energetics(self):

        """
        Compute all time and space averaged diagnostic
        quantites including the available, total and reference
        potential energies by using the reference height/CDF

        z*(b) = int_b fB(d) db = F_B(b)

        """
       
        print('------  Computing Energetics ------- \n ')

        f     = h5py.File(self.file + 'scalar_data/scalar_data_s1.h5', mode='r')    
        file  = h5py.File(self.file + 'snapshots/snapshots_s3.h5'    , mode='r')
        times = file['tasks/buoyancy/'].dims[0]['sim_time'][:]
        print('Averaging Window T = [%f,%f]'%( times[-self.frames], times[-1] ))
        indx  = np.where(f['scales/sim_time'][()] > times[-self.frames])

        Lx = 4.0; Lz = 1.0; V = Lx*Lz;# Domain size

        try:
            Disp_U = np.mean(f['tasks/dU^2(t)/Re'][:,0,0][indx])/V
        except:
            Disp_U = np.mean(f['tasks/dU^2(t)_div_Re'][:,0,0][indx])/V
        wb_avg = np.mean(f['tasks/<wB>'      ][:,0,0][indx])/V
 
        B_avg  = np.mean(f['tasks/<B>'    ][:,0,0][indx])/V
        Disp_B = np.mean(f['tasks/dB^2(t)'][:,0,0][indx])/V

        KE     = np.mean(f['tasks/Eu(t)'  ][:,0,0][indx])/V# <u.u>
        BE     = np.mean(f['tasks/Eb(t)'  ][:,0,0][indx])/V# <b*b>

        # Spectral TPE
        TPE = (-1./V)*np.mean(f['tasks/<zB>'][:,0,0][indx])
        
        # PDF TPE
        Ibz = np.outer(self.b,self.z)
        db = abs(self.b[1] - self.b[0]) 
        TPE_pdf = -np.sum( np.trapz(Ibz*self.fBZ,x=self.z,axis=1) )*db # use trapz as z is non-uniform grid

        error = abs(TPE_pdf - TPE)/abs(TPE) 
        if error > 1e-02:
            print(f"TPE resdiual error must be less than 1e-03 but got : {error}")
            #raise ValueError(f"TPE resdiual error must be less than 1e-03 but got : {error}")

        z_b = np.cumsum(self.fB)*db
        RPE = -np.sum(z_b*self.b*self.fB)*db
        APE = (TPE - RPE);

        self.stats = {'TPE':TPE,'APE':APE,'RPE':RPE,
                      '(1/2)<|U|^2>':.5*KE,'<|∇U|^2>/Re':Disp_U,
                      '(1/2)<|B|^2>':.5*BE,'<|∇B|^2>':   Disp_B,
                      '<WB>':       wb_avg,'<B>':        B_avg }

        # print('\n~~~~~~~~~~~~~~~~~~~~~')
        # print('Integral Quantities \n')
        # for (key,val) in self.stats.items():
        #     print(key,'=%e'%val)
        # print('~~~~~~~~~~~~~~~~~~~~~ \n')

        print('  APE     &      RPE    &      E_k   &      <WB>   &       <e_U>/Re &   (1/2)<|B|^2> &      <e_B> &    <B> ')
        print('%1.3e &  %1.3e &  %1.3e &   %1.3e &      %1.3e &      %1.3e &  %1.3e &  %1.3e '%(APE,RPE,.5*KE,wb_avg,Disp_U,.5*BE,Disp_B,B_avg) )
        print('~~~~~~~~~~~~~~~~~~~~~ \n')

        return TPE,APE;
    
    # Interpolation
    def Interpolate(self):

        # Interpolate both in z onto a finer grid
        zq = np.linspace(min(self.z),max(self.z),self.N);
        
        # PDFs
        fBZ = np.zeros((self.N,self.N))
        fWZ = np.zeros((self.N,self.N))
        for i in range(self.N):
            fBZ[i,:] = interp1d(self.z,self.fBZ[i,:])(zq)
            fWZ[i,:] = interp1d(self.z,self.fWZ[i,:])(zq)
        self.fBZ = fBZ;
        self.fWZ = fWZ;
        self.fZ  = interp1d(self.z,self.fZ)(zq)

        # Expectations
        for key,value in zip(self.Expectations.keys(),self.Expectations.values()):
            
            E_BZ = np.zeros((self.N,self.N)) # E[Φ|B=b,Z=z]
            E_WZ = np.zeros((self.N,self.N)) # E[Φ|W=w,Z=z]
            for i in range(self.N):
                E_BZ[i,:] = interp1d(self.z,value['2D']['bz'][i,:])(zq)
                E_WZ[i,:] = interp1d(self.z,value['2D']['wz'][i,:])(zq)
            
            value['2D']['bz'] = E_BZ
            value['2D']['wz'] = E_WZ
        
        # Grid
        self.z  = zq

        return None;

    # Plottting ------------------------
    
    def Plot_Pdfs(       self     ,interval=None,figname=None,dpi=200,Nlevels=15,norm_2d='log'):

        """
        Return a base plot of the PDFs ontop of which we can overlay the expectations
        """

        fig, axs = plt.subplots(3, 2,figsize=(8,6),constrained_layout=True)
        
        if interval != None:
            
            idx1 = np.where(self.b < interval['b'][1]);
            idx2 = np.where(self.b > interval['b'][0]);
            b_idx = np.intersect1d(idx1,idx2)

            idx1 = np.where(self.w < interval['w'][1]);
            idx2 = np.where(self.w > interval['w'][0]); 
            w_idx = np.intersect1d(idx1,idx2)

            idx1 = np.where(self.z < interval['z'][1]); 
            idx2 = np.where(self.z > interval['z'][0]); 
            z_idx = np.intersect1d(idx1,idx2)

        else:
            b_idx = np.arange(0,len(self.b),1)
            w_idx = np.arange(0,len(self.w),1)
            z_idx = np.arange(0,len(self.z),1)

        # f_B ---------------
        axs[0, 0].plot(self.b[b_idx],self.fB[b_idx],'r')
        axs[0, 0].set_ylim([0.,1.01*max(self.fB[b_idx])])
        axs[0, 0].fill_between(x=self.b[b_idx],y1=self.fB[b_idx],color= "r",alpha= 0.2)
        axs[0, 0].tick_params('x', labelbottom=False)
        name_00 = {'x':r'$B$','y':r'$f_B(b)$'}
      
        axs[1, 0].sharex(axs[0, 0])
        axs[1, 0].contourf(self.b[b_idx], self.z[z_idx], self.fBZ[b_idx,:][:,z_idx].T,cmap='Reds',levels=Nlevels ,norm=norm_2d)
        axs[1, 0].tick_params('x', labelbottom=False)
        name_10 = {'x':r'$B$','y':r'$Z$'}

        axs[2, 0].sharex(axs[0, 0])
        axs[2, 0].contourf(self.b[b_idx], self.w[w_idx], self.fWB[w_idx,:][:,b_idx]  ,cmap='Reds',levels=Nlevels,norm=norm_2d)
        name_20 = {'x':r'$B$','y':r'$W$'}
        axs[2, 0].set_xlabel(r'$B$')

        # f_W -------------
        axs[0,1].plot(self.w[w_idx],self.fW[w_idx],'r')
        axs[0,1].set_ylim([0.,1.01*max(self.fW[w_idx])])
        axs[0,1].fill_between(x=self.w[w_idx],y1=self.fW[w_idx],color= "r",alpha= 0.2)
        axs[0,1].tick_params('x', labelbottom=False)
        name_01 = {'x':r'$W$','y':r'$f_W(w)$'}
      
        axs[1, 1].sharex(axs[0, 1])
        axs[1, 1].contourf(self.w[w_idx], self.z[z_idx], self.fWZ[w_idx,:][:,z_idx].T,cmap='Reds',levels=10*Nlevels,norm=norm_2d)
        axs[1, 1].tick_params('x', labelbottom=False)
        name_11 = {'x':r'$W$','y':r'$Z$'}

        axs[2, 1].sharex(axs[0, 1])
        axs[2, 1].contourf(self.w[w_idx], self.b[b_idx], self.fWB[w_idx,:][:,b_idx].T,cmap='Reds',levels=Nlevels,norm=norm_2d)
        name_21 = {'x':r'$W$','y':r'$B$'}
        axs[2, 1].set_xlabel(r'$W$')

        
        Names = np.asarray( [[name_00,name_01],[name_10,name_11], [name_20,name_21]] );
        for ax,name in zip(axs.flat,Names.flat):
            ax.set(ylabel=name['y'])

        if figname != None:
            fig.savefig(figname, dpi=dpi)
        plt.close(fig)

        return fig,axs;

    def Plot_Expectation(self,term,interval=None,figname=None,dpi=200,Nlevels=15,norm='linear',sigma_smooth=1.):

        """
        Using the plots of the pdfs overlay the expectations in terms:
        - contours for surface plots
        - hatched lines for graph plots
        """

        if interval != None:
            
            idx1 = np.where(self.b < interval['b'][1]);
            idx2 = np.where(self.b > interval['b'][0]);
            b_idx = np.intersect1d(idx1,idx2)

            idx1 = np.where(self.w < interval['w'][1]);
            idx2 = np.where(self.w > interval['w'][0]); 
            w_idx = np.intersect1d(idx1,idx2)

            idx1 = np.where(self.z < interval['z'][1]); 
            idx2 = np.where(self.z > interval['z'][0]); 
            z_idx = np.intersect1d(idx1,idx2)

        else:
            b_idx = np.arange(0,len(self.b),1)
            w_idx = np.arange(0,len(self.w),1)
            z_idx = np.arange(0,len(self.z),1)


        fig,axs = self.Plot_Pdfs(interval=interval,figname=figname,dpi=dpi,Nlevels=15,norm_2d='log');

        E_1D = self.Expectations[term]['1D']
        E_2D = self.Expectations[term]['2D']

        # B ---------------------
        twin_00 = axs[0,0].twinx()
        twin_00.plot(self.b[b_idx],E_1D['b'][b_idx], 'b-',label=str(r'$E['+term+'\|b]$'))
        twin_00.set_ylabel(r'$E['+term+'\|b]$');

        Z = gaussian_filter(E_2D['bz'][b_idx,:][:,z_idx], sigma=sigma_smooth,truncate=3.0)
        try:
            Level = np.linspace(Z.flatten().min(),Z.flatten().max(),Nlevels)
            CS_10  = axs[1, 0].contour(self.b[b_idx], self.z[z_idx], Z.T, levels = Level,norm=norm,cmap='Blues')
            axs[1, 0].clabel(CS_10, inline=False, fontsize=1) 
        except:
            CS_10  = axs[1, 0].contour(self.b[b_idx], self.z[z_idx], Z.T, levels = Nlevels,norm=norm,cmap='Blues')
            axs[1, 0].clabel(CS_10, inline=False, fontsize=1) 

        Z = gaussian_filter(E_2D['wb'][w_idx,:][:,b_idx], sigma=sigma_smooth,truncate=3.0)
        try:
            Level = np.linspace(Z.flatten().min(),Z.flatten().max(),Nlevels)
            CS_20  = axs[2, 0].contour(self.b[b_idx], self.w[w_idx], Z , levels = Level,norm=norm,cmap='Blues')
            axs[2, 0].clabel(CS_20, inline=False, fontsize=1) 
        except:
            CS_20  = axs[2, 0].contour(self.b[b_idx], self.w[w_idx],  Z , levels = Nlevels,norm=norm,cmap='Blues')
            axs[2, 0].clabel(CS_20, inline=False, fontsize=1) 

        # W ---------------------
        twin_01 = axs[0,1].twinx()
        twin_01.plot(self.w[w_idx],E_1D['w'][w_idx], 'b-',label=str(r'$E['+term+'\|w]$'))
        twin_01.set_ylabel(r'$E['+term+'\|w]$');

        Z = gaussian_filter(E_2D['wz'][w_idx,:][:,z_idx], sigma=sigma_smooth,truncate=3.0)
        try:
            Level = np.linspace(Z.flatten().min(),Z.flatten().max(),Nlevels)
            CS_11  = axs[1, 1].contour(self.w[w_idx], self.z[z_idx], Z.T, levels = Level,norm=norm,cmap='Blues')
            axs[1, 1].clabel(CS_11, inline=False, fontsize=1)
        except:
            CS_11  = axs[1, 1].contour(self.w[w_idx], self.z[z_idx], Z.T, levels = Nlevels,norm=norm,cmap='Blues')
            axs[1, 1].clabel(CS_11, inline=False, fontsize=1)     

        Z = gaussian_filter(E_2D['wb'][w_idx,:][:,b_idx], sigma=sigma_smooth,truncate=3.0)
        try:
            Level = np.linspace(Z.flatten().min(),Z.flatten().max(),Nlevels)
            CS_21  = axs[2, 1].contour(self.w[w_idx], self.b[b_idx], Z.T, levels = Level,norm=norm,cmap='Blues')
            axs[2, 1].clabel(CS_21, inline=False, fontsize=1) 
        except:
            CS_21  = axs[2, 1].contour(self.w[w_idx], self.b[b_idx], Z.T, levels = Nlevels,norm=norm,cmap='Blues')
            axs[2, 1].clabel(CS_21, inline=False, fontsize=1) 

        if figname != None:
            fig.savefig(figname, dpi=dpi)
        #plt.show()
        plt.close()

        return None;

class PDF_Master(PDF_Gen_Master):

    """
    A simple base class plotting PDFs and expectations 
    """
    
    def __init__(self,file_dir,N_pts=2**10,frames=10, down_scale = 8,method='HIST'):

         # Initialise the base clas
        super().__init__(file_dir,N_pts,frames,down_scale,method)

        return None;

    # Paper figures

    # ----- section f_BZ

    def Plot_fBZ(self,interval = None, figname=None,dpi=200,Nlevels=15,norm_2d='log'):

        """
        Return a base plot of the PDFs ontop of which we can overlay the expectations
        """

        fig, axs = plt.subplots(2,1,figsize=(12,8),height_ratios=[2,3],sharex=True,constrained_layout=True)
        
        if interval != None:
            
            idx1 = np.where(self.b < interval['b'][1]);
            idx2 = np.where(self.b > interval['b'][0]);
            b_idx = np.intersect1d(idx1,idx2)

            idx1 = np.where(self.w < interval['w'][1]);
            idx2 = np.where(self.w > interval['w'][0]); 
            w_idx = np.intersect1d(idx1,idx2)

            idx1 = np.where(self.z < interval['z'][1]); 
            idx2 = np.where(self.z > interval['z'][0]); 
            z_idx = np.intersect1d(idx1,idx2)

        else:
            b_idx = np.arange(0,len(self.b),1)
            w_idx = np.arange(0,len(self.w),1)
            z_idx = np.arange(0,len(self.z),1)

        # 1D PDFs ---------------
        axs[0].plot(self.b[b_idx],self.fB[b_idx],'r')
        axs[0].set_ylim([0.,1.01*max(self.fB[b_idx])])
        axs[0].fill_between(x=self.b[b_idx],y1=self.fB[b_idx],color= "r",alpha= 0.2)
        axs[0].set_ylabel(r'$f_B(b)$',color='r')
        axs[0].tick_params(axis="y",labelcolor="r")

        # 2D PDFs ---------------
        #axs[1].pcolormesh(self.b[b_idx], self.z[z_idx], self.fBZ[b_idx,:][:,z_idx].T,cmap='Reds' ,norm='linear')
        axs[1].contourf(self.b[b_idx], self.z[z_idx], self.fBZ[b_idx,:][:,z_idx].T,cmap='Reds',levels=Nlevels ,norm=norm_2d)
        axs[1].set_xlabel(r"$b$")
        axs[1].set_ylabel(r"$z$")

        if figname != None:
            fig.savefig(figname, dpi=dpi)
        #plt.show()
        plt.close(fig)

        return fig,axs;

    def Plot_EBZ(self,term,interval=None,figname=None,dpi=200,Nlevels=15,norm='log',sigma_smooth=2):

        """
        Using the plots of the pdfs overlay the expectations in terms:
        - contours for surface plots
        - hatched lines for graph plots
        """
        
        #'''

        if interval != None:
            
            idx1 = np.where(self.b < interval['b'][1]);
            idx2 = np.where(self.b > interval['b'][0]);
            b_idx = np.intersect1d(idx1,idx2)

            idx1 = np.where(self.w < interval['w'][1]);
            idx2 = np.where(self.w > interval['w'][0]); 
            w_idx = np.intersect1d(idx1,idx2)

            idx1 = np.where(self.z < interval['z'][1]); 
            idx2 = np.where(self.z > interval['z'][0]); 
            z_idx = np.intersect1d(idx1,idx2)

        else:
            b_idx = np.arange(0,len(self.b),1)
            w_idx = np.arange(0,len(self.w),1)
            z_idx = np.arange(0,len(self.z),1)


        fig,axs = self.Plot_fBZ(interval=interval,figname=figname,dpi=dpi);

        # 1D Expectationss ---------------
        E_1D = self.Expectations[term]['1D']

        twin_00 = axs[0].twinx()
        twin_00.plot(self.b[b_idx],E_1D['b'][b_idx], 'b')
        twin_00.set_ylabel(r'$E\{ |\nabla B|^2 | b \}$',color="b");
        twin_00.tick_params(axis="y",labelcolor="b")

        # 2D Expectations ---------------
        E_2D = self.Expectations[term]['2D']
        
        Z = gaussian_filter(E_2D['bz'][b_idx,:][:,z_idx], sigma=sigma_smooth,truncate=3.0)
        try:
            Level = np.linspace(Z.flatten().min(),Z.flatten().max(),Nlevels)
            CS_10  = axs[1].contour(self.b[b_idx], self.z[z_idx],Z.T, levels = Level,norm=norm,cmap='Blues')
            axs[1].clabel(CS_10, inline=False, fontsize=1) 
        except:
            CS_10  = axs[1].contour(self.b[b_idx], self.z[z_idx],Z.T, levels =Nlevels,norm=norm,cmap='Blues')
            axs[1].clabel(CS_10, inline=False, fontsize=1) 

        if figname != None:
            fig.savefig(figname, dpi=dpi)
        #plt.show()
        plt.close(fig)

        return None;

    def Decomposition(self,interval=None,figname=None,dpi=200,Nlevels=10):

        """
        
        """

        if interval != None:
            
            idx1 = np.where(self.b < interval['b'][1]);
            idx2 = np.where(self.b > interval['b'][0]);
            b_idx = np.intersect1d(idx1,idx2)

            idx1 = np.where(self.w < interval['w'][1]);
            idx2 = np.where(self.w > interval['w'][0]); 
            w_idx = np.intersect1d(idx1,idx2)

            idx1 = np.where(self.z < interval['z'][1]); 
            idx2 = np.where(self.z > interval['z'][0]); 
            z_idx = np.intersect1d(idx1,idx2)

        else:
            b_idx = np.arange(0,len(self.b),1)
            w_idx = np.arange(0,len(self.w),1)
            z_idx = np.arange(0,len(self.z),1)

        # Calculate the terms
        #Φ = self.Expectations['\|dB\|^2']['1D']['b']
        Φ  = gaussian_filter(self.Expectations['\|dB\|^2']['1D']['b'], sigma=3,truncate=3.0)
        fB = gaussian_filter(self.fB                                 , sigma=3,truncate=3.0)
        h = self.b[b_idx][1] - self.b[b_idx][0]

        ddf = (fB[b_idx][2:] - 2.*fB[b_idx][1:-1] + fB[b_idx][:-2])/(h**2);
        Φ_norm = np.linalg.norm(Φ[b_idx][1:-1]*ddf,2)

        dΦ = ( Φ[b_idx][2:]  -  Φ[b_idx][:-2])/(2.*h)
        df = (fB[b_idx][2:]  - fB[b_idx][:-2])/(2.*h)
        dΦ_norm = np.linalg.norm(dΦ*df,2)
        
        ddΦ = (Φ[b_idx][2:] - 2.*Φ[b_idx][1:-1] + Φ[b_idx][:-2])/(h**2);
        ddΦ_norm = np.linalg.norm(ddΦ*fB[b_idx][1:-1],2)


        fig,ax=plt.subplots(3,sharex=True)

        # Plot PDFs
        ax[2].plot(self.b[b_idx],self.fB[b_idx],'r-',linewidth=1)
        ax[2].fill_between(x=self.b[b_idx],y1=self.fB[b_idx],color= "r",alpha= 0.2)
        
        ax[1].plot(self.b[b_idx],self.fB[b_idx],'r-',linewidth=1)
        ax[1].fill_between(x=self.b[b_idx],y1=self.fB[b_idx],color= "r",alpha= 0.2)
        
        ax[0].plot(self.b[b_idx],self.fB[b_idx],'r-',linewidth=1)
        ax[0].fill_between(x=self.b[b_idx],y1=self.fB[b_idx],color= "r",alpha= 0.2)

        ax[2].set_xlabel(r'$b$',fontsize=20)
        ax[2].set_ylabel(r'$f_B(b)$',fontsize=20)
        ax[1].set_ylabel(r'$f_B(b)$',fontsize=20)
        ax[0].set_ylabel(r'$f_B(b)$',fontsize=20)

        ax[2].set_xlim(interval['b'])
        ax[1].set_xlim(interval['b'])
        ax[0].set_xlim(interval['b'])

       
        # Plot E[Φ|B=b]
        twin2 = ax[2].twinx()
        twin2.plot(self.b[b_idx][1:-1],-1.*(Φ[b_idx][1:-1]*ddf)/Φ_norm,'b-',label=r"$-E\{|\nabla B|^2|b\} \frac{\partial^2 f_B}{\partial b^2}$")
        twin2.legend(loc=1,fontsize=14)
        twin2.set_yticks([])
        
        twin1 = ax[1].twinx()
        twin1.plot(self.b[b_idx][1:-1],-1.*(dΦ*df)/dΦ_norm,'b-',label=r"$-E\{|\nabla B|^2|b\}' \frac{\partial f_B}{\partial b}$")
        twin1.legend(loc=3,fontsize=14)
        twin1.set_yticks([])

        twin0 = ax[0].twinx()
        twin0.plot(self.b[b_idx][1:-1],-1.*(ddΦ*fB[b_idx][1:-1])/ddΦ_norm, 'b-',label=r"$-E\{|\nabla B|^2|b\}'' f_B$")
        twin0.legend(loc=3,fontsize=14)
        twin0.set_yticks([])

        print('|Φ| =',Φ_norm)
        print('|dΦ/db| =',dΦ_norm)
        print('|d^2Φ/db^2| =',ddΦ_norm)

        plt.tight_layout()
        if figname != None:
            fig.savefig(figname, dpi=dpi)
        #plt.show()
        plt.close(fig)

        return None;

      # Check these
    
    def boundary_terms(self,figname):

        """
        Processes the data from the dedalus format for KDEpy
        """   

        raise NotImplementedError('This has not yet been corrected to work with D2 nor verified \n')
    
        print('Calculating the boundary terms as not yet available ...')
        file = h5py.File(self.file  + 'snapshots/snapshots_s1.h5', mode='r')

        z_cheb = file['tasks/buoyancy'].dims[2][0][:]; Nz = len(z_cheb)
        x_cheb = file['tasks/buoyancy'].dims[1][0][:]; Nx = len(x_cheb)
        Z      = np.outer(np.ones(Nx),z_cheb)

        # Y = [Φ,B,Z], y = [φ,b,z]
        Φ_data = [];
        B_data = [];
        Z_data = [];
        for i in range(1,self.frames+1,1):
    
            # Integrate over x eval
            # [t,x,z]
            dBz = file['tasks/grad_b'][-i,1,:,:]
            B   = file['tasks/buoyancy'][-i,:,:]
        
            Φ_data.append(dBz.flatten());
            B_data.append(  B.flatten());
            Z_data.append(  Z.flatten());

        Φ = np.concatenate(Φ_data)  
        B = np.concatenate(B_data)  
        Z = np.concatenate(Z_data)

        weights = self.weights #Weights()

        data = np.vstack( (B,Z,Φ) ).T;
        
        grid, points = self.Fit(data=data,weights=weights,scale_bw=[8,8,200])                
        x1,x2,φ = np.unique(grid[:, 0]), np.unique(grid[:, 1]), np.unique(grid[:, 2])
        p_ijk = points.reshape(self.pts,self.pts,self.pts) # f_XΦ(x_1,x_2,φ)        
        p_ij  = np.trapz(p_ijk, x=φ, axis = 2);            # f_X( x_1,x_2  )

        # p_ijk,edges = np.histogramdd(data, bins=(32,64,32), range=None, density=True, weights=weights)
        # x1,x2,φ     = edges[0],edges[1],edges[2]        
        # x1 = (x1[1:] + x1[:-1])/2.
        # x2 = (x2[1:] + x2[:-1])/2.
        # φ  = ( φ[1:] +  φ[:-1])/2.
        # p_ij       = np.trapz(p_ijk, x=φ, axis = 2);

        # plt.pcolor(x1,x2,p_ij.T)
        # plt.xlabel(r'b')
        # plt.ylabel(r'z')
        # plt.show()
    
        #Check expectationsNone)
        p_i     = np.trapz(p_ij,x = x2, axis = 1)
        Sum_pij = np.trapz(p_i ,x = x1, axis = 0)
        print('Sum_pij',Sum_pij,'\n')

        f_X  = p_ij[:,:]; f_X[f_X < 1e-12] = 0.; # Set small values to zero to force a np.nan
    
        # Smaller Grid
        E_BZ = np.zeros(p_ij.shape)
        for k in range( p_ijk.shape[2] ):

            f_XΦ= p_ijk[:,:,k]; f_XΦ[f_XΦ < 1e-12] = 0.;              
            E_BZ += (f_XΦ/f_X)*φ[k] # = int_φ f_Φ|X(φ|x_1,x_2)*φ dφ
        
        E_BZ = np.nan_to_num(E_BZ,0)
        # plt.xlabel(r'b')
        # plt.ylabel(r'z')
        # plt.pcolor(x1,x2,E_BZ.T)
        # plt.show()

        #E{Φ|B,Z}f_B|z |z={0,1} = E{Φ|B,z=1}f_B|z=1 - E{Φ|B,z=0}f_B|z=0
        self.Sbfb = (E_BZ[:,-1]*f_X[:,-1] - E_BZ[:,0]*f_X[:,0]) # = S*(b)f*B(b) = S(b)f_B(b)

        print('self.Sbfb',self.Sbfb.shape)
        print('self.b',self.b.shape)
        # ~~~~~~~~~~~~~~~~~~~~~ # ~~~~~~~~~~~~~~~~~~~~~ # ~~~~~~~~~~~~~~~~~~~~~
        
        fig = plt.figure()
        plt.plot(x1,self.Sbfb,'k')
        plt.xlabel(r'$b$'         ,fontsize=26)
        plt.ylabel(r'$S(b)f_B(b)$',fontsize=26)
       
        # h = self.b[1] - self.b_boundary[0]
        # dS = self.Sbfb[2:] - self.Sbfb[:-2]/(2.*h)
        # plt.plot(self.b[1:-1],-dS,'r:',label=r'$-\partial_b(S(b) f_B(b))$')
        # plt.legend(loc=1,fontsize=16)    
        
        plt.xlim([min(x1),max(x1)])
        plt.tight_layout()
        if figname != None:
            fig.figure.savefig(figname, dpi=200)
        plt.show()
        plt.close()

        return None;

    # ----- section f_WZ

    def Plot_fWZ(self,interval = None, figname=None,dpi=200,Nlevels=100,norm_2d='log'):

        """
        Return a base plot of the PDFs ontop of which we can overlay the expectations
        """

        fig, axs = plt.subplots(2,1,figsize=(12,8),height_ratios=[2,3],sharex=True,constrained_layout=True)
        
        if interval != None:
            
            idx1 = np.where(self.b < interval['b'][1]);
            idx2 = np.where(self.b > interval['b'][0]);
            b_idx = np.intersect1d(idx1,idx2)

            idx1 = np.where(self.w < interval['w'][1]);
            idx2 = np.where(self.w > interval['w'][0]); 
            w_idx = np.intersect1d(idx1,idx2)

            idx1 = np.where(self.z < interval['z'][1]); 
            idx2 = np.where(self.z > interval['z'][0]); 
            z_idx = np.intersect1d(idx1,idx2)

        else:
            b_idx = np.arange(0,len(self.b),1)
            w_idx = np.arange(0,len(self.w),1)
            z_idx = np.arange(0,len(self.z),1)

        # 1D PDFs ---------------
        axs[0].plot(self.w[w_idx],self.fW[w_idx],'r')
        axs[0].set_ylim([0.,1.01*max(self.fW[w_idx])])
        axs[0].fill_between(x=self.w[w_idx],y1=self.fW[w_idx],color= "r",alpha= 0.2)
        axs[0].set_ylabel(r'$f_W(w)$',color='r')
        axs[0].tick_params(axis="y",labelcolor="r")
        #axs[0].set_xticks([])

        # 2D PDFs ---------------
        #axs[1].pcolormesh(self.w[w_idx], self.z[z_idx], self.fWZ[w_idx,:][:,z_idx].T,cmap='Reds' ,norm='linear')
        axs[1].contourf(self.w[w_idx], self.z[z_idx], self.fWZ[w_idx,:][:,z_idx].T,cmap='Reds',levels=10*Nlevels,norm=norm_2d)
        axs[1].set_xlabel(r"$w$")
        axs[1].set_ylabel(r"$z$")

        if figname != None:
            fig.savefig(figname, dpi=dpi)
        #plt.show()
        #plt.close(fig)

        return fig,axs;

    def Plot_EWZ(self,term,Ra,interval=None,figname=None,dpi=200,Nlevels=15,norm='log',sigma_smooth=2):

        """
        Using the plots of the pdfs overlay the expectations in terms:
        - contours for surface plots
        - hatched lines for graph plots
        """
        
        if interval != None:
            
            idx1 = np.where(self.w < interval['w'][1]);
            idx2 = np.where(self.w > interval['w'][0]); 
            w_idx = np.intersect1d(idx1,idx2)

            idx1 = np.where(self.z < interval['z'][1]); 
            idx2 = np.where(self.z > interval['z'][0]); 
            z_idx = np.intersect1d(idx1,idx2)

        else:
            w_idx = np.arange(0,len(self.w),1)
            z_idx = np.arange(0,len(self.z),1)


        fig,axs = self.Plot_fWZ(interval=interval,figname=figname,dpi=dpi);

        #'''
        # 1D Expectationss ---------------
        twin_00 = axs[0].twinx()
        if   term == '\| dW \|^2':
            
            Re   = np.sqrt(Ra)
            E_1D = self.Expectations[term]['1D']['w']/Re
            twin_00.plot(self.w[w_idx],E_1D[w_idx], 'b-')
            twin_00.set_ylabel(r'$E\{ |\nabla W|^2 | w \}/Re$',color="b");

            E_2D = self.Expectations[term]['2D']['wz']

        elif term == '\partial_z P':
            
            E_1D = -1.*self.Expectations[term]['1D']['w']
            twin_00.set_ylabel(r'$-E\{ \partial_z P | w \}$',color="b");
            twin_00.plot(self.w[w_idx],E_1D[w_idx], 'b-')

            E_2D = -1.*self.Expectations[term]['2D']['wz']

        elif term == 'B':
            
            # E{B|w} = int_b f_B|WZ(b|w)*b db
            E_1D = np.trapz(self.fWB*self.b,x=self.b,axis=1)/self.fW
            twin_00.set_ylabel(r'$E\{ B | w \}$',color="b");
            twin_00.plot(self.w[w_idx],E_1D[w_idx], 'b-')

            # E{B|w,z} = int_b f_B|WZ(b|w,z)*b db
            W,B,Z, dPGrad,B2Grad,WBGrad,W2Grad = self.data
            Nz   = len(Z)
            E_WZ = np.zeros((self.N,Nz)) # f_WZΦ
            for j in range(Nz):
                
                # HIST
                f_XΦ,w,φ = np.histogram2d(x=W[:,j].flatten(),y=B[:,j].flatten(),bins=self.N,density=True)
                φ = .5*(φ[1:]+φ[:-1]); dφ = φ[1] - φ[0];
                w = .5*(w[1:]+w[:-1]); dw = w[1] - w[0];
                f_X      =  np.sum(  f_XΦ,axis=1)*dφ
                E        = (np.sum(φ*f_XΦ,axis=1)*dφ)/f_X;
                E_WZ[:,j]= interp1d(w,E,bounds_error=False)(self.w)

            # Interpolate it in z
            E_2D = np.zeros((self.N,self.N)) # E[Φ|W=w,Z=z]
            for i in range(self.N):
                E_2D[i,:] = interp1d(Z,E_WZ[i,:])(self.z)

        elif term == 'both':

            # E{B|w} = int_b f_B|WZ(b|w)*b db
            E_1D = np.trapz(self.fWB*self.b,x=self.b,axis=1)/self.fW
            
            # E = E{B|w} - E{P_z|w}
            E_1D -= self.Expectations['\partial_z P']['1D']['w']

            E_1D = gaussian_filter(E_1D, sigma=sigma_smooth,truncate=3.0)

            twin_00.set_ylabel(r'$E\{ B - \partial_z P | w \} $',color="b");
            twin_00.plot(self.w[w_idx],E_1D[w_idx], 'b-')


             # E{B|w,z} = int_b f_B|WZ(b|w,z)*b db
            W,B,Z, dPGrad,B2Grad,WBGrad,W2Grad = self.data
            Nz   = len(Z)
            E_WZ = np.zeros((self.N,Nz)) # f_WZΦ
            for j in range(Nz):
                
                # HIST
                f_XΦ,w,φ = np.histogram2d(x=W[:,j].flatten(),y=B[:,j].flatten(),bins=self.N,density=True)
                φ = .5*(φ[1:]+φ[:-1]); dφ = φ[1] - φ[0];
                w = .5*(w[1:]+w[:-1]); dw = w[1] - w[0];
                f_X      =  np.sum(  f_XΦ,axis=1)*dφ
                E        = (np.sum(φ*f_XΦ,axis=1)*dφ)/f_X;
                E_WZ[:,j]= interp1d(w,E,bounds_error=False)(self.w)

            # Interpolate it in z
            E_2D = np.zeros((self.N,self.N)) # E[Φ|W=w,Z=z]
            for i in range(self.N):
                E_2D[i,:] = interp1d(Z,E_WZ[i,:])(self.z)            

            E_2D -= self.Expectations['\partial_z P']['2D']['wz']
            
        twin_00.tick_params(axis="y",labelcolor="b")

        # 2D Expectations ---------------
        Z = gaussian_filter(E_2D[w_idx,:][:,z_idx], sigma=sigma_smooth,truncate=3.0)
        try:
            Level = np.linspace(Z.flatten().min(),Z.flatten().max(),Nlevels)
            CS_10  = axs[1].contour(self.w[w_idx], self.z[z_idx], Z.T, levels =Level,norm=norm,cmap='Blues')
            axs[1].clabel(CS_10, inline=False, fontsize=1) 
        except:
            CS_10  = axs[1].contour(self.w[w_idx], self.z[z_idx], Z.T, levels =Nlevels,norm=norm,cmap='Blues')
            axs[1].clabel(CS_10, inline=False, fontsize=1) 

        #'''
        if figname != None:
            fig.savefig(figname, dpi=dpi)
        plt.show()
        plt.close(fig)

        return None;

    # ----- section f_WB

    def Plot_fWB(self,interval = None, figname=None,dpi=200,Nlevels=100,norm_2d='log'):

        """
        Return a base plot of the PDFs ontop of which we can overlay the expectations
        """
        from matplotlib.ticker import NullFormatter, MaxNLocator
        from numpy import linspace
        plt.ion()

        if interval != None:
            
            idx1 = np.where(self.b < interval['b'][1]);
            idx2 = np.where(self.b > interval['b'][0]);
            b_idx = np.intersect1d(idx1,idx2)

            idx1 = np.where(self.w < interval['w'][1]);
            idx2 = np.where(self.w > interval['w'][0]); 
            w_idx = np.intersect1d(idx1,idx2)

        else:
            b_idx = np.arange(0,len(self.b),1)
            w_idx = np.arange(0,len(self.w),1)

        # Coords
        x    = self.w[w_idx] 
        y    = self.b[b_idx]

        # Data
        f_x  = self.fW[w_idx]
        f_y  = self.fB[b_idx]
        f_xy = self.fWB[w_idx,:][:,b_idx]

        # Set up your x and y labels
        xlabel = r'$w$'
        ylabel = r'$b$'

        fxlabel = r'$f_W(w)$'
        fylabel = r'$f_B(b)$'

        # Set up default x and y limits
        xlims = [min(x),max(x)]
        ylims = [min(y),max(y)]
        
        # Define the locations for the axes
        left, width = 0.12, 0.55
        bottom, height = 0.12, 0.55
        bottom_h = left_h = left+width+0.02
        
        # Set up the geometry of the three plots
        rect_temperature = [left, bottom, width, height] # dimensions of temp plot
        rect_histx = [left, bottom_h, width, 0.25] # dimensions of x-histogram
        rect_histy = [left_h, bottom, 0.25, height] # dimensions of y-histogram
        
        # Set up the size of the figure
        fig = plt.figure(1, figsize=(12,9))
        
        # Make the three plots
        axTemperature = plt.axes(rect_temperature) # temperature plot
        axHistx       = plt.axes(rect_histx) # x histogram
        axHisty       = plt.axes(rect_histy) # y histogram

        # Remove the inner axes numbers of the histograms
        nullfmt = NullFormatter()
        axHistx.xaxis.set_major_formatter(nullfmt)
        axHisty.yaxis.set_major_formatter(nullfmt)
        
        # Find the min/max of the data
        xmin = min(xlims)
        xmax = max(xlims)
        ymin = min(ylims)
        ymax = max(y)

        # Make the 'main' 2D plot
        axTemperature.contourf(x,y,f_xy.T,cmap='Reds',levels=Nlevels,norm=norm_2d)
        axTemperature.set_xlabel(xlabel,fontsize=25)
        axTemperature.set_ylabel(ylabel,fontsize=25)

        #Make the tickmarks pretty
        ticklabels = axTemperature.get_xticklabels()
        for label in ticklabels:
            label.set_fontsize(20)
            label.set_family('serif')
        
        ticklabels = axTemperature.get_yticklabels()
        for label in ticklabels:
            label.set_fontsize(20)
            label.set_family('serif')
        
        #Set up the plot limits
        axTemperature.set_xlim(xlims)
        axTemperature.set_ylim(ylims)

        # Make the 1D plots
        axHistx.plot(x,f_x,'r')
        axHistx.set_ylim([0.,1.01*max(f_x)])
        axHistx.fill_between(x=x,y1=f_x,color= "r",alpha= 0.2)
        
        axHisty.plot(f_y,y,'r')
        axHisty.set_xlim([0.,1.01*max(f_y)])
        axHisty.fill_between(x=f_y,y1=y,color= "r",alpha= 0.2)
        
        axHistx.set_ylabel(fxlabel,fontsize=25)
        axHisty.set_xlabel(fylabel,fontsize=25)

        #Set up the histogram limits
        axHistx.set_xlim( min(x), max(x) )
        axHisty.set_ylim( min(y), max(y) )
        
        #Make the tickmarks pretty
        ticklabels = axHistx.get_yticklabels()
        for label in ticklabels:
            label.set_fontsize(20)
            label.set_family('serif')
        
        #Make the tickmarks pretty
        ticklabels = axHisty.get_xticklabels()
        for label in ticklabels:
            label.set_fontsize(20)
            label.set_family('serif')
        
        #Cool trick that changes the number of tickmarks for the histogram axes
        axHisty.xaxis.set_major_locator(MaxNLocator(4))
        axHisty.yaxis.set_major_locator(MaxNLocator(6))
        axTemperature.yaxis.set_major_locator(MaxNLocator(6))


        axHistx.yaxis.set_major_locator(MaxNLocator(4))
        axHistx.xaxis.set_major_locator(MaxNLocator(6))
        axTemperature.xaxis.set_major_locator(MaxNLocator(6))
        
        #Show the plot
        plt.draw()
        
        # Save to a File
        #filename = 'myplot'
        #plt.savefig(figname, dpi=dpi,format = 'png', transparent=True)

        # if figname != None:
        #     fig.savefig(figname, dpi=dpi)
        #plt.show()
        #plt.close(fig)

        return fig,[axTemperature, axHistx, axHisty];

    def Plot_EWB(self,term,Ra,interval=None,figname=None,dpi=200,Nlevels=15,sigma_smooth=2):

        """
        Using the plots of the pdfs overlay the expectations in terms:
        - contours for surface plots
        - hatched lines for graph plots
        """
        
        #'''
        from matplotlib.ticker import NullFormatter, MaxNLocator
        if interval != None:
            
            idx1 = np.where(self.b < interval['b'][1]);
            idx2 = np.where(self.b > interval['b'][0]);
            b_idx = np.intersect1d(idx1,idx2)

            idx1 = np.where(self.w < interval['w'][1]);
            idx2 = np.where(self.w > interval['w'][0]); 
            w_idx = np.intersect1d(idx1,idx2)
            
        else:
            b_idx = np.arange(0,len(self.b),1)
            w_idx = np.arange(0,len(self.w),1)

        fig,axs = self.Plot_fWB(interval=interval,figname=figname,dpi=dpi);


        # Coords
        x    = self.w[w_idx] 
        y    = self.b[b_idx]

        if term == 'both':

            # E = E{B - dP_z|W=w}
            f_x  = np.trapz(self.fWB*self.b,x=self.b,axis=1)/self.fW
            f_x -= self.Expectations['\partial_z P']['1D']['w']
            f_x  = gaussian_filter(f_x[w_idx], sigma=sigma_smooth,truncate=3.0)

            # E = E{B - dP_z|B=b}
            f_y = self.b - self.Expectations['\partial_z P']['1D']['b']
            f_y = gaussian_filter(f_y[b_idx], sigma=sigma_smooth,truncate=3.0)

            # E{B - dP_z|W=w,B=b} = B - E{dP_z|W=w,B=b}
            B    = np.outer(np.ones(len(self.w)),self.b)
            E_2D = B - self.Expectations['\partial_z P']['2D']['wb']
            f_xy = gaussian_filter( E_2D[w_idx,:][:,b_idx], sigma=sigma_smooth,truncate=3.0)

        else:
            
            # Data
            f_x  = self.Expectations[term]['1D']['w'][w_idx] 
            f_y  = self.Expectations[term]['1D']['b'][b_idx]
            f_xy = gaussian_filter( self.Expectations[term]['2D']['wb'][w_idx,:][:,b_idx], sigma=sigma_smooth,truncate=3.0)
        
            f_x /= (Ra**0.5);
            f_y /= (Ra**0.5);
            f_xy/= (Ra**0.5);

        axTemperature, axHistx, axHisty = axs[0],axs[1],axs[2];

        # Main 2D Plot
        try:
            Level = np.linspace(f_xy.flatten().min(),f_xy.flatten().max(),Nlevels)
            CS_10 = axTemperature.contour(x,y,f_xy.T,levels= Level,norm='linear',cmap='Blues')
            axTemperature.clabel(CS_10, inline=False, fontsize=1)
        except:
            CS_10 = axTemperature.contour(x,y,f_xy.T,levels=Nlevels,norm='linear',cmap='Blues')
            axTemperature.clabel(CS_10, inline=False, fontsize=1)


        # 1D Expectationss ---------------
        if term == 'both':
            
            twin_x = axHistx.twinx()
            twin_x.plot(x,f_x, 'b-.')
            twin_x.set_ylabel(r'$E\{-|w\}$',color="b",fontsize=20);
            twin_x.tick_params(axis="y",labelcolor="b")

            twin_y = axHisty.twiny()
            twin_y.plot(f_y, y, 'b-.')
            twin_y.set_xlabel(r'$E\{-|b\}$',color="b",fontsize=20);
            twin_y.tick_params(axis="x",labelcolor="b")

        else:
            twin_x = axHistx.twinx()
            twin_x.plot(x,f_x, 'b-.')
            twin_x.set_ylabel(r'$E\{-|w\}/Re$',color="b",fontsize=20);
            twin_x.tick_params(axis="y",labelcolor="b")

            twin_y = axHisty.twiny()
            twin_y.plot(f_y, y, 'b-.')
            twin_y.set_xlabel(r'$E\{-|b\}/Re$',color="b",fontsize=20);
            twin_y.tick_params(axis="x",labelcolor="b")


        #Make the tickmarks pretty
        ticklabels = twin_x.get_yticklabels()
        for label in ticklabels:
            label.set_fontsize(20)
            label.set_family('serif')
        
        #Make the tickmarks pretty
        ticklabels = twin_y.get_xticklabels()
        for label in ticklabels:
            label.set_fontsize(20)
            label.set_family('serif')
        
        #Cool trick that changes the number of tickmarks for the histogram axes
        twin_y.xaxis.set_major_locator(MaxNLocator(3))
        twin_x.yaxis.set_major_locator(MaxNLocator(3))

        if figname != None:
            fig.savefig(figname, dpi=dpi)
        #plt.show()
        plt.close(fig)

        return None;

if __name__ == "__main__":

    # %%
    #%matplotlib inline

    pdf = PDF_Master(file_dir='Plumes5e08',N_pts=2**8,frames=800,method='HIST')
    pdf.Scalings('PLUME')
    
    pdf.Generate_PDF()        
    pdf.Energetics()
    pdf.Spectra()
    pdf.Generate_Expectations()
    pdf.Interpolate()

    # %%
    pdf.Plot_Pdfs(interval=pdf.domain,figname='PDF_RBC.png',Nlevels=15,norm_2d='log')

    # %%  
    for key,save in zip(pdf.Expectations.keys(),pdf.Save_Handles):    
        print('key=',key)
        pdf.Plot_Expectation(term=key,interval=pdf.domain,figname=save+'.png',Nlevels=20,norm='log',sigma_smooth=3)


    # %%
    Files         = glob.glob('/data/pmannix/PDF_DNS_Data/Sim**')
    Names_Frames  = {'RBC':200,'STEP':1000,'PLUME':400,'IC_Noise':750,'IC':750,'RBC_Sine':800}
    parent_dir = '/home/pmannix/Dstratify/DNS_RBC/'

    # %%
    # for file,(name,frames) in zip(Files,Names_Frames.items()): 
        
    #     print('Case %s \n'%name)
        
    #     pdf = PDF_Master(file_dir=file,N_pts=2**10,frames=frames,method='HIST')
    #     pdf.Scalings(name)
        
    #     pdf.Generate_PDF()        
    #     pdf.Energetics()
    #     pdf.Generate_Expectations()
    #     pdf.Interpolate()

    #     with open('mypickle'+name+'.pickle', 'wb') as f:
    #         pickle.dump(pdf, f)

    # # %%
    # for (name,frames) in Names_Frames.items(): 

    #     print('Case %s \n'%name)

    #     with open(parent_dir+'mypickle'+name+'.pickle','rb') as f:
    #         pdf = pickle.load(f)

    #     os.mkdir(parent_dir+name+'_Convection') 
    #     os.chdir(parent_dir+name+'_Convection')
        
    #     pdf.Plot_Pdfs(interval=pdf.domain,figname='PDF_'+name+'.png',Nlevels=15,norm_2d='log')
        
    #     for key,save in zip(pdf.Expectations.keys(),pdf.Save_Handles):    
    #         print('key=',key)
    #         pdf.Plot_Expectation(term=key,interval=pdf.domain,figname=save+'.png',Nlevels=20,norm='log',sigma_smooth=3)

    #     os.chdir(parent_dir)


    # # %%
    # #Paper figures section f_BZ
    # #~~~~~~~~~~~~~~ # ~~~~~~~~~~~~~~
    # name = 'f_BZ'
    # print(name)
    # #os.mkdir(parent_dir+name) 
    # os.chdir(parent_dir+name)

    # with open(parent_dir+'mypickleIC_Noise.pickle','rb') as f:
    #     pdf = pickle.load(f)
    # pdf.Decomposition(interval=pdf.domain,figname='Compare_Terms_fB_IC.png',)
    # pdf.Plot_EBZ(interval = pdf.domain, term='\|dB\|^2', figname='IC_E_BZ_and_f_BZ.png',dpi=200,Nlevels=15,sigma_smooth=2)

    # with open(parent_dir+ 'mypickleRBC.pickle','rb') as f:
    #     pdf = pickle.load(f)
    # pdf.Decomposition(interval=pdf.domain,figname='Compare_Terms_fB_RBC.png',)
    # pdf.Plot_EBZ(interval = pdf.domain, term='\|dB\|^2', figname='RBC_E_BZ_and_f_BZ.png',dpi=200,Nlevels=15,sigma_smooth=2)

    # os.chdir(parent_dir)

    # # %%
    # # # Paper figures section f_WZ
    # # # ~~~~~~~~~~~~~~ # ~~~~~~~~~~~~~~
    # name = 'f_WZ'
    # print(name)
    # #os.mkdir(parent_dir+name) 
    # os.chdir(parent_dir+name)

    # with open(parent_dir +'mypickleRBC_Sine.pickle','rb') as f:
    #     pdf = pickle.load(f)
    # Ra = 1e10;
    # pdf.Plot_EWZ(Ra = Ra,interval = pdf.domain, term='B'           , figname='HC_E_B___WZ_and_f_WZ.png',dpi=200,Nlevels=40,sigma_smooth=3,norm='linear')
    # pdf.Plot_EWZ(Ra = Ra,interval = pdf.domain, term='\partial_z P', figname='HC_E_dPZ_WZ_and_f_WZ.png',dpi=200,Nlevels=40,sigma_smooth=3,norm='linear')
    # pdf.Plot_EWZ(Ra = Ra,interval = pdf.domain, term='\| dW \|^2'  , figname='HC_E_dW2_WZ_and_f_WZ.png',dpi=400,Nlevels=40,sigma_smooth=3,norm='linear')
    # pdf.Plot_EWZ(Ra = Ra,interval = pdf.domain, term='both'        , figname='HC_E_BPZ_WZ_and_f_WZ.png',dpi=200,Nlevels=40,sigma_smooth=3,norm='linear')

    # with open(parent_dir +'mypickleIC.pickle','rb') as f:
    #     pdf = pickle.load(f)
    # Ra = 1e11;    
    # pdf.Plot_EWZ(Ra = Ra,interval = pdf.domain, term='B'           , figname='IC_E_B___WZ_and_f_WZ.png',dpi=200,Nlevels=40,sigma_smooth=3,norm='linear')
    # pdf.Plot_EWZ(Ra = Ra,interval = pdf.domain, term='\partial_z P', figname='IC_E_dPZ_WZ_and_f_WZ.png',dpi=200,Nlevels=40,sigma_smooth=3,norm='linear')
    # pdf.Plot_EWZ(Ra = Ra,interval = pdf.domain, term='\| dW \|^2'  , figname='IC_E_dW2_WZ_and_f_WZ.png',dpi=400,Nlevels=40,sigma_smooth=3,norm='linear')
    # pdf.Plot_EWZ(Ra = Ra,interval = pdf.domain, term='both'        , figname='IC_E_BPZ_WZ_and_f_WZ.png',dpi=200,Nlevels=40,sigma_smooth=3,norm='linear')
    # os.chdir(parent_dir)

    # # %%
    # # Paper figures section f_WB
    # # ~~~~~~~~~~~~~~ # ~~~~~~~~~~~~~~
    # name = 'f_WB'
    # print(name)
    # #os.mkdir(parent_dir+name) 
    # os.chdir(parent_dir+name)

    # with open(parent_dir +'mypickleIC_Noise.pickle','rb') as f:
    #     pdf = pickle.load(f)
    # Ra = 1e11;  
    # pdf.Plot_EWB(Ra = Ra,interval = pdf.domain, term='both', figname='IC_E_BdPz_WB_and_f_WB.png',dpi=200,Nlevels=15,sigma_smooth=4)
    # pdf.Plot_EWB(Ra = Ra,interval = pdf.domain, term='\|dB\|^2', figname='IC_E_dB2_WB_and_f_WB.png',dpi=200,Nlevels=15,sigma_smooth=4)
    # pdf.Plot_EWB(Ra = Ra,interval = pdf.domain, term=' dW^T dB', figname='IC_E_dWdB_WB_and_f_WB.png',dpi=200,Nlevels=15,sigma_smooth=4)
    # pdf.Plot_EWB(Ra = Ra,interval = pdf.domain, term='\| dW \|^2', figname='IC_E_dW2_WB_and_f_WB.png',dpi=200,Nlevels=15,sigma_smooth=4)

    # with open(parent_dir +'mypickleRBC.pickle','rb') as f:
    #     pdf = pickle.load(f)
    # Ra = 1e10;
    # pdf.Plot_EWB(Ra = Ra,interval = pdf.domain, term='both', figname='RBC_E_BdPz_WB_and_f_WB.png',dpi=200,Nlevels=15,sigma_smooth=4)
    # pdf.Plot_EWB(Ra = Ra,interval = pdf.domain, term='\|dB\|^2', figname='RBC_E_dB2_WB_and_f_WB.png',dpi=200,Nlevels=15,sigma_smooth=4)
    # pdf.Plot_EWB(Ra = Ra,interval = pdf.domain, term=' dW^T dB', figname='RBC_E_dWdB_WB_and_f_WB.png',dpi=200,Nlevels=15,sigma_smooth=4)
    # pdf.Plot_EWB(Ra = Ra,interval = pdf.domain, term='\| dW \|^2', figname='RBC_E_dW2_WB_and_f_WB.png',dpi=200,Nlevels=15,sigma_smooth=4)

    # os.chdir(parent_dir)


# %%
