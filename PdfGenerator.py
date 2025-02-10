from matplotlib import rc
#rc('text', usetex=True)
rc('font', family='serif')
rc('font', size=16.0)

import matplotlib.pyplot as plt
import h5py, warnings, glob, os, pickle
from   scipy.interpolate import interp1d

import numpy as np
np.seterr(divide='ignore', invalid='ignore')


class PdfGenerator(object):

    """
    Calculates PDFs and expectations from DNS_Data

    - joint PDF(s) f_Y(y)
    - conditional expectations E[ (ð_iY_j)^T(ð_jY_k) |Y=y]

    which [if Y = (B,Z) dim = 2D] or [if Y=(W,B,Z) dim = 3D]
    """

    def __init__(self, file_dir, N_pts=2**8, frames=10):
        """Initialise the class to hold the grid, PDFs and Expectations."""

        self.file = file_dir + '/';
        self.frames = frames
        self.N = N_pts
        self.data = self.load_data()

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

        # Expectations -----------------
        self.Plot_Titles = [r'$E\{∂z P |Y = y\}$',r'E\{ |∂xB|^2 + |∂zB|^2 |Y = y}',r'$E\{ (∂xB)(∂xW ) + (∂z B)(∂z W ) | Y = y \}$',r'$E\{ |∂xW|^2 + |∂zW|^2 |Y = y\}$',r'$E\{B|Y=y\}',r'$E\{W|Y=y\}']
        self.Save_Handles = ['dz Pressure','Grad buoyancy squared','Cross Grad Buoyancy Velocity','Grad velocity squared', 'Buoyancy', 'Velocity']
        self.Expectations = {r'\partial_z P':{},r'\|\nabla B\|^2':{},r'\nabla W^T \nabla B':{},r'\|\nabla W \|^2':{}, r'B':{}, r'W':{}}

        for key,_ in self.Expectations.items():
    
            self.Expectations[key] = {'1D':{},'2D':{}}
            self.Expectations[key]['1D'] = {'w':np.zeros(self.N),'b':np.zeros(self.N),'z':np.zeros(self.N)}
            self.Expectations[key]['2D'] = {'wb':np.zeros((self.N,self.N)),'wz':np.zeros((self.N,self.N)),'bz':np.zeros((self.N,self.N))}

        return None

    def load_data(self):
        """Load the data from the dedalus format."""    
        
        print('------  Loading Data ------- \n ')

        file  = h5py.File(self.file + 'snapshots/snapshots_s1.h5', mode='r')
       
        # Y = [W,B,Z], y = [w,b,z] 
        w_split = []; 
        b_split = []; 
        z_cheb = file['tasks/buoyancy'].dims[2][0][:]
        z_data = np.linspace(z_cheb[0],z_cheb[-1],2**10)
        
        dpGrad_split = [];
        b2Grad_split = []; 
        wbGrad_split = []; 
        w2Grad_split = []; 

        for i in range(1,self.frames+1,1):
        
            # PDF variables ---------------------
            w_cheb = file['tasks/w'       ][-i,:,:]
            b_cheb = file['tasks/buoyancy'][-i,:,:]

            # Expectation variables -------------
            try:
                dP_z = file['tasks/grad_p'][-i,1,:,:]# d/dz
                dB_x = file['tasks/grad_b'][-i,0,:,:]# d/dx
                dB_z = file['tasks/grad_b'][-i,1,:,:]# d/dz
                dW_x = file['tasks/grad_w'][-i,0,:,:]# d/dx
                dW_z = file['tasks/grad_w'][-i,1,:,:]# d/dz
            except:
                dP_z = file['tasks/grad_pz'][-i,:,:]# d/dz
                dB_x = file['tasks/grad_bx'][-i,:,:]# d/dx
                dB_z = file['tasks/grad_bz'][-i,:,:]# d/dz
                dW_x = file['tasks/grad_wx'][-i,:,:]# d/dx
                dW_z = file['tasks/grad_wz'][-i,:,:]# d/dz
                
            R_00 = (dB_x**2   + dB_z**2  )
            R_01 = (dB_x*dW_x + dB_z*dW_z)
            R_11 = (dW_x**2   + dW_z**2  )
            
            w_split.append( self.interp(w_cheb, z_cheb, z_data) )
            b_split.append( self.interp(b_cheb, z_cheb, z_data) )
            dpGrad_split.append( self.interp(dP_z, z_cheb, z_data) )
            b2Grad_split.append( self.interp(R_00, z_cheb, z_data) )
            wbGrad_split.append( self.interp(R_01, z_cheb, z_data) )
            w2Grad_split.append( self.interp(R_11, z_cheb, z_data) )

        w_data = np.concatenate(w_split)
        b_data = np.concatenate(b_split)

        dpGrad_data = np.concatenate(dpGrad_split)
        b2Grad_data = np.concatenate(b2Grad_split)
        wbGrad_data = np.concatenate(wbGrad_split)
        w2Grad_data = np.concatenate(w2Grad_split)
        
        return w_data,b_data,z_data,    dpGrad_data,b2Grad_data,wbGrad_data,w2Grad_data

    def interp(self, y, x, x_new):
        """Interpolate the Chebyshev points onto a uniform grid."""

        y_new = interp1d(x, y, axis=1, fill_value="extrapolate")

        return y_new(x_new)

    def generate_pdf(self):
        """Generate all possible 1D and 2D PDFs."""
        
        print('------  Generating PDFs ------- \n ')

        # Get Data & weights
        W,B,Z, dPGrad,B2Grad,WBGrad,W2Grad = self.data

        # 1D PDFs
        pdfs_1D = ['fW', 'fB']
        grid_1D = ['w',  'b']
        data_1D = [W, B]
        for pdf_name,grid_name,X in zip(pdfs_1D,grid_1D,data_1D):
            
            points, bin_edges = np.histogram(X.flatten(), bins=self.N, density=True)
            grid = 0.5*(bin_edges[1:] + bin_edges[:-1])
            
            setattr(self,pdf_name,points)
            setattr(self,grid_name,grid)

        setattr(self,'fZ',np.ones(len(Z)))
        setattr(self,'z',Z)

        data_range = [[min(self.w), max(self.w)], [min(self.b), max(self.b)]]
        fWB = np.histogram2d(x=W.flatten(), y=B.flatten(), bins=self.N, range=data_range, density=True)[0]
        setattr(self,'fWB',fWB)

        fWZ = np.zeros((len(self.w),len(self.z)));
        fBZ = np.zeros((len(self.b),len(self.z)));
        for j,_ in enumerate(self.z):
            
            # HIST
            fWZ[:,j] = np.histogram(W[:,j], bins=self.N, range=(min(self.w),max(self.w)), density=True)[0]
            fBZ[:,j] = np.histogram(B[:,j], bins=self.N, range=(min(self.b),max(self.b)), density=True)[0]

        setattr(self,'fWZ',fWZ)
        setattr(self,'fBZ',fBZ)

        return None

    def generate_expectation(self, testing=False):
        """
        Compute the 1D & 2D conditional expectations by creating the 2D pdf
        
            E[Φ|B = b] = int φfΦ|B (φ|b)dφ = int φ fBΦ(b,φ)/fB(b) dφ,

        and integrating out the independent variable φ. The same
        approach is used to compute 2D conditional expectations.
        """

        print('------  Generating Conditional Expectations ------- \n ')

        W,B,Z,  dPGrad,B2Grad,WBGrad,W2Grad = self.data
        Term_E = [dPGrad,B2Grad,WBGrad,W2Grad,B,W]
        Nz = len(self.z)

        if not np.any(self.fB):
            self.generate_pdf()

        # x2 1D E{Φ|W=w},E{Φ|B=b}
        data_1D = [W,B]
        for value, Φ in zip(self.Expectations.values(),Term_E):
            
            for key_1D,X_i in zip(value['1D'].keys(),data_1D):
                
                f_XΦ,_,φ = np.histogram2d(X_i.flatten(),Φ.flatten(),bins=self.N,density=True)
                φ = .5*(φ[1:]+φ[:-1]); dφ = φ[1] - φ[0];

                # E{Φ|X} = int_φ f_Φ|X(φ|x)*φ dφ
                if testing:
                    value['1D'][key_1D] = np.nansum(φ*f_XΦ*dφ ,axis=1)
                else:
                    f_X                 = np.nansum(  f_XΦ*dφ ,axis=1)    # f_X(x)
                    value['1D'][key_1D] = np.nansum(φ*f_XΦ*dφ ,axis=1)/f_X
        
        
        # 2D E{Φ|W=w,B=b}
        for value, Φ in zip(self.Expectations.values(),Term_E):
            
            data = [W.flatten(),B.flatten(),Φ.flatten()]
            f_XΦ,Edges = np.histogramdd(data,bins=self.N,density=True) # f_XΦ(x_1,x_2,φ)
            φ = .5*(Edges[2][1:]+ Edges[2][:-1]); dφ = φ[1] - φ[0];
            
            if testing:
                value['2D']['wb'] = np.nansum(φ*f_XΦ*dφ,axis=2)
            else:
                f_X  =  np.nansum(  f_XΦ*dφ,axis=2)    # f_X(x)
                value['2D']['wb'] = np.nansum(φ*f_XΦ*dφ,axis=2)/f_X


        # 2D E{Φ|B=b, Z=z}
        for value, Φ in zip(self.Expectations.values(),Term_E):
            
            data_range = [[min(self.b), max(self.b)], [min(Φ.flatten()), max(Φ.flatten())]]
            E_X = np.zeros((self.N,Nz))
            for j in range(Nz):
                
                # HIST
                f_XΦ,_,φ = np.histogram2d(x=B[:,j].flatten(),y=Φ[:,j].flatten(),bins=self.N,density=True, range=data_range)
                φ = .5*(φ[1:]+φ[:-1]); dφ = φ[1] - φ[0];
                if testing:
                    E_X[:,j] = np.nansum(φ*f_XΦ*dφ,axis=1)
                else:
                    f_X      = np.nansum(  f_XΦ*dφ,axis=1)
                    E_X[:,j] = np.nansum(φ*f_XΦ*dφ,axis=1)/f_X;
                
            value['2D']['bz'] = E_X


        # 2D E{Φ|W=w, Z=z}
        for value, Φ in zip(self.Expectations.values(),Term_E):
            
            data_range = [[min(self.w), max(self.w)], [min(Φ.flatten()), max(Φ.flatten())]]
            E_X     = np.zeros((self.N,Nz)) # f_WZΦ
            for j in range(Nz):
                
                f_XΦ,_,φ = np.histogram2d(x=W[:,j].flatten(),y=Φ[:,j].flatten(),bins=self.N,density=True, range=data_range)
                φ = .5*(φ[1:]+φ[:-1]); dφ = φ[1] - φ[0];
                if testing:
                    E_X[:,j] = np.nansum(φ*f_XΦ*dφ,axis=1)
                else:
                    f_X      = np.nansum(  f_XΦ*dφ,axis=1)
                    E_X[:,j] = np.nansum(φ*f_XΦ*dφ,axis=1)/f_X;

            value['2D']['wz'] = E_X

        return None
    
    def spectra(self):
        """Plot the time-averaged spectra of the Kinetic energy and buoyancy variance."""


        f  = h5py.File(self.file + 'scalar_data/scalar_data_s1.h5', mode='r')
        
        fig, (ax1,ax2) = plt.subplots(ncols=2,figsize=(8,4),constrained_layout=True)
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
        # plt.show()
        plt.close(fig)


        # Shape time,x,z
        Eu     = f['tasks/Eu(t)'][:,0,0]
        Eb     = f['tasks/Eb(t)'][:,0,0]
        wB_avg = f['tasks/<wB>'][:,0,0]
        B_avg  = f['tasks/<B>' ][:,0,0]
        t      = f['scales/sim_time'][()]
        
        fig, axs = plt.subplots(nrows=2,ncols=2,figsize=(8,4),constrained_layout=True)
        axs[0, 0].plot(t,Eu,'b-',label=r'$E_u$')
        axs[0, 0].set_title(r'$E_u$')
        axs[0, 1].plot(t,Eb,'r-',label=r'$E_b$')
        axs[0, 1].set_title(r'$E_b$')
        axs[1, 0].plot(t,wB_avg,'b:',label=r'$\langle wB \rangle$')
        axs[1, 0].set_title(r'$\langle wB \rangle$')
        axs[1, 1].plot(t, B_avg,'r:',label=r'$\langle B  \rangle$')
        axs[1, 1].set_title(r'$\langle B  \rangle$')
        fig.savefig('EnergyTimeSeries.png',dpi=200)
        # plt.show()
        plt.close(fig)

        return None;

    def energetics(self,name="0"):
        """Compute all time and space averaged diagnostic quantites including the available, total and 
        reference potential energies by using the reference height/CDF z*(b) = int_b fB(d) db = F_B(b)"""
       
        # Check the pdfs if not done already
        if not np.any(self.fB):
            self.generate_pdf()

        print('------  Computing Energetics ------- \n ')

        f     = h5py.File(self.file + 'scalar_data/scalar_data_s1.h5', mode='r')    
        file  = h5py.File(self.file + 'snapshots/snapshots_s1.h5'    , mode='r')
        times = file['tasks/buoyancy/'].dims[0]['sim_time'][:]
        x = file['tasks/buoyancy'].dims[1][0][:]
        z = file['tasks/buoyancy'].dims[2][0][:]
        Lx = np.round(x[-1] - x[0])
        Lz = np.round(z[-1] - z[0])
        V = Lx*Lz  # Domain size
        dt = f['scales/timestep'][0]


        indx  = np.where(f['scales/sim_time'][()] > times[-self.frames])
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
        dz = abs(self.z[1] - self.z[0]) 
        db = abs(self.b[1] - self.b[0]) 
        TPE_pdf = -np.nansum(Ibz*self.fBZ * db * dz)

        error = abs(TPE_pdf - TPE)/abs(TPE) 
        if error > 1e-02:
            warnings.warn(f"TPE resdiual error must be less than 1% but got : {error} \n")

        z_b = np.cumsum(self.fB)*db
        RPE = -np.sum(z_b*self.b*self.fB)*db
        APE = (TPE - RPE)

        self.stats = {'TPE':TPE,'APE':APE,'RPE':RPE,
                      '(1/2)<|U|^2>':.5*KE,'<|∇U|^2>/Re':Disp_U,
                      '(1/2)<|B|^2>':.5*BE,'<|∇B|^2>':   Disp_B,
                      '<WB>':       wb_avg,'<B>':        B_avg }


        with open("Diagnostics_" + name + ".txt", "w") as text_file:
            
            indx  = np.where(times > times[-self.frames])
            print('Nx,Nz,∆t = %d,%d,%e'%(len(x),len(z),dt),file=text_file); 
            print('Available frames = ',len(times),'Used frames = ',len(times[indx]),'∆T frames = ',abs(times[1]-times[0]),'\n',file=text_file)
            print('  APE     &      RPE    &      E_k   &      <WB>   &       <e_U>/Re &   (1/2)<|B|^2> &      <e_B> &    <B> ',file=text_file)
            print('%1.3e &  %1.3e &  %1.3e &   %1.3e &      %1.3e &      %1.3e &  %1.3e &  %1.3e '%(APE,RPE,.5*KE,wb_avg,Disp_U,.5*BE,Disp_B,B_avg),file=text_file)
            print('~~~~~~~~~~~~~~~~~~~~~ \n',file=text_file)

        return TPE, APE    


if __name__ == "__main__":

    # Generate the pdf objects for all plots
    
    files = glob.glob('/data/pmannix/PDF_DNS_Data/' + '/Sim**')
    
    names_frames  = {'ICR':5000, 'IC':5000, 'RBC':1000, 'PLUME':1700, 'SINE':1200, 'STEP':2000}
    
    for file,(name,frames) in zip(files,names_frames.items()):
        
        # check names match
        if name == file.split('/')[-1].split('_')[1]:

            os.chdir(file)
            print('## Simulation case: ',name,'## \n')

            pdf = PdfGenerator(file_dir=file,N_pts=2**8,frames=frames)       

            pdf.generate_pdf()
            pdf.energetics(name)
            pdf.spectra()
            pdf.generate_expectation()

            # Remove loaded data snapshots before saving
            delattr(pdf, "data")

            with open(name + '_pickled.pickle', 'wb') as f:
                pickle.dump(pdf, f)


    # To copy all the files generated to your data folder use
    # $ cd /data/pmannix/PDF_DNS_Data
    # $ cp -vr **/*.pickle /home/pmannix/Stratification-DNS/data/
    # $ cp -vr **/*.txt /home/pmannix/Stratification-DNS/data/