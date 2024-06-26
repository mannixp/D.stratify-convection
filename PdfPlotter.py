from matplotlib import rc
rc('text', usetex=True)
rc('font', family='serif')
rc('font', size=16.0)

import matplotlib.pyplot as plt
import glob, os, pickle
import numpy as np
from scipy.ndimage import gaussian_filter1d
from PdfGenerator import PdfGenerator

class PdfPlotter(object):
    """A base class plotting PDFs and expectations."""
    
    def __init__(self, pdf, interval=None):
        """
        Initialise the Density object.
        
        Parameters
        ----------
        ptp : class 'PdfGenerator'
            PdfGenerator object corresponding to the simulation case.
        interval : dictionary
            Domain of each random variables pdf {'w':(-1,1),'b':(0,q),'z':(0,1)}.
        """
        self.pdf      = pdf
        self.interval = interval

    def calculate_interval(self):
        """Calculate the indices in order to select the interval."""

        if self.interval != None:
            
            idx1 = np.where(self.pdf.b < self.interval['b'][1])
            idx2 = np.where(self.pdf.b > self.interval['b'][0])
            b_idx = np.intersect1d(idx1,idx2)

            idx1 = np.where(self.pdf.w < self.interval['w'][1])
            idx2 = np.where(self.pdf.w > self.interval['w'][0])
            w_idx = np.intersect1d(idx1,idx2)

            idx1 = np.where(self.pdf.z < self.interval['z'][1])
            idx2 = np.where(self.pdf.z > self.interval['z'][0])
            z_idx = np.intersect1d(idx1,idx2)

        else:
            b_idx = np.arange(0,len(self.pdf.b),1)
            w_idx = np.arange(0,len(self.pdf.w),1)
            z_idx = np.arange(0,len(self.pdf.z),1)

        return b_idx, w_idx, z_idx
    
    def plot_pdf(self,figname=None,Nlevels=15,norm='log'):
        """Return a base plot of the PDFs ontop of which we can overlay the expectations"""

        b_idx, w_idx, z_idx = self.calculate_interval()
        fig, axs = plt.subplots(3, 2,figsize=(8,6),constrained_layout=True)
        
        # f_B ---------------
        axs[0, 0].plot(self.pdf.b[b_idx],self.pdf.fB[b_idx],'r')
        axs[0, 0].set_ylim([0.,1.01*max(self.pdf.fB[b_idx])])
        axs[0, 0].fill_between(x=self.pdf.b[b_idx],y1=self.pdf.fB[b_idx],color= "r",alpha= 0.2)
        axs[0, 0].tick_params('x', labelbottom=False)
        name_00 = {'x':r'$B$','y':r'$f_B(b)$'}
      
        axs[1, 0].sharex(axs[0, 0])
        axs[1, 0].contourf(self.pdf.b[b_idx], self.pdf.z[z_idx], self.pdf.fBZ[b_idx,:][:,z_idx].T,cmap='Reds',levels=Nlevels ,norm=norm)
        axs[1, 0].tick_params('x', labelbottom=False)
        name_10 = {'x':r'$B$','y':r'$Z$'}

        axs[2, 0].sharex(axs[0, 0])
        axs[2, 0].contourf(self.pdf.b[b_idx], self.pdf.w[w_idx], self.pdf.fWB[w_idx,:][:,b_idx]  ,cmap='Reds',levels=Nlevels,norm=norm)
        name_20 = {'x':r'$B$','y':r'$W$'}
        axs[2, 0].set_xlabel(r'$B$')

        # f_W -------------
        axs[0,1].plot(self.pdf.w[w_idx],self.pdf.fW[w_idx],'r')
        axs[0,1].set_ylim([0.,1.01*max(self.pdf.fW[w_idx])])
        axs[0,1].fill_between(x=self.pdf.w[w_idx],y1=self.pdf.fW[w_idx],color= "r",alpha= 0.2)
        axs[0,1].tick_params('x', labelbottom=False)
        name_01 = {'x':r'$W$','y':r'$f_W(w)$'}
      
        axs[1, 1].sharex(axs[0, 1])
        axs[1, 1].contourf(self.pdf.w[w_idx], self.pdf.z[z_idx], self.pdf.fWZ[w_idx,:][:,z_idx].T,cmap='Reds',levels=10*Nlevels,norm=norm)
        axs[1, 1].tick_params('x', labelbottom=False)
        name_11 = {'x':r'$W$','y':r'$Z$'}

        axs[2, 1].sharex(axs[0, 1])
        axs[2, 1].contourf(self.pdf.w[w_idx], self.pdf.b[b_idx], self.pdf.fWB[w_idx,:][:,b_idx].T,cmap='Reds',levels=Nlevels,norm=norm)
        name_21 = {'x':r'$W$','y':r'$B$'}
        axs[2, 1].set_xlabel(r'$W$')

        Names = np.asarray( [[name_00,name_01], [name_10,name_11], [name_20,name_21]])
        for ax,name in zip(axs.flat,Names.flat):
            ax.set(ylabel=name['y'])

        if figname != None:
            fig.savefig(figname, dpi=200)
        #plt.show()
        #plt.close(fig)

        return fig, axs

    def plot_expectation(self,term,figname=None,Nlevels=15,norm='log',sigma_smooth=1.):

        """
        Using the plots of the pdfs overlay the expectations in terms:
        - contours for surface plots
        - hatched lines for graph plots
        """

        b_idx, w_idx, z_idx = self.calculate_interval()
        fig,axs = self.plot_pdf(figname=figname)

        E_1D = self.pdf.Expectations[term]['1D']
        E_2D = self.pdf.Expectations[term]['2D']

        # B ---------------------
        twin_00 = axs[0,0].twinx()
        twin_00.plot(self.pdf.b[b_idx],E_1D['b'][b_idx], 'b-',label=str(r'$E['+term+'\|b]$'))
        twin_00.set_ylabel(r'$E['+term+'\|b]$')

        Z = gaussian_filter1d(E_2D['bz'][b_idx,:][:,z_idx], sigma=sigma_smooth,truncate=3.0)
        CS_10  = axs[1, 0].contour(self.pdf.b[b_idx], self.pdf.z[z_idx], Z.T, levels=Nlevels, norm=norm, cmap='Blues')
        axs[1, 0].clabel(CS_10, inline=False, fontsize=1)

        Z = gaussian_filter1d(E_2D['wb'][w_idx,:][:,b_idx], sigma=sigma_smooth,truncate=3.0)
        CS_20  = axs[2, 0].contour(self.pdf.b[b_idx], self.pdf.w[w_idx],  Z, levels=Nlevels, norm=norm, cmap='Blues')
        axs[2, 0].clabel(CS_20, inline=False, fontsize=1) 

        # W ---------------------
        twin_01 = axs[0,1].twinx()
        twin_01.plot(self.pdf.w[w_idx],E_1D['w'][w_idx], 'b-',label=str(r'$E['+term+'\|w]$'))
        twin_01.set_ylabel(r'$E['+term+'\|w]$')

        Z = gaussian_filter1d(E_2D['wz'][w_idx,:][:,z_idx], sigma=sigma_smooth,truncate=3.0)
        CS_11  = axs[1, 1].contour(self.pdf.w[w_idx], self.pdf.z[z_idx], Z.T, levels = Nlevels,norm=norm,cmap='Blues')
        axs[1, 1].clabel(CS_11, inline=False, fontsize=1)     

        Z = gaussian_filter1d(E_2D['wb'][w_idx,:][:,b_idx], sigma=sigma_smooth,truncate=3.0)
        CS_21  = axs[2, 1].contour(self.pdf.w[w_idx], self.pdf.b[b_idx], Z.T, levels = Nlevels,norm=norm,cmap='Blues')
        axs[2, 1].clabel(CS_21, inline=False, fontsize=1) 

        if figname != None:
            fig.savefig(figname, dpi=200)
        #plt.show()
        plt.close()

        return None

    # ----- section f_BZ

    # def plot_fBZ(self,interval = None, figname=None,Nlevels=15,norm_2d='log'):

    #     """
    #     Return a base plot of the PDFs ontop of which we can overlay the expectations
    #     """

    #     fig, axs = plt.subplots(2,1,figsize=(12,8),height_ratios=[2,3],sharex=True,constrained_layout=True)
        
    #     b_idx, w_idx, z_idx = self.calculate_interval(interval)

    #     # 1D PDFs ---------------
    #     axs[0].plot(self.b[b_idx],self.fB[b_idx],'r')
    #     axs[0].set_ylim([0.,1.01*max(self.fB[b_idx])])
    #     axs[0].fill_between(x=self.b[b_idx],y1=self.fB[b_idx],color= "r",alpha= 0.2)
    #     axs[0].set_ylabel(r'$f_B(b)$',color='r')
    #     axs[0].tick_params(axis="y",labelcolor="r")

    #     # 2D PDFs ---------------
    #     #axs[1].pcolormesh(self.b[b_idx], self.z[z_idx], self.fBZ[b_idx,:][:,z_idx].T,cmap='Reds' ,norm='linear')
    #     axs[1].contourf(self.b[b_idx], self.z[z_idx], self.fBZ[b_idx,:][:,z_idx].T,cmap='Reds',levels=Nlevels ,norm=norm_2d)
    #     axs[1].set_xlabel(r"$b$")
    #     axs[1].set_ylabel(r"$z$")

    #     if figname != None:
    #         fig.savefig(figname, dpi=200)
    #     #plt.show()
    #     plt.close(fig)

    #     return fig,axs;

    # def Plot_EBZ(self,term,interval=None,figname=None,Nlevels=15,norm='log',sigma_smooth=2):

    #     """
    #     Using the plots of the pdfs overlay the expectations in terms:
    #     - contours for surface plots
    #     - hatched lines for graph plots
    #     """
        
    #     #'''

    #     if interval != None:
            
    #         idx1 = np.where(self.b < interval['b'][1]);
    #         idx2 = np.where(self.b > interval['b'][0]);
    #         b_idx = np.intersect1d(idx1,idx2)

    #         idx1 = np.where(self.w < interval['w'][1]);
    #         idx2 = np.where(self.w > interval['w'][0]); 
    #         w_idx = np.intersect1d(idx1,idx2)

    #         idx1 = np.where(self.z < interval['z'][1]); 
    #         idx2 = np.where(self.z > interval['z'][0]); 
    #         z_idx = np.intersect1d(idx1,idx2)

    #     else:
    #         b_idx = np.arange(0,len(self.b),1)
    #         w_idx = np.arange(0,len(self.w),1)
    #         z_idx = np.arange(0,len(self.z),1)


    #     fig,axs = self.Plot_fBZ(interval=interval,figname=figname,dpi=200);

    #     # 1D Expectationss ---------------
    #     E_1D = self.Expectations[term]['1D']

    #     twin_00 = axs[0].twinx()
    #     twin_00.plot(self.b[b_idx],E_1D['b'][b_idx], 'b')
    #     twin_00.set_ylabel(r'$E\{ |\nabla B|^2 | b \}$',color="b");
    #     twin_00.tick_params(axis="y",labelcolor="b")

    #     # 2D Expectations ---------------
    #     E_2D = self.Expectations[term]['2D']
        
    #     Z = gaussian_filter(E_2D['bz'][b_idx,:][:,z_idx], sigma=sigma_smooth,truncate=3.0)
    #     try:
    #         Level = np.linspace(Z.flatten().min(),Z.flatten().max(),Nlevels)
    #         CS_10  = axs[1].contour(self.b[b_idx], self.z[z_idx],Z.T, levels = Level,norm=norm,cmap='Blues')
    #         axs[1].clabel(CS_10, inline=False, fontsize=1) 
    #     except:
    #         CS_10  = axs[1].contour(self.b[b_idx], self.z[z_idx],Z.T, levels =Nlevels,norm=norm,cmap='Blues')
    #         axs[1].clabel(CS_10, inline=False, fontsize=1) 

    #     if figname != None:
    #         fig.savefig(figname, dpi=200)
    #     #plt.show()
    #     plt.close(fig)

    #     return None;

    # def Decomposition(self,interval=None,figname=None,Nlevels=10):

    #     """
        
    #     """

    #     if interval != None:
            
    #         idx1 = np.where(self.b < interval['b'][1]);
    #         idx2 = np.where(self.b > interval['b'][0]);
    #         b_idx = np.intersect1d(idx1,idx2)

    #         idx1 = np.where(self.w < interval['w'][1]);
    #         idx2 = np.where(self.w > interval['w'][0]); 
    #         w_idx = np.intersect1d(idx1,idx2)

    #         idx1 = np.where(self.z < interval['z'][1]); 
    #         idx2 = np.where(self.z > interval['z'][0]); 
    #         z_idx = np.intersect1d(idx1,idx2)

    #     else:
    #         b_idx = np.arange(0,len(self.b),1)
    #         w_idx = np.arange(0,len(self.w),1)
    #         z_idx = np.arange(0,len(self.z),1)

    #     # Calculate the terms
    #     #Φ = self.Expectations['\|dB\|^2']['1D']['b']
    #     Φ  = gaussian_filter(self.Expectations['\|dB\|^2']['1D']['b'], sigma=3,truncate=3.0)
    #     fB = gaussian_filter(self.fB                                 , sigma=3,truncate=3.0)
    #     h = self.b[b_idx][1] - self.b[b_idx][0]

    #     ddf = (fB[b_idx][2:] - 2.*fB[b_idx][1:-1] + fB[b_idx][:-2])/(h**2);
    #     Φ_norm = np.linalg.norm(Φ[b_idx][1:-1]*ddf,2)

    #     dΦ = ( Φ[b_idx][2:]  -  Φ[b_idx][:-2])/(2.*h)
    #     df = (fB[b_idx][2:]  - fB[b_idx][:-2])/(2.*h)
    #     dΦ_norm = np.linalg.norm(dΦ*df,2)
        
    #     ddΦ = (Φ[b_idx][2:] - 2.*Φ[b_idx][1:-1] + Φ[b_idx][:-2])/(h**2);
    #     ddΦ_norm = np.linalg.norm(ddΦ*fB[b_idx][1:-1],2)


    #     fig,ax=plt.subplots(3,sharex=True)

    #     # Plot PDFs
    #     ax[2].plot(self.b[b_idx],self.fB[b_idx],'r-',linewidth=1)
    #     ax[2].fill_between(x=self.b[b_idx],y1=self.fB[b_idx],color= "r",alpha= 0.2)
        
    #     ax[1].plot(self.b[b_idx],self.fB[b_idx],'r-',linewidth=1)
    #     ax[1].fill_between(x=self.b[b_idx],y1=self.fB[b_idx],color= "r",alpha= 0.2)
        
    #     ax[0].plot(self.b[b_idx],self.fB[b_idx],'r-',linewidth=1)
    #     ax[0].fill_between(x=self.b[b_idx],y1=self.fB[b_idx],color= "r",alpha= 0.2)

    #     ax[2].set_xlabel(r'$b$',fontsize=20)
    #     ax[2].set_ylabel(r'$f_B(b)$',fontsize=20)
    #     ax[1].set_ylabel(r'$f_B(b)$',fontsize=20)
    #     ax[0].set_ylabel(r'$f_B(b)$',fontsize=20)

    #     ax[2].set_xlim(interval['b'])
    #     ax[1].set_xlim(interval['b'])
    #     ax[0].set_xlim(interval['b'])

       
    #     # Plot E[Φ|B=b]
    #     twin2 = ax[2].twinx()
    #     twin2.plot(self.b[b_idx][1:-1],-1.*(Φ[b_idx][1:-1]*ddf)/Φ_norm,'b-',label=r"$-E\{|\nabla B|^2|b\} \frac{\partial^2 f_B}{\partial b^2}$")
    #     twin2.legend(loc=1,fontsize=14)
    #     twin2.set_yticks([])
        
    #     twin1 = ax[1].twinx()
    #     twin1.plot(self.b[b_idx][1:-1],-1.*(dΦ*df)/dΦ_norm,'b-',label=r"$-E\{|\nabla B|^2|b\}' \frac{\partial f_B}{\partial b}$")
    #     twin1.legend(loc=3,fontsize=14)
    #     twin1.set_yticks([])

    #     twin0 = ax[0].twinx()
    #     twin0.plot(self.b[b_idx][1:-1],-1.*(ddΦ*fB[b_idx][1:-1])/ddΦ_norm, 'b-',label=r"$-E\{|\nabla B|^2|b\}'' f_B$")
    #     twin0.legend(loc=3,fontsize=14)
    #     twin0.set_yticks([])

    #     print('|Φ| =',Φ_norm)
    #     print('|dΦ/db| =',dΦ_norm)
    #     print('|d^2Φ/db^2| =',ddΦ_norm)

    #     plt.tight_layout()
    #     if figname != None:
    #         fig.savefig(figname, dpi=200)
    #     #plt.show()
    #     plt.close(fig)

    #     return None;

    #   # Check these
    
    # # ----- section f_WZ

    # def Plot_fWZ(self,interval = None, figname=None,Nlevels=100,norm_2d='log'):

    #     """
    #     Return a base plot of the PDFs ontop of which we can overlay the expectations
    #     """

    #     fig, axs = plt.subplots(2,1,figsize=(12,8),height_ratios=[2,3],sharex=True,constrained_layout=True)
        
    #     if interval != None:
            
    #         idx1 = np.where(self.b < interval['b'][1]);
    #         idx2 = np.where(self.b > interval['b'][0]);
    #         b_idx = np.intersect1d(idx1,idx2)

    #         idx1 = np.where(self.w < interval['w'][1]);
    #         idx2 = np.where(self.w > interval['w'][0]); 
    #         w_idx = np.intersect1d(idx1,idx2)

    #         idx1 = np.where(self.z < interval['z'][1]); 
    #         idx2 = np.where(self.z > interval['z'][0]); 
    #         z_idx = np.intersect1d(idx1,idx2)

    #     else:
    #         b_idx = np.arange(0,len(self.b),1)
    #         w_idx = np.arange(0,len(self.w),1)
    #         z_idx = np.arange(0,len(self.z),1)

    #     # 1D PDFs ---------------
    #     axs[0].plot(self.w[w_idx],self.fW[w_idx],'r')
    #     axs[0].set_ylim([0.,1.01*max(self.fW[w_idx])])
    #     axs[0].fill_between(x=self.w[w_idx],y1=self.fW[w_idx],color= "r",alpha= 0.2)
    #     axs[0].set_ylabel(r'$f_W(w)$',color='r')
    #     axs[0].tick_params(axis="y",labelcolor="r")
    #     #axs[0].set_xticks([])

    #     # 2D PDFs ---------------
    #     #axs[1].pcolormesh(self.w[w_idx], self.z[z_idx], self.fWZ[w_idx,:][:,z_idx].T,cmap='Reds' ,norm='linear')
    #     axs[1].contourf(self.w[w_idx], self.z[z_idx], self.fWZ[w_idx,:][:,z_idx].T,cmap='Reds',levels=10*Nlevels,norm=norm_2d)
    #     axs[1].set_xlabel(r"$w$")
    #     axs[1].set_ylabel(r"$z$")

    #     if figname != None:
    #         fig.savefig(figname, dpi=200)
    #     #plt.show()
    #     #plt.close(fig)

    #     return fig,axs;

    # def Plot_EWZ(self,term,Ra,interval=None,figname=None,Nlevels=15,norm='log',sigma_smooth=2):

    #     """
    #     Using the plots of the pdfs overlay the expectations in terms:
    #     - contours for surface plots
    #     - hatched lines for graph plots
    #     """
        
    #     if interval != None:
            
    #         idx1 = np.where(self.w < interval['w'][1]);
    #         idx2 = np.where(self.w > interval['w'][0]); 
    #         w_idx = np.intersect1d(idx1,idx2)

    #         idx1 = np.where(self.z < interval['z'][1]); 
    #         idx2 = np.where(self.z > interval['z'][0]); 
    #         z_idx = np.intersect1d(idx1,idx2)

    #     else:
    #         w_idx = np.arange(0,len(self.w),1)
    #         z_idx = np.arange(0,len(self.z),1)


    #     fig,axs = self.Plot_fWZ(interval=interval,figname=figname,dpi=200);

    #     #'''
    #     # 1D Expectationss ---------------
    #     twin_00 = axs[0].twinx()
    #     if   term == '\| dW \|^2':
            
    #         Re   = np.sqrt(Ra)
    #         E_1D = self.Expectations[term]['1D']['w']/Re
    #         twin_00.plot(self.w[w_idx],E_1D[w_idx], 'b-')
    #         twin_00.set_ylabel(r'$E\{ |\nabla W|^2 | w \}/Re$',color="b");

    #         E_2D = self.Expectations[term]['2D']['wz']

    #     elif term == '\partial_z P':
            
    #         E_1D = -1.*self.Expectations[term]['1D']['w']
    #         twin_00.set_ylabel(r'$-E\{ \partial_z P | w \}$',color="b");
    #         twin_00.plot(self.w[w_idx],E_1D[w_idx], 'b-')

    #         E_2D = -1.*self.Expectations[term]['2D']['wz']

    #     elif term == 'B':
            
    #         # E{B|w} = int_b f_B|WZ(b|w)*b db
    #         E_1D = np.trapz(self.fWB*self.b,x=self.b,axis=1)/self.fW
    #         twin_00.set_ylabel(r'$E\{ B | w \}$',color="b");
    #         twin_00.plot(self.w[w_idx],E_1D[w_idx], 'b-')

    #         # E{B|w,z} = int_b f_B|WZ(b|w,z)*b db
    #         W,B,Z, dPGrad,B2Grad,WBGrad,W2Grad = self.data
    #         Nz   = len(Z)
    #         E_WZ = np.zeros((self.N,Nz)) # f_WZΦ
    #         for j in range(Nz):
                
    #             # HIST
    #             f_XΦ,w,φ = np.histogram2d(x=W[:,j].flatten(),y=B[:,j].flatten(),bins=self.N,density=True)
    #             φ = .5*(φ[1:]+φ[:-1]); dφ = φ[1] - φ[0];
    #             w = .5*(w[1:]+w[:-1]); dw = w[1] - w[0];
    #             f_X      =  np.sum(  f_XΦ,axis=1)*dφ
    #             E        = (np.sum(φ*f_XΦ,axis=1)*dφ)/f_X;
    #             E_WZ[:,j]= interp1d(w,E,bounds_error=False)(self.w)

    #         # Interpolate it in z
    #         E_2D = np.zeros((self.N,self.N)) # E[Φ|W=w,Z=z]
    #         for i in range(self.N):
    #             E_2D[i,:] = interp1d(Z,E_WZ[i,:])(self.z)

    #     elif term == 'both':

    #         # E{B|w} = int_b f_B|WZ(b|w)*b db
    #         E_1D = np.trapz(self.fWB*self.b,x=self.b,axis=1)/self.fW
            
    #         # E = E{B|w} - E{P_z|w}
    #         E_1D -= self.Expectations['\partial_z P']['1D']['w']

    #         E_1D = gaussian_filter(E_1D, sigma=sigma_smooth,truncate=3.0)

    #         twin_00.set_ylabel(r'$E\{ B - \partial_z P | w \} $',color="b");
    #         twin_00.plot(self.w[w_idx],E_1D[w_idx], 'b-')


    #          # E{B|w,z} = int_b f_B|WZ(b|w,z)*b db
    #         W,B,Z, dPGrad,B2Grad,WBGrad,W2Grad = self.data
    #         Nz   = len(Z)
    #         E_WZ = np.zeros((self.N,Nz)) # f_WZΦ
    #         for j in range(Nz):
                
    #             # HIST
    #             f_XΦ,w,φ = np.histogram2d(x=W[:,j].flatten(),y=B[:,j].flatten(),bins=self.N,density=True)
    #             φ = .5*(φ[1:]+φ[:-1]); dφ = φ[1] - φ[0];
    #             w = .5*(w[1:]+w[:-1]); dw = w[1] - w[0];
    #             f_X      =  np.sum(  f_XΦ,axis=1)*dφ
    #             E        = (np.sum(φ*f_XΦ,axis=1)*dφ)/f_X;
    #             E_WZ[:,j]= interp1d(w,E,bounds_error=False)(self.w)

    #         # Interpolate it in z
    #         E_2D = np.zeros((self.N,self.N)) # E[Φ|W=w,Z=z]
    #         for i in range(self.N):
    #             E_2D[i,:] = interp1d(Z,E_WZ[i,:])(self.z)            

    #         E_2D -= self.Expectations['\partial_z P']['2D']['wz']
            
    #     twin_00.tick_params(axis="y",labelcolor="b")

    #     # 2D Expectations ---------------
    #     Z = gaussian_filter(E_2D[w_idx,:][:,z_idx], sigma=sigma_smooth,truncate=3.0)
    #     try:
    #         Level = np.linspace(Z.flatten().min(),Z.flatten().max(),Nlevels)
    #         CS_10  = axs[1].contour(self.w[w_idx], self.z[z_idx], Z.T, levels =Level,norm=norm,cmap='Blues')
    #         axs[1].clabel(CS_10, inline=False, fontsize=1) 
    #     except:
    #         CS_10  = axs[1].contour(self.w[w_idx], self.z[z_idx], Z.T, levels =Nlevels,norm=norm,cmap='Blues')
    #         axs[1].clabel(CS_10, inline=False, fontsize=1) 

    #     #'''
    #     if figname != None:
    #         fig.savefig(figname, dpi=200)
    #     plt.show()
    #     plt.close(fig)

    #     return None;

    # # ----- section f_WB

    # def Plot_fWB(self,interval = None, figname=None,Nlevels=100,norm_2d='log'):

    #     """
    #     Return a base plot of the PDFs ontop of which we can overlay the expectations
    #     """
    #     from matplotlib.ticker import NullFormatter, MaxNLocator
    #     from numpy import linspace
    #     plt.ion()

    #     if interval != None:
            
    #         idx1 = np.where(self.b < interval['b'][1]);
    #         idx2 = np.where(self.b > interval['b'][0]);
    #         b_idx = np.intersect1d(idx1,idx2)

    #         idx1 = np.where(self.w < interval['w'][1]);
    #         idx2 = np.where(self.w > interval['w'][0]); 
    #         w_idx = np.intersect1d(idx1,idx2)

    #     else:
    #         b_idx = np.arange(0,len(self.b),1)
    #         w_idx = np.arange(0,len(self.w),1)

    #     # Coords
    #     x    = self.w[w_idx] 
    #     y    = self.b[b_idx]

    #     # Data
    #     f_x  = self.fW[w_idx]
    #     f_y  = self.fB[b_idx]
    #     f_xy = self.fWB[w_idx,:][:,b_idx]

    #     # Set up your x and y labels
    #     xlabel = r'$w$'
    #     ylabel = r'$b$'

    #     fxlabel = r'$f_W(w)$'
    #     fylabel = r'$f_B(b)$'

    #     # Set up default x and y limits
    #     xlims = [min(x),max(x)]
    #     ylims = [min(y),max(y)]
        
    #     # Define the locations for the axes
    #     left, width = 0.12, 0.55
    #     bottom, height = 0.12, 0.55
    #     bottom_h = left_h = left+width+0.02
        
    #     # Set up the geometry of the three plots
    #     rect_temperature = [left, bottom, width, height] # dimensions of temp plot
    #     rect_histx = [left, bottom_h, width, 0.25] # dimensions of x-histogram
    #     rect_histy = [left_h, bottom, 0.25, height] # dimensions of y-histogram
        
    #     # Set up the size of the figure
    #     fig = plt.figure(1, figsize=(12,9))
        
    #     # Make the three plots
    #     axTemperature = plt.axes(rect_temperature) # temperature plot
    #     axHistx       = plt.axes(rect_histx) # x histogram
    #     axHisty       = plt.axes(rect_histy) # y histogram

    #     # Remove the inner axes numbers of the histograms
    #     nullfmt = NullFormatter()
    #     axHistx.xaxis.set_major_formatter(nullfmt)
    #     axHisty.yaxis.set_major_formatter(nullfmt)
        
    #     # Find the min/max of the data
    #     xmin = min(xlims)
    #     xmax = max(xlims)
    #     ymin = min(ylims)
    #     ymax = max(y)

    #     # Make the 'main' 2D plot
    #     axTemperature.contourf(x,y,f_xy.T,cmap='Reds',levels=Nlevels,norm=norm_2d)
    #     axTemperature.set_xlabel(xlabel,fontsize=25)
    #     axTemperature.set_ylabel(ylabel,fontsize=25)

    #     #Make the tickmarks pretty
    #     ticklabels = axTemperature.get_xticklabels()
    #     for label in ticklabels:
    #         label.set_fontsize(20)
    #         label.set_family('serif')
        
    #     ticklabels = axTemperature.get_yticklabels()
    #     for label in ticklabels:
    #         label.set_fontsize(20)
    #         label.set_family('serif')
        
    #     #Set up the plot limits
    #     axTemperature.set_xlim(xlims)
    #     axTemperature.set_ylim(ylims)

    #     # Make the 1D plots
    #     axHistx.plot(x,f_x,'r')
    #     axHistx.set_ylim([0.,1.01*max(f_x)])
    #     axHistx.fill_between(x=x,y1=f_x,color= "r",alpha= 0.2)
        
    #     axHisty.plot(f_y,y,'r')
    #     axHisty.set_xlim([0.,1.01*max(f_y)])
    #     axHisty.fill_between(x=f_y,y1=y,color= "r",alpha= 0.2)
        
    #     axHistx.set_ylabel(fxlabel,fontsize=25)
    #     axHisty.set_xlabel(fylabel,fontsize=25)

    #     #Set up the histogram limits
    #     axHistx.set_xlim( min(x), max(x) )
    #     axHisty.set_ylim( min(y), max(y) )
        
    #     #Make the tickmarks pretty
    #     ticklabels = axHistx.get_yticklabels()
    #     for label in ticklabels:
    #         label.set_fontsize(20)
    #         label.set_family('serif')
        
    #     #Make the tickmarks pretty
    #     ticklabels = axHisty.get_xticklabels()
    #     for label in ticklabels:
    #         label.set_fontsize(20)
    #         label.set_family('serif')
        
    #     #Cool trick that changes the number of tickmarks for the histogram axes
    #     axHisty.xaxis.set_major_locator(MaxNLocator(4))
    #     axHisty.yaxis.set_major_locator(MaxNLocator(6))
    #     axTemperature.yaxis.set_major_locator(MaxNLocator(6))


    #     axHistx.yaxis.set_major_locator(MaxNLocator(4))
    #     axHistx.xaxis.set_major_locator(MaxNLocator(6))
    #     axTemperature.xaxis.set_major_locator(MaxNLocator(6))
        
    #     #Show the plot
    #     plt.draw()
        
    #     # Save to a File
    #     #filename = 'myplot'
    #     #plt.savefig(figname, dpi=200,format = 'png', transparent=True)

    #     # if figname != None:
    #     #     fig.savefig(figname, dpi=200)
    #     #plt.show()
    #     #plt.close(fig)

    #     return fig,[axTemperature, axHistx, axHisty];

    # def Plot_EWB(self,term,Ra,interval=None,figname=None,Nlevels=15,sigma_smooth=2):

    #     """
    #     Using the plots of the pdfs overlay the expectations in terms:
    #     - contours for surface plots
    #     - hatched lines for graph plots
    #     """
        
    #     #'''
    #     from matplotlib.ticker import NullFormatter, MaxNLocator
    #     if interval != None:
            
    #         idx1 = np.where(self.b < interval['b'][1]);
    #         idx2 = np.where(self.b > interval['b'][0]);
    #         b_idx = np.intersect1d(idx1,idx2)

    #         idx1 = np.where(self.w < interval['w'][1]);
    #         idx2 = np.where(self.w > interval['w'][0]); 
    #         w_idx = np.intersect1d(idx1,idx2)
            
    #     else:
    #         b_idx = np.arange(0,len(self.b),1)
    #         w_idx = np.arange(0,len(self.w),1)

    #     fig,axs = self.Plot_fWB(interval=interval,figname=figname,dpi=200);


    #     # Coords
    #     x    = self.w[w_idx] 
    #     y    = self.b[b_idx]

    #     if term == 'both':

    #         # E = E{B - dP_z|W=w}
    #         f_x  = np.trapz(self.fWB*self.b,x=self.b,axis=1)/self.fW
    #         f_x -= self.Expectations['\partial_z P']['1D']['w']
    #         f_x  = gaussian_filter(f_x[w_idx], sigma=sigma_smooth,truncate=3.0)

    #         # E = E{B - dP_z|B=b}
    #         f_y = self.b - self.Expectations['\partial_z P']['1D']['b']
    #         f_y = gaussian_filter(f_y[b_idx], sigma=sigma_smooth,truncate=3.0)

    #         # E{B - dP_z|W=w,B=b} = B - E{dP_z|W=w,B=b}
    #         B    = np.outer(np.ones(len(self.w)),self.b)
    #         E_2D = B - self.Expectations['\partial_z P']['2D']['wb']
    #         f_xy = gaussian_filter( E_2D[w_idx,:][:,b_idx], sigma=sigma_smooth,truncate=3.0)

    #     else:
            
    #         # Data
    #         f_x  = self.Expectations[term]['1D']['w'][w_idx] 
    #         f_y  = self.Expectations[term]['1D']['b'][b_idx]
    #         f_xy = gaussian_filter( self.Expectations[term]['2D']['wb'][w_idx,:][:,b_idx], sigma=sigma_smooth,truncate=3.0)
        
    #         f_x /= (Ra**0.5);
    #         f_y /= (Ra**0.5);
    #         f_xy/= (Ra**0.5);

    #     axTemperature, axHistx, axHisty = axs[0],axs[1],axs[2];

    #     # Main 2D Plot
    #     try:
    #         Level = np.linspace(f_xy.flatten().min(),f_xy.flatten().max(),Nlevels)
    #         CS_10 = axTemperature.contour(x,y,f_xy.T,levels= Level,norm='linear',cmap='Blues')
    #         axTemperature.clabel(CS_10, inline=False, fontsize=1)
    #     except:
    #         CS_10 = axTemperature.contour(x,y,f_xy.T,levels=Nlevels,norm='linear',cmap='Blues')
    #         axTemperature.clabel(CS_10, inline=False, fontsize=1)


    #     # 1D Expectationss ---------------
    #     if term == 'both':
            
    #         twin_x = axHistx.twinx()
    #         twin_x.plot(x,f_x, 'b-.')
    #         twin_x.set_ylabel(r'$E\{-|w\}$',color="b",fontsize=20);
    #         twin_x.tick_params(axis="y",labelcolor="b")

    #         twin_y = axHisty.twiny()
    #         twin_y.plot(f_y, y, 'b-.')
    #         twin_y.set_xlabel(r'$E\{-|b\}$',color="b",fontsize=20);
    #         twin_y.tick_params(axis="x",labelcolor="b")

    #     else:
    #         twin_x = axHistx.twinx()
    #         twin_x.plot(x,f_x, 'b-.')
    #         twin_x.set_ylabel(r'$E\{-|w\}/Re$',color="b",fontsize=20);
    #         twin_x.tick_params(axis="y",labelcolor="b")

    #         twin_y = axHisty.twiny()
    #         twin_y.plot(f_y, y, 'b-.')
    #         twin_y.set_xlabel(r'$E\{-|b\}/Re$',color="b",fontsize=20);
    #         twin_y.tick_params(axis="x",labelcolor="b")


    #     #Make the tickmarks pretty
    #     ticklabels = twin_x.get_yticklabels()
    #     for label in ticklabels:
    #         label.set_fontsize(20)
    #         label.set_family('serif')
        
    #     #Make the tickmarks pretty
    #     ticklabels = twin_y.get_xticklabels()
    #     for label in ticklabels:
    #         label.set_fontsize(20)
    #         label.set_family('serif')
        
    #     #Cool trick that changes the number of tickmarks for the histogram axes
    #     twin_y.xaxis.set_major_locator(MaxNLocator(3))
    #     twin_x.yaxis.set_major_locator(MaxNLocator(3))

    #     if figname != None:
    #         fig.savefig(figname, dpi=200)
    #     #plt.show()
    #     plt.close(fig)

    #     return None;


if __name__ == "__main__":


    for file in glob.glob('/data/pmannix/PDF_DNS_Data/' + '/Sim**'):
        
        os.chdir(file)
        name = file.split('/')[-1].split('_')[1]
        print('## Simulation case: ',name,'## \n')
    
        with open(name + '_pickled.pickle','rb') as f:
            pdf = pickle.load(f)
            plotter = PdfPlotter(pdf)
            plotter.plot_pdf(figname='pdf_'+name+'.png')
            for key,save in zip(pdf.Expectations.keys(),pdf.Save_Handles):
                plotter.plot_expectation(term=key, figname=save+'.png', Nlevels=25, sigma_smooth=3)

