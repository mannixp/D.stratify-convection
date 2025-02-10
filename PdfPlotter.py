from matplotlib import rc
rc('text', usetex=True)
rc('font', family='serif')
rc('font', size=20.0)

import matplotlib.pyplot as plt
import glob, os, pickle
import numpy as np
from scipy.ndimage import gaussian_filter1d
from PdfGenerator import PdfGenerator


from matplotlib.colors import BoundaryNorm

def cmap_and_norm(Z):

    # Flatten the input
    Z = Z.flatten()

    # define the colormap
    cmap = plt.cm.bwr_r

    # extract all colors from the .jet map
    cmaplist = [cmap(i) for i in range(cmap.N)]
    
    # create the new map
    cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)

    # define the bins and normalize and forcing 0 to be part of the colorbar!
    lim = np.max( [abs(np.max(Z)), abs(np.min(Z))] )
    bounds = np.linspace(-lim,lim,100)
    idx    = np.searchsorted(bounds,0)
    bounds = np.insert(bounds,idx,0)
    norm   = BoundaryNorm(bounds, cmap.N)

    return norm, cmap

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

            # idx1 = np.where(self.pdf.z < self.interval['z'][1])
            # idx2 = np.where(self.pdf.z > self.interval['z'][0])
            # z_idx = np.intersect1d(idx1,idx2)

        else:
            b_idx = np.arange(0,len(self.pdf.b),1)
            w_idx = np.arange(0,len(self.pdf.w),1)
        z_idx = np.arange(0,len(self.pdf.z),1)

        return b_idx, w_idx, z_idx
    
    def plot_pdf(self, figname=None, Nlevels=15, norm='log'):
        """Return a base plot of the PDFs ontop of which we can overlay the expectations"""

        b_idx, w_idx, z_idx = self.calculate_interval()
        fig, axs = plt.subplots(3, 2,figsize=(8,6),constrained_layout=True)
        
        # f_B ---------------
        axs[0, 0].plot(self.pdf.b[b_idx],self.pdf.fB[b_idx],'r')
        axs[0, 0].set_ylim([0.,1.01*max(self.pdf.fB[b_idx])])
        axs[0, 0].fill_between(x=self.pdf.b[b_idx],y1=self.pdf.fB[b_idx],color= "r",alpha= 0.2)
        axs[0, 0].tick_params('x', labelbottom=False)
        name_00 = {'x':r'$b$','y':r'$f_B(b)$'}
      
        axs[1, 0].sharex(axs[0, 0])
        axs[1, 0].contourf(self.pdf.b[b_idx], self.pdf.z[z_idx], self.pdf.fBZ[b_idx,:][:,z_idx].T,cmap='Reds',levels=Nlevels ,norm=norm)
        axs[1, 0].tick_params('x', labelbottom=False)
        name_10 = {'x':r'$b$','y':r'$z$'}

        axs[2, 0].sharex(axs[0, 0])
        axs[2, 0].contourf(self.pdf.b[b_idx], self.pdf.w[w_idx], self.pdf.fWB[w_idx,:][:,b_idx]  ,cmap='Reds',levels=Nlevels,norm=norm)
        name_20 = {'x':r'$b$','y':r'$w$'}
        axs[2, 0].set_xlabel(r'$b$')

        # f_W -------------
        axs[0,1].plot(self.pdf.w[w_idx],self.pdf.fW[w_idx],'r')
        axs[0,1].set_ylim([0.,1.01*max(self.pdf.fW[w_idx])])
        axs[0,1].fill_between(x=self.pdf.w[w_idx],y1=self.pdf.fW[w_idx],color= "r",alpha= 0.2)
        axs[0,1].tick_params('x', labelbottom=False)
        name_01 = {'x':r'$w$','y':r'$f_W(w)$'}
      
        axs[1, 1].sharex(axs[0, 1])
        axs[1, 1].contourf(self.pdf.w[w_idx], self.pdf.z[z_idx], self.pdf.fWZ[w_idx,:][:,z_idx].T,cmap='Reds',levels=10*Nlevels,norm=norm)
        axs[1, 1].tick_params('x', labelbottom=False)
        name_11 = {'x':r'$w$','y':r'$z$'}

        axs[2, 1].sharex(axs[0, 1])
        axs[2, 1].contourf(self.pdf.w[w_idx], self.pdf.b[b_idx], self.pdf.fWB[w_idx,:][:,b_idx].T,cmap='Reds',levels=Nlevels,norm=norm)
        name_21 = {'x':r'$w$','y':r'$b$'}
        axs[2, 1].set_xlabel(r'$w$')

        Names = np.asarray( [[name_00,name_01], [name_10,name_11], [name_20,name_21]])
        for ax,name in zip(axs.flat,Names.flat):
            ax.set(ylabel=name['y'])

        if figname != None:
            fig.savefig(figname, dpi=100)
        #plt.show()
        #plt.close(fig)

        return fig, axs

    def plot_expectation(self, term, figname=None, Nlevels=15, norm='linear', sigma_smooth=1):

        """
        Using the plots of the pdfs overlay the expectations in terms:
        - contours for surface plots
        - hatched lines for graph plots
        """

        b_idx, w_idx, z_idx = self.calculate_interval()
        fig,axs = self.plot_pdf()

        E_1D = self.pdf.Expectations[term]['1D']
        E_2D = self.pdf.Expectations[term]['2D']

        # B ---------------------
        twin_00 = axs[0,0].twinx()
        twin_00.plot(self.pdf.b[b_idx],E_1D['b'][b_idx], 'b-',label=str(r'$E['+term+'\|b]$'))
        twin_00.set_ylabel(r'$E_B['+term+']$')

        Z = gaussian_filter1d(E_2D['bz'][b_idx,:][:,z_idx], sigma=sigma_smooth,truncate=3.0)
        CS_10  = axs[1, 0].contour(self.pdf.b[b_idx], self.pdf.z[z_idx], Z.T, levels=Nlevels, norm=norm, cmap='Blues')
        axs[1, 0].clabel(CS_10, inline=False, fontsize=1)

        Z = gaussian_filter1d(E_2D['wb'][w_idx,:][:,b_idx], sigma=sigma_smooth,truncate=3.0)
        CS_20  = axs[2, 0].contour(self.pdf.b[b_idx], self.pdf.w[w_idx],  Z, levels=Nlevels, norm=norm, cmap='Blues')
        axs[2, 0].clabel(CS_20, inline=False, fontsize=1) 

        # W ---------------------
        twin_01 = axs[0,1].twinx()
        twin_01.plot(self.pdf.w[w_idx],E_1D['w'][w_idx], 'b-',label=str(r'$E['+term+'\|w]$'))
        twin_01.set_ylabel(r'$E_W['+term+']$')

        Z = gaussian_filter1d(E_2D['wz'][w_idx,:][:,z_idx], sigma=sigma_smooth,truncate=3.0)
        CS_11  = axs[1, 1].contour(self.pdf.w[w_idx], self.pdf.z[z_idx], Z.T, levels = Nlevels,norm=norm,cmap='Blues')
        axs[1, 1].clabel(CS_11, inline=False, fontsize=1)     

        Z = gaussian_filter1d(E_2D['wb'][w_idx,:][:,b_idx], sigma=sigma_smooth,truncate=3.0)
        CS_21  = axs[2, 1].contour(self.pdf.w[w_idx], self.pdf.b[b_idx], Z.T, levels = Nlevels,norm=norm,cmap='Blues')
        axs[2, 1].clabel(CS_21, inline=False, fontsize=1) 

        if figname != None:
            fig.savefig(figname, dpi=100)
        #plt.show()
        plt.close()

        return None

    # ----- section f_BZ -----

    def plot_fBZ(self, figname=None, Nlevels=15, norm='log'):
        """Return a plot of f_B,f_BZ on which we can overlay the expectations."""

        b_idx, w_idx, z_idx = self.calculate_interval()
        fig, axs = plt.subplots(4,1,figsize=(12,8),height_ratios=[1.5,.5,1.5,.5],sharex=True,constrained_layout=True)
        
        # 1D PDFs ---------------
        #axs[0].plot(self.pdf.b[b_idx],self.pdf.fB[b_idx],'r')

        db = self.pdf.b[b_idx][1] - self.pdf.b[b_idx][0]
        b_edges = np.hstack( [self.pdf.b[b_idx] - .5*db*np.ones(len(self.pdf.b[b_idx])), self.pdf.b[b_idx][-1] + .5*db ])
        axs[0].stairs(self.pdf.fB[b_idx][1:-1], b_edges[1:-1])

        axs[0].set_ylim([0.,1.01*max(self.pdf.fB[b_idx])])
        axs[0].fill_between(x=self.pdf.b[b_idx],y1=self.pdf.fB[b_idx],color= "r",alpha= 0.2)
        axs[0].set_ylabel(r'$f_B$',color='r', fontsize=30)
        axs[0].tick_params(axis="y",labelcolor="r")


        # 2D PDFs ---------------
        axs[2].contourf(self.pdf.b[b_idx], self.pdf.z[z_idx], self.pdf.fBZ[b_idx,:][:,z_idx].T,cmap='Reds', levels=Nlevels, norm=norm)
        axs[2].set_ylabel(r"$z$", fontsize=30)
        
        # top inset
        idx1 = np.where(self.pdf.z < 1)
        idx2 = np.where(self.pdf.z > 0.95)
        z_idx = np.intersect1d(idx1,idx2)
        axs[1].contourf(self.pdf.b[b_idx], self.pdf.z[z_idx], self.pdf.fBZ[b_idx,:][:,z_idx].T,cmap='Reds', levels=Nlevels, norm=norm)

        # bottom inset
        idx1 = np.where(self.pdf.z < 0.05)
        idx2 = np.where(self.pdf.z > 0)
        z_idx = np.intersect1d(idx1,idx2)
        axs[3].contourf(self.pdf.b[b_idx], self.pdf.z[z_idx], self.pdf.fBZ[b_idx,:][:,z_idx].T,cmap='Reds', levels=Nlevels, norm=norm)
        axs[3].set_xlabel(r"$b$", fontsize=30)
        
        # Add mean E[B]
        μ = np.trapz(y=self.pdf.b*self.pdf.fB,x=self.pdf.b)
        y = np.linspace(np.min(self.pdf.fB),np.max(self.pdf.fB),50)
        axs[0].plot(μ*np.ones(50),y,'k--',linewidth=2)
        
        # Add mean E_Z[B]
        z = self.pdf.z
        b = np.outer(self.pdf.b,np.ones(len(z)))
        f = self.pdf.fBZ
        EB_Z = np.trapz(y=b*f,x=b,axis=0)
        idx1 = np.where(EB_Z < self.interval['b'][1])
        idx2 = np.where(EB_Z > self.interval['b'][0])
        idx = np.intersect1d(idx1,idx2)
        axs[2].plot(EB_Z[idx], z[idx],'k--',linewidth=2)

        for ax in axs:
            ax.tick_params(axis='both', labelsize=30)

        if figname != None:
            fig.savefig(figname, dpi=100)
            plt.close(fig)

        return fig, axs

    def plot_EBZ(self, term, figname=None, Nlevels=15, norm='log', sigma_smooth=2):
        """
        Using the plots of the pdfs overlay the expectations in terms:
        - contours for surface plots
        - hatched lines for graph plots
        """

        b_idx, w_idx, z_idx = self.calculate_interval()
        fig, axs = self.plot_fBZ()

        E_1D = self.pdf.Expectations[term]['1D']
        E_2D = self.pdf.Expectations[term]['2D']

        twin_00 = axs[0].twinx()
        twin_00.plot(self.pdf.b[b_idx], E_1D['b'][b_idx], 'b')
        #twin_00.set_ylabel(r'$E_B[|\nabla B|^2 ]$', color="b", fontsize=30)
        twin_00.tick_params(axis="y", labelcolor="b", labelsize=30)

        Z = gaussian_filter1d(E_2D['bz'][b_idx,:][:,z_idx], sigma=sigma_smooth, truncate=3.0)
        norm, cmap = cmap_and_norm(Z)

        CS_10  = axs[2].contour(self.pdf.b[b_idx], self.pdf.z[z_idx], Z.T, levels=Nlevels, norm=norm, cmap=cmap)
        axs[2].clabel(CS_10, inline=False, fontsize=1)

        # top inset
        idx1 = np.where(self.pdf.z < 1)
        idx2 = np.where(self.pdf.z > 0.95)
        z_idx = np.intersect1d(idx1,idx2)

        #Z = gaussian_filter1d(E_2D['bz'][b_idx,:][:,z_idx], sigma=sigma_smooth, truncate=3.0)
        CS_10  = axs[1].contour(self.pdf.b[b_idx], self.pdf.z[z_idx], Z[:,z_idx].T, levels=Nlevels, norm=norm, cmap=cmap)

        # bottom inset
        idx1 = np.where(self.pdf.z < 0.05)
        idx2 = np.where(self.pdf.z > 0)
        z_idx = np.intersect1d(idx1,idx2)
        
        #Z = gaussian_filter1d(E_2D['bz'][b_idx,:][:,z_idx], sigma=sigma_smooth, truncate=3.0)
        CS_10  = axs[3].contour(self.pdf.b[b_idx], self.pdf.z[z_idx], Z[:,z_idx].T, levels=Nlevels, norm=norm, cmap=cmap)

        if figname != None:
            fig.savefig(figname, dpi=100)
        plt.close(fig)
        #plt.show()

        return None

    def Decomposition(self, figname=None, sigma_smooth=2):
        """Plot the term ∂2/∂b^2(E{|∇B|^2|B=b}f_B) expanded in term of its components."""

        b_idx, w_idx, z_idx = self.calculate_interval()

        # Calculate the terms
        Φ  = gaussian_filter1d(self.pdf.Expectations[r'\|\nabla B\|^2']['1D']['b'], sigma=sigma_smooth, truncate=3.0)
        fB = gaussian_filter1d(self.pdf.fB, sigma=sigma_smooth, truncate=3.0)
        h = self.pdf.b[b_idx][1] - self.pdf.b[b_idx][0]

        ddf = (fB[b_idx][2:] - 2.*fB[b_idx][1:-1] + fB[b_idx][:-2])/(h**2)
        Φ_norm = np.linalg.norm(Φ[b_idx][1:-1]*ddf,2)

        dΦ = ( Φ[b_idx][2:]  -  Φ[b_idx][:-2])/(2.*h)
        df = (fB[b_idx][2:]  - fB[b_idx][:-2])/(2.*h)
        dΦ_norm = np.linalg.norm(dΦ*df,2)
        
        ddΦ = (Φ[b_idx][2:] - 2.*Φ[b_idx][1:-1] + Φ[b_idx][:-2])/(h**2)
        ddΦ_norm = np.linalg.norm(ddΦ*fB[b_idx][1:-1],2)


        fig,ax=plt.subplots(3,sharex=True)

        # Plot PDFs

        b  = self.pdf.b[b_idx]
        db = b[1] - b[0]
        b_edges = np.hstack( [b - .5*db*np.ones(len(b)), b[-1] + .5*db ])
        values  = self.pdf.fB[b_idx]
        
        for ax_i in ax:
            ax_i.stairs(values, edges=b_edges)
            ax_i.fill_between(x=self.pdf.b[b_idx],y1=self.pdf.fB[b_idx],color= "r",alpha= 0.2)
            ax_i.set_ylabel(r'$f_B$',fontsize=20)
            ax_i.set_xlim(self.interval['b'])
        
        ax[2].set_xlabel(r'$b$',fontsize=20)
        
        # Plot E[Φ|B=b]
        twin2 = ax[2].twinx()
        twin2.plot(self.pdf.b[b_idx][1:-1],1.*(Φ[b_idx][1:-1]*ddf)/Φ_norm,'b-',label=r"$E_B[|\nabla B|^2] \frac{\partial^2 f_B}{\partial b^2}$")
        twin2.legend(loc=1,fontsize=14)
        #twin2.set_yticks([])
        
        twin1 = ax[1].twinx()
        twin1.plot(self.pdf.b[b_idx][1:-1],1.*(dΦ*df)/dΦ_norm,'b-',label=r"$E_B[|\nabla B|^2]' \frac{\partial f_B}{\partial b}$")
        twin1.legend(loc=3,fontsize=14)
        #twin1.set_yticks([])

        twin0 = ax[0].twinx()
        twin0.plot(self.pdf.b[b_idx][1:-1],1.*(ddΦ*fB[b_idx][1:-1])/ddΦ_norm, 'b-',label=r"$E_B[|\nabla B|^2]'' f_B$")
        twin0.legend(loc=3,fontsize=14)
        #twin0.set_yticks([])

        print('|Φ| =',Φ_norm)
        print('|dΦ/db| =',dΦ_norm)
        print('|d^2Φ/db^2| =',ddΦ_norm)

        plt.tight_layout()
        if figname != None:
            fig.savefig(figname, dpi=100)
        plt.close(fig)

        return None
    

    # ----- section f_WZ

    def plot_fWZ(self, figname=None, Nlevels=100, norm='log'):

        """
        Return a base plot of the PDFs ontop of which we can overlay the expectations
        """

        b_idx, w_idx, z_idx = self.calculate_interval()
        fig, axs = plt.subplots(2,1,figsize=(12,8),height_ratios=[2,3],sharex=True,constrained_layout=True)
        
        # 1D PDFs ---------------
        #axs[0].plot(self.pdf.w[w_idx],self.pdf.fW[w_idx],'r')
        
        w = self.pdf.w[w_idx]
        dw = w[1] - w[0]
        w_edges = np.hstack( [w - .5*dw*np.ones(len(w)), w[-1] + .5*dw ])
        values  = self.pdf.fW[w_idx]
        axs[0].stairs(values[1:-1], edges=w_edges[1:-1])

        axs[0].set_ylim([0.,1.01*max(self.pdf.fW[w_idx])])
        axs[0].fill_between(x=self.pdf.w[w_idx],y1=self.pdf.fW[w_idx],color= "r",alpha= 0.2)
        axs[0].set_ylabel(r'$f_W$',color='r', fontsize=30)
        axs[0].tick_params(axis="y",labelcolor="r")
        axs[0].tick_params(axis='both', labelsize=30)


        # 2D PDFs ---------------
        #axs[1].pcolormesh(self.w[w_idx], self.z[z_idx], self.fWZ[w_idx,:][:,z_idx].T,cmap='Reds' ,norm='linear')
        axs[1].contourf(self.pdf.w[w_idx], self.pdf.z[z_idx], self.pdf.fWZ[w_idx,:][:,z_idx].T, cmap='Reds', levels=10*Nlevels, norm=norm)
        axs[1].set_xlabel(r"$w$", fontsize=30)
        axs[1].set_ylabel(r"$z$", fontsize=30)
        axs[1].tick_params(axis='both', labelsize=30)

        if figname != None:
            fig.savefig(figname, dpi=100)
            plt.close(fig)

        return fig, axs

    def plot_EWZ(self, term, Ra=1, figname=None, Nlevels=15, norm='linear', sigma_smooth=2):
        """
        Using the plots of the pdfs overlay the expectations in terms:
        - contours for surface plots
        - hatched lines for graph plots
        """
        
        b_idx, w_idx, z_idx = self.calculate_interval()
        fig, axs = self.plot_fWZ()

        twin_00 = axs[0].twinx()
        if term == r'\|\nabla W \|^2': 
            E_1D = self.pdf.Expectations[term]['1D']['w']/np.sqrt(Ra)
            E_2D = self.pdf.Expectations[term]['2D']['wz']/np.sqrt(Ra)
            #twin_00.set_ylabel(r'$E_W[ |\nabla W|^2 ]/Re$', color="b", fontsize=30)

        elif term == r'\partial_z P': 
            E_1D = -1.*self.pdf.Expectations[term]['1D']['w']
            E_2D = -1.*self.pdf.Expectations[term]['2D']['wz']
            #twin_00.set_ylabel(r'$E_W[-\partial_z P ]$', color="b", fontsize=30)

        elif term == r'B':   
            E_1D = self.pdf.Expectations[term]['1D']['w']
            E_2D = self.pdf.Expectations[term]['2D']['wz']
            #twin_00.set_ylabel(r'$E_W[B]$', color="b", fontsize=30)

        elif term == 'both':
            E_1D = self.pdf.Expectations[r'B']['1D']['w'] - self.pdf.Expectations['\partial_z P']['1D']['w']
            E_2D = self.pdf.Expectations[r'B']['2D']['wz'] - self.pdf.Expectations['\partial_z P']['2D']['wz']
            #twin_00.set_ylabel(r'$E_W[B - \partial_z P]$', color="b", fontsize=30)

        # plot 1D Expectation 
        twin_00.plot(self.pdf.w[w_idx], E_1D[w_idx], 'b-')
        twin_00.tick_params(axis="y",labelcolor="b", labelsize=30)

        # plot 2D Expectations
        Z = gaussian_filter1d(E_2D[w_idx,:][:,z_idx], sigma=sigma_smooth, truncate=3.0)
        norm, cmap = cmap_and_norm(Z)
        CS_10 = axs[1].contour(self.pdf.w[w_idx], self.pdf.z[z_idx], Z.T, levels=Nlevels, norm=norm, cmap=cmap)
        axs[1].clabel(CS_10, inline=False, fontsize=1) 

        if figname != None:
            fig.savefig(figname, dpi=100)
        plt.close(fig)

        return None

    # ----- section f_WB

    def plot_fWB(self, figname=None, Nlevels=100, norm='log'):
        """Return a base plot of the PDFs ontop of which we can overlay the expectations."""

        b_idx, w_idx, z_idx = self.calculate_interval()
        from matplotlib.ticker import NullFormatter, MaxNLocator
        from numpy import linspace
        #plt.ion()


        # Coords
        x = self.pdf.w[w_idx] 
        y = self.pdf.b[b_idx]

        # Data
        f_x  = self.pdf.fW[w_idx]
        f_y  = self.pdf.fB[b_idx]
        f_xy = self.pdf.fWB[w_idx,:][:,b_idx]

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
        axTemperature.contourf(x, y, f_xy.T, cmap='Reds', levels=Nlevels, norm=norm)
        axTemperature.set_xlabel(xlabel,fontsize=30)
        axTemperature.set_ylabel(ylabel,fontsize=30)

        #Make the tickmarks pretty
        ticklabels = axTemperature.get_xticklabels()
        for label in ticklabels:
            label.set_fontsize(24)
            label.set_family('serif')
        
        ticklabels = axTemperature.get_yticklabels()
        for label in ticklabels:
            label.set_fontsize(24)
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
        
        axHistx.set_ylabel(fxlabel,fontsize=30)
        axHisty.set_xlabel(fylabel,fontsize=30)

        #Set up the histogram limits
        axHistx.set_xlim( min(x), max(x) )
        axHisty.set_ylim( min(y), max(y) )
        
        #Make the tickmarks pretty
        ticklabels = axHistx.get_yticklabels()
        for label in ticklabels:
            label.set_fontsize(30)
            label.set_family('serif')
        
        #Make the tickmarks pretty
        ticklabels = axHisty.get_xticklabels()
        for label in ticklabels:
            label.set_fontsize(30)
            label.set_family('serif')
        
        #Cool trick that changes the number of tickmarks for the histogram axes
        axHisty.xaxis.set_major_locator(MaxNLocator(4))
        axHisty.yaxis.set_major_locator(MaxNLocator(6))
        axTemperature.yaxis.set_major_locator(MaxNLocator(6))

        axHistx.yaxis.set_major_locator(MaxNLocator(4))
        axHistx.xaxis.set_major_locator(MaxNLocator(6))
        axTemperature.xaxis.set_major_locator(MaxNLocator(6))
        
        #Show the plot
        #plt.draw()
        
        if figname != None:
            fig.savefig(figname, dpi=100)
            plt.close(fig)

        return fig, [axTemperature, axHistx, axHisty]

    def plot_EWB(self, term, Ra=1, figname=None, Nlevels=15, sigma_smooth=2):
        """
        Using the plots of the pdfs overlay the expectations in terms:
        - contours for surface plots
        - hatched lines for graph plots
        """
        
        from matplotlib.ticker import NullFormatter, MaxNLocator
        
        b_idx, w_idx, z_idx = self.calculate_interval()
        fig, axs = self.plot_fWB()

        # Coords
        x = self.pdf.w[w_idx] 
        y = self.pdf.b[b_idx]

        if term == 'both':

            # E = E{B - dP_z|W=w}
            f_x = self.pdf.Expectations['B']['1D']['w'] - self.pdf.Expectations['\partial_z P']['1D']['w']
            f_x = gaussian_filter1d(f_x[w_idx], sigma=sigma_smooth, truncate=3.0)

            # E = E{B - dP_z|B=b}
            f_y = self.pdf.b - self.pdf.Expectations['\partial_z P']['1D']['b']
            f_y = gaussian_filter1d(f_y[b_idx], sigma=sigma_smooth, truncate=3.0)

            # E{B - dP_z|W=w,B=b} = B - E{dP_z|W=w,B=b}
            E_2D = self.pdf.Expectations['B']['2D']['wb'] - self.pdf.Expectations['\partial_z P']['2D']['wb']
            f_xy = gaussian_filter1d(E_2D[w_idx,:][:,b_idx], sigma=sigma_smooth, truncate=3.0)

        else:
            
            dx = self.pdf.w[0] - self.pdf.w[1]
            dy = self.pdf.b[0] - self.pdf.b[1]
            print(term + '/(Ra^1/2) = ', np.nansum(self.pdf.Expectations[term]['2D']['wb'][:,:]/np.sqrt(Ra))*dx*dy)

            # Data
            f_x  = self.pdf.Expectations[term]['1D']['w'][w_idx] 
            f_y  = self.pdf.Expectations[term]['1D']['b'][b_idx]
            f_xy = gaussian_filter1d(self.pdf.Expectations[term]['2D']['wb'][w_idx,:][:,b_idx], sigma=sigma_smooth, truncate=3)
        
            f_x /= (Ra**0.5)
            f_y /= (Ra**0.5)
            f_xy/= (Ra**0.5)

        axTemperature, axHistx, axHisty = axs[0], axs[1], axs[2]

        norm, cmap = cmap_and_norm(f_xy)
        CS_10 = axTemperature.contour(x, y, f_xy.T, levels=Nlevels, norm=norm, cmap=cmap)
        axTemperature.clabel(CS_10, inline=False, fontsize=1)

        # 1D Expectationss ---------------
        twin_x = axHistx.twinx()
        twin_x.plot(x,f_x, 'b-.')
        twin_x.set_ylabel(r'$E_W[-]$',color="b",fontsize=30);
        twin_x.tick_params(axis="y",labelcolor="b")
        twin_x.ticklabel_format(axis='y',style='sci',scilimits=(0,0))

        twin_y = axHisty.twiny()
        twin_y.plot(f_y, y, 'b-.')
        twin_y.set_xlabel(r'$E_B[-]$',color="b",fontsize=30);
        twin_y.tick_params(axis="x",labelcolor="b")
        twin_y.ticklabel_format(axis='x',style='sci',scilimits=(0,0))

        #Make the tickmarks pretty
        ticklabels = twin_x.get_yticklabels()
        for label in ticklabels:
            label.set_fontsize(30)
            label.set_family('serif')
        
        #Make the tickmarks pretty
        ticklabels = twin_y.get_xticklabels()
        for label in ticklabels:
            label.set_fontsize(30)
            label.set_family('serif')
        
        #Cool trick that changes the number of tickmarks for the histogram axes
        twin_y.xaxis.set_major_locator(MaxNLocator(3))
        twin_x.yaxis.set_major_locator(MaxNLocator(3))


        if figname != None:
            fig.savefig(figname, dpi=100)
            plt.close(fig)
        

        # # Test plot
        # print(term)
        # from mpl_toolkits.mplot3d import Axes3D  
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # X, Y = np.meshgrid(x,y)
        # ax.plot_surface(X, Y, f_xy.T, norm=norm, cmap=cmap)
        # ax.set_xlabel('w Label')
        # ax.set_ylabel('z Label')
        # plt.show()

        return None


if __name__ == "__main__":

    intervals = {'IC':{'w':(-0.05,0.05),'b':(0.004,0.008)}, 'ICR':{'w':(-0.05,0.05),'b':(0.004,0.008)}, 'RBC':{'w':(-1,1),'b':(0.4,0.6)}, 
                 'PLUME':{'w':(-0.1,0.1),'b':(-0.01,0.01)}, 'SINE':{'w':(-.5,.5),'b':(0.05,.25)}, 'STEP':{'w':(-1,1),'b':(0.2,.8)}}

    # Generate all the plots in the Appendix
    for file in glob.glob("./data/*.pickle"):
        
        name = file.split('/')[-1].split('_')[0]
        print('## Simulation case: ',name,'## \n')

        os.mkdir(name)        
        with open(file,'rb') as f:
            
            os.chdir(name)
            pdf = pickle.load(f)
            plotter = PdfPlotter(pdf, interval=intervals[name])
            plotter.plot_pdf(figname='pdf_'+name+'.png')
            for key,save in zip(pdf.Expectations.keys(),pdf.Save_Handles):
                plotter.plot_expectation(term=key, figname=save+'.png', Nlevels=25, sigma_smooth=3)
            os.chdir("../")
