from matplotlib import rc
rc('text', usetex=True)
rc('font', family='serif')
rc('font', size=20.0)
import matplotlib.pyplot as plt

import os, pickle
import numpy as np
from scipy.ndimage import gaussian_filter1d

from PdfGenerator import PdfGenerator
from PdfPlotter import PdfPlotter

Dir = '/home/pmannix/Stratification-DNS/'

# Paper figures section f_BZ
# ~~~~~~~~~~~~~~ # ~~~~~~~~~~~~~~
name = 'figures_fBZ'
print(name)
os.mkdir(Dir+name) 
os.chdir(Dir+name)

with open(Dir+'data/IC_pickled.pickle', 'rb') as f:
    plotter = PdfPlotter(pdf=pickle.load(f), interval=None)
    plotter.Decomposition(figname='Compare_Terms_fB_IC.png')
    plotter.plot_EBZ(term=r'\|\nabla B\|^2', figname='IC_E_BZ_and_f_BZ.png', Nlevels=15, sigma_smooth=2)

with open(Dir+'data/RBC_pickled.pickle', 'rb') as f:
    plotter = PdfPlotter(pdf=pickle.load(f), interval=None)
    plotter.Decomposition(figname='Compare_Terms_fB_RBC.png')
    plotter.plot_EBZ(term=r'\|\nabla B\|^2', figname='RBC_E_BZ_and_f_BZ.png', Nlevels=15, sigma_smooth=2)

os.chdir(Dir)

# Paper figures section f_WZ
# ~~~~~~~~~~~~~~ # ~~~~~~~~~~~~~~
name = 'figures_fWZ'
print(name)
os.mkdir(Dir+name) 
os.chdir(Dir+name)

with open(Dir+'data/SINE_pickled.pickle', 'rb') as f:
    plotter = PdfPlotter(pdf=pickle.load(f), interval=None)
    plotter.plot_EWZ(term=r'B', figname='HC_E_B___WZ_and_f_WZ.png', Nlevels=30, sigma_smooth=2)
    plotter.plot_EWZ(term=r'\partial_z P', figname='HC_E_dPZ_WZ_and_f_WZ.png', Nlevels=30, sigma_smooth=2)
    plotter.plot_EWZ(term=r'\|\nabla W \|^2', figname='HC_E_dB2_WZ_and_f_WZ.png', Nlevels=30, sigma_smooth=2)
    plotter.plot_EWZ(term='both', figname='HC_E_BPZ_WZ_and_f_WZ.png', Nlevels=30, sigma_smooth=2)

with open(Dir+'data/IC_pickled.pickle', 'rb') as f:
    plotter = PdfPlotter(pdf=pickle.load(f), interval=None)
    plotter.plot_EWZ(term=r'B', figname='IC_E_B___WZ_and_f_WZ.png')
    plotter.plot_EWZ(term=r'\partial_z P', figname='IC_E_dPZ_WZ_and_f_WZ.png')
    plotter.plot_EWZ(term=r'\|\nabla W \|^2', figname='IC_E_dB2_WZ_and_f_WZ.png')
    plotter.plot_EWZ(term='both', figname='IC_E_BPZ_WZ_and_f_WZ.png',)

os.chdir(Dir)

# Paper figures section f_WB
# ~~~~~~~~~~~~~~ # ~~~~~~~~~~~~~~
name = 'figures_fWB'
print(name)
os.mkdir(Dir+name) 
os.chdir(Dir+name)

with open(Dir+'data/IC_pickled.pickle', 'rb') as f:
    plotter = PdfPlotter(pdf=pickle.load(f), interval=None)
    plotter.plot_EWB(term=r'\partial_z P', figname='IC_E_dPz_WB_and_f_WB.png')
    plotter.plot_EWB(term=r'\|\nabla B\|^2', figname='IC_E_dB2_WB_and_f_WB.png')
    plotter.plot_EWB(term=r'\nabla W^T \nabla B', figname='IC_E_dWdB_WB_and_f_WB.png')
    plotter.plot_EWB(term=r'\|\nabla W \|^2', figname='IC_E_dW2_WB_and_f_WB.png')
    plotter.plot_EWB(term='both', figname='IC_E_BdPz_WB_and_f_WB.png')

with open(Dir+'data/RBC_pickled.pickle', 'rb') as f:
    plotter = PdfPlotter(pdf=pickle.load(f), interval=None)
    plotter.plot_EWB(term=r'\partial_z P', figname='RBC_E_dPz_WB_and_f_WB.png')
    plotter.plot_EWB(term=r'\|\nabla B\|^2', figname='RBC_E_dB2_WB_and_f_WB.png')
    plotter.plot_EWB(term=r'\nabla W^T \nabla B', figname='RBC_E_dWdB_WB_and_f_WB.png')
    plotter.plot_EWB(term=r'\|\nabla W \|^2', figname='RBC_E_dW2_WB_and_f_WB.png')
    plotter.plot_EWB(term='both', figname='RBC_E_BdPz_WB_and_f_WB.png')

os.chdir(Dir)

