import os, pickle
from PdfGenerator import PdfGenerator
from PdfPlotter import PdfPlotter

Dir = '/home/pmannix/Stratification-DNS/'


intervals = {'IC':{'w':(-0.05,0.05),'b':(0.004,0.008)}, 'ICR':{'w':(-0.05,0.05),'b':(0.004,0.008)}, 'RBC':{'w':(-1,1),'b':(0.4,0.6)}, 
                'PLUME':{'w':(-0.1,0.1),'b':(-0.01,0.01)}, 'SINE':{'w':(-.5,.5),'b':(0.05,.25)}, 'STEP':{'w':(-1,1),'b':(0.2,.8)}}


# Paper figures section f_BZ
# ~~~~~~~~~~~~~~ # ~~~~~~~~~~~~~~
# name = 'figures_fBZ'
# print(name)
# os.mkdir(Dir+name) 
# os.chdir(Dir+name)

# with open(Dir+'data/IC_pickled.pickle', 'rb') as f:
#     plotter = PdfPlotter(pdf=pickle.load(f), interval=intervals['IC'])
#     plotter.Decomposition(figname='Compare_Terms_fB_IC.png')
#     plotter.plot_EBZ(term=r'\|\nabla B\|^2', figname='IC_E_BZ_and_f_BZ.png', Nlevels=50, sigma_smooth=1)

# with open(Dir+'data/RBC_pickled.pickle', 'rb') as f:
#     plotter = PdfPlotter(pdf=pickle.load(f), interval=intervals['RBC'])
#     plotter.Decomposition(figname='Compare_Terms_fB_RBC.png')
#     plotter.plot_EBZ(term=r'\|\nabla B\|^2', figname='RBC_E_BZ_and_f_BZ.png', Nlevels=50, sigma_smooth=1)

# os.chdir(Dir)

# # Paper figures section f_WZ
# # ~~~~~~~~~~~~~~ # ~~~~~~~~~~~~~~
# name = 'figures_fWZ'
# print(name)
# os.mkdir(Dir+name) 
# os.chdir(Dir+name)

# with open(Dir+'data/SINE_pickled.pickle', 'rb') as f:
#     plotter = PdfPlotter(pdf=pickle.load(f), interval=intervals['SINE'])
#     plotter.plot_EWZ(term=r'B', figname='HC_E_B___WZ_and_f_WZ.png', Nlevels=50, sigma_smooth=1)
#     plotter.plot_EWZ(term=r'\partial_z P', figname='HC_E_dPZ_WZ_and_f_WZ.png', Nlevels=100, sigma_smooth=1)
#     plotter.plot_EWZ(term=r'\|\nabla W \|^2', figname='HC_E_dW2_WZ_and_f_WZ.png', Nlevels=100, sigma_smooth=1, Ra=1e10, norm='log')
#     plotter.plot_EWZ(term='both', figname='HC_E_BPZ_WZ_and_f_WZ.png', Nlevels=100, sigma_smooth=1)

# with open(Dir+'data/IC_pickled.pickle', 'rb') as f:
#     plotter = PdfPlotter(pdf=pickle.load(f), interval=intervals['IC'])
#     plotter.plot_EWZ(term=r'B', figname='IC_E_B___WZ_and_f_WZ.png', Nlevels=30, sigma_smooth=1)
#     plotter.plot_EWZ(term=r'\partial_z P', figname='IC_E_dPZ_WZ_and_f_WZ.png', Nlevels=30, sigma_smooth=1)
#     plotter.plot_EWZ(term=r'\|\nabla W \|^2', figname='IC_E_dW2_WZ_and_f_WZ.png', Nlevels=30, sigma_smooth=1, Ra=1e11, norm='log')
#     plotter.plot_EWZ(term='both', figname='IC_E_BPZ_WZ_and_f_WZ.png', Nlevels=30, sigma_smooth=1)

# os.chdir(Dir)

# Paper figures section f_WB
# ~~~~~~~~~~~~~~ # ~~~~~~~~~~~~~~
name = 'figures_fWB'
print(name)
os.mkdir(Dir+name) 
os.chdir(Dir+name)

with open(Dir+'data/IC_pickled.pickle', 'rb') as f:
    plotter = PdfPlotter(pdf=pickle.load(f), interval=intervals['IC'])
    #plotter.plot_EWB(term=r'\partial_z P', figname='IC_E_dPz_WB_and_f_WB.png', Nlevels=30, sigma_smooth=1)
    plotter.plot_EWB(term=r'\|\nabla B\|^2', figname='IC_E_dB2_WB_and_f_WB.png', Nlevels=30, sigma_smooth=1, Ra=1e11)
    plotter.plot_EWB(term=r'\nabla W^T \nabla B', figname='IC_E_dWdB_WB_and_f_WB.png', Nlevels=30, sigma_smooth=1, Ra=1e11)
    plotter.plot_EWB(term=r'\|\nabla W \|^2', figname='IC_E_dW2_WB_and_f_WB.png', Nlevels=30, sigma_smooth=1, Ra=1e11)
    plotter.plot_EWB(term='both', figname='IC_E_BdPz_WB_and_f_WB.png', Nlevels=30, sigma_smooth=1)

with open(Dir+'data/RBC_pickled.pickle', 'rb') as f:
    plotter = PdfPlotter(pdf=pickle.load(f), interval=intervals['RBC'])
    #plotter.plot_EWB(term=r'\partial_z P', figname='RBC_E_dPz_WB_and_f_WB.png', Nlevels=30, sigma_smooth=1)
    plotter.plot_EWB(term=r'\|\nabla B\|^2', figname='RBC_E_dB2_WB_and_f_WB.png', Nlevels=30, sigma_smooth=1, Ra=1e10)
    plotter.plot_EWB(term=r'\nabla W^T \nabla B', figname='RBC_E_dWdB_WB_and_f_WB.png', Nlevels=30, sigma_smooth=1, Ra=1e10)
    plotter.plot_EWB(term=r'\|\nabla W \|^2', figname='RBC_E_dW2_WB_and_f_WB.png', Nlevels=30, sigma_smooth=1, Ra=1e10)
    plotter.plot_EWB(term='both', figname='RBC_E_BdPz_WB_and_f_WB.png', Nlevels=30, sigma_smooth=1)

os.chdir(Dir)

