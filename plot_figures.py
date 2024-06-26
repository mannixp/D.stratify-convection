"""

Generate the figures for the paper

"""


# %%
#%matplotlib inline

from PdfGenerator import PDF_Master
import pickle

from matplotlib import rc

#rc('text', usetex=True)
rc('font', family='serif')
rc('font', size=25.0)

import os
Dir = '/home/pmannix/Dstratify/DNS_RBC/'

# %%

# Paper figures section f_BZ
# ~~~~~~~~~~~~~~ # ~~~~~~~~~~~~~~
# name = 'figures_fBZ'
# print(name)
# os.mkdir(Dir+name) 
# os.chdir(Dir+name)

# with open(Dir +'mypickleIC_Noise.pickle','rb') as f:
#     pdf = pickle.load(f)
# pdf.Decomposition(interval=pdf.domain,figname='Compare_Terms_fB_IC.png',)
# pdf.Plot_EBZ(interval = pdf.domain, term='\|dB\|^2', figname='IC_E_BZ_and_f_BZ.png',dpi=200,Nlevels=15,sigma_smooth=2)

# with open(Dir + 'mypickleRBC.pickle','rb') as f:
#     pdf = pickle.load(f)
# pdf.Decomposition(interval=pdf.domain,figname='Compare_Terms_fB_RBC.png',)
# pdf.Plot_EBZ(interval = pdf.domain, term='\|dB\|^2', figname='RBC_E_BZ_and_f_BZ.png',dpi=200,Nlevels=15,sigma_smooth=2)

# os.chdir(Dir)

# Paper figures section f_WZ
# ~~~~~~~~~~~~~~ # ~~~~~~~~~~~~~~
name = 'figures_fWZ'
print(name)
os.mkdir(Dir+name) 
os.chdir(Dir+name)

with open(Dir +'mypickleRBC_Sine.pickle','rb') as f:
    pdf = pickle.load(f)
pdf.Plot_EWZ(interval = pdf.domain, term='B'           , figname='HC_E_B___WZ_and_f_WZ.png',dpi=200,Nlevels=30,sigma_smooth=2)
pdf.Plot_EWZ(interval = pdf.domain, term='\partial_z P', figname='HC_E_dPZ_WZ_and_f_WZ.png',dpi=200,Nlevels=30,sigma_smooth=2)
pdf.Plot_EWZ(interval = pdf.domain, term='\| dW \|^2'  , figname='HC_E_dB2_WZ_and_f_WZ.png',dpi=200,Nlevels=30,sigma_smooth=2)

with open(Dir +'mypickleIC.pickle','rb') as f:
    pdf = pickle.load(f)
pdf.Plot_EWZ(interval = pdf.domain, term='B'           , figname='IC_E_B___WZ_and_f_WZ.png')
pdf.Plot_EWZ(interval = pdf.domain, term='\partial_z P', figname='IC_E_dPZ_WZ_and_f_WZ.png')
pdf.Plot_EWZ(interval = pdf.domain, term='\| dW \|^2'  , figname='IC_E_dB2_WZ_and_f_WZ.png')

os.chdir(Dir)

# Paper figures section f_WB
# ~~~~~~~~~~~~~~ # ~~~~~~~~~~~~~~

# with open('mypickleIC.pickle','rb') as f:
#     pdf_A = pickle.load(f)
# pdf_A.Plot_EWB(interval = pdf_A.domain, term='\partial_z P', figname='IC_E_dPz_WB_and_f_WB.png')
# pdf_A.Plot_EWB(interval = pdf_A.domain, term='\|dB\|^2', figname='IC_E_dB2_WB_and_f_WB.png')
# pdf_A.Plot_EWB(interval = pdf_A.domain, term=' dW^T dB', figname='IC_E_dWdB_WB_and_f_WB.png')
# pdf_A.Plot_EWB(interval = pdf_A.domain, term='\| dW \|^2', figname='IC_E_dW2_WB_and_f_WB.png')

# with open('mypickleRBC.pickle','rb') as f:
#     pdf_A = pickle.load(f)
# pdf_A.Plot_EWB(interval = pdf_A.domain, term='\partial_z P', figname='RBC_E_dPz_WB_and_f_WB.png')
# pdf_A.Plot_EWB(interval = pdf_A.domain, term='\|dB\|^2', figname='RBC_E_dB2_WB_and_f_WB.png')
# pdf_A.Plot_EWB(interval = pdf_A.domain, term=' dW^T dB', figname='RBC_E_dWdB_WB_and_f_WB.png')
# pdf_A.Plot_EWB(interval = pdf_A.domain, term='\| dW \|^2', figname='RBC_E_dW2_WB_and_f_WB.png')

# %%
