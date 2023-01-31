##NOTE: this check doesn't include fast parameters. Do check full prediction of emulator, please use lsst_emu_cs_lcdm.py in cobaya.likelihood


import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import torch
from cocoa_emu import Config, NNEmulator, nn_pca_emulator
from cocoa_emu.sampling import EmuSampler

# just for convinience
from datetime import datetime

#compute using double
torch.set_default_dtype(torch.double)

def get_chi2(dv_predict, dv_exact, mask, cov_inv):
    delta_dv = (dv_predict - np.float32(dv_exact))[mask]
    chi2 = np.matmul( np.matmul(np.transpose(delta_dv), np.float32(cov_inv)) , delta_dv  )   
    return chi2

# adjust config
configfile = './projects/lsst_y1/train_emulator_dzl.yaml'
config = Config(configfile)

# open validation samples
# !!! Watch thin factor !!!
samples_validation = np.load('./projects/lsst_y1/emulator_output/chains/vali_post_T1_3x2_samples_0.npy')[::60]
dv_validation      = np.load('./projects/lsst_y1/emulator_output/chains/vali_post_T1_3x2_data_vectors_0.npy')[::60]

# output dim for full 3x2
# adjust as needed
OUTPUT_DIM=1560
BIN_SIZE=OUTPUT_DIM
mask=config.mask

cov            = config.cov[0:OUTPUT_DIM, 0:OUTPUT_DIM]
cov_inv        = np.linalg.inv(config.cov[0:OUTPUT_DIM, 0:OUTPUT_DIM])
cov_inv_masked = np.linalg.inv(config.cov[0:OUTPUT_DIM, 0:OUTPUT_DIM][mask][:,mask])

logA   = samples_validation[:,0]
ns     = samples_validation[:,1]
H0     = samples_validation[:,2]
Omegab = samples_validation[:,3]
Omegac = samples_validation[:,4]
# dz1    = samples_validation[:,5]
# dz2    = samples_validation[:,6]
# dz3    = samples_validation[:,7]
# dz4    = samples_validation[:,8]
# dz5    = samples_validation[:,9]

bin_count = 0
start_idx = 0
end_idx   = 0

# set needed parameters to initialize emulator
device=torch.device('cpu')
torch.set_num_threads(32) # `Intra-op parallelism
evecs=0

#=====   open trained emulators   =====#
emu_cs = nn_pca_emulator(config.n_dim, BIN_SIZE, config.dv_fid, config.dv_std, cov_inv, cov_inv, evecs, device)
emu_22 = nn_pca_emulator(config.n_dim, BIN_SIZE, config.dv_fid, config.dv_std, cov_inv, cov_inv, evecs, device) 
emu_cs.load('projects/lsst_y1/emulator_output/models/model_T16')
emu_22.load('projects/lsst_y1/emulator_output/models/model_2x2pt_1x10000')
emu_cs.model.double()
emu_22.model.double()
print('emulator(s) loaded\n')

chi2_list=np.zeros(len(samples_validation))
count=0
start_time=datetime.now()
time_prev=start_time
predicted_dv = np.zeros(OUTPUT_DIM)

for j in range(len(samples_validation)):
    _j=j+1

    # get params and true dv
    theta = torch.Tensor(samples_validation[j])
    dv_truth = dv_validation[j]

    # reconstruct dv
    dv_cs = emu_cs.predict(theta[:12])[0]
    predicted_dv[:780] = dv_cs

    dv_22 = emu_22.predict(theta[:17])[0]
    predicted_dv[780:] = dv_22

    # compute chi2
    chi2 = get_chi2(predicted_dv, dv_truth, mask, cov_inv_masked)

    #count how many points have "poor" prediction.
    chi2_list[j] = chi2
    if chi2>1:
       count += 1

    # progress check
    if j%10==0:
        runtime=datetime.now()-start_time
        print('\rprogress: '+str(j)+'/'+str(len(samples_validation))+\
            ' | runtime: '+str(runtime)+\
            ' | remaining time: '+str(runtime*(len(samples_validation)/_j - 1))+\
            ' | time/it: '+str(runtime/_j),end='')

#summary
print("\naverage chi2 is: ", np.average(chi2_list))
print("Warning: This can be different from the training-validation loss. It depends on the mask file you use.")
print("points with chi2 > 1: "+str(count)+" ( "+str((count*100)/len(samples_validation))+"% )")

###PLOT chi2 start
cmap = plt.cm.get_cmap('coolwarm')

num_bins = 100
plt.xlabel(r'$\chi^2$')
plt.ylabel('distribution')
plt.xscale('log')

plt.hist(chi2_list, num_bins, 
                            density = 1, 
                            color ='green',
                            alpha = 0.7)


plt.savefig("validation_chi2.pdf")

####PLOT chi2 end

#####PLOT 2d start######
plt.figure().clear()

#plt.scatter(logA, Omegam, c=chi2_list, label=r'$\chi^2$ between emulator and cocoa', s = 2, cmap=cmap)
plt.scatter(logA, Omegac, c=chi2_list, label=r'$\chi^2$ between emulator and cocoa (T=8 chain)', s = 2, cmap=cmap,norm=matplotlib.colors.LogNorm())
#plt.scatter(Omegam, Omegam_growth, c=chi2_list, label=r'$\chi^2$ between emulator and cocoa', s = 2, cmap=cmap,norm=matplotlib.colors.LogNorm())
#plt.scatter(Omegam, Omegam_growth, c=chi2_list, label=r'$\chi^2$ between emulator and cocoa', s = 2, cmap=cmap)
#plt.scatter(logA, Omegam_growth, c=chi2_list, label=r'$\chi^2$ between emulator and cocoa', s = 2, cmap=cmap,norm=matplotlib.colors.LogNorm())
#plt.scatter(H0, Omegab, c=chi2_list, label=r'$\chi^2$ between emulator and cocoa', s = 2, cmap=cmap,norm=matplotlib.colors.LogNorm())

cb = plt.colorbar()

plt.xlabel(r'$\log A$')
plt.ylabel(r'$\Omega_c h^2$')

plt.legend()
plt.savefig("validation.pdf")

#####PLOT 2d end######


##### PLOT 3d start###

# fig = plt.figure(figsize = (10, 7))
# ax = plt.axes(projection ="3d")

# # Creating plot
# ax.scatter3D(logA, Omegam, Omegam_growth, c = chi2_list, s = 2, cmap=cmap, norm=matplotlib.colors.LogNorm())
# plt.title("simple 3D scatter plot")

# ax.azim = 150
# ax.elev = 15

##### PLOT 3d end###