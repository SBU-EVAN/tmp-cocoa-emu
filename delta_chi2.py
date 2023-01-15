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

#torch.set_default_dtype(torch.double)

##3x2 setting: separate cosmic shear and 2x2pt
OUTPUT_DIM=780
BIN_SIZE=OUTPUT_DIM


def get_chi2(dv_predict, dv_exact, mask, cov_inv):
    ## GPU emulators works well with float32
    #print('===')
    #print(dv_predict)
    #print(dv_exact)
    #print('===')
    delta_dv = (dv_predict - np.float32(dv_exact))[mask]
    chi2 = np.matmul( np.matmul(np.transpose(delta_dv), np.float32(cov_inv)) , delta_dv  )   
    return chi2


#os.environ["OMP_NUM_THREADS"] = "1"

configfile = './projects/lsst_y1/train_emulator.yaml'
config = Config(configfile)

samples_validation = np.load('./projects/lsst_y1/emulator_output/chains/vali_post_T1_samples_0.npy')#[::60000]
dv_validation      = np.load('./projects/lsst_y1/emulator_output/chains/vali_post_T1_data_vectors_0.npy')#[::60000]
#samples_validation = np.load('./projects/lsst_y1/emulator_output/chains/train_post_600k_T8_samples_0.npy')[::6000]
#dv_validation      = np.load('./projects/lsst_y1/emulator_output/chains/train_post_600k_T8_data_vectors_0.npy')[::6000]

if config.probe =='cosmic_shear':
    dv_validation = dv_validation[:,:OUTPUT_DIM]
    mask = config.mask[0:OUTPUT_DIM]
else:
    print("3x2 not tested")
    quit()
    
mask = np.ones(OUTPUT_DIM,dtype=bool)
cov            = config.cov[0:OUTPUT_DIM, 0:OUTPUT_DIM]
cov_inv        = np.linalg.inv(config.cov[0:OUTPUT_DIM, 0:OUTPUT_DIM])
#print(cov[mask][:,mask].shape)
cov_inv_masked = np.linalg.inv(config.cov[0:OUTPUT_DIM, 0:OUTPUT_DIM][mask][:,mask])

#onfigfile = './projects/lsst_y1/train_emulator.yaml'
#config_test = Config(configfile)
#cov_test= config_test.cov[0:OUTPUT_DIM, 0:OUTPUT_DIM]

# do diagonalizatieigensys = np.linalg.eig(cov)
eigensys = np.linalg.eig(cov)
evals = eigensys[0]
evecs = eigensys[1]
cov_inv_masked = np.diag(1/evals)
cov_inv_masked = evecs @ cov_inv_masked @ np.linalg.inv(evecs)


logA   = samples_validation[:,0]
ns     = samples_validation[:,1]
H0     = samples_validation[:,2]
Omegab = samples_validation[:,3]
Omegac = samples_validation[:,4]
dz1    = samples_validation[:,5]
dz2    = samples_validation[:,6]
dz3    = samples_validation[:,7]
dz4    = samples_validation[:,8]
dz5    = samples_validation[:,9]

print('number of points to plot',len(samples_validation))

bin_count = 0
start_idx = 0
end_idx   = 0

# #Loop over the models glue them together
# #It's more intuitive to take one sample at a time, but that would require too many loading of the emulator
# #The loop below is to get dv_predict of ALL samples, bin by bin.
# for i in range(1):
#     device='cpu'
#     emu = nn_pca_emulator(config.n_dim, BIN_SIZE, config.dv_fid, config.dv_std, cov, cov, 0, 0, device) #should privde dv_max instead of dv_fid, but emu.load will make it correct
#     emu.load('projects/lsst_y1/emulator_output/models/model')
#     print('emulator loaded')
#     tmp = []
#     for j in range(len(samples_validation)):

#         theta = torch.Tensor(samples_validation[j])
#         dv_emu = emu.predict(theta,evecs)[0]

#         tmp.append(dv_emu)
#     tmp = np.array(tmp)

#     if i==0:
#         dv_predict = tmp
#     else:
#         dv_predict = np.append(dv_predict, tmp, axis = 1)


# print("testing", np.shape(dv_predict))

# chi2_list = []
# count=0
# count2=0
# for i in range(len(dv_predict)):
#     chi2 = get_chi2(dv_predict[i], dv_validation[i], mask, cov_inv_masked)
#     chi2_list.append(chi2)
#     print(chi2)
#     if chi2>1:
#         count +=1


device=torch.device('cpu')
#torch.set_num_interop_threads(60) # Inter-op parallelism
torch.set_num_threads(32) # `Intra-op parallelism
emu = nn_pca_emulator(config.n_dim, BIN_SIZE, config.dv_fid, config.dv_std, cov_inv, cov_inv, 0, 0, device) #should privde dv_max instead of dv_fid, but emu.load will make it correct
emu.load('projects/lsst_y1/emulator_output/models/model')
#print(emu.X_mean)
#print(samples_validation[0])
print('emulator loaded\n')
chi2_list=np.zeros(len(samples_validation))
count=0
start_time=datetime.now()
time_prev=start_time
predicted_dvs = np.zeros((len(samples_validation),BIN_SIZE))

for j in range(len(samples_validation)):
    _j=j+1

    theta = torch.Tensor(samples_validation[j])
    dv_emu = emu.predict(theta,evecs)[0]
    predicted_dvs[j] = dv_emu

    chi2 = get_chi2(dv_emu, dv_validation[j], mask, cov_inv_masked)
    chi2_list[j]=chi2
    if chi2>1:
       count +=1

    if j%10==0:
        runtime=datetime.now()-start_time
        print('\rprogress: '+str(j)+'/'+str(len(samples_validation))+\
            ' | runtime: '+str(runtime)+\
            ' | remaining time: '+str(runtime*(len(samples_validation)/_j - 1))+\
            ' | time/it: '+str(runtime/_j),end='')

chi2_list = np.array(chi2_list)
print(chi2_list)
#print("testing",chi2_list)
print("\naverage chi2 is: ", np.average(chi2_list))
print("Warning: This can be different from the training-validation loss. It depends on the mask file you use.")
print("points with chi2 > 1: ", count)

cmap = plt.cm.get_cmap('coolwarm')

### DEBUG ###
#print(predicted_dvs.shape)
#print(cov_inv_masked.shape)
#print(predicted_dvs[:,mask].shape)
#diff = predicted_dvs[:,mask] - dv_validation[:,mask]
#chi2_array = np.diag(diff @ cov_inv_masked @ np.transpose(diff))
#chi2_list=chi2_array
print(chi2_list)
print("\naverage chi2 is: ", np.average(chi2_list))
print("Warning: This can be different from the training-validation loss. It depends on the mask file you use.")
print("points with chi2 > 1: ", len(np.where(chi2_list>1)[0]))

###PLOT chi2 start

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
plt.scatter(logA, Omegac, c=chi2_list, label=r'$\chi^2$ between emulator and cocoa', s = 2, cmap=cmap,norm=matplotlib.colors.LogNorm())
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