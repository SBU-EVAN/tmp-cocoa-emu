import sys,os
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(".."))

from cocoa_emu import Config
#from cocoa_emu.emulator import NNEmulator, GPEmulator  #KZ: not working, no idea why
from cocoa_emu import NNEmulator, nn_pca_emulator  #KZ: not working, no idea why

debug=False

configfile = sys.argv[1]
config = Config(configfile)

# Training set
train_samples_files = sys.argv[2]
file = sys.argv[2]

### CONCATENATE TRAINING DATA
#train_samples = []
#train_data_vectors = []

#for file in train_samples_files:
train_samples=np.load(file+'_samples_0.npy')#.append(np.load(file+'_samples_0.npy'))
train_data_vectors=np.load(file+'_data_vectors_0.npy')#.append(np.load(file+'_data_vectors_0.npy'))
if debug:
    print('(debug)')
    print('lhs')
    #print(train_samples[0])
    #print(train_data_vectors[0])
    print('(end debug)')

#train_samples = np.array([subsubarr for subarr in train_samples for subsubarr in subarr])#train_samples_file1
#train_data_vectors = np.array([subsubarr for subarr in train_data_vectors for subsubarr in subarr])#train_data_vectors_file1
###

# this script do the same thing for as train_emulator.py, but instead calculate the data_vactors for training, a set of

print("length of samples from LHS: ", train_samples.shape)

if config.probe=='cosmic_shear':
    print("training for cosmic shear only")
    OUTPUT_DIM = 780
    train_data_vectors = train_data_vectors[:,:OUTPUT_DIM]
    cov_inv = np.linalg.inv(config.cov)[0:OUTPUT_DIM, 0:OUTPUT_DIM] #NO mask here for cov_inv enters training
    mask_cs = config.mask[0:OUTPUT_DIM]
    
    dv_fid =config.dv_fid[0:OUTPUT_DIM]
    dv_std = config.dv_std[0:OUTPUT_DIM]
elif config.probe=='3x2pt':
    print("trianing for 3x2pt")
    train_data_vectors = train_data_vectors
    cov_inv = np.linalg.inv(config.cov) #NO mask here for cov_inv enters training
    OUTPUT_DIM = config.output_dim #config will do it automatically, check config.py
    dv_fid =config.dv_fid
    dv_std = config.dv_std
else:
    print('probe not defnied')
    quit()

def get_chi_sq_cut(train_data_vectors, chi2_cut):
    chi_sq_list = []
    for dv in train_data_vectors:
        if config.probe=='cosmic_shear':
            delta_dv = (dv - config.dv_obs[0:OUTPUT_DIM])[mask_cs] #technically this should be masked(on a fiducial scale cut), but the difference is small
            chi_sq = delta_dv @ cov_inv[mask_cs][:,mask_cs] @ delta_dv
        elif config.probe=='3x2pt':
            delta_dv = (dv - config.dv_obs)[config.mask]
            chi_sq = delta_dv @ config.masked_inv_cov @ delta_dv


        chi_sq_list.append(chi_sq)
    chi_sq_arr = np.array(chi_sq_list)
    select_chi_sq = (chi_sq_arr < chi2_cut)
    return select_chi_sq


# ====================chi2 cut for train dvs===========================
# select_chi_sq = get_chi_sq_cut(train_data_vectors, config.chi_sq_cut)
print("not applying chi2 cut to lhs")
# select_chi_sq = get_chi_sq_cut(train_data_vectors, 1e6)
# selected_obj = np.sum(select_chi_sq)
# total_obj    = len(select_chi_sq)
        
# train_data_vectors = train_data_vectors[select_chi_sq]
# train_samples      = train_samples[select_chi_sq]

print("training LHC samples after chi2 cut: ", len(train_samples))

#adding points from chains here to avoid chi2 cut
if len(sys.argv) > 3:
    print("training with posterior samples")
    train_samples_file2      = np.load(sys.argv[3]+'_samples_0.npy')
    train_data_vectors_file2 = np.load(sys.argv[3]+'_data_vectors_0.npy')[:,:OUTPUT_DIM]
    if debug:
        print('(debug)')
        print('posterior')
        #print(train_samples_file2[0])
        #print(train_data_vectors_file2[0])
        print('(end debug)')
    
    train_samples = np.vstack((train_samples, train_samples_file2))
    train_data_vectors = np.vstack((train_data_vectors, train_data_vectors_file2))
    print("posterior samples contains: ", len(train_samples_file2))

print("Total samples enter the training: ", len(train_samples))

##Normalize the data vectors for training based on the maximum##
#dv_max = np.abs(train_data_vectors).max(axis=0)
#train_data_vectors = train_data_vectors / dv_max


###============= Setting up validation set ============
validation_samples      = np.load('./projects/lsst_y1/emulator_output/chains/vali_post_T1_samples_0.npy')[::60]
validation_data_vectors = np.load('./projects/lsst_y1/emulator_output/chains/vali_post_T1_data_vectors_0.npy')[::60,:OUTPUT_DIM]
#====================chi2 cut for test dvs===========================
select_chi_sq = get_chi_sq_cut(validation_data_vectors, 7000)
selected_obj = np.sum(select_chi_sq)
total_obj    = len(select_chi_sq)

if debug:
    print('(debug)')
    print('validation')
    #print(validation_samples[0])
    #print(validation_data_vectors[0])
    print('(end debug)')
        
validation_data_vectors = validation_data_vectors[select_chi_sq]
validation_samples      = validation_samples[select_chi_sq]

print("validation samples after chi2 cut: ", len(validation_samples))


##### shuffeling #####
def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

train_samples, train_data_vectors = unison_shuffled_copies(train_samples, train_data_vectors)
validation_samples, validation_data_vectors = unison_shuffled_copies(validation_samples, validation_data_vectors)

# Convert to eigenbasis if PCA
if config.do_PCA:
    #open data
    lsst_cov = config.cov[0:OUTPUT_DIM,0:OUTPUT_DIM] #np.loadtxt('lsst_y1_cov.txt')
    lsst_fid = config.dv_fid[0:OUTPUT_DIM] #np.loadtxt('lsst_y1_fid.txt')

    #print(lsst_cov.shape)
    #print(lsst_fid.shape)

    # do diagonalization
    eigensys = np.linalg.eig(lsst_cov)
    evals = eigensys[0]
    evecs = eigensys[1]

    # truncate PCAs
    # we need to keep ALL indices, cant forget unmodelled dimensions add to loss.
    n_PCA = config.n_PCA
    sorted_idxs = np.argsort(1/evals)
    pc_idxs = sorted_idxs[:n_PCA]
    non_pc_idxs = sorted_idxs[n_PCA:] 
    #pca_vecs = evecs[pc_idxs]

    print('pca idxs   :',len(pc_idxs))
    print('nonpca idxs:',len(non_pc_idxs))

    # Now we change basis to the eigenbasis
    dv = np.array([dv_fid for _ in range(len(train_data_vectors))])
    train_data_vectors = np.transpose((np.linalg.inv(evecs) @ np.transpose(train_data_vectors - dv)))#[pc_idxs])
    dv = np.array([dv_fid for _ in range(len(validation_data_vectors))])
    validation_data_vectors = np.transpose((np.linalg.inv(evecs) @ np.transpose(validation_data_vectors - dv)))#[pc_idxs])

    cov_inv_pc = np.diag(1/evals[pc_idxs])
    cov_inv_npc = np.diag(1/evals[non_pc_idxs])
    dv_std = np.sqrt(evals)

    print('cov inv pc shape:',cov_inv_pc.shape)
    print('cov inv non-pc shape:',cov_inv_npc.shape)
    # fix output dim
    OUTPUT_DIM = n_PCA

#    if debug:
    #print('[ debug ]   PCA bais training DV array shape:', train_data_vectors.shape)#, file=sys.stderr)

#print("Training emulator...")
# cuda or cpu
if torch.cuda.is_available():
    device = 'gpu'
else:
    device = 'cpu'
    torch.set_num_interop_threads(60) # Inter-op parallelism
    torch.set_num_threads(60) # Intra-op parallelism

print('Using device: ',device)
    
TS = torch.Tensor(train_samples)
#TS.to(device)
TDV = torch.Tensor(train_data_vectors)
#TDV.to(device)
VS = torch.Tensor(validation_samples)
#VS.to(device)
VDV = torch.Tensor(validation_data_vectors)
#VDV.to(device)

emu = nn_pca_emulator(config.n_dim, OUTPUT_DIM, 
                        dv_fid, dv_std, cov_inv_pc,cov_inv_npc, 
                        pc_idxs, non_pc_idxs,
                        device, reduce_lr=False)#, PCA_vecs=pca_vecs)
emu.train(TS, TDV, VS, VDV, batch_size=config.batch_size, n_epochs=config.n_epochs)
print("model saved to ",str(config.savedir))
#emu.save(config.savedir + '/model')


print("DONE!!")   

