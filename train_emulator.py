import sys,os
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(".."))

from cocoa_emu import Config
from cocoa_emu import NNEmulator, nn_pca_emulator 

debug=False # flag to enable debug messages
thin_int=1  # thins data vectors when shuffling. Not needed anymore

# open yaml config. 
configfile = sys.argv[1]
config = Config(configfile)

# Training set filenames
train_samples_files = sys.argv[2]
file = sys.argv[2]

#Open first dv file.
print('Opening:',file)
train_samples=np.load(file+'_samples_0.npy')#.append(np.load(file+'_samples_0.npy'))
train_data_vectors=np.load(file+'_data_vectors_0.npy')#.append(np.load(file+'_data_vectors_0.npy'))
if debug:
    print('(debug)')
    print('first file')
    print(train_samples[0])
    print(train_data_vectors[0])
    print('(end debug)')

#print("length of samples from LHS: ", train_samples.shape)

#### adjust validation root directories to your sample directory
if config.probe=='cosmic_shear':
    print("training for cosmic shear only")
    start=0
    stop=780
    validation_root='./projects/lsst_y1/emulator_output/chains/vali_post_T1'
    OUTPUT_DIM = 780
    train_data_vectors = train_data_vectors[:,:OUTPUT_DIM]
    cov_inv = np.linalg.inv(config.cov)[0:OUTPUT_DIM, 0:OUTPUT_DIM] #NO mask here for cov_inv enters training
    mask_cs = config.mask[0:OUTPUT_DIM]
    dv_fid =config.dv_fid[0:OUTPUT_DIM]
    dv_std = config.dv_std[0:OUTPUT_DIM]
elif config.probe=='3x2pt':
    # 3x2pt is generally very difficult.
    print("trianing for 3x2pt")
    start=0
    stop=1560
    validation_root='./projects/lsst_y1/emulator_output/chains/vali_post_T1_3x2'
    train_data_vectors = train_data_vectors
    cov_inv = np.linalg.inv(config.cov) #NO mask here for cov_inv enters training
    OUTPUT_DIM = config.output_dims #config will do it automatically, check config.py
    dv_fid =config.dv_fid
    dv_std = config.dv_std
elif config.probe=='2x2pt':
    print("training for 2x2")
    start=780
    stop=1560
    validation_root='./projects/lsst_y1/emulator_output/chains/vali_post_T1_3x2'
    OUTPUT_DIM = 780
    train_data_vectors = train_data_vectors[:,780:]
    cov_inv = np.linalg.inv(config.cov)[780:, 780:] #NO mask here for cov_inv enters training
    mask_2x2 = config.mask[780:]
    config.n_dim=17
    
    dv_fid =config.dv_fid[780:]
    dv_std = config.dv_std[780:]
else:
    print('probe not defined')
    quit()

def get_chi_sq_cut(train_data_vectors, chi2_cut):
    ### Use to apply chi2 cuts to data vectors which enter training. Not strictly necessary but training on a high chi2 range is more difficult
    chi_sq_list = []
    for dv in train_data_vectors:
        if config.probe=='cosmic_shear':
            delta_dv = (dv - config.dv_obs[0:OUTPUT_DIM])[mask_cs] #technically this should be masked(on a fiducial scale cut), but the difference is small
            chi_sq = delta_dv @ cov_inv[mask_cs][:,mask_cs] @ delta_dv
        elif config.probe=='3x2pt':
            delta_dv = (dv - config.dv_obs)[config.mask]
            chi_sq = delta_dv @ config.masked_inv_cov @ delta_dv
        if config.probe=='2x2pt':
            delta_dv = (dv - config.dv_obs[start:stop])[mask_2x2]
            chi_sq = delta_dv @ cov_inv[mask_2x2][:,mask_2x2] @ delta_dv

        chi_sq_list.append(chi_sq)
    chi_sq_arr = np.array(chi_sq_list)
    select_chi_sq = (chi_sq_arr < chi2_cut)
    return select_chi_sq


# ====================chi2 cut for train dvs===========================
# select_chi_sq = get_chi_sq_cut(train_data_vectors, config.chi_sq_cut)
# select_chi_sq = get_chi_sq_cut(train_data_vectors, 1e6)
# selected_obj = np.sum(select_chi_sq)
# total_obj    = len(select_chi_sq)
        
# train_data_vectors = train_data_vectors[select_chi_sq]
# train_samples      = train_samples[select_chi_sq]

#print("training LHC samples after chi2 cut: ", len(train_samples))

# Now add more training samples from files named 'file_samples_1','file_samples_2', etc.
if len(sys.argv) > 3:
    for i in range(3,len(sys.argv)):
        print("Opening File:",sys.argv[i])
        train_samples_file2      = np.load(sys.argv[i]+'_samples_'+str(i-2)+'.npy')[:,:12]
        train_data_vectors_file2 = np.load(sys.argv[i]+'_data_vectors_'+str(i-2)+'.npy')[:,start:stop]
        train_samples = np.vstack((train_samples, train_samples_file2))
        train_data_vectors = np.vstack((train_data_vectors, train_data_vectors_file2))
    
    #print("posterior samples contains: ", len(train_samples_file2))

##Normalize the data vectors for training based on the maximum##
#dv_max = np.abs(train_data_vectors).max(axis=0)
#train_data_vectors = train_data_vectors / dv_max


###============= Setting up validation set ============
validation_samples      = np.load(validation_root+'_samples_0.npy')[::60,:12] # careful with thinning!
validation_data_vectors = np.load(validation_root+'_data_vectors_0.npy')[::60,start:stop] #Thin only to number of validation dvs you want!
print(validation_samples.shape)
print(validation_data_vectors.shape)
#====================chi2 cut for validation dvs===========================
select_chi_sq = get_chi_sq_cut(validation_data_vectors, 7000)
selected_obj = np.sum(select_chi_sq)
total_obj    = len(select_chi_sq)
        
#validation_data_vectors = validation_data_vectors[select_chi_sq]
#validation_samples      = validation_samples[select_chi_sq]

##### shuffeling
def unison_shuffled_copies(a, b, thin=False):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    #print(p)
    samples = a[p]
    dvs     = b[p]
    return samples[0::thin_int], dvs[0::thin_int]

train_samples, train_data_vectors = unison_shuffled_copies(train_samples, train_data_vectors,thin=True)
validation_samples, validation_data_vectors = unison_shuffled_copies(validation_samples, validation_data_vectors)

# Convert to eigenbasis if PCA is true in config.
# This greatly simplifies the information the NN needs to learn.
if config.do_PCA:
    #open data
    lsst_cov = config.cov[start:stop,start:stop] #np.loadtxt('lsst_y1_cov.txt')
    lsst_fid = config.dv_fid[start:stop] #np.loadtxt('lsst_y1_fid.txt')
    dv_fid=lsst_fid

    # do diagonalization
    eigensys = np.linalg.eigh(lsst_cov)
    evals = eigensys[0]
    evecs = eigensys[1]

    # ensure n_PCA does not exceed number of dimensions
    n_PCA = config.n_PCA
    if n_PCA>OUTPUT_DIM:
        n_PCA=OUTPUT_DIM
        
    #sorted_idxs = np.argsort(1/evals)

    # Now we change basis to the eigenbasis
    dv = np.array([dv_fid for _ in range(len(train_data_vectors))])
    train_data_vectors = np.transpose((np.linalg.inv(evecs) @ np.transpose(train_data_vectors - dv)))#[pc_idxs])
    dv = np.array([dv_fid for _ in range(len(validation_data_vectors))])
    validation_data_vectors = np.transpose((np.linalg.inv(evecs) @ np.transpose(validation_data_vectors - dv)))#[pc_idxs])
    #np.save('validation_dvs_transformed.npy',validation_data_vectors)

    cov_inv_pc = np.diag(1/evals)#[pc_idxs])
    cov_inv_npc = np.diag(1/evals)#[non_pc_idxs])
    dv_std = np.sqrt(evals)*10
    print('N_PCs:', n_PCA)
    print('cov inv pc shape:',cov_inv_pc.shape)
    print('cov inv non-pc shape:',cov_inv_npc.shape)
    # fix output dim
    OUTPUT_DIM = n_PCA

#    if debug:
    #print('[ debug ]   PCA bais training DV array shape:', train_data_vectors.shape)#, file=sys.stderr)
print("Number of training points:  ", len(train_samples))
print("Number of validation points:", len(validation_samples))
#print("Training emulator...")
# cuda or cpu
print('cuda device avaliable?  ',torch.cuda.is_available())
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
    torch.set_num_interop_threads(32) # Inter-op parallelism
    torch.set_num_threads(32) # `Intra-op parallelism

print('Using device: ',device)
    
TS = torch.Tensor(train_samples)
TDV = torch.Tensor(train_data_vectors)
VS = torch.Tensor(validation_samples)
VDV = torch.Tensor(validation_data_vectors)

emu = nn_pca_emulator(12, n_PCA, #config.n_dim
                        dv_fid, dv_std, cov_inv_pc,cov_inv_npc,
                        evecs, 
                        #pc_idxs, non_pc_idxs,
                        device, reduce_lr=True, dtype='double') # dtype is not strictly necessary at all. Remove for the default float
emu.train(TS, TDV, VS, VDV, batch_size=10000,n_epochs=250)# n_epochs=config.n_epochs)
print("model saved to ",str(config.savedir))
emu.save(config.savedir + '/model_'+str(config.probe))


print("DONE!!")   

