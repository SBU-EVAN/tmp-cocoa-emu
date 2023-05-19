from cocoa_emu import Config
import numpy as np
import getdist

def add_shear_calib(m, datavector):
    for i in range(5):
        factor = (1 + m[i])**shear_calib_mask[i]
        datavector = factor * datavector
    return datavector

def compute_chi2(theta,dvec):
    shear = theta[12:17]
    dv = dvec[:780]#add_shear_calib(shear,dvec[:780])
    dv_diff_masked = (dv - dv_fid)[mask]
    chi2 = dv_diff_masked @ cov_inv_masked @ dv_diff_masked
    return chi2

def get_inv_cov_masked(cov,mask):
    n_rows,n_columns = cov.shape
    for i in range(n_rows):
        for j in range(n_columns):
            if( i!=j and j>=i):
                _mask_row = mask[i]
                _mask_col = mask[j]
                cov[i,j] = cov[i,j]*_mask_row*_mask_col
                cov[j,i] = cov[i,j]

    cov_inv = np.linalg.inv(cov)
    cov_inv_masked = cov_inv[mask][:,mask]

    return(cov_inv_masked)

configfile = './projects/lsst_y1/cs_mcmc_omm_2_test.yaml'
config = Config(configfile)

shear_calib_mask = config.shear_calib_mask[:,:780]
print(config.shear_calib_mask)
mask             = config.mask[:780]
covmat           = config.cov[:780,:780]
dv_fid           = config.dv_fid[:780]
cov_inv_masked   = get_inv_cov_masked(covmat,mask)

mask_first = np.linalg.inv(covmat[mask][:,mask])
mask_after = np.linalg.inv(covmat)[mask][:,mask]

#open samples and dvs
datavecs = np.load('./projects/lsst_y1/emulator_output/chains/cocoa_chain_training_lkl_dvs.npy')
samples  = np.load('./projects/lsst_y1/emulator_output/chains/cocoa_chain_training_lkl_samples.npy')
print(datavecs)
print('computing chi2')
chi2_arr = np.array([compute_chi2(samples[i],datavecs[i]) for i in range(len(samples))])
print(chi2_arr)
np.save('chi2_of_cocoa_different_config.npy',chi2_arr)

#open chain for chi2
# chain = getdist.mcsamples.loadMCSamples('./cocoa_w_chi2',no_cache=True)
# chi2_chain = chain.getParams().chi2_cocoa
# np.save('chi2_from_cocoa_chain.npy',chi2_chain)

# print(chi2_arr[:13000]-chi2_chain[:13000])
