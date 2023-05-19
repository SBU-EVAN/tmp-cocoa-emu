import numpy as np
import getdist
from getdist import plots
import matplotlib.pyplot as plt

chi2_train_config = np.load('chi2_of_cocoa_different_config.npy')
#chi2_chain_config = np.load('backup_chi2_of_cocoa_different_config.npy')#'chi2_from_cocoa_chain.npy')
chi2_cocoa_config = np.load('chi2_from_cocoa_chain.npy')
print(chi2_train_config)
plt.hist(chi2_train_config, 50, density=True, facecolor='r', alpha=0.5, label='mcmc config')#'training config')
#plt.hist(chi2_chain_config, 50, density=True, facecolor='b', alpha=0.5, label='mask before inverse')#'cobaya config')
plt.hist(chi2_cocoa_config, 50, density=True, facecolor='g', alpha=0.5, label='cobaya')
plt.xlabel('Chi2')
plt.legend()
plt.savefig('chi2_testing.pdf')