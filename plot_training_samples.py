import numpy as np
import getdist
from getdist import plots
import matplotlib.pyplot as plt

# debug file to plot samples used for training in a triangle plot

base_path = './projects/lsst_y1/emulator_output/'
names = ['logA','ns','H0','omegab','omegac']


#hypercube = np.load(base_path+'dvs_for_training_100k/train_samples_0.npy')
#print(hypercube.shape)
#hypercube_chain = getdist.mcsamples.MCSamples(samples=hypercube[:,:5],names=names,labels=names)

mcmc_chain0 = getdist.mcsamples.loadMCSamples(base_path+'/chains/gaussian_approx_none_600k_T16',no_cache=True)
mean0 = np.mean(mcmc_chain0.samples[:,:5],axis=0)

print(mean0)

chain_array = [mcmc_chain0,mcmc_chain1]#,mcmc_chain2,mcmc_chain3,mcmc_chain4,mcmc_chain5,mcmc_chain6]
g = plots.get_subplot_plotter()
g.settings.num_plot_contours = 2
g.triangle_plot(chain_array,
               filled=True,
               params=names)
g.export('training_samples.pdf')