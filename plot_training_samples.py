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
mcmc_chain1 = getdist.mcsamples.loadMCSamples(base_path+'/chains/gaussian_approx_+5sigma8_600k_T16',no_cache=True)
#mcmc_chain2 = getdist.mcsamples.loadMCSamples(base_path+'/chains/gaussian_approx_+4omegam_600k_T8',no_cache=True)
#mcmc_chain3 = getdist.mcsamples.loadMCSamples(base_path+'/chains/gaussian_approx_-4sigma8_600k_T8',no_cache=True)
#mcmc_chain4 = getdist.mcsamples.loadMCSamples(base_path+'/chains/gaussian_approx_+4sigma8_600k_T8',no_cache=True)
#mcmc_chain5 = getdist.mcsamples.loadMCSamples(base_path+'/chains/gaussian_approx_-4pc1_600k_T8',no_cache=True)
#mcmc_chain6 = getdist.mcsamples.loadMCSamples(base_path+'/chains/gaussian_approx_+4pc1_600k_T8',no_cache=True)

mean0 = np.mean(mcmc_chain0.samples[:,:5],axis=0)
mean1 = np.mean(mcmc_chain1.samples[:,:5],axis=0)
#mean2 = np.mean(mcmc_chain2.samples[:,:5],axis=0)
#mean3 = np.mean(mcmc_chain3.samples[:,:5],axis=0)
#mean4 = np.mean(mcmc_chain4.samples[:,:5],axis=0)
#mean5 = np.mean(mcmc_chain5.samples[:,:5],axis=0)
#mean6 = np.mean(mcmc_chain6.samples[:,:5],axis=0)

print(mean0)
print(mean1)
# print(mean2)
# print(mean3)
# print(mean4)
# print(mean5)
# print(mean6)

chain_array = [mcmc_chain0,mcmc_chain1]#,mcmc_chain2,mcmc_chain3,mcmc_chain4,mcmc_chain5,mcmc_chain6]
g = plots.get_subplot_plotter()
g.settings.num_plot_contours = 2
g.triangle_plot(chain_array,
               filled=True,
               params=names)
g.export('training_samples.pdf')

#python3 train_emulator.py \
#        ./projects/lsst_y1/train_emulator.yaml \
#        ./projects/lsst_y1/emulator_output/chains/train_post_T4 \
#        ./projects/lsst_y1/emulator_output/chains/train_post_T8 \
#        ./projects/lsst_y1/emulator_output/chains/train_post_T8_1