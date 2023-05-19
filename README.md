# tmp-emu

Temporary repo for datavector emulator files. Run cocoatorch.sh to setup the conda environment.

Does not require cocoa.
Many files related to the LSST analysis are included as a reference of which files to replace and which directory they belong to.

The architecture is as follows:

- linear layer (input dimension x internal dimension)
- activation function ( tanh )
- n_layers are the number of stacked bottleneck blocks ("ResBottle")
- linear layer (internal dimension x output dimension)
- affine layer to scale the output 

Each ResBottle is structure as follows (y is the input to the block):

- linear layer (internal dimension x internal dimension / N)
- batch normalization
- activation
- linear layer (internal dimension / N x internal dimension / N)
- batch normalization
- activation
- linear layer (internal dimension / N x internal dimension / N) 
- batch normalization
- add y to the batch-normed result
- activation

In general, you can run with

    conda activate cocoatorch
    python3 train_emulator.py \
  	    (path/to/training .yaml) \
  		(path/to/training samples \
  		--auto \
  		(internal dimension) \
  		(N to divide the internal dimension) \
  		(n_layers of residual bottleneck blocks)
          
The recommended starting point is internal dimension=64, N=8, and n_layer=1.

In *projects/lsst_y1* there is a config file containing the parameters, priors, and training hyperparameters which you can edit as needed. The model is in cocoa_emu/nn_emulator.py . Lastly, use train_emulator.py to train the network. It opens the data, randomly shuffles, and then diagonalizes the basis before passing that to the neural network. Calling the predict function of the model returns the data vector in the data basis (NOT the diagonal basis)

---
# Debugging Effort

An inconsistency between the $\chi^2$ reported during mcmc to the $\chi^2$ computed manually using datavectors from *get_dv_from_chain.py* has been found. Here's a summary of debugging efforts and a summary of the code involved.

### Training the emulator

We train our emulator on datavectors generated from *get_dv_from_chain.py* using the config *./projects/lsst_y1/dv_from_chain.yaml* . When running a chain using our emulator+emcee, the posteriors are different from the cosmolike+cobaya chain. The emulator is trained without scale cuts and without shear calibration.

We take only the cosmolike+cobaya chain cosmologies for the testing thus far. If we compute $\chi^2$ using the emulators datavector prediction, we see a drop in $\chi^2$.

plot

### Removing the emulator

I can remove the emulator from the tests by passing the test cosmologies into *get_dv_from_chain.py*. When I do this the $\chi^2$ matches the emulator prediction.

plot

At this point, we can rule out bad predictions from the emulator. Thus there must be some difference in the training datavectors.

### Looking for differences in config

A few differences in the config were found. First is *kmax_boltzmann: 5* in the training set. The cosmolike+cobaya chain used *kmax_boltzmann: 20*.













