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

In projects/lsst_y1 there is a config file containing the parameters, priors, and training hyperparameters which you can edit as needed. The model is in cocoa_emu/nn_emulator.py . Lastly, use train_emulator.py to train the network. It opens the data, randomly shuffles, and then diagonalizes the basis before passing that to the neural network. Calling the predict function of the model returns the data vector in the data basis (NOT the diagonal basis)

