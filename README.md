# tmp-emu

Temporary repo for datavector emulator files. Run cocoatorch.sh to setup the conda environment.

Does not require cocoa.
Many files related to the LSST analysis are included as a reference of which files to replace and which directory they belong to.

In projects/lsst_y1 there is a config file containing the parameters, priors, and training hyperparameters which you can edit as needed. The model is in cocoa_emu/nn_emulator.py . Lastly, use train_emulator.py to train the network. It opens the data, randomly shuffles, and then diagonalizes the basis before passing that to the neural network. Calling the predict function of the model returns the data vector in the data basis (NOT the diagonal basis)
