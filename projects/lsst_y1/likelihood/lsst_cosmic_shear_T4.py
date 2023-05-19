from cobaya.likelihoods.lsst_y1._cosmolike_prototype_base import _cosmolike_prototype_base
import cosmolike_lsst_y1_interface as ci
import numpy as np

class lsst_cosmic_shear_T4(_cosmolike_prototype_base):
    def initialize(self):
        super(lsst_cosmic_shear_T4,self).initialize(probe="xi")

    def logp(self, **params_values):
        datavector = self.internal_get_datavector(**params_values)
        return 0.25 * self.compute_logp(datavector) #temperature T=4
    
    def get_datavector(self, **params_values):        
        datavector = self.internal_get_datavector(**params_values)
        return np.array(datavector)

    def internal_get_datavector(self, **params_values):
        if self.create_baryon_pca:
            self.generate_baryonic_PCA(**params_values)
            self.force_cache_false = True

        self.set_cosmo_related()
                
        self.set_source_related(**params_values)
        
        if self.create_baryon_pca:
            self.force_cache_false = False

        # datavector C++ returns a list (not numpy array)
        datavector = np.array(ci.compute_data_vector_masked())
        
        if self.use_baryon_pca:
            self.set_baryon_related(**params_values)
            datavector = self.add_baryon_pcs_to_datavector(datavector)

        if self.print_datavector:
          size = len(datavector)
          out = np.zeros(shape=(size, 2))
          out[:,0] = np.arange(0, size)
          out[:,1] = datavector
          fmt = '%d', '%1.8e'
          np.savetxt(self.print_datavector_file, out, fmt = fmt)

        return datavector
