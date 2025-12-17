import numpy as np
from scipy.stats import multivariate_normal

class MyMultivariateGaussianEstimator:
    def __init__(self):
        self.mean = None
        self.covariance = None
        self.covariance_inverse = None
        self.ln_constant = None

    def fit(self, data):
        """Ajusta el estimador a los datos"""
        self.mean = np.mean(data, axis=0)
        self.covariance = np.cov(data.T)
        self.covariance_inverse = inv(self.covariance)
        # Calcular término constante
        n_features = data.shape[1]
        self.ln_constant = -0.5 * (n_features * np.log(2 * np.pi) + np.log(np.linalg.det(self.covariance)))

    def get_covariance_inverse(self):
        return self.covariance_inverse

    def get_mean_vector(self):
        return self.mean

    def get_ln_constant(self):
        return self.ln_constant
