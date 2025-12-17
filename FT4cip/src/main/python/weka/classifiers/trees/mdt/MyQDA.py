import numpy as np
import scipy.linalg as la
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array
import warnings

# Import previously defined classes
from my_multivariate_gaussian_estimator import MyMultivariateGaussianEstimator

class MyQDA(QuadraticDiscriminantAnalysis):
    """
    Extended Quadratic Discriminant Analysis with additional functionality.
    Translated from Java to Python.
    """

    def __init__(self, ridge=1e-4):
        """
        Initialize MyQDA.

        Args:
            ridge: Ridge parameter for covariance estimation
        """
        super().__init__(reg_param=ridge)
        self.ridge = ridge
        self.m_estimators = None
        self.m_log_priors = None
        self.m_remove_useless = None
        self.m_data = None

    def get_m_estimators(self):
        """
        Get the multivariate Gaussian estimators.

        Returns:
            List[MyMultivariateGaussianEstimator]: List of estimators for each class
        """
        return self.m_estimators

    def get_m_log_priors(self):
        """
        Get the log prior probabilities.

        Returns:
            numpy.ndarray: Array of log priors for each class
        """
        return self.m_log_priors

    def discriminant_test(self):
        """
        Perform discriminant analysis test and print results.
        This method analyzes the discriminant function between two classes.
        """
        if (self.m_estimators is None or len(self.m_estimators) < 2 or
            self.m_estimators[0] is None or self.m_estimators[1] is None):
            print("Not enough estimators for discriminant test")
            return

        left_estimator = self.m_estimators[1]
        right_estimator = self.m_estimators[0]

        # Get covariance inverses and means
        left_covariance_inverse = left_estimator.get_covariance_inverse()
        right_covariance_inverse = right_estimator.get_covariance_inverse()
        left_mean = left_estimator.get_mean_vector()
        right_mean = right_estimator.get_mean_vector()
        left_constant = left_estimator.get_ln_constant()
        right_constant = right_estimator.get_ln_constant()

        # Calculate cut point
        cut_point = left_constant - right_constant

        # Add quadratic terms
        try:
            right_quad_term = 0.5 * right_mean.T @ right_covariance_inverse @ right_mean
            left_quad_term = 0.5 * left_mean.T @ left_covariance_inverse @ left_mean
            cut_point += right_quad_term - left_quad_term
        except Exception as e:
            warnings.warn(f"Error calculating quadratic terms: {e}")

        print(f"Cut point: {cut_point}")

        # Calculate linear coefficients
        try:
            linear_coefficients = (right_covariance_inverse @ right_mean -
                                 left_covariance_inverse @ left_mean)
            print(f"Linear coefficients: {linear_coefficients}")
        except Exception as e:
            warnings.warn(f"Error calculating linear coefficients: {e}")

        # Calculate quadratic coefficients
        try:
            quadratic_coefficients = right_covariance_inverse - left_covariance_inverse
            print(f"Quadratic coefficients shape: {quadratic_coefficients.shape}")
        except Exception as e:
            warnings.warn(f"Error calculating quadratic coefficients: {e}")

    def fit(self, X, y, sample_weight=None):
        """
        Build the QDA classifier.

        Args:
            X: Feature matrix
            y: Target values
            sample_weight: Sample weights

        Returns:
            self: Fitted estimator
        """
        X, y = check_X_y(X, y)

        # Remove constant attributes (simplified implementation)
        X = self._remove_constant_attributes(X)

        # Remove samples with missing target values
        X, y = self._remove_missing_target(X, y)

        # Get class information
        classes = np.unique(y)
        n_classes = len(classes)

        if n_classes < 2:
            raise ValueError("Need at least 2 classes for QDA")

        # Calculate class counts and weights
        class_counts = np.zeros(n_classes, dtype=int)
        sum_weights_per_class = np.zeros(n_classes)

        for i, cls in enumerate(classes):
            mask = (y == cls)
            class_counts[i] = np.sum(mask)
            if sample_weight is not None:
                sum_weights_per_class[i] = np.sum(sample_weight[mask])
            else:
                sum_weights_per_class[i] = class_counts[i]

        # Organize data by class
        data_by_class = []
        weights_by_class = []

        for i, cls in enumerate(classes):
            mask = (y == cls)
            class_data = X[mask]
            data_by_class.append(class_data)

            if sample_weight is not None:
                class_weights = sample_weight[mask]
            else:
                class_weights = np.ones(class_data.shape[0])
            weights_by_class.append(class_weights)

        # Create estimators for each class
        self.m_estimators = [None] * n_classes

        for i in range(n_classes):
            if sum_weights_per_class[i] > 0:
                estimator = MyMultivariateGaussianEstimator()
                estimator.set_ridge(self.ridge)
                estimator.estimate(data_by_class[i], weights_by_class[i])
                self.m_estimators[i] = estimator

        # Calculate log priors
        total_weight = np.sum(sum_weights_per_class)
        self.m_log_priors = np.zeros(n_classes)

        for i in range(n_classes):
            if sum_weights_per_class[i] > 0:
                self.m_log_priors[i] = (np.log(sum_weights_per_class[i]) -
                                      np.log(total_weight))

        # Store class information
        self.classes_ = classes
        self.m_data = X  # Store feature data for reference

        # Also call parent fit for scikit-learn compatibility
        try:
            super().fit(X, y, sample_weight)
        except Exception as e:
            warnings.warn(f"Parent fit failed: {e}")

        return self

    def _remove_constant_attributes(self, X):
        """
        Remove constant attributes from the data.

        Args:
            X: Feature matrix

        Returns:
            numpy.ndarray: Feature matrix with constant attributes removed
        """
        # Simplified implementation - remove columns with zero variance
        variances = np.var(X, axis=0)
        non_constant_mask = variances > 1e-10
        return X[:, non_constant_mask]

    def _remove_missing_target(self, X, y):
        """
        Remove samples with missing target values.

        Args:
            X: Feature matrix
            y: Target values

        Returns:
            tuple: (X_clean, y_clean)
        """
        # In numpy, we assume no missing values in y for simplicity
        # In practice, you might want to handle NaN values
        valid_mask = ~np.isnan(y)
        return X[valid_mask], y[valid_mask]

    def predict_proba(self, X):
        """
        Predict class probabilities for X.

        Args:
            X: Feature matrix

        Returns:
            numpy.ndarray: Class probabilities
        """
        X = check_array(X)

        if self.m_estimators is None:
            raise ValueError("Classifier not fitted")

        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        probabilities = np.zeros((n_samples, n_classes))

        for i in range(n_classes):
            if self.m_estimators[i] is not None:
                # Calculate log probability for each class
                for j in range(n_samples):
                    try:
                        log_prob = self.m_estimators[i].log_probability(X[j])
                        log_prob += self.m_log_priors[i]
                        probabilities[j, i] = log_prob
                    except Exception as e:
                        warnings.warn(f"Error calculating probability: {e}")
                        probabilities[j, i] = -np.inf

        # Convert log probabilities to probabilities
        # Subtract max for numerical stability
        max_log_probs = np.max(probabilities, axis=1, keepdims=True)
        probabilities = np.exp(probabilities - max_log_probs)

        # Normalize
        row_sums = np.sum(probabilities, axis=1, keepdims=True)
        probabilities = probabilities / row_sums

        return probabilities

    def predict(self, X):
        """
        Predict class labels for X.

        Args:
            X: Feature matrix

        Returns:
            numpy.ndarray: Predicted class labels
        """
        probabilities = self.predict_proba(X)
        return self.classes_[np.argmax(probabilities, axis=1)]


# Implementation of MyMultivariateGaussianEstimator for completeness
class MyMultivariateGaussianEstimator:
    """
    Multivariate Gaussian Estimator with additional functionality.
    """

    def __init__(self, ridge=1e-4):
        self.ridge = ridge
        self.covariance = None
        self.covariance_inverse = None
        self.mean = None
        self.ln_constant = None
        self.n_features = None

    def set_ridge(self, ridge):
        """Set ridge parameter for covariance estimation."""
        self.ridge = ridge

    def estimate(self, data, weights=None):
        """
        Estimate the multivariate Gaussian parameters.

        Args:
            data: Input data (n_samples x n_features)
            weights: Sample weights
        """
        if weights is None:
            weights = np.ones(data.shape[0])

        # Calculate weighted mean
        self.n_features = data.shape[1]
        total_weight = np.sum(weights)
        self.mean = np.average(data, axis=0, weights=weights)

        # Calculate weighted covariance
        centered_data = data - self.mean
        self.covariance = np.cov(centered_data.T, aweights=weights)

        # Apply ridge regularization
        self.covariance += np.eye(self.n_features) * self.ridge

        try:
            # Calculate inverse covariance
            self.covariance_inverse = la.inv(self.covariance)

            # Calculate log constant term
            sign, log_det = np.linalg.slogdet(self.covariance)
            self.ln_constant = -0.5 * (self.n_features * np.log(2 * np.pi) + log_det)

        except la.LinAlgError as e:
            warnings.warn(f"Matrix inversion failed: {e}")
            # Use pseudo-inverse as fallback
            self.covariance_inverse = la.pinv(self.covariance)
            self.ln_constant = -np.inf

    def get_covariance_inverse(self):
        """Get the inverse covariance matrix."""
        return self.covariance_inverse

    def get_mean_vector(self):
        """Get the mean vector."""
        return self.mean

    def get_ln_constant(self):
        """Get the log constant term."""
        return self.ln_constant

    def log_probability(self, x):
        """
        Calculate log probability density for a sample.

        Args:
            x: Input sample

        Returns:
            float: Log probability density
        """
        if self.covariance_inverse is None or self.mean is None:
            return -np.inf

        try:
            diff = x - self.mean
            exponent = -0.5 * diff.T @ self.covariance_inverse @ diff
            return self.ln_constant + exponent
        except Exception as e:
            warnings.warn(f"Error calculating log probability: {e}")
            return -np.inf

    def probability(self, x):
        """
        Calculate probability density for a sample.

        Args:
            x: Input sample

        Returns:
            float: Probability density
        """
        log_prob = self.log_probability(x)
        return np.exp(log_prob) if log_prob > -np.inf else 0.0


# Example usage and demonstration
def demonstrate_myqda():
    """Demonstrate usage of MyQDA"""

    # Generate sample data
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=1000,
        n_features=4,
        n_informative=4,
        n_redundant=0,
        n_classes=2,
        random_state=42
    )

    # Create and fit MyQDA
    myqda = MyQDA(ridge=1e-4)
    myqda.fit(X, y)

    # Get estimators and priors
    estimators = myqda.get_m_estimators()
    log_priors = myqda.get_m_log_priors()

    print(f"Number of estimators: {len(estimators)}")
    print(f"Log priors: {log_priors}")

    # Perform discriminant test
    print("\nDiscriminant Test Results:")
    myqda.discriminant_test()

    # Make predictions
    y_pred = myqda.predict(X)
    accuracy = np.mean(y_pred == y)
    print(f"\nAccuracy: {accuracy:.4f}")

    # Get probabilities
    y_proba = myqda.predict_proba(X)
    print(f"Probability shape: {y_proba.shape}")

if __name__ == "__main__":
    demonstrate_myqda()
