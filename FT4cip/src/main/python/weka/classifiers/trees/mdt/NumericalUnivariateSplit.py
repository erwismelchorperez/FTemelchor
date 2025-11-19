import numpy as np
from typing import List, Optional, Any
import warnings

# Placeholder imports for WEKA equivalents
from weka_placeholder_classes import (
    GainRatioSplit, Distribution, GainRatioSplitCrit,
    Instances, Instance, Attribute, Utils, InfoGainSplitCrit
)

class NumericalUnivariateSplit(GainRatioSplit):
    """
    Univariate split for numerical attributes.
    Translated from Java to Python.
    """

    def __init__(self, attribute: Attribute, info_gain: bool = False):
        """
        Initialize NumericalUnivariateSplit.

        Args:
            attribute: The numerical attribute to split on
            info_gain: Whether to use information gain
        """
        super().__init__(info_gain)
        self.attribute = attribute
        self.current_index = -1
        self.last_class_value = 0.0
        self.cut_point = 0.0
        self.split_index = -1
        self.best_eval = -float('inf')
        self.m_distribution = None
        self.m_num_subsets = 0

    def build_classifier(self, instances: Instances) -> None:
        """
        Build the classifier by finding the best split point.

        Args:
            instances: Training instances
        """
        try:
            # Sort instances by the attribute
            sorted_instances = self._sort_instances_by_attribute(instances, self.attribute)

            # Initialize distribution
            self.m_distribution = Distribution(2, instances.num_classes())
            self.m_distribution.add_range(1, sorted_instances, 0, sorted_instances.num_instances())

            self.current_index = -1
            self.split_index = -1
            self.last_class_value = self._find_next_class(sorted_instances, 0)

            # Calculate default entropy
            default_ent = -float('inf')
            if self.info_gain:
                default_ent = self.info_gain_crit.old_ent(self.m_distribution)

            self.best_eval = -float('inf')

            # Find the best split point
            while self._find_next(sorted_instances):
                current_eval = 0.0
                if self.info_gain:
                    current_eval = self.info_gain_crit.split_crit_value(
                        self.m_distribution, instances.sum_of_weights(), default_ent)
                else:
                    current_eval = self.split_criterion.split_crit_value(self.m_distribution)

                if current_eval > self.best_eval:
                    self.split_index = self.current_index
                    self.best_eval = current_eval

            # Calculate cut point
            if self.split_index < len(sorted_instances.data) - 1:
                if self._is_missing(sorted_instances.data[self.split_index + 1], self.attribute):
                    self.cut_point = self._get_value(sorted_instances.data[self.split_index], self.attribute)
                else:
                    value1 = self._get_value(sorted_instances.data[self.split_index], self.attribute)
                    value2 = self._get_value(sorted_instances.data[self.split_index + 1], self.attribute)
                    self.cut_point = (value1 + value2) / 2.0
            else:
                self.cut_point = self._get_value(sorted_instances.data[self.split_index], self.attribute)

            # Create final distribution for the best split
            self.m_distribution = Distribution(2, instances.num_classes())
            self.m_distribution.add_range(0, sorted_instances, 0, self.split_index + 1)
            self.m_distribution.add_range(1, sorted_instances, self.split_index + 1,
                                        sorted_instances.num_instances())
            self.m_num_subsets = 2

            # Calculate gain ratio if using information gain
            if self.info_gain:
                self.best_eval -= (Utils.log2(self.split_index) / instances.sum_of_weights())
                self.m_gain_ratio = self.split_criterion.split_crit_value(
                    self.m_distribution, instances.sum_of_weights(), self.best_eval)

        except Exception as e:
            warnings.warn(f"Error building numerical univariate split: {e}")

    def _sort_instances_by_attribute(self, instances: Instances, attribute: Attribute) -> Instances:
        """
        Sort instances by the given attribute.

        Args:
            instances: Instances to sort
            attribute: Attribute to sort by

        Returns:
            Instances: Sorted instances
        """
        # Create a list of (value, instance) tuples
        sorted_data = sorted(
            [(self._get_value(instance, attribute), instance) for instance in instances.data],
            key=lambda x: x[0]
        )

        # Create new instances with sorted data
        sorted_instances = Instances()
        sorted_instances.data = [instance for _, instance in sorted_data]
        sorted_instances.class_index = instances.class_index
        return sorted_instances

    def _get_value(self, instance: Instance, attribute: Attribute) -> float:
        """
        Get the numerical value of the attribute for the given instance.

        Args:
            instance: The instance
            attribute: The attribute

        Returns:
            float: The attribute value
        """
        # Placeholder implementation
        # In practice, you'd use instance.value(attribute)
        return instance.values[attribute.index]

    def _is_missing(self, instance: Instance, attribute: Attribute) -> bool:
        """
        Check if the attribute value is missing for the instance.

        Args:
            instance: The instance
            attribute: The attribute

        Returns:
            bool: True if value is missing
        """
        # Placeholder implementation
        # In practice, you'd use instance.isMissing(attribute)
        return False

    def _find_next(self, instances: Instances) -> bool:
        """
        Find the next potential split point.

        Args:
            instances: Sorted instances

        Returns:
            bool: True if a valid split point was found
        """
        if self.current_index >= instances.num_instances() - 1:
            return False

        for self.current_index in range(self.current_index + 1, instances.num_instances() - 1):
            instance = instances.data[self.current_index]

            # Skip if value is missing
            if self._is_missing(instance, self.attribute):
                return False

            # Shift distribution (placeholder implementation)
            # self.m_distribution.shift(1, 0, instance)

            current_value = self._get_value(instance, self.attribute)
            next_value = self._get_value(instances.data[self.current_index + 1], self.attribute)

            if current_value != next_value:
                next_class_value = self._find_next_class(instances, self.current_index + 1)
                if (self.last_class_value != next_class_value or
                    next_class_value == -1):
                    self.last_class_value = next_class_value
                    return True

        return False

    def _find_next_class(self, instances: Instances, index: int) -> float:
        """
        Find the next distinct class value.

        Args:
            instances: Sorted instances
            index: Starting index

        Returns:
            float: Next class value, or -1 if conflicting classes found
        """
        if index >= instances.num_instances():
            return -1

        current_class = instances.data[index].class_value
        current_value = self._get_value(instances.data[index], self.attribute)

        idx = index + 1
        while (idx < instances.num_instances() and
               current_value == self._get_value(instances.data[idx], self.attribute)):
            if instances.data[idx].class_value != current_class:
                return -1
            idx += 1

        return current_class

    def left_side(self, instances: Instances) -> str:
        """
        Return left side of the split description.

        Args:
            instances: Training instances

        Returns:
            str: Left side description
        """
        return self.attribute.name

    def right_side(self, subset_index: int, instances: Instances) -> str:
        """
        Return right side of the split description.

        Args:
            subset_index: Index of the subset (0 or 1)
            instances: Training instances

        Returns:
            str: Right side description
        """
        if subset_index == 0:
            return f" <= {self.cut_point:.6f}"
        else:
            return f" > {self.cut_point:.6f}"

    def source_expression(self, subset_index: int, instances: Instances) -> Optional[str]:
        """
        Return source expression for the split.

        Args:
            subset_index: Index of the subset
            instances: Training instances

        Returns:
            str: Source expression, or None if not implemented
        """
        return None

    def weights(self, instance: Instance) -> List[float]:
        """
        Return weights for the instance.

        Args:
            instance: Instance to get weights for

        Returns:
            list: Empty list (not implemented)
        """
        return []

    def which_subset(self, instance: Instance) -> int:
        """
        Determine which subset the instance belongs to.

        Args:
            instance: Instance to classify

        Returns:
            int: Subset index (0 if <= cut point, 1 otherwise)
        """
        try:
            instance_value = self._get_value(instance, self.attribute)

            if instance_value <= self.cut_point:
                return 0
            else:
                return 1
        except Exception as e:
            warnings.warn(f"Error determining subset: {e}")
            return 0  # Default to left subset

    def get_revision(self) -> Optional[str]:
        """
        Get revision string.

        Returns:
            str: Revision string, or None if not implemented
        """
        return None

    @property
    def num_subsets(self) -> int:
        """Get number of subsets."""
        return self.m_num_subsets

    def distribution(self):
        """Get the distribution of the split."""
        return self.m_distribution

# Enhanced implementation with scikit-learn compatibility
class EnhancedNumericalUnivariateSplit(NumericalUnivariateSplit):
    """
    Enhanced version with additional functionality and scikit-learn compatibility.
    """

    def __init__(self, attribute: Attribute, info_gain: bool = False):
        super().__init__(attribute, info_gain)
        self.feature_importances_ = None
        self.split_quality = 0.0
        self._sorted_values = None
        self._sorted_classes = None

    def fit(self, X, y, sample_weight=None):
        """
        Fit the split using numpy arrays (scikit-learn compatibility).

        Args:
            X: Feature matrix (numpy array)
            y: Target values
            sample_weight: Sample weights
        """
        # Convert to Instances format (simplified)
        instances = self._array_to_instances(X, y, sample_weight)
        self.build_classifier(instances)

        # Calculate feature importance
        self._calculate_feature_importance()

        return self

    def _array_to_instances(self, X, y, sample_weight=None):
        """
        Convert numpy arrays to Instances format.

        Args:
            X: Feature matrix
            y: Target values
            sample_weight: Sample weights

        Returns:
            Instances: WEKA Instances object
        """
        instances = Instances()
        instances.class_index = X.shape[1]  # Assume class is last

        for i in range(len(X)):
            instance = Instance(len(X[i]) + 1)  # +1 for class
            for j, value in enumerate(X[i]):
                instance.set_value(j, value)
            instance.set_value(len(X[i]), y[i])  # Set class value

            if sample_weight is not None:
                instance.set_weight(sample_weight[i])
            else:
                instance.set_weight(1.0)

            instances.data.append(instance)

        return instances

    def _calculate_feature_importance(self):
        """Calculate feature importance based on split quality."""
        if self.m_distribution is not None:
            # Simple importance based on gain ratio
            self.feature_importances_ = np.array([self.m_gain_ratio])
            self.split_quality = self.m_gain_ratio
        else:
            self.feature_importances_ = np.array([0.0])
            self.split_quality = 0.0

    def predict(self, X):
        """
        Predict subsets for instances (scikit-learn compatibility).

        Args:
            X: Feature matrix

        Returns:
            numpy.ndarray: Predicted subsets
        """
        predictions = []
        for i in range(len(X)):
            # Create instance from row
            instance = Instance(len(X[i]))
            for j, value in enumerate(X[i]):
                instance.set_value(j, value)

            subset = self.which_subset(instance)
            predictions.append(subset)

        return np.array(predictions)

    def get_split_quality(self) -> float:
        """
        Get the quality of the split.

        Returns:
            float: Split quality metric
        """
        return self.split_quality

    def get_split_info(self) -> dict:
        """
        Get detailed information about the split.

        Returns:
            dict: Split information
        """
        return {
            'attribute': self.attribute.name,
            'cut_point': self.cut_point,
            'gain_ratio': self.m_gain_ratio,
            'split_index': self.split_index,
            'num_subsets': self.num_subsets
        }

# Utility function for working with numerical splits
def create_numerical_split_from_data(attribute_name: str, data: np.ndarray,
                                    target: np.ndarray, use_info_gain: bool = False) -> NumericalUnivariateSplit:
    """
    Create a numerical split from numpy data.

    Args:
        attribute_name: Name of the attribute
        data: Feature data (1D array for this attribute)
        target: Target values
        use_info_gain: Whether to use information gain

    Returns:
        NumericalUnivariateSplit: Configured split object
    """
    # Create attribute
    attribute = Attribute(attribute_name)
    attribute.is_numeric = True

    # Create enhanced split
    split = EnhancedNumericalUnivariateSplit(attribute, use_info_gain)

    # Fit the split
    X = data.reshape(-1, 1)  # Reshape to 2D
    split.fit(X, target)

    return split

# Example usage and demonstration
def demonstrate_numerical_split():
    """Demonstrate usage of NumericalUnivariateSplit"""

    # Create sample data
    age_data = np.array([25, 30, 35, 40, 45, 50, 55, 60])
    target = np.array([0, 0, 0, 1, 1, 1, 1, 1])

    # Create numerical split
    age_attribute = Attribute("age")
    age_attribute.is_numeric = True

    split = NumericalUnivariateSplit(age_attribute, info_gain=True)

    # Create instances (simplified)
    instances = Instances()
    instances.class_index = 1  # Class is second attribute

    for i, age in enumerate(age_data):
        instance = Instance(2)  # 2 attributes: age and class
        instance.set_value(0, age)
        instance.set_value(1, target[i])
        instances.data.append(instance)

    # Build the split
    split.build_classifier(instances)

    # Print results
    print(f"Best cut point: {split.cut_point:.2f}")
    print(f"Gain ratio: {split.m_gain_ratio:.4f}")
    print(f"Left side: {split.left_side(instances)}")
    print(f"Right side for subset 0: {split.right_side(0, instances)}")
    print(f"Right side for subset 1: {split.right_side(1, instances)}")

    # Test classification
    test_instance = Instance(2)
    test_instance.set_value(0, 32)
    subset = split.which_subset(test_instance)
    print(f"Test instance age 32 belongs to subset: {subset}")

    test_instance.set_value(0, 42)
    subset = split.which_subset(test_instance)
    print(f"Test instance age 42 belongs to subset: {subset}")

    # Get split information
    if isinstance(split, EnhancedNumericalUnivariateSplit):
        split_info = split.get_split_info()
        print(f"Split info: {split_info}")

def advanced_numerical_split_demo():
    """Advanced demonstration with real-world data pattern"""

    # Create data with clear separation
    np.random.seed(42)

    # Class 0: lower values
    class0 = np.random.normal(50, 10, 100)
    # Class 1: higher values
    class1 = np.random.normal(70, 10, 100)

    data = np.concatenate([class0, class1])
    target = np.concatenate([np.zeros(100), np.ones(100)])

    # Create and fit split
    split = create_numerical_split_from_data(
        "feature",
        data,
        target,
        use_info_gain=True
    )

    print(f"Optimal cut point: {split.cut_point:.2f}")
    print(f"Split quality (gain ratio): {split.m_gain_ratio:.4f}")

    # Test the split
    test_values = [45, 60, 75]
    for val in test_values:
        test_instance = Instance(1)
        test_instance.set_value(0, val)
        subset = split.which_subset(test_instance)
        print(f"Value {val} -> Subset {subset}")

if __name__ == "__main__":
    print("Basic Numerical Split Demo:")
    demonstrate_numerical_split()

    print("\nAdvanced Numerical Split Demo:")
    advanced_numerical_split_demo()
