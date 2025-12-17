import numpy as np
from typing import Optional

# Placeholder import for WEKA Distribution
from weka_placeholder_classes import Distribution

class TwoingSplitCriterion:
    """
    Twoing split criterion for evaluating binary splits.
    Translated from Java to Python.
    """

    def split_crit_value(self, bags: Distribution) -> float:
        """
        Calculate the Twoing criterion value for a distribution.

        Args:
            bags: The distribution to evaluate

        Returns:
            float: The Twoing criterion value
        """
        # Preconditions
        if bags.num_bags() != 2:
            raise ValueError("Twoing Distance need only two child nodes (binary split)")

        parent_sum = bags.total()
        if parent_sum == bags.num_correct():
            return 0.0

        total = parent_sum

        # Calculate bag proportions
        SL = bags.per_bag(0)  # Left bag sum
        SR = bags.per_bag(1)  # Right bag sum

        # Base Twoing factor: 0.25 * (SL/total) * (SR/total)
        twoing = 0.25 * (SL / total) * (SR / total)

        # Calculate class distribution differences
        aux = 0.0
        num_classes = bags.num_classes()

        for i in range(num_classes):
            # Proportion of class i in left bag
            left_prop = bags.per_class_per_bag(0, i) / SL if SL > 0 else 0.0
            # Proportion of class i in right bag
            right_prop = bags.per_class_per_bag(1, i) / SR if SR > 0 else 0.0

            aux += abs(left_prop - right_prop)

        # Square the sum of absolute differences
        twoing *= (aux ** 2.0)

        return twoing

    def get_revision(self) -> Optional[str]:
        """
        Get revision string.

        Returns:
            str: Revision string, or None if not implemented
        """
        return None

# Enhanced implementation with additional features
class EnhancedTwoingSplitCriterion(TwoingSplitCriterion):
    """
    Enhanced Twoing split criterion with additional functionality.
    """

    def __init__(self):
        super().__init__()
        self.last_calculation_details = {}

    def split_crit_value(self, bags: Distribution) -> float:
        """
        Calculate Twoing criterion with detailed tracking.

        Args:
            bags: The distribution to evaluate

        Returns:
            float: The Twoing criterion value
        """
        try:
            value = super().split_crit_value(bags)

            # Store calculation details
            self.last_calculation_details = {
                'twoing_value': value,
                'num_bags': bags.num_bags(),
                'total_instances': bags.total(),
                'num_classes': bags.num_classes(),
                'left_bag_size': bags.per_bag(0) if bags.num_bags() > 0 else 0,
                'right_bag_size': bags.per_bag(1) if bags.num_bags() > 1 else 0
            }

            return value

        except Exception as e:
            self.last_calculation_details = {'error': str(e)}
            raise

    def get_calculation_details(self) -> dict:
        """
        Get details of the last calculation.

        Returns:
            dict: Calculation details
        """
        return self.last_calculation_details.copy()

    def evaluate_split_quality(self, bags: Distribution) -> dict:
        """
        Evaluate split quality with multiple metrics.

        Args:
            bags: The distribution to evaluate

        Returns:
            dict: Split quality metrics
        """
        twoing_value = self.split_crit_value(bags)

        # Calculate additional metrics
        total = bags.total()
        SL = bags.per_bag(0)
        SR = bags.per_bag(1)

        # Balance metric (closer to 0.5 is more balanced)
        balance_metric = min(SL, SR) / total if total > 0 else 0.0

        # Purity metric (higher is better)
        left_purity = max([bags.per_class_per_bag(0, i) / SL for i in range(bags.num_classes())]) if SL > 0 else 0.0
        right_purity = max([bags.per_class_per_bag(1, i) / SR for i in range(bags.num_classes())]) if SR > 0 else 0.0
        purity_metric = (left_purity + right_purity) / 2.0

        return {
            'twoing_value': twoing_value,
            'balance_metric': balance_metric,
            'purity_metric': purity_metric,
            'left_bag_size': SL,
            'right_bag_size': SR,
            'total_instances': total
        }

# Utility functions for working with Twoing criterion
class TwoingUtilities:
    """
    Utility functions for Twoing split criterion.
    """

    @staticmethod
    def normalize_twoing_values(values: list) -> np.ndarray:
        """
        Normalize a list of Twoing values to [0, 1] range.

        Args:
            values: List of Twoing values

        Returns:
            numpy.ndarray: Normalized values
        """
        if not values:
            return np.array([])

        values_array = np.array(values)
        if np.max(values_array) == np.min(values_array):
            return np.ones_like(values_array)

        return (values_array - np.min(values_array)) / (np.max(values_array) - np.min(values_array))

    @staticmethod
    def find_best_split(splits: list, distributions: list) -> int:
        """
        Find the best split based on Twoing criterion.

        Args:
            splits: List of split objects
            distributions: List of corresponding distributions

        Returns:
            int: Index of the best split
        """
        if len(splits) != len(distributions):
            raise ValueError("Number of splits must match number of distributions")

        criterion = TwoingSplitCriterion()
        best_index = -1
        best_value = -float('inf')

        for i, distribution in enumerate(distributions):
            try:
                value = criterion.split_crit_value(distribution)
                if value > best_value:
                    best_value = value
                    best_index = i
            except Exception:
                continue

        return best_index

    @staticmethod
    def compare_splits(distribution1: Distribution, distribution2: Distribution) -> float:
        """
        Compare two splits using Twoing criterion.

        Args:
            distribution1: First distribution
            distribution2: Second distribution

        Returns:
            float: Difference (distribution1 - distribution2)
        """
        criterion = TwoingSplitCriterion()
        value1 = criterion.split_crit_value(distribution1)
        value2 = criterion.split_crit_value(distribution2)
        return value1 - value2

# Example implementation of Distribution class for testing
class ExampleDistribution(Distribution):
    """
    Example implementation of Distribution for testing Twoing criterion.
    """

    def __init__(self, class_counts_per_bag):
        """
        Initialize with class counts per bag.

        Args:
            class_counts_per_bag: List of lists, where each inner list contains
                                 class counts for a bag
        """
        self.class_counts = class_counts_per_bag
        self.num_bags_val = len(class_counts_per_bag)
        self.num_classes_val = len(class_counts_per_bag[0]) if class_counts_per_bag else 0

    def num_bags(self) -> int:
        return self.num_bags_val

    def num_classes(self) -> int:
        return self.num_classes_val

    def total(self) -> float:
        total_sum = 0.0
        for bag in self.class_counts:
            total_sum += sum(bag)
        return total_sum

    def num_correct(self) -> float:
        # For simplicity, assume the majority class in each bag is correct
        correct = 0.0
        for bag in self.class_counts:
            if bag:
                correct += max(bag)
        return correct

    def per_bag(self, bag_index: int) -> float:
        if 0 <= bag_index < self.num_bags_val:
            return sum(self.class_counts[bag_index])
        return 0.0

    def per_class_per_bag(self, bag_index: int, class_index: int) -> float:
        if (0 <= bag_index < self.num_bags_val and
            0 <= class_index < self.num_classes_val):
            return self.class_counts[bag_index][class_index]
        return 0.0

# Example usage and demonstration
def demonstrate_twoing_criterion():
    """Demonstrate usage of TwoingSplitCriterion"""

    # Create example distributions
    # Distribution 1: Good split (clear separation)
    good_split = ExampleDistribution([
        [80, 20],  # Bag 0: mostly class 0
        [10, 90]   # Bag 1: mostly class 1
    ])

    # Distribution 2: Poor split (similar distributions)
    poor_split = ExampleDistribution([
        [55, 45],  # Bag 0: mixed
        [45, 55]   # Bag 1: mixed
    ])

    # Distribution 3: Perfect split
    perfect_split = ExampleDistribution([
        [100, 0],  # Bag 0: all class 0
        [0, 100]   # Bag 1: all class 1
    ])

    # Create criterion
    criterion = TwoingSplitCriterion()
    enhanced_criterion = EnhancedTwoingSplitCriterion()

    # Calculate Twoing values
    good_value = criterion.split_crit_value(good_split)
    poor_value = criterion.split_crit_value(poor_split)
    perfect_value = criterion.split_crit_value(perfect_split)

    print("Twoing Criterion Demonstration:")
    print(f"Good split Twoing value: {good_value:.6f}")
    print(f"Poor split Twoing value: {poor_value:.6f}")
    print(f"Perfect split Twoing value: {perfect_value:.6f}")

    # Use enhanced criterion for detailed analysis
    good_quality = enhanced_criterion.evaluate_split_quality(good_split)
    print(f"\nGood split quality metrics: {good_quality}")

    # Compare splits
    comparison = TwoingUtilities.compare_splits(good_split, poor_split)
    print(f"\nComparison (good - poor): {comparison:.6f}")

    # Find best split
    splits = [good_split, poor_split, perfect_split]
    best_index = TwoingUtilities.find_best_split(['split1', 'split2', 'split3'], splits)
    print(f"Best split index: {best_index}")

def advanced_twoing_analysis():
    """Advanced analysis using Twoing criterion"""

    # Create multiple distributions with varying quality
    distributions = []
    qualities = []

    for i in range(5):
        # Vary the separation between classes
        separation = (i + 1) * 0.2
        class0_bag0 = int(100 * (0.5 + separation/2))
        class1_bag0 = 100 - class0_bag0
        class0_bag1 = int(100 * (0.5 - separation/2))
        class1_bag1 = 100 - class0_bag1

        dist = ExampleDistribution([
            [class0_bag0, class1_bag0],
            [class0_bag1, class1_bag1]
        ])
        distributions.append(dist)

        # Calculate quality
        criterion = TwoingSplitCriterion()
        quality = criterion.split_crit_value(dist)
        qualities.append(quality)

    # Normalize qualities
    normalized = TwoingUtilities.normalize_twoing_values(qualities)

    print("\nAdvanced Twoing Analysis:")
    for i, (quality, norm_quality) in enumerate(zip(qualities, normalized)):
        print(f"Distribution {i+1}: Twoing={quality:.6f}, Normalized={norm_quality:.6f}")

if __name__ == "__main__":
    demonstrate_twoing_criterion()
    advanced_twoing_analysis()
