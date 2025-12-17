from abc import ABC, abstractmethod

class FeatureSelectionSplit(ABC):
    """
    Interface for feature selection split criteria.
    Translated from Java to Python.

    This interface defines a method to determine when to break
    the feature selection process during tree construction.
    """

    @abstractmethod
    def break_feature_selection(self) -> bool:
        """
        Determines whether to break the feature selection process.

        Returns:
            bool: True if feature selection should be stopped, False otherwise
        """
        pass
