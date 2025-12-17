"""
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
"""

import math
import numpy as np
from typing import List, Optional, Tuple, Any, Dict
from abc import ABC, abstractmethod
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import warnings
from collections import defaultdict

# Placeholder imports for equivalent Python classes
class Instances:
    def __init__(self, data=None, capacity=0):
        self.data = data if data else []
        self.class_index = -1
        self.attributes = []

    def numInstances(self) -> int:
        return len(self.data)

    def numAttributes(self) -> int:
        return len(self.attributes)

    def numClasses(self) -> int:
        return len(set(instance.class_value for instance in self.data))

    def trainCV(self, folds: int, fold: int):
        # Simplified cross-validation split
        return self

    def testCV(self, folds: int, fold: int):
        # Simplified cross-validation split
        return self

    def stratify(self, folds: int):
        # Placeholder for stratification
        pass

    def classAttribute(self):
        # Placeholder
        return None

    def sumOfWeights(self) -> float:
        return sum(instance.weight for instance in self.data)

    def setClassIndex(self, index: int):
        self.class_index = index

class Instance:
    def __init__(self, num_attributes):
        self.values = [0.0] * num_attributes
        self.weight = 1.0
        self.class_value = 0

    def value(self, index: int) -> float:
        return self.values[index]

    def setValue(self, index: int, value: float):
        self.values[index] = value

    def setWeight(self, weight: float):
        self.weight = weight

    def numAttributes(self) -> int:
        return len(self.values)

    def setDataset(self, dataset):
        self.dataset = dataset

class Attribute:
    def __init__(self, name, is_numeric=True):
        self.name = name
        self.is_numeric = is_numeric

class ModelSelection:
    def selectModel(self, data: Instances):
        # Placeholder implementation
        return ClassifierSplitModel()

class ClassifierSplitModel:
    def numSubsets(self) -> int:
        return 2  # Default binary split

    def split(self, data: Instances) -> List[Instances]:
        # Placeholder - split data into subsets
        return [Instances(data.data[:len(data.data)//2]),
                Instances(data.data[len(data.data)//2:])]

    def whichSubset(self, instance: Instance) -> int:
        # Placeholder - determine which subset instance belongs to
        return 0

    def distribution(self):
        # Placeholder
        return Distribution()

    def leftSide(self, data: Instances) -> str:
        return "left_side"

    def rightSide(self, i: int, data: Instances) -> str:
        return f"right_side_{i}"

class Distribution:
    def numIncorrect(self) -> float:
        return 0.0

    def perBag(self, i: int) -> float:
        return 1.0

    def total(self) -> float:
        return 1.0

class CompareQuadraticNode:
    """
    Auxiliary class for comparing QuadraticLMTNodes
    """
    def compare(self, o1: 'QuadraticLMTNode', o2: 'QuadraticLMTNode') -> int:
        if o1.m_alpha < o2.m_alpha:
            return -1
        if o1.m_alpha > o2.m_alpha:
            return 1
        return 0

class QuadraticLMTNode:
    """
    Class for logistic model tree structure using Quadratic Discriminant Analysis.
    Translated from Java to Python.
    """

    # Static variable
    m_numFoldsPruning = 5

    def __init__(self, modelSelection: ModelSelection, minNumInstances: int, nominalToBinary):
        self.m_modelSelection = modelSelection
        self.m_minNumInstances = minNumInstances
        self.m_nominalToBinary = nominalToBinary

        # Initialize instance variables
        self.m_totalInstanceWeight = 0.0
        self.m_id = 0
        self.m_leafModelNum = 0
        self.m_alpha = 0.0
        self.m_numIncorrectModel = 0.0
        self.m_numIncorrectTree = 0.0
        self.m_numInstances = 0
        self.m_localModel = None
        self.m_sons = None
        self.m_isLeaf = True
        self.m_train = None
        self.m_numClasses = 0
        self.qda_model = None
        self.filtered_qda = None
        self.filter_qda = None
        self.m_numericDataHeader = None
        self.m_numericData = None

    def buildClassifier(self, data: Instances) -> None:
        """
        Method for building a logistic model tree (only called for the root node).
        Grows an initial tree and prunes it back using the CART pruning scheme.
        """
        # Cross-validate alpha-parameter for CART-pruning
        cvData = Instances(data)
        cvData.stratify(self.m_numFoldsPruning)

        alphas = [None] * self.m_numFoldsPruning
        errors = [None] * self.m_numFoldsPruning

        for i in range(self.m_numFoldsPruning):
            # For every fold, grow tree on training set...
            train = cvData.trainCV(self.m_numFoldsPruning, i)
            test = cvData.testCV(self.m_numFoldsPruning, i)

            self.buildTree(train, train.numInstances())

            numNodes = self.getNumInnerNodes()
            alphas[i] = [0.0] * (numNodes + 2)
            errors[i] = [0.0] * (numNodes + 2)

            # ... then prune back and log alpha-values and errors on test set
            self.prune(alphas[i], errors[i], test)

        # Don't need CV data anymore
        cvData = None

        # Build tree using all the data
        self.buildTree(data, data.numInstances())
        numNodes = self.getNumInnerNodes()

        treeAlphas = [0.0] * (numNodes + 2)

        # Prune back and log alpha-values
        iterations = self.prune(treeAlphas, None, None)

        treeErrors = [0.0] * (numNodes + 2)

        for i in range(iterations + 1):
            # Compute midpoint alphas
            alpha = math.sqrt(treeAlphas[i] * treeAlphas[i + 1])
            error = 0.0

            # Compute error estimate from cross-validation
            for k in range(self.m_numFoldsPruning):
                l = 0
                while alphas[k][l] <= alpha:
                    l += 1
                error += errors[k][l - 1]

            treeErrors[i] = error

        # Find best alpha
        best = -1
        bestError = float('inf')
        for i in range(iterations, -1, -1):
            if treeErrors[i] < bestError:
                bestError = treeErrors[i]
                best = i

        bestAlpha = math.sqrt(treeAlphas[best] * treeAlphas[best + 1])

        # "Unprune" final tree
        self.unprune()

        # CART-prune it with best alpha
        self.prune(bestAlpha)

    def buildTree(self, data: Instances, totalInstanceWeight: float) -> None:
        """
        Method for building the tree structure. Builds a QDA model, splits the
        node and recursively builds tree for child nodes.
        """
        # Save some stuff
        self.m_totalInstanceWeight = totalInstanceWeight
        self.m_train = data
        self.m_isLeaf = True
        self.m_sons = None

        self.m_numInstances = self.m_train.numInstances()
        self.m_numClasses = self.m_train.numClasses()

        # Get numeric version of the training data
        self.m_numericData = self.getNumericData(self.m_train)

        # Build QDA model with feature selection
        remaining_attributes = list(range(self.m_train.numAttributes()))
        if self.m_train.class_index in remaining_attributes:
            remaining_attributes.remove(self.m_train.class_index)

        # Filter numeric attributes only
        remaining_attributes = [i for i in remaining_attributes
                              if self._is_attribute_numeric(i)]

        selected_attributes = []
        best_incorrect = float('inf')

        while remaining_attributes:
            best_attribute = -1
            candidate_attributes = selected_attributes.copy()

            for i, attr_idx in enumerate(remaining_attributes):
                current_candidate = candidate_attributes + [attr_idx, self.m_train.class_index]

                try:
                    # Filter data to selected attributes
                    filtered_data = self._filter_attributes(self.m_train, current_candidate)

                    # Build QDA model
                    candidate_model = QuadraticDiscriminantAnalysis()
                    X, y = self._instances_to_arrays(filtered_data)
                    candidate_model.fit(X, y)

                    # Evaluate model
                    y_pred = candidate_model.predict(X)
                    incorrect = 1 - accuracy_score(y, y_pred)

                    if incorrect < best_incorrect:
                        best_incorrect = incorrect
                        best_attribute = i
                        self.qda_model = candidate_model
                        self.filtered_qda = filtered_data
                        # Store filter information
                        self.filter_qda = current_candidate

                except Exception as e:
                    continue

            if best_attribute >= 0:
                selected_attributes.append(remaining_attributes[best_attribute])
                remaining_attributes.pop(best_attribute)
            else:
                break

        # Store performance of model at this node
        if self.qda_model is not None and self.filtered_qda is not None:
            X, y = self._instances_to_arrays(self.filtered_qda)
            y_pred = self.qda_model.predict(X)
            self.m_numIncorrectModel = (1 - accuracy_score(y, y_pred)) * len(y)
        else:
            self.m_numIncorrectModel = float('inf')

        # Split node if more than minNumInstances...
        grow = False
        if self.m_numInstances > self.m_minNumInstances:
            self.m_localModel = self.m_modelSelection.selectModel(self.m_train)
            grow = (self.m_localModel.numSubsets() > 1)

        if grow:
            # Create and build children of node
            self.m_isLeaf = False
            localInstances = self.m_localModel.split(self.m_train)

            # Clean up
            self.cleanup()

            self.m_sons = [None] * self.m_localModel.numSubsets()
            for i in range(len(self.m_sons)):
                self.m_sons[i] = QuadraticLMTNode(
                    self.m_modelSelection, self.m_minNumInstances, self.m_nominalToBinary
                )
                self.m_sons[i].buildTree(localInstances[i], self.m_totalInstanceWeight)
                localInstances[i] = None
        else:
            self.cleanup()

    def _is_attribute_numeric(self, attr_index: int) -> bool:
        """Check if attribute is numeric (placeholder implementation)"""
        # In practice, you'd check the actual attribute type
        return True

    def _filter_attributes(self, data: Instances, attribute_indices: List[int]) -> Instances:
        """Filter instances to include only specified attributes"""
        # Simplified implementation - in practice you'd implement proper filtering
        filtered = Instances()
        filtered.data = data.data  # This is a simplification
        return filtered

    def _instances_to_arrays(self, data: Instances) -> Tuple[np.ndarray, np.ndarray]:
        """Convert Instances to numpy arrays for scikit-learn"""
        if not data.data:
            return np.array([]), np.array([])

        X = []
        y = []

        for instance in data.data:
            features = []
            for i in range(instance.numAttributes()):
                if i != data.class_index:
                    features.append(instance.value(i))
            X.append(features)
            y.append(instance.class_value)

        return np.array(X), np.array(y)

    def prune(self, alpha: float) -> None:
        """
        Prunes the tree using the CART pruning scheme with given cost-complexity parameter.
        """
        comparator = CompareQuadraticNode()

        # Determine training error of models and subtrees, calculate alpha-values
        self.treeErrors()
        self.calculateAlphas()

        # Get list of all inner nodes in the tree
        nodeList = self.getNodes()

        prune_flag = (len(nodeList) > 0)

        while prune_flag:
            # Select node with minimum alpha
            nodeToPrune = min(nodeList, key=lambda x: x.m_alpha)

            # Prune if its alpha is smaller than alpha
            if nodeToPrune.m_alpha > alpha:
                break

            nodeToPrune.m_isLeaf = True
            nodeToPrune.m_sons = None

            # Update tree errors and alphas
            self.treeErrors()
            self.calculateAlphas()

            nodeList = self.getNodes()
            prune_flag = (len(nodeList) > 0)

    def prune(self, alphas: List[float], errors: List[float], test: Instances) -> int:
        """
        Method for performing one fold in cross-validation of cost-complexity parameter.
        """
        comparator = CompareQuadraticNode()

        # Determine training error and calculate alpha-values
        self.treeErrors()
        self.calculateAlphas()

        # Get list of all inner nodes
        nodeList = self.getNodes()

        prune_flag = (len(nodeList) > 0)

        # alpha_0 is always zero (unpruned tree)
        alphas[0] = 0.0

        # Error of unpruned tree
        if errors is not None and test is not None:
            test_error = self._evaluate_on_test(test)
            errors[0] = test_error

        iteration = 0
        while prune_flag:
            iteration += 1

            # Get node with minimum alpha
            nodeToPrune = min(nodeList, key=lambda x: x.m_alpha)

            nodeToPrune.m_isLeaf = True
            # Note: Don't set m_sons to None to allow unpruning

            # Get alpha-value of node
            alphas[iteration] = nodeToPrune.m_alpha

            # Log error
            if errors is not None and test is not None:
                test_error = self._evaluate_on_test(test)
                errors[iteration] = test_error

            # Update errors/alphas
            self.treeErrors()
            self.calculateAlphas()

            nodeList = self.getNodes()
            prune_flag = (len(nodeList) > 0)

        # Set last alpha to 1 to indicate end
        alphas[iteration + 1] = 1.0
        return iteration

    def _evaluate_on_test(self, test: Instances) -> float:
        """Evaluate current model on test data and return error rate"""
        try:
            correct = 0
            total = test.numInstances()

            for instance in test.data:
                pred_probs = self.distributionForInstance(instance)
                pred_class = np.argmax(pred_probs)
                if pred_class == instance.class_value:
                    correct += 1

            return 1 - (correct / total) if total > 0 else 1.0
        except:
            return 1.0  # Return worst-case error if evaluation fails

    def unprune(self) -> None:
        """Method to 'unprune' the tree. Sets all leaf fields to False."""
        if self.m_sons is not None:
            self.m_isLeaf = False
            for son in self.m_sons:
                son.unprune()

    def getNumInnerNodes(self) -> int:
        """Method to count the number of inner nodes in the tree"""
        if self.m_isLeaf:
            return 0
        numNodes = 1
        for son in self.m_sons:
            numNodes += son.getNumInnerNodes()
        return numNodes

    def getNumLeaves(self) -> int:
        """Returns the number of leaves in the tree."""
        if not self.m_isLeaf:
            numLeaves = 0
            numEmptyLeaves = 0

            for son in self.m_sons:
                numLeaves += son.getNumLeaves()
                if son.m_isLeaf and not son.hasModels():
                    numEmptyLeaves += 1

            if numEmptyLeaves > 1:
                numLeaves -= (numEmptyLeaves - 1)
            return numLeaves
        else:
            return 1

    def treeErrors(self) -> None:
        """Updates the numIncorrectTree field for all nodes."""
        if self.m_isLeaf:
            self.m_numIncorrectTree = self.m_numIncorrectModel
        else:
            self.m_numIncorrectTree = 0
            for son in self.m_sons:
                son.treeErrors()
                self.m_numIncorrectTree += son.m_numIncorrectTree

    def calculateAlphas(self) -> None:
        """Updates the alpha field for all nodes."""
        if not self.m_isLeaf:
            errorDiff = self.m_numIncorrectModel - self.m_numIncorrectTree

            if errorDiff <= 0:
                # Split increases training error, prune instantly
                self.m_isLeaf = True
                self.m_sons = None
                self.m_alpha = float('inf')
            else:
                # Compute alpha
                errorDiff /= self.m_totalInstanceWeight
                self.m_alpha = errorDiff / (self.getNumLeaves() - 1)

                for son in self.m_sons:
                    son.calculateAlphas()
        else:
            # Alpha = infinite for leaves (do not want to prune)
            self.m_alpha = float('inf')

    def getNodes(self) -> List['QuadraticLMTNode']:
        """Return a list of all inner nodes in the tree"""
        nodeList = []
        self._getNodesHelper(nodeList)
        return nodeList

    def _getNodesHelper(self, nodeList: List['QuadraticLMTNode']) -> None:
        """Fills a list with all inner nodes in the tree"""
        if not self.m_isLeaf:
            nodeList.append(self)
            for son in self.m_sons:
                son._getNodesHelper(nodeList)

    def hasModels(self) -> bool:
        """Returns true if the model at this node has changed compared to parent."""
        return self.qda_model is not None

    def modelDistributionForInstance(self, instance: Instance) -> List[float]:
        """
        Returns the class probabilities for an instance according to the QDA model at the node.
        """
        # Apply nominal to binary filter
        # self.m_nominalToBinary.input(instance)  # Placeholder
        # instance = self.m_nominalToBinary.output()  # Placeholder

        # Apply QDA filter
        # self.filter_qda.input(instance)  # Placeholder
        # instance = self.filter_qda.output()  # Placeholder

        if self.qda_model is None or self.filtered_qda is None:
            # Return uniform distribution if no model
            return [1.0 / self.m_numClasses] * self.m_numClasses

        try:
            # Extract features for QDA model
            features = []
            for i in range(instance.numAttributes()):
                if i != self.filtered_qda.class_index and i in self.filter_qda:
                    features.append(instance.value(i))

            features = np.array(features).reshape(1, -1)
            return self.qda_model.predict_proba(features)[0].tolist()
        except:
            # Return uniform distribution in case of error
            return [1.0 / self.m_numClasses] * self.m_numClasses

    def distributionForInstance(self, instance: Instance) -> List[float]:
        """
        Returns the class probabilities for an instance given by the tree.
        """
        if self.m_isLeaf:
            # Leaf: use QDA model
            return self.modelDistributionForInstance(instance)
        else:
            # Sort into appropriate child node
            branch = self.m_localModel.whichSubset(instance)
            return self.m_sons[branch].distributionForInstance(instance)

    def numLeaves(self) -> int:
        """Returns the number of leaves (normal count)."""
        if self.m_isLeaf:
            return 1
        numLeaves = 0
        for son in self.m_sons:
            numLeaves += son.numLeaves()
        return numLeaves

    def numNodes(self) -> int:
        """Returns the number of nodes."""
        if self.m_isLeaf:
            return 1
        numNodes = 1
        for son in self.m_sons:
            numNodes += son.numNodes()
        return numNodes

    def __str__(self) -> str:
        """Returns a description of the tree structure and models."""
        # Assign numbers to models at leaves
        self.assignLeafModelNumbers(0)

        try:
            text_parts = []

            if self.m_isLeaf:
                text_parts.append(f": LM_{self.m_leafModelNum}:{self.getModelParameters()}")
            else:
                text_parts.append(self.dumpTree(0))

            text_parts.append(f"\n\nNumber of Leaves  : \t{self.numLeaves()}\n")
            text_parts.append(f"\nSize of the Tree : \t{self.numNodes()}\n")

            return "".join(text_parts)
        except Exception as e:
            return f"Can't print tree: {str(e)}"

    def getModelParameters(self) -> str:
        """Returns a string describing the model parameters."""
        return f"{self.m_numIncorrectModel}, {self.m_numIncorrectTree}, {self.m_alpha} ({self.m_numInstances})"

    def dumpTree(self, depth: int) -> str:
        """Helper method for printing tree structure."""
        text_parts = [f" : {self.m_numIncorrectModel}, {self.m_numIncorrectTree}, {self.m_alpha}"]

        for i, son in enumerate(self.m_sons):
            text_parts.append("\n")
            text_parts.append("|   " * depth)
            text_parts.append(self.m_localModel.leftSide(self.m_train))
            text_parts.append(self.m_localModel.rightSide(i, self.m_train))

            if son.m_isLeaf:
                text_parts.append(f": LM_{son.m_leafModelNum}:{son.getModelParameters()}")
            else:
                text_parts.append(son.dumpTree(depth + 1))

        return "".join(text_parts)

    def assignIDs(self, lastID: int) -> int:
        """Assigns unique IDs to all nodes in the tree."""
        currLastID = lastID + 1
        self.m_id = currLastID

        if self.m_sons is not None:
            for son in self.m_sons:
                currLastID = son.assignIDs(currLastID)

        return currLastID

    def assignLeafModelNumbers(self, leafCounter: int) -> int:
        """Assigns numbers to the models at the leaves of the tree."""
        if not self.m_isLeaf:
            self.m_leafModelNum = 0
            for son in self.m_sons:
                leafCounter = son.assignLeafModelNumbers(leafCounter)
        else:
            leafCounter += 1
            self.m_leafModelNum = leafCounter

        return leafCounter

    def modelsToString(self) -> str:
        """Returns a string describing the model at the node."""
        text_parts = []
        if self.m_isLeaf:
            model_str = str(self.qda_model) if self.qda_model else "No model"
            text_parts.append(f"LM_{self.m_leafModelNum}:\n{model_str}")
        else:
            for son in self.m_sons:
                text_parts.append("\n" + son.modelsToString())

        return "".join(text_parts)

    def cleanup(self):
        """Cleanup to save memory."""
        # Simplified cleanup - in practice you might want to clear large data structures
        pass

    def getNumericData(self, data: Instances) -> Instances:
        """Converts data to numeric version (placeholder implementation)."""
        if self.m_numericDataHeader is None:
            self.m_numericDataHeader = Instances(data, 0)
            # Placeholder: set up numeric data header

        numericData = Instances(self.m_numericDataHeader, data.numInstances())
        numericData.data = data.data  # Simplified
        return numericData

# Utility class similar to Java's UnsafeInstance
class UnsafeInstance(Instance):
    """
    Instance class with unsafe setValue operation.
    """
    def __init__(self, original_instance: Instance):
        super().__init__(original_instance.numAttributes())
        # Copy values from original instance
        for i in range(original_instance.numAttributes()):
            self.values[i] = original_instance.value(i)
        self.weight = original_instance.weight
        self.class_value = original_instance.class_value

    def setValue(self, attIndex: int, value: float):
        """Unsafe setValue method - directly modifies the value."""
        self.values[attIndex] = value

    def copy(self):
        """Return self instead of making a copy."""
        return self
