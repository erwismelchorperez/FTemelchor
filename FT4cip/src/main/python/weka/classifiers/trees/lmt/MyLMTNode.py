"""
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import math
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Any
import sys
from collections import defaultdict

# Placeholder imports for WEKA equivalents in Python
# These would need to be replaced with actual Python implementations
class Instances:
    def numInstances(self) -> int:
        pass

    def numClasses(self) -> int:
        pass

    def trainCV(self, folds: int, fold: int):
        pass

    def testCV(self, folds: int, fold: int):
        pass

    def stratify(self, folds: int):
        pass

class Instance:
    pass

class Distribution:
    def numIncorrect(self) -> float:
        pass

    def perBag(self, i: int) -> float:
        pass

    def total(self) -> float:
        pass

class ModelSelection:
    def selectModel(self, data: Instances):
        pass

class ClassifierSplitModel:
    def numSubsets(self) -> int:
        pass

    def split(self, data: Instances) -> List[Instances]:
        pass

    def whichSubset(self, instance: Instance) -> int:
        pass

    def distribution(self) -> Distribution:
        pass

    def leftSide(self, data: Instances) -> str:
        pass

    def rightSide(self, i: int, data: Instances) -> str:
        pass

class NoSplit(ClassifierSplitModel):
    def __init__(self, distribution: Distribution):
        self.distribution = distribution

class LogisticBase:
    def __init__(self):
        self.m_fixedNumIterations = -1
        self.m_errorOnProbabilities = False
        self.m_maxIterations = 200
        self.m_numFoldsBoosting = 5
        self.m_numDecimalPlaces = 2
        self.m_regressions = None
        self.m_numericData = None
        self.m_numericDataHeader = None
        self.m_numParameters = 0
        self.m_numRegressions = 0

    def setWeightTrimBeta(self, beta: float):
        self.m_weightTrimBeta = beta

    def setUseAIC(self, use_aic: bool):
        self.m_useAIC = use_aic

    def getWeightTrimBeta(self) -> float:
        return self.m_weightTrimBeta

    def getUseAIC(self) -> bool:
        return self.m_useAIC

    def getNumRegressions(self) -> int:
        return self.m_numRegressions

    def buildClassifier(self, data: Instances):
        pass

    def performBoosting(self, iterations: int):
        pass

    def performBoostingInfCriterion(self):
        pass

    def performBoostingCV(self):
        pass

    def getFs(self, data):
        pass

    def probs(self, fs):
        pass

    def initRegressions(self):
        pass

    def copyRegressions(self, regressions):
        pass

    def cleanup(self):
        pass

class WekaClassifierEvaluator:
    def __init__(self, classifier):
        self.classifier = classifier

    def Evaluate(self, data):
        pass

    def getAuc(self) -> float:
        return 0.0

    def getAcc(self) -> float:
        return 0.0

class MyCompareNode:
    """
    Auxiliary class for comparing MyLMTNodes
    """
    def compare(self, o1: 'MyLMTNode', o2: 'MyLMTNode') -> int:
        if o1.m_alpha < o2.m_alpha:
            return -1
        if o1.m_alpha > o2.m_alpha:
            return 1
        return 0

class MyLMTNode(LogisticBase):
    """
    Class for logistic model tree structure.
    Translated from Java to Python
    """

    # Static variable
    m_numFoldsPruning = 5

    def __init__(self, modelSelection: ModelSelection, numBoostingIterations: int,
                 fastRegression: bool, errorOnProbabilities: bool, minNumInstances: int,
                 weightTrimBeta: float, useAIC: bool, nominalToBinary, numDecimalPlaces: int,
                 trainLogistic: bool, prune: bool, optimizeAUC: bool):
        super().__init__()

        self.m_modelSelection = modelSelection
        self.m_fixedNumIterations = numBoostingIterations
        self.m_fastRegression = fastRegression
        self.m_errorOnProbabilities = errorOnProbabilities
        self.m_minNumInstances = minNumInstances
        self.setWeightTrimBeta(weightTrimBeta)
        self.setUseAIC(useAIC)
        self.m_nominalToBinary = nominalToBinary
        self.m_numDecimalPlaces = numDecimalPlaces
        self.trainLogistic = trainLogistic
        self.prune_flag = prune
        self.optimizeAUC = optimizeAUC

        # Initialize instance variables
        self.m_dataHeader = None
        self.m_logistic = None
        self.m_trained_logistic = False
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
        self.m_CF = 0.10
        self.m_train = None
        self.m_numClasses = 0

    def buildClassifier(self, data: Instances) -> None:
        """
        Method for building a logistic model tree (only called for the root node).
        Grows an initial logistic model tree and prunes it back using the CART pruning scheme.
        """
        # heuristic to avoid cross-validating the number of LogitBoost iterations
        # at every node: build standalone logistic model and take its optimum number of iterations
        if self.trainLogistic and self.m_fastRegression and (self.m_fixedNumIterations < 0):
            self.m_fixedNumIterations = self.tryLogistic(data)

        if self.prune_flag:
            self.buildWithPruning(data)
        else:
            self.buildTree(data, None, data.numInstances(), 0, None)

    def buildWithPruning(self, data: Instances) -> None:
        """Build tree with pruning using cross-validation"""
        # Need to cross-validate alpha-parameter for CART-pruning
        cvData = Instances(data)  # Assuming copy constructor
        cvData.stratify(self.m_numFoldsPruning)

        alphas = [None] * self.m_numFoldsPruning
        errors = [None] * self.m_numFoldsPruning

        for i in range(self.m_numFoldsPruning):
            # for every fold, grow tree on training set...
            train = cvData.trainCV(self.m_numFoldsPruning, i)
            test = cvData.testCV(self.m_numFoldsPruning, i)

            dist = Distribution(test)

            self.buildTree(train, None, train.numInstances(), 0, None)

            numNodes = self.getNumInnerNodes()
            alphas[i] = [0.0] * (numNodes + 2)
            errors[i] = [0.0] * (numNodes + 2)

            # ... then prune back and log alpha-values and errors on test set
            self.prune(alphas[i], errors[i], test)

        # don't need CV data anymore
        cvData = None

        # build tree using all the data
        self.buildTree(data, None, data.numInstances(), 0, None)
        numNodes = self.getNumInnerNodes()

        treeAlphas = [0.0] * (numNodes + 2)
        treeErrors = [0.0] * (numNodes + 2)

        iterations = self.prune(treeAlphas, treeErrors, data)

        for i in range(iterations + 1):
            # compute midpoint alphas
            alpha = math.sqrt(treeAlphas[i] * treeAlphas[i + 1])
            error = 0.0

            # compute error estimate for final trees from the midpoint-alphas
            for k in range(self.m_numFoldsPruning):
                l = 0
                while alphas[k][l] <= alpha:
                    l += 1
                error += errors[k][l - 1]

            treeErrors[i] = error

        # find best alpha
        best = -1
        bestError = float('inf')
        for i in range(iterations, -1, -1):
            if treeErrors[i] < bestError:
                bestError = treeErrors[i]
                best = i

        bestAlpha = math.sqrt(treeAlphas[best] * treeAlphas[best + 1])

        # "unprune" final tree (faster than regrowing it)
        self.unprune()

        # CART-prune it with best alpha
        self.prune(bestAlpha)

    def buildTree(self, data: Instances, higherRegressions, totalInstanceWeight: float,
                 higherNumParameters: float, numericDataHeader: Instances) -> None:
        """
        Method for building the tree structure. Builds a logistic model, splits the
        node and recursively builds tree for child nodes.
        """
        # save some stuff
        self.m_totalInstanceWeight = totalInstanceWeight
        self.m_train = data  # no need to copy the data here
        self.m_isLeaf = True
        self.m_sons = None

        self.m_numInstances = self.m_train.numInstances()
        self.m_numClasses = self.m_train.numClasses()

        # init
        if self.trainLogistic:
            self.buildLogistic(numericDataHeader, higherRegressions, higherNumParameters)

        grow = False
        # split node if more than minNumInstances...
        if self.m_numInstances > self.m_minNumInstances:
            m_localModel = self.m_modelSelection.selectModel(self.m_train)
            # ... and valid split found
            grow = (m_localModel.numSubsets() > 1)
        else:
            m_localModel = self.m_modelSelection.selectModel(self.m_train)
            grow = False

        if grow:
            # create and build children of node
            self.m_isLeaf = False
            localInstances = m_localModel.split(self.m_train)

            # don't need data anymore, so clean up
            self.cleanup()

            self.m_sons = [None] * m_localModel.numSubsets()
            for i in range(len(self.m_sons)):
                self.m_sons[i] = MyLMTNode(
                    self.m_modelSelection, self.m_fixedNumIterations,
                    self.m_fastRegression, self.m_errorOnProbabilities,
                    self.m_minNumInstances, self.getWeightTrimBeta(),
                    self.getUseAIC(), self.m_nominalToBinary,
                    self.m_numDecimalPlaces, self.trainLogistic,
                    self.prune_flag, self.optimizeAUC
                )
                if self.trainLogistic:
                    self.m_sons[i].buildTree(
                        localInstances[i], self.copyRegressions(self.m_regressions),
                        self.m_totalInstanceWeight, self.m_numParameters,
                        self.m_numericDataHeader
                    )
                else:
                    self.m_sons[i].buildTree(
                        localInstances[i], None,
                        self.m_totalInstanceWeight, self.m_numParameters,
                        self.m_numericDataHeader
                    )
                localInstances[i] = None
        else:
            self.cleanup()

    def buildLogistic(self, numericDataHeader: Instances, higherRegressions,
                     higherNumParameters: float) -> None:
        """Build logistic model at this node"""
        self.m_numericDataHeader = numericDataHeader
        self.m_numericData = self.getNumericData(self.m_train)

        if higherRegressions is None:
            self.m_regressions = self.initRegressions()
        else:
            self.m_regressions = higherRegressions

        self.m_numParameters = higherNumParameters
        self.m_numRegressions = 0

        # build logistic model
        if self.m_numInstances >= self.m_numFoldsBoosting:
            if self.m_fixedNumIterations > 0:
                self.performBoosting(self.m_fixedNumIterations)
            elif self.getUseAIC():
                self.performBoostingInfCriterion()
            else:
                self.performBoostingCV()

        self.m_numParameters += self.m_numRegressions

        # Evaluate the model - placeholder for actual evaluation
        # eval = Evaluation(self.m_train)
        # eval.evaluateModel(self, self.m_train)
        # self.m_numIncorrectModel = eval.incorrect()
        self.m_numIncorrectModel = 0.0  # Placeholder

    def prune(self, alpha: float) -> None:
        """
        Prunes a logistic model tree using the CART pruning scheme, given a
        cost-complexity parameter alpha.
        """
        # determine training error of logistic models and subtrees, and calculate alpha-values
        self.treeErrors()
        self.calculateAlphas()

        # get list of all inner nodes in the tree
        nodeList = self.getNodes()

        prune = (len(nodeList) > 0)
        comparator = MyCompareNode()

        while prune:
            # select node with minimum alpha
            nodeToPrune = min(nodeList, key=lambda x: x.m_alpha)

            # want to prune if its alpha is smaller than alpha
            if nodeToPrune.m_alpha > alpha:
                break

            nodeToPrune.m_isLeaf = True
            nodeToPrune.m_sons = None

            # update tree errors and alphas
            self.treeErrors()
            self.calculateAlphas()

            nodeList = self.getNodes()
            prune = (len(nodeList) > 0)

        # discard references to models at internal nodes because they are not needed
        for node in self.getNodes():
            if not node.m_isLeaf:
                node.m_regressions = None

    def prune(self, alphas: List[float], errors: List[float], test: Instances) -> int:
        """
        Method for performing one fold in the cross-validation of the
        cost-complexity parameter.
        """
        comparator = MyCompareNode()

        # determine training error of logistic models and subtrees, and calculate alpha-values
        self.treeErrors()
        self.calculateAlphas()

        # get list of all inner nodes in the tree
        nodeList = self.getNodes()

        prune = (len(nodeList) > 0)

        # alpha_0 is always zero (unpruned tree)
        alphas[0] = 0.0

        # error of unpruned tree
        if errors is not None:
            evaluator = WekaClassifierEvaluator(self)
            evaluator.Evaluate(test)
            if self.optimizeAUC:
                errors[0] = 1 - evaluator.getAuc()
            else:
                errors[0] = 1 - evaluator.getAcc()

        iteration = 0
        while prune:
            iteration += 1

            # get node with minimum alpha
            nodeToPrune = min(nodeList, key=lambda x: x.m_alpha)

            nodeToPrune.m_isLeaf = True
            # Do not set m_sons null, want to unprune

            # get alpha-value of node
            alphas[iteration] = nodeToPrune.m_alpha

            # log error
            if errors is not None:
                evaluator = WekaClassifierEvaluator(self)
                evaluator.Evaluate(test)
                if self.optimizeAUC:
                    errors[iteration] = 1 - evaluator.getAuc()
                else:
                    errors[iteration] = 1 - evaluator.getAcc()

            # update errors/alphas
            self.treeErrors()
            self.calculateAlphas()

            nodeList = self.getNodes()
            prune = (len(nodeList) > 0)

        # set last alpha 1 to indicate end
        alphas[iteration + 1] = 1.0
        return iteration

    def unprune(self) -> None:
        """Method to 'unprune' a logistic model tree. Sets all leaf-fields to False."""
        if self.m_sons is not None:
            self.m_isLeaf = False
            for son in self.m_sons:
                son.unprune()

    def tryLogistic(self, data: Instances) -> int:
        """
        Determines the optimum number of LogitBoost iterations to perform by
        building a standalone logistic regression function on the training data.
        """
        # convert nominal attributes
        filteredData = self.m_nominalToBinary.useFilter(data)  # Placeholder

        logistic = LogisticBase(0, True, self.m_errorOnProbabilities)  # Placeholder

        # limit LogitBoost to 200 iterations (speed)
        logistic.setMaxIterations(200)
        logistic.setWeightTrimBeta(self.getWeightTrimBeta())
        logistic.setUseAIC(self.getUseAIC())
        logistic.buildClassifier(filteredData)

        # return best number of iterations
        return logistic.getNumRegressions()

    def getNumInnerNodes(self) -> int:
        """Method to count the number of inner nodes in the tree"""
        if self.m_isLeaf:
            return 0
        numNodes = 1
        for son in self.m_sons:
            numNodes += son.getNumInnerNodes()
        return numNodes

    def getNumLeaves(self) -> int:
        """
        Returns the number of leaves in the tree. Leaves are only counted if their
        logistic model has changed compared to the one of the parent node.
        """
        if not self.m_isLeaf:
            numLeaves = 0
            numEmptyLeaves = 0

            for son in self.m_sons:
                numLeaves += son.getNumLeaves()
                if self.trainLogistic and son.m_isLeaf and not son.hasModels():
                    numEmptyLeaves += 1

            if numEmptyLeaves > 1:
                numLeaves -= (numEmptyLeaves - 1)
            return numLeaves
        else:
            return 1

    def treeErrors(self) -> None:
        """Updates the numIncorrectTree field for all nodes."""
        if self.m_isLeaf:
            if self.trainLogistic:
                self.m_numIncorrectTree = self.m_numIncorrectModel
            else:
                self.m_numIncorrectTree = self.m_localModel.distribution().numIncorrect()
        else:
            self.m_numIncorrectTree = 0
            for son in self.m_sons:
                son.treeErrors()
                self.m_numIncorrectTree += son.m_numIncorrectTree

    def calculateAlphas(self) -> None:
        """Updates the alpha field for all nodes."""
        if not self.m_isLeaf:
            if self.trainLogistic:
                errorDiff = self.m_numIncorrectModel - self.m_numIncorrectTree
            else:
                errorDiff = self.m_localModel.distribution().numIncorrect() - self.m_numIncorrectTree

            if errorDiff <= 0:
                # split increases training error, prune it instantly
                self.m_isLeaf = True
                self.m_sons = None
                self.m_alpha = float('inf')
            else:
                # compute alpha
                errorDiff /= self.m_totalInstanceWeight
                self.m_alpha = errorDiff / (self.getNumLeaves() - 1)

                for son in self.m_sons:
                    son.calculateAlphas()
        else:
            # alpha = infinite for leaves (do not want to prune)
            self.m_alpha = float('inf')

    def getNodes(self) -> List['MyLMTNode']:
        """Return a list of all inner nodes in the tree"""
        nodeList = []
        self.getNodesHelper(nodeList)
        return nodeList

    def getNodesHelper(self, nodeList: List['MyLMTNode']) -> None:
        """Fills a list with all inner nodes in the tree"""
        if not self.m_isLeaf:
            nodeList.append(self)
            for son in self.m_sons:
                son.getNodesHelper(nodeList)

    def hasModels(self) -> bool:
        """
        Returns true if the logistic regression model at this node has changed
        compared to the one at the parent node.
        """
        return self.m_numRegressions > 0

    def modelDistributionForInstance(self, instance: Instance) -> List[float]:
        """
        Returns the class probabilities for an instance according to the logistic
        model at the node.
        """
        # make copy and convert nominal attributes
        self.m_nominalToBinary.input(instance)  # Placeholder
        instance = self.m_nominalToBinary.output()  # Placeholder

        # set numeric pseudo-class
        instance.setDataset(self.m_numericDataHeader)

        return self.probs(self.getFs(instance))

    def distributionForInstance(self, instance: Instance) -> List[float]:
        """
        Returns the class probabilities for an instance given by the logistic model tree.
        """
        probs = []

        if self.m_isLeaf:
            # leaf: use logistic model
            if self.trainLogistic:
                probs = self.modelDistributionForInstance(instance)
            else:
                probs = [0.0] * instance.numClasses()
                for i in range(len(probs)):
                    probs[i] = self.m_localModel.classProb(i, instance, -1)
        else:
            # sort into appropriate child node
            branch = self.m_localModel.whichSubset(instance)
            probs = self.m_sons[branch].distributionForInstance(instance)

        return probs

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
        """Returns a description of the logistic model tree"""
        if self.trainLogistic:
            self.assignLeafModelNumbers(0)

        try:
            text = []

            if self.m_isLeaf:
                leaf_text = ": "
                if self.trainLogistic:
                    leaf_text += f"LM_{self.m_leafModelNum}:{self.getModelParameters()}"
                else:
                    leaf_text += self.m_localModel.dumpLabel(0, self.m_train)
                text.append(leaf_text)
            else:
                text.append(self.dumpTree(0))

            text.append(f"\n\nNumber of Leaves  : \t{self.numLeaves()}\n")
            text.append(f"\nSize of the Tree : \t{self.numNodes()}\n")

            # This prints logistic models after the tree
            if self.trainLogistic:
                text.append(self.modelsToString())

            return "".join(text)
        except Exception as e:
            return f"Can't print logistic model tree: {str(e)}"

    def getModelParameters(self) -> str:
        """
        Returns a string describing the number of LogitBoost iterations performed
        at this node.
        """
        numModels = int(self.m_numParameters)
        return f"{self.m_numRegressions}/{numModels} ({self.m_numInstances})"

    def dumpTree(self, depth: int) -> str:
        """Help method for printing tree structure."""
        text = []
        for i in range(len(self.m_sons)):
            text.append("\n")
            text.append("|   " * depth)
            text.append(self.m_localModel.leftSide(self.m_train))
            text.append(self.m_localModel.rightSide(i, self.m_train))
            if self.m_sons[i].m_isLeaf:
                text.append(": ")
                if self.trainLogistic:
                    text.append(f"LM_{self.m_sons[i].m_leafModelNum}:{self.m_sons[i].getModelParameters()}")
                else:
                    text.append(self.m_localModel.dumpLabel(i, self.m_train))
            else:
                text.append(self.m_sons[i].dumpTree(depth + 1))
        return "".join(text)

    def assignIDs(self, lastID: int) -> int:
        """Assigns unique IDs to all nodes in the tree"""
        currLastID = lastID + 1
        self.m_id = currLastID
        if self.m_sons is not None:
            for son in self.m_sons:
                currLastID = son.assignIDs(currLastID)
        return currLastID

    def assignLeafModelNumbers(self, leafCounter: int) -> int:
        """Assigns numbers to the logistic regression models at the leaves of the tree"""
        if not self.m_isLeaf:
            self.m_leafModelNum = 0
            for son in self.m_sons:
                leafCounter = son.assignLeafModelNumbers(leafCounter)
        else:
            leafCounter += 1
            self.m_leafModelNum = leafCounter
        return leafCounter

    def modelsToString(self) -> str:
        """Returns a string describing the logistic regression function at the node."""
        text = []
        if self.m_isLeaf:
            text.append(f"LM_{self.m_leafModelNum}:{super().__str__()}")
        else:
            for son in self.m_sons:
                text.append("\n" + son.modelsToString())
        return "".join(text)

    def isTrainLogistic(self) -> bool:
        return self.trainLogistic

    def setTrainLogistic(self, trainLogistic: bool) -> None:
        self.trainLogistic = trainLogistic
