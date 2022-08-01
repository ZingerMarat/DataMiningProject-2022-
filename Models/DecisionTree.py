import math
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

'''
# Topaz Aakal -> 318644549
# Marat Zinger -> 332689405
# Afik Danan -> 208900175
'''


class decisionTree_sklearn:
    """A class for decision tree algorithm in the library sklearn"""

    def __init__(self, x_train, x_test, y_train, y_test):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test

    def model(self, max_depth):
        """function that builds the model and activates it"""
        decisionTree_en = DecisionTreeClassifier(criterion='entropy', max_depth=max_depth, random_state=3,
                                                 splitter='best')  # create Decision Tree classifier object
        model = decisionTree_en.fit(self.x_train, self.y_train)  # fitting the object to the training set

        pred = model.predict(self.x_test)  # predict the response for test dataset

        # calculation of evaluation indices on test data
        accuracy = accuracy_score(self.y_test, pred)
        precision = precision_score(self.y_test, pred, average='weighted', zero_division=True)
        recall = recall_score(self.y_test, pred, average='weighted', labels=np.unique(pred), zero_division=True)
        f1Score = f1_score(self.y_test, pred, average='weighted')
        cm = confusion_matrix(self.y_test, pred)

        R = self.y_test.value_counts().argmax()
        maj = (self.y_test.values == R).sum()
        majority = (maj / len(self.y_test)) * 100

        # calculation of evaluation indices on train data
        pred_train = model.predict(self.x_train)
        cm_train = confusion_matrix(self.y_train, pred_train)

        return {'accuracy': accuracy * 100, 'precision': precision * 100, 'recall': recall * 100,
                'f1Score': f1Score * 100,
                'confusion_matrix': cm, 'confusion_matrix_train': cm_train, "model": model, "majority": majority}


class DecisionTree:
    """A class for decision tree algorithm in our implementation"""

    def __init__(self, df, classifier):
        self.df = df.copy()
        self.classifier = classifier
        self.root = None

    def decisionTreeImplementation(self, maxDepth):
        """implementation of a decision tree using entropy"""
        def getMajority():
            """function that returns the classification value that represents the majority"""
            values, counts = np.unique(self.df[self.classifier], return_counts=True)
            UniqueVals = []
            for first, sec in zip(values, counts):
                UniqueVals.append((first, sec))
            max = 0
            maxKey = ''
            for i in range(len(UniqueVals)):
                if UniqueVals[i][1] > max:
                    max = UniqueVals[i][1]
                    maxKey = UniqueVals[i][0]
            return maxKey

        def entropyCalc(data):
            """function for calculate the entropy of a single column"""
            unique = {}
            for i in data:
                if i not in unique.keys():
                    unique[i] = 1
                else:
                    unique[i] += 1
            entropy = 0
            for key in unique:
                prob = unique[key] / len(data)
                entropy += prob * (math.log(prob, 2))
            return -1 * entropy

        def conditionalEntropy(data1, data2):
            """function for calculate conditional entropy between two columns"""
            unique1 = {}
            for i in data1:
                if i not in unique1.keys():
                    unique1[i] = 1
                else:
                    unique1[i] += 1

            unique2 = {}
            for i in data2:
                if i not in unique2.keys():
                    unique2[i] = 1
                else:
                    unique2[i] += 1

            unique3 = {}
            for i in unique1:
                for j in unique2:
                    unique3[(i, j)] = 0
            for first, sec in zip(data1, data2):
                unique3[(first, sec)] += 1

            pMutual = {}
            pConditional = {}

            for i in unique3:
                pMutual[i] = unique3[i] / len(data1)
            for i in unique3:
                pConditional[i] = unique3[i] / unique2[i[1]]

            entropy = 0

            for i in pMutual.keys():
                entropy += pMutual[i] * math.log(pConditional[i], 2) if pConditional[i] != 0 else 0

            return -1 * entropy

        def informationGain(data1, data2):
            """function for calculate the information gain of a column"""
            return entropyCalc(data1) - conditionalEntropy(data1, data2)

        def getMaxGain(df, classifier):
            """function that returns the column with the max information gain"""
            maxGain = 0
            maxGainCol = ''
            for col in df.columns:
                if col != classifier:
                    gain = informationGain(df[col], df[classifier])
                    if gain > maxGain:
                        maxGain = gain
                        maxGainCol = col
            return maxGainCol

        def get_subtable(df, splitNode, value):
            """function that returns the subtree of a particular node"""
            return df[df[splitNode] == value].reset_index(drop=True)

        def buildTree(df, className, depth=maxDepth, tree=None):
            '''function for build the ID3 Decision Tree.'''
            classifier = className

            splitNode = getMaxGain(df, classifier)  # saving the column with the maximum information gain
            attValue = np.unique(df[splitNode])  # saving the unique values of that column

            if tree is None:  # if the tree has not yet been built, we create an empty dictionary
                tree = {}
                tree[splitNode] = {}

            for value in attValue:  # we will go over the unique values of that column
                if depth == 0:  # stopping condition: if we have reached maximum depth
                    tree[splitNode][value] = getMajority()  # insert the majority value into the leaf
                else:
                    # otherwise, we will build the subtree
                    subtable = get_subtable(df, splitNode, value)
                    clValue, counts = np.unique(subtable[classifier], return_counts=True)

                    if len(counts) == 1:
                        # if we have reached a situation of only one value in the column - we will stop the algorithm and save the leaf
                        tree[splitNode][value] = clValue[0]
                    else:
                        # otherwise, we will continue to run the algorithm on the subtree
                        tree[splitNode][value] = buildTree(subtable, classifier,
                                                           depth - 1)  # calling the function recursively

            return tree

        self.root = buildTree(self.df, self.classifier)  # saving the final tree
        return self.root

    def __predict_for(self, row, node):
        """function for predicting results by using a query function"""
        tree_copy = {}

        def find(a):
            nonlocal tree_copy
            for key, val in a.items():
                if key in tree_copy.keys():
                    return tree_copy[key][val]
            return tree_copy

        def query(kwargs):
            """function that gets arguments representing a query and returns the result"""
            nonlocal tree_copy, node
            tree_copy = node
            result = find(kwargs)
            while isinstance(result, dict):
                tree_copy = result
                result = find(kwargs)
            return result

        return query(row)  # running a query on a particular row

    def predict(self, X):
        """function that returns Y predicted, X should be a 2-D np array"""
        Y = np.array([0 for i in range(len(X))], dtype='int64')
        for i in range(len(X)):
            Y[i] = self.__predict_for(X.iloc[i].to_dict(), self.root)
        return Y


class ourDecisionTree:
    """A class for decision tree algorithm in our implementation"""

    def __init__(self, x_train, x_test, y_train, y_test):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test

    def model(self, max_depth):
        """function that builds the model and activates it"""

        # adjustment of the data to our model
        classifier = self.y_train.name
        df_train = self.x_train.copy()
        df_train.insert(len(self.x_train.columns), classifier, self.y_train)
        df_test = self.x_test.copy()
        df_test.insert(len(self.x_test.columns), classifier, self.y_test)

        model = DecisionTree(df_train, classifier)  # creat the tree object
        model.decisionTreeImplementation(max_depth)  # running the model on the object

        # predict the response for test dataset
        pred = model.predict(self.x_test)

        # calculation of evaluation indices on test data
        accuracy = accuracy_score(self.y_test, pred)
        precision = precision_score(self.y_test, pred)
        recall = recall_score(self.y_test, pred)
        f1Score = f1_score(self.y_test, pred)
        cm = confusion_matrix(self.y_test, pred)

        R = self.y_test.value_counts().argmax()
        maj = (self.y_test.values == R).sum()
        majority = (maj / len(self.y_test)) * 100

        # calculation of evaluation indices on train data
        pred_train = model.predict(self.x_train)
        cm_train = confusion_matrix(self.y_train, pred_train)

        return {'accuracy': accuracy * 100, 'precision': precision * 100, 'recall': recall * 100,
                'f1Score': f1Score * 100,
                'confusion_matrix': cm, 'confusion_matrix_train': cm_train, "model": model, "majority": majority}
