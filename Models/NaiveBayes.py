import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

'''
# Topaz Aakal -> 318644549
# Marat Zinger -> 332689405
# Afik Danan -> 208900175
'''

class naiveBayes_sklearn:
    """A class for Naive Bayes algorithm in the library sklearn"""
    def __init__(self, x_train, x_test, y_train, y_test):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test


    def model(self):
        """function that builds the model and activates it"""
        nb = GaussianNB() # create naive bayes classifier object
        model = nb.fit(self.x_train, self.y_train)  # fitting the object to the training set

        pred = model.predict(self.x_test) # predict the response for test dataset

        # calculation of evaluation indices on test data
        accuracy = accuracy_score(self.y_test, pred)
        precision = precision_score(self.y_test, pred, average='weighted', zero_division=True)
        recall = recall_score(self.y_test, pred, average='weighted', labels=np.unique(pred), zero_division=True)
        f1Score = f1_score(self.y_test, pred, average='weighted')
        cm = confusion_matrix(self.y_test, pred)

        # calculation of evaluation indices on train data
        pred_train = model.predict(self.x_train)
        cm_train = confusion_matrix(self.y_train, pred_train)

        R = self.y_test.value_counts().argmax()
        maj = (self.y_test.values == R).sum()
        majority = (maj / len(self.y_test)) * 100

        return {'accuracy': accuracy * 100, 'precision': precision * 100, 'recall': recall * 100, 'f1Score': f1Score * 100,
                'confusion_matrix': cm, 'confusion_matrix_train': cm_train, "model": model, "majority": majority}


class NaiveBayes:
    """Class for the implementation of the model - Naive Bayes"""

    def __init__(self, df_train, df_test, classifier):
        self.classVal1 = None
        self.classVal2 = None
        self.df = df_train.copy()
        self.df_test = df_test.copy()
        self.classifier = classifier
        self.prob_classifier = {}  # contains the probabilities of the classification column
        self.conditional_probs = {}  # contains the conditional probabilities of the other columns
        self.class_values = []  # contains the classification values (for example: yes/no)

    def NaiveBayesImplementation(self):
        # create the list: class_values of the train set
        for key, val in self.df.groupby(self.classifier):
            self.class_values.append(key)

        self.classVal1 = self.class_values[0]  # the name of the first classification value
        self.classVal2 = self.class_values[1]  # the name of the second classification value

        # calculate the classifier probabilities
        class_col = self.df.groupby(self.classifier).size() / len(self.df)
        i = 0
        for key, item in self.df.groupby(self.classifier):
            if key != "Unnamed: 0":
                self.prob_classifier[key] = class_col[i]
                i += 1

        # calculate the conditional probabilities
        for col in self.df.columns:
            if col != self.classifier:
                self.conditional_probs[str(col)] = (self.df.groupby([self.classifier, col]).size()) / (
                    self.df.groupby(self.classifier).size())

    def laplaceCorr(self, col_name, class_value, value):
        """function to calculate Laplace correction"""

        gr = self.df.groupby([self.classifier, col_name])
        numberOfValues = self.df.groupby(self.classifier).get_group(class_value)[col_name].count()

        for val in self.df[col_name].unique():
            if val != value:
                try:
                    previous_group_size = gr.get_group((int(class_value), int(val))).groupby(self.classifier).size()
                    update_denominator = previous_group_size[int(class_value)] + numberOfValues + 1
                    update_numerator = (gr.get_group((int(class_value), int(val))).groupby(col_name).size() + 1)
                    # updating the existing conditional_probs table
                    self.conditional_probs[col_name][int(class_value)][int(val)] = update_numerator[
                                                                                  int(val)] / update_denominator
                except KeyError:
                    continue

        # the probability requested
        return 1 / update_denominator

    def getColumnProb(self, col, class_value, col_value):
        """function that returns the conditional probability of the classification column with another specific column"""
        try:
            return self.conditional_probs[col][class_value][col_value]
        except KeyError:
            return self.laplaceCorr(col, class_value, col_value)


    def predict(self, X):
        """function that accepts the test set and predicts the results"""
        pred = []

        # go through the columns of the test data except the classification column and calculate the probability of each row
        for index, row in X.iterrows():
            predSumCV1, predSumCV2 = 1, 1

            for col in self.df_test.columns:
                if col != self.classifier and col != "Unnamed: 0":
                    predSumCV1 *= self.getColumnProb(col, self.classVal1, row[col])
                    predSumCV2 *= self.getColumnProb(col, self.classVal2, row[col])

            # at the end we will multiply by the probability of the classification values according to bayesian method
            predSumCV1 *= self.prob_classifier[self.classVal1]
            predSumCV2 *= self.prob_classifier[self.classVal2]

            if predSumCV1 > predSumCV2:
                pred.append(self.class_values[0])
            else:
                pred.append(self.class_values[1])

        return pred


class ourNaivebayes:
    """A class for naive bayes algorithm in our implementation"""
    def __init__(self, x_train, x_test, y_train, y_test):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test

    def model(self):
        """function that builds the model and activates it"""

        # adjustment of the data to our model
        classifier = self.y_train.name
        df_train = self.x_train.copy()
        df_train.insert(len(self.x_train.columns), classifier, self.y_train)
        df_test = self.x_test.copy()
        df_test.insert(len(self.x_test.columns), classifier, self.y_test)

        model= NaiveBayes(df_train, df_test, classifier) # creat the naive bayes object
        model.NaiveBayesImplementation() # running the model on the object

        pred = model.predict(self.x_test) # predict the response for test dataset

        # calculation of evaluation indices on test data
        accuracy = accuracy_score(self.y_test, pred)
        precision = precision_score(self.y_test, pred, average='weighted', zero_division=True)
        recall = recall_score(self.y_test, pred, average='weighted', labels=np.unique(pred), zero_division=True)
        f1Score = f1_score(self.y_test, pred, average='weighted', labels=np.unique(pred), zero_division=True)
        cm = confusion_matrix(self.y_test, pred)

        R = self.y_test.value_counts().argmax()
        maj = (self.y_test.values == R).sum()

        majority = (maj / len(self.y_test)) * 100

        pred_train = model.predict(self.x_train)
        cm_train = confusion_matrix(self.y_train, pred_train)

        return {'accuracy': accuracy * 100, 'precision': precision * 100, 'recall': recall * 100, 'f1Score': f1Score * 100,
                'confusion_matrix': cm, 'confusion_matrix_train': cm_train, "model": model, "majority": majority}