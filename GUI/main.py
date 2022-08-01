import csv
from pathlib import Path
from tkinter.messagebox import showerror, showinfo

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from GUI.Plot import printPlot
from Models.DecisionTree import decisionTree_sklearn, ourDecisionTree
from Models.KNN import KNN
from Models.Kmeans import Kmeans
from Models.NaiveBayes import naiveBayes_sklearn, ourNaivebayes
from PreProcessing.CleanData import Cleaning
from PreProcessing.Discretization import Discretization
from PreProcessing.MissingValuesHandlers import missingValHandler
from PreProcessing.Normalize import Normalization

'''
# Topaz Aakal -> 318644549
# Marat Zinger -> 332689405
# Afik Danan -> 208900175
'''

TrainFile = None
TestFile = None

TrainFile_Clean = None
TestFile_Clean = None

x_train = None
x_test = None
y_train = None
y_test = None

x_y_train = None
x_y_test = None

window1 = None
models = {}
choices = {}

results = {}


# main function
def main():
    from GUI.window import Window
    global window1
    window1 = Window()
    window1.run()


# displaying graphs with model results
def showResults():
    printPlot(results)


# separating data into training and test
def splitTheData():
    global x_train, x_test, y_train, y_test, x_y_train, x_y_test

    # receiving data from the user
    df = window1.dataframe.copy()
    classifier = choices["classification_column"]

    # train and test without classifier
    x = df.drop(classifier, axis=1)
    y = df[classifier]

    # split data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=10)  ## 20/80

    # train and test with classifier
    x_y_train = x_train.copy()
    x_y_test = x_test.copy()
    x_y_train.insert(len(x_train.columns), classifier, y_train)
    x_y_test.insert(len(x_test.columns), classifier, y_test)


# data preprocessing depending on the values selected by the user
def preprocess():
    global x_train, x_test, y_train, y_test, x_y_train, x_y_test

    Test = x_y_test
    Train = x_y_train

    # set classifier name
    classifier = y_train.name

    # path to store test_clean and train_clean
    path_test = Path(__file__).parent / "../Files/Test_clean.csv"
    path_train = Path(__file__).parent / "../Files/Train_clean.csv"

    # fill in the missing values in a class column
    cleaning_test = Cleaning(Test, classifier)
    cleaning_train = Cleaning(Train, classifier)

    # deleting rows with Nan value
    if choices['deleting_rows'] == 'Yes':
        Test = cleaning_test.cleanNa()
        Test = cleaning_test.cleanClassifier()

        Train = cleaning_train.cleanNa()
        Train = cleaning_train.cleanClassifier()

    # data after deleting missing values
    missing_test = missingValHandler(Test, classifier)
    missing_train = missingValHandler(Train, classifier)

    # completing missing values
    if choices['completing_missing_values'] == 'Classification value':
        Test = missing_test.missingVals(1)
        Train = missing_train.missingVals(1)
    elif choices['completing_missing_values'] == 'All data':
        Test = missing_test.missingVals(2)
        Train = missing_train.missingVals(2)

    # prepare data for normalization
    normalize_test = Normalization(Test)
    normalize_train = Normalization(Train)

    # data normalization
    if choices['normalization'] == 'Yes':
        Test = normalize_test.norm()
        Train = normalize_train.norm()

    # prepare data for discretization
    discrete_test = Discretization(Test, classifier, choices['num_of_bins'])
    discrete_train = Discretization(Train, classifier, choices['num_of_bins'])

    # data discretization
    if choices['discretization'] == 'Equal depth':
        Test = discrete_test.equalFreqDisc()
        Train = discrete_train.equalFreqDisc()
    elif choices['discretization'] == 'Equal width':
        Test = discrete_test.equalWDisc()
        Train = discrete_train.equalWDisc()
    elif choices['discretization'] == 'Entropy':
        Test = discrete_test.entropyBasedDiscretizer()
        Train = discrete_train.entropyBasedDiscretizer()

    # save data after preprocess
    Test.to_csv(path_test)
    Train.to_csv(path_train)


# create the model according to the user's choice
def runTheModels():
    global x_train, x_test, y_train, y_test, x_y_train, x_y_test, TestFile_Clean, TrainFile_Clean, results
    classifier = y_train.name

    # preprocess
    preprocess()

    # paths to train and test file
    path_test = Path(__file__).parent / "../Files/Test_clean.csv"
    path_train = Path(__file__).parent / "../Files/Train_clean.csv"

    # opening of the train and test
    TestFile_Clean = open(path_test, "r")
    TrainFile_Clean = open(path_train, "r")
    TrainFile = pd.read_csv(TrainFile_Clean)
    TestFile = pd.read_csv(TestFile_Clean)

    # encoding of train and test files
    TrainFile = TrainFile.apply(LabelEncoder().fit_transform)
    TestFile = TestFile.apply(LabelEncoder().fit_transform)

    # drop class column in train file
    x_train = TrainFile.drop(classifier, axis=1)
    y_train = TrainFile[classifier]

    # drop class column in test file
    x_test = TestFile.drop(classifier, axis=1)
    y_test = TestFile[classifier]

    # model creating
    if choices['model_selected'] == 'OurDT':
        dt = ourDecisionTree(x_train, x_test, y_train, y_test)
        results = dt.model(choices["depth_of_tree"])

    elif choices['model_selected'] == 'SklearnDT':
        dt = decisionTree_sklearn(x_train, x_test, y_train, y_test)
        results = dt.model(choices["depth_of_tree"])

    elif choices['model_selected'] == 'OurNB':
        naive = ourNaivebayes(x_train, x_test, y_train, y_test)
        results = naive.model()

    elif choices['model_selected'] == 'SklearnNB':
        naive = naiveBayes_sklearn(x_train, x_test, y_train, y_test)
        results = naive.model()

    elif choices['model_selected'] == 'KNN':
        knn = KNN(x_train, x_test, y_train, y_test)
        results = knn.model(choices['num_of_neighbors'])

    elif choices['model_selected'] == 'K-MEANS':
        kmeans = Kmeans(x_train, x_test, y_train, y_test)
        results = kmeans.model(choices['num_of_clusters'])

    if results is not None:
        writeTheResults(results, choices)
        showinfo(title='OK', message="Done! " + choices['model_selected'])


# writing results to excel file named results
def writeTheResults(modelResults, choices):
    flag = False

    # EVALUATION
    accuracy = modelResults['accuracy']
    recall = modelResults['recall']
    precision = modelResults['precision']
    majority = modelResults['majority']
    f1score = modelResults['f1Score']

    # NORMALIZATION & DISCRETIZAION
    normalization = 'Yes' if choices['normalization'] == 'Yes' else 'No'
    discretization = 'Yes' if choices['discretization'] != 'Without' else 'No'

    numOfBining = 'None' if choices['discretization'] == 'Without' else choices['num_of_bins']
    discretType = 'Equal-width' if choices['discretization'] == 'Equal width' else 'Equal-depth' if choices[
                                                                                                        'discretization'] == 'Equal depth' else 'Entropy Based' if \
        choices['discretization'] == 'Entropy' else 'None'

    # COMPLETING DATA
    dataComplete = 'All Data' if choices['completing_missing_values'] == 'All data' else 'Classification Column'

    # THE MODEL SELECTED
    model = 'Our NaiveBayes' if choices['model_selected'] == 'OurNB' else 'Sklearn NaiveBayes' if choices[
                                                                                                      'model_selected'] == 'SklearnNB' else 'Our Decision Tree' if \
        choices['model_selected'] == 'OurDT' else 'Sklearn Decision Tree' if choices[
                                                                                 'model_selected'] == 'SklearnDT' else 'KNN' if \
        choices['model_selected'] == 'KNN' else 'K-Means'

    # DECISION TREE
    depth = choices['depth_of_tree'] if (
            choices['model_selected'] == 'OurDT' or choices['model_selected'] == 'SklearnDT') else 'Not Relevant'

    # KNN
    neighbors = choices['num_of_neighbors'] if choices['model_selected'] == 'KNN' else 'Not Relevant'

    # K-MEANS
    numOfClusters = choices['num_of_clusters'] if choices['model_selected'] == 'K-MEANS' else 'Not Relevant'

    results = [model, discretization, discretType, numOfBining, normalization, dataComplete, depth, neighbors,
               numOfClusters, majority, accuracy, recall, precision, f1score]

    file_name = "results.csv"
    file_path = "../Files/" + file_name
    path = Path(__file__).parent / file_path
    results_file = path

    while not flag:
        try:
            if results_file.is_file():  # file already exists
                with open(results_file, 'a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(results)

            else:  # file now created in the first time
                with open(results_file, 'a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(
                        ['Model Selected', 'Discretizaion', 'Discretizaion Type', 'Number Of Bins', 'Normalization',
                         'Data Completed By', 'Max Tree Depth', 'Number Of Neighbors',
                         'Number Of Clusters', 'Majority', 'Accuracy', 'Recall', 'Precision', 'F1Score'])
                    writer.writerow(results)
            flag = True

            file.close()
        except:
            showerror("Error", "Close the results file and try again!")


main()
