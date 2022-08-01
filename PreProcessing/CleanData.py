'''
# Topaz Aakal -> 318644549
# Marat Zinger -> 332689405
# Afik Danan -> 208900175
'''


class Cleaning:
    ''' our cleaning class '''

    def __init__(self, df, classifier):
        self.classifier = classifier
        self.df = df

    def cleanNa(self):
        ''' function to clean the classify column from missing values'''
        self.df.dropna(how='all', inplace=False)
        return self.df

    def cleanClassifier(self):
        """ gets a dataframe and the classifier column and return dataframe
        without missing values in the classifier column """
        return self.df[self.df[self.classifier].notna()]
