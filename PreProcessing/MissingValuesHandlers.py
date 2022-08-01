import numpy as np
from sklearn.impute import SimpleImputer

from PreProcessing.AuxiliaryFunctions import colType

'''
# Topaz Aakal -> 318644549
# Marat Zinger -> 332689405
# Afik Danan -> 208900175
'''


class missingValHandler:
    ''' class for completing missing values '''

    def __init__(self, df, classifier):
        self.classifier = classifier
        self.df = df.copy()
        self.continuous, self.discrete, self.categorical = colType(df)

    def missingVals(self, mode):
        ''' the controller '''
        if mode == 1:
            return self.fillUsingClass()
        if mode == 2:
            return self.fillUsingAll()

    def fillUsingClass(self):  # groupby class and Fill using all
        '''this function will fill the missing value according to the classify column '''
        for col in self.df.columns:
            if col in self.continuous:
                self.df[col] = self.df.groupby(self.classifier)[col].transform(
                    lambda grp: grp.fillna(np.mean(grp)))  # every NaN will fill with mean value

            elif col in self.discrete:
                self.df[col] = self.df.groupby(self.classifier)[col].transform(
                    lambda x: x.fillna(x.mode().iloc[0]))  # every NaN will fill with most frequent value
                self.df[col] = self.df[col].astype("int64")  # converts to int


            elif col in self.categorical:
                self.df[col] = self.df.groupby(self.classifier)[col].transform(
                    lambda x: x.fillna(x.mode().iloc[0]))  # every NaN will fill with most frequent value

        return self.df

    def fillUsingAll(self):  # Sklearn
        '''this function will fill the missing value according to all the values in the column '''
        for col in self.df.columns:
            if col in self.continuous:
                imputer = SimpleImputer(missing_values=np.nan, strategy='mean')  # every NaN will fill with mean value
                self.df[col] = imputer.fit_transform(self.df[col].values.reshape(-1, 1))[:, 0]

            elif col in self.discrete:
                imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
                self.df[col] = imputer.fit_transform(self.df[col].values.reshape(-1, 1))[:,
                               0]  # every NaN will fill with most frequent value
                self.df[col] = self.df[col].astype("int64")  # converts to int

            elif col in self.categorical:
                imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
                self.df[col] = imputer.fit_transform(self.df[col].values.reshape(-1, 1))[:,
                               0]  # every NaN will fill with most frequent value

        return self.df
