from sklearn import preprocessing

from PreProcessing.AuxiliaryFunctions import colType

'''
# Topaz Aakal -> 318644549
# Marat Zinger -> 332689405
# Afik Danan -> 208900175
'''


class Normalization:
    ''' A class fo normalizing data'''

    def __init__(self, df):
        self.df = df

    def norm(self):
        ''' the function will normalize the dataframe using Min-Max Scale, values will be between 0-1 '''
        continuous, discrete, categorical = colType(self.df)
        colToNorm = continuous + discrete

        if len(colToNorm) > 0:
            scaler = preprocessing.MinMaxScaler()  # from sklearn
            self.df[colToNorm] = scaler.fit_transform(self.df[colToNorm])  # updating the column

        return self.df
