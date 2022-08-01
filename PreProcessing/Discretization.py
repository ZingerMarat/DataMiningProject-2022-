import entropy_based_binning as ebb
import numpy as np
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer

from PreProcessing.AuxiliaryFunctions import colType

'''
# Topaz Aakal -> 318644549
# Marat Zinger -> 332689405
# Afik Danan -> 208900175
'''


class Discretization:
    ''' a class for discretize the dataframe  '''

    def __init__(self, df, classifier, bins):
        self.classifier = classifier
        self.df = df.copy()  # so we won't affect the original
        self.bins = bins
        self.continuous, self.discrete, self.categorical = colType(df)

    def equalWDisc(self):
        """ params: dataframe , size of bin [type: int]
            Equal-Width Discretizer.
        """
        for col in (self.continuous + self.discrete):
            if col != self.classifier and (
                    col in self.discrete and len(self.df[col].unique()) > self.bins) or col in self.continuous:
                arr = np.array(self.df[col])
                arr = arr.reshape((len(arr), 1))
                bins = KBinsDiscretizer(n_bins=self.bins, encode='ordinal', strategy='uniform')  # from sklearn
                self.df[col] = bins.fit_transform(arr)
                self.df[col] = self.df[col].astype("int64")  # converts to int
        return self.df

    def equalFreqDisc(self):
        """ params:dataframe , size of bin [type: int]
            Equal-frequency Discretizer.
        """
        for col in (self.continuous + self.discrete):
            if col != self.classifier and (
                    col in self.discrete and len(self.df[col].unique()) > self.bins) or col in self.continuous:
                inBin = int(len(self.df[col]) / self.bins)
                for i in range(self.bins):
                    for j in range(i * inBin, (i + 1) * inBin):
                        if j >= len(self.df[col]):
                            break
                        self.df.at[j, col] = i
                self.df[col] = self.df[col].astype("int64")  # converts to int
        return self.df

    def entropyBasedDiscretizer(self):
        ''' using entropy to choose the best binning and discretize '''
        for name in self.continuous:
            x1 = self.df[name].astype("int64").to_numpy()
            a1 = ebb.bin_array(x1, nbins=self.bins, axis=0)
            list1 = a1.tolist()
            d1 = pd.DataFrame({name: list1})
            self.df.update(d1)
            self.df[name] = self.df[name].astype("int64")
        return self.df
