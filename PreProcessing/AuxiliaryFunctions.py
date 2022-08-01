import math

from pandas.core.dtypes.common import is_numeric_dtype

'''
# Topaz Aakal -> 318644549
# Marat Zinger -> 332689405
# Afik Danan -> 208900175
'''


def castColToInt(series):
    ''' this function converts the column to int64 '''
    series_int = series.astype("int64", copy=False)
    return series_int


def castToInt_check(series):
    ''' this function will check if it is possible to convert a column to int64, so we can tell it is discrete  '''
    if (is_numeric_dtype(series)):
        for i in range(len(series)):
            try:
                if (math.isnan(series[i])):  # in case of missing values ignore NaN
                    continue
                elif not float((series[i]).item()).is_integer():  # loop all the value and check if their all integers
                    return False
            except KeyError:
                continue
        return True
    return False


def colType(df):
    ''' this function classify the columns by their type '''
    categorical = []
    continuous = []
    discrete = []

    for col in df.columns:
        if (castToInt_check(df[col])):  # true if can be converted to integer
            if df[col].isna().sum() == 0:
                castColToInt(df[col])
            discrete.append(col)

        elif df[col].dtype == "float64":  # continuous features are with float values
            continuous.append(col)
        else:
            categorical.append(col)  # if not either than categorical

    return continuous, discrete, categorical
