
'''
In statistics, imputation is the process of replacing missing data with substituted values

Missing value create a lot of problem id were not handled properly in machine learning 

I the real world all the data contains missing va;ues, like NaNs, etc.

Missing datapoints can be replaced by the mean value of that varaible. Their are also many more techniques for replacing missing values. 


Sklearn have a class Imputer to handle this problem. 

'''

import numpy as np
from sklearn.preprocessing import Imputer
imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
imp.fit([[1, 2], [np.nan, 3], [7, 6]])

X = [[np.nan, 2], [6, np.nan], [7, 6]]

print('NaN Value will be replaced my te mean value')
print(imp.transform(X))  







