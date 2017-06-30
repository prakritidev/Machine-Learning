# one hot encoding used when we have to change categorical data into numberucal 
#["male", "from US", "uses Internet Explorer"] could be expressed as [0, 1, 3]


from sklearn import preprocessing
import numpy as np

X = [[0, 0, 3], 
	[1, 1, 0], 
	[0, 2, 1], 
	[1, 0, 2]]
enc = preprocessing.OneHotEncoder()
enc.fit(X)  


print(enc.transform([[0, 1, 3]]).toarray())
