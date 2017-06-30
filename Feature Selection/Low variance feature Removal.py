'''
Why remove low variance Features?(Quick and Dirty Solution)

1. They are considered to have less predictive power.
2. Can also cause numerical problems and cause a model to crash.

For more detail read the post of Thiago G. Martins Data Scientist at Yahoo!

https://tgmstat.wordpress.com/2014/03/06/near-zero-variance-predictors/

This link will give you insights whether you should thow the data or not.

However, throwing data away should be avoided, if possible. One solution for the near-variance predictor is to collect more data
'''

# VarianceThreshold removes all features whose variance doesn’t meet some threshold

from sklearn.feature_selection import VarianceThreshold


X = [[0, 0, 1],
	 [0, 1, 0], 
	 [1, 0, 0], # This column ahve only 1 data point this will be removed in the output. 
	 [0, 1, 1], 
	 [0, 1, 0], 
	 [0, 1, 1]]
sel = VarianceThreshold(threshold=(.8 * (1 - .8)))

print(sel.fit_transform(X))

print(' VarianceThreshold has removed the first column, which has a probability p = 5/6 > .8 of containing a zero.')