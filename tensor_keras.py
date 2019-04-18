from __future__ import print_function, unicode_literals
from numpy.random import seed
from sklearn.preprocessing import normalize
seed(7)
from tensorflow import set_random_seed
set_random_seed(7)
import pandas as pd
from keras.models import Sequential
import sklearn
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
import numpy as np
from scipy.stats import kendalltau, spearmanr, pearsonr
from six import string_types
from six.moves import xrange as range
from sklearn.metrics import confusion_matrix, f1_score, SCORERS

from qwk import quadratic_weighted_kappa


import time
start = time.time()

# COLUMNS1 = ['pos_unique','misspelled_words','exesntial_there','superlative_adj',
# 'predeterminants','coordinating_conjuctions', 'words','characters','min_st_sum',
# 'max_st_sum','c_centrality','diameter',
#     'density_diff','center','top_words_comp','common_length','eigen','grammar_fn','common_length1','singular_s',
#     'plural_s','vb','vbd','vbg','vbn','vbp','vbz','ing']
# LABEL = ['score']
X = pd.read_csv("/home/compute/work/aee/essay_evaluation_codes/domain6/4/all.csv", skipinitialspace=True,skiprows=0)

# split into input (X) and output (Y) variables
X_new =pd.read_csv("/home/compute/work/aee/essay_evaluation_codes/domain6/4/predict.csv",
	 skipinitialspace=True,skiprows=0)


# split into input (X) and output (Y) variables
X=X.values

Y=X[:, -1]
X=X[:,1:-1]
print(X)
X.astype(float)
Y.astype(float)


X_new=X_new.values

Y_new=X_new[:,-1]

X_new=X_new[:,1:-1]


# X.astype(float)
X_new.astype(float)
Y_new.astype(float)

print ('X')
print (X)
print ('Y')
print(Y) 
print("x_new")
print (X_new)
print("ynew")
print (Y_new)
# define the wider model
# define wider model
def wider_model():
	# create model
	model = Sequential()
	model.add(Dense(30, input_dim=36, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal'))
	# Compile model
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model
# define base model
def baseline_model():
	# create mode
	model = Sequential()

	model.add(Dense(30, input_dim=36, kernel_initializer='normal', activation='relu'))
	model.add(Dense(20, kernel_initializer='normal', activation='relu'))
	model.add(Dense(15,kernel_initializer='normal',activation='relu'))
	model.add(Dense(6, kernel_initializer='normal'))

	model.add(Dense(1, kernel_initializer='normal'))
	# Compile model
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasRegressor(build_fn=baseline_model, epochs=200, batch_size=50, verbose=0)))
pipeline = Pipeline(estimators)
kfold = KFold(n_splits=10, random_state=seed)
results = cross_val_score(pipeline, X, Y, cv=kfold)
print("Wider: %.2f (%.2f) MSE" % (results.mean(), results.std()))
#prediction


# X_new =scaler.fit_transform(X_new)
pipeline.fit(X,Y)
ynew = pipeline.predict(X_new)


# show the inputs and predicted outputs
ynew=np.around(ynew, decimals=0)
for i in range(len(ynew)):
	print("X=%s, Predicted=%s" % (Y_new[i], ynew[i]))
# pred=pred.tolist()
# Y_new=Y_new.tolist()
qwk = quadratic_weighted_kappa(Y_new,ynew)
print(qwk)

end = time.time()
print(end - start)
