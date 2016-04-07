# Libraries
import numpy as np
import pandas as pd
from xgboost.sklearn import XGBClassifier
from sklearn.preprocessing import LabelEncoder

# Variables
le = LabelEncoder()

# Retrieve train and test datasets
def load_data(base_dir = '.'):
	print("Loading data")
	trn = pd.read_csv(base_dir + '/train_users_2.csv')
	tst = pd.read_csv(base_dir + '/test_users.csv')
	return trn, tst

# Filter, clean, convert attributes
def preprocess(trn, tst):
	print("Preprocessing")
	lbls = trn['country_destination'].values
	trn = trn.drop(['country_destination'], axis = 1)
	
	# Concatenate data into one struture
	dat = pd.concat((trn, tst), axis = 0, ignore_index = True)
	dat = dat.drop(['id', 'date_first_booking'], axis = 1)
	dat = dat.drop(['date_account_created'], axis = 1)
	dat = dat.drop(['timestamp_first_active'], axis = 1)
	
	# Replace missing data with -1
	dat = dat.fillna(-1)
	
	# Clean age
	age = dat.age.values
	dat['age'] = np.where(np.logical_and(15 < age, age < 99), age, -1)
	
	# Convert nominal attributes to numerical data
	feats = ['gender', 'signup_method', 'signup_flow', 'language', 'affiliate_channel', 'affiliate_provider', 
                 'first_affiliate_tracked', 'signup_app', 'first_device_type', 'first_browser']
	for feat in feats:
		dat_dummy = pd.get_dummies(dat[feat], prefix = feat)
		dat = dat.drop([feat], axis = 1)
		dat = pd.concat((dat, dat_dummy), axis = 1)
	
	# Split train and test data
	split = trn.shape[0]
	vals  = dat.values
	X = vals[: split]
	y = le.fit_transform(lbls)
	X_tst  = vals[split :]
	id_tst = tst['id']
	
	return X, y, X_tst, id_tst
	
# Build classifier
def build_model(X, y):
	print("Fitting classifier")
	clf = XGBClassifier(max_depth = 5, learning_rate = 0.25, n_estimators = 32,
                            objective = 'multi:softprob', subsample = 0.6, colsample_bytree = 0.6)
	clf.fit(X, y)
	return clf
	
# Predict outputs for test data
def predict(clf, X_tst, id_tst):
	y_pred = clf.predict_proba(X_tst)
	
	# Take top 5 most likely countries
	ids = []
	dst = []
	for i, idn in enumerate(id_tst):
		ids += [idn] * 5
		srt  = np.argsort(y_pred[i])[::-1]
		dst += le.inverse_transform(srt)[: 5].tolist()
	return ids, dst

# Write submission
def output(ids, dst):
	print("Writing output")
	sub = pd.DataFrame(np.column_stack((ids, dst)),
                           columns = ['id', 'country'])
	sub.to_csv('sub.csv', index = False)

# Combine all steps
def pipeline():
	trn, tst = load_data()
	X, y, X_tst, id_tst = preprocess(trn, tst)
	clf = build_model(X, y)
	ids, dst = predict(clf, X_tst, id_tst)
	output(ids, dst)

if __name__ == "__main__":
	pipeline()
