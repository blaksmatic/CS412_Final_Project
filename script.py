import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from xgboost.sklearn import XGBClassifier

#Loading data
df_train = pd.read_csv('./train_users_2.csv')
df_test = pd.read_csv('./test_users.csv')
labels = df_train['country_destination'].values
df_train = df_train.drop(['country_destination'], axis=1)
id_test = df_test['id']
piv_train = df_train.shape[0]

df_all = pd.concat((df_train, df_test), axis=0, ignore_index=True)

#Removing id and date_first_booking
df_all = df_all.drop(['id', 'date_first_booking'], axis=1)
df_all = df_all.drop(['date_account_created'], axis=1)
df_all = df_all.drop(['timestamp_first_active'], axis=1)
df_all = df_all.drop(['first_device_type'], axis=1)
df_all = df_all.drop(['first_browser'], axis=1)
df_all = df_all.drop(['signup_app'], axis=1)

#Filling none data
df_all = df_all.fillna(-1)
#Cleaning data: age
av = df_all.age.values
df_all['age'] = np.where(np.logical_or(av<12, av>80), -1, av)

#use pandas to convert classified data into numerical data
ohe_feats = ['gender', 'signup_method', 'signup_flow', 'language', 'affiliate_channel', 'affiliate_provider', 'first_affiliate_tracked']
for f in ohe_feats:
	df_all_dummy = pd.get_dummies(df_all[f], prefix=f)
	df_all = df_all.drop([f], axis=1)
	df_all = pd.concat((df_all, df_all_dummy), axis=1)

#Splitting train and test
vals = df_all.values
X = vals[:piv_train]
le = LabelEncoder()
y = le.fit_transform(labels)   
X_test = vals[piv_train:]

#Classifier
xgb = XGBClassifier(max_depth=4, learning_rate=0.25, n_estimators=25,
					objective='multi:softprob', subsample=0.6, colsample_bytree=0.6)                  
xgb.fit(X, y)
y_pred = xgb.predict_proba(X_test)  

#Taking the 5 classes with highest probabilities
ids = []  #list of ids
cts = []  #list of countries
for i in range(len(id_test)):
	idx = id_test[i]
	ids += [idx] * 5
	cts += le.inverse_transform(np.argsort(y_pred[i])[::-1])[:5].tolist()

#Generate submission
sub = pd.DataFrame(np.column_stack((ids, cts)), columns=['id', 'country'])
sub.to_csv('sub.csv',index=False)