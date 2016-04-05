# Libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from xgboost.sklearn import XGBClassifier
 
# Variables
le = LabelEncoder()
 
# Retrieve train and test datasets
def load_data(base_dir = '.'):
    print("Loading data")
    df_train = pd.read_csv(base_dir + '/train_users_2.csv')
    df_test = pd.read_csv(base_dir + '/test_users.csv')
    return df_train, df_test
 
# Removing id and date_first_booking
def preprocess(df_train, df_test):
    print("Preprocessing")
    lbls = df_train['country_destination'].values
    df_train = df_train.drop(['country_destination'], axis = 1)
     
    # Concatenating data into one struture
    df_all = pd.concat((df_train, df_test), axis = 0, ignore_index=True)
    df_all = df_all.drop(['id', 'date_first_booking'], axis = 1)
    df_all = df_all.drop(['date_account_created'], axis = 1)
    df_all = df_all.drop(['timestamp_first_active'], axis = 1)
     
    # Filling empty data
    df_all = df_all.fillna(-1)
     
    # Cleaning age
    avg = df_all.age.values
    df_all['age'] = np.where(np.logical_or(avg < 15, avg > 99), -1, avg)
     
    # Convert nominal attributes to numerical data
    feats = ['gender', 'signup_method', 'signup_flow', 'language', 'affiliate_channel', 'affiliate_provider', 
                 'first_affiliate_tracked', 'signup_app', 'first_device_type', 'first_browser']
    for feat in feats:
        df_all_dummy = pd.get_dummies(df_all[feat], prefix=feat)
        df_all = df_all.drop([feat], axis = 1)
        df_all = pd.concat((df_all, df_all_dummy), axis = 1)
     
    # Split train and test data
    split = df_train.shape[0]
    vals = df_all.values
    X = vals[: split]
    y = le.fit_transform(lbls) 
    X_test = vals[split :]
    id_test = df_test['id']
     
    return X, y, X_test, id_test
     
# Build classifier
def build_model(X, y):
    print("Fitting classifier")
    xgb = XGBClassifier(max_depth = 4, learning_rate = 0.25, n_estimators = 25,
                            objective = 'multi:softprob', subsample = 0.6, colsample_bytree = 0.6)
    xgb.fit(X, y)
    return xgb
     
# Predict outputs for test data
def predict(xgb, X_test, id_test):
    y_pred = xgb.predict_proba(X_test)
     
    # Take top 5 most likely countries
    ids = []
    cts = []
    for i, idx in enumerate(id_test):
        ids += [idx] * 5
        cts += le.inverse_transform(np.argsort(y_pred[i])[::-1])[:5].tolist()
    return ids, cts
 
# Write submission
def output(ids, cts):
    print("Writing output")
    sub = pd.DataFrame(np.column_stack((ids, cts)), columns = ['id', 'country'])
    sub.to_csv('sub.csv', index = False)
 
# Combine all steps
def pipeline():
    df_train, df_test = load_data()
    X, y, X_test, id_test = preprocess(df_train, df_test)
    xgb = build_model(X, y)
    ids, cts = predict(xgb, X_test, id_test)
    output(ids, cts)
 
if __name__ == "__main__":
    pipeline()
