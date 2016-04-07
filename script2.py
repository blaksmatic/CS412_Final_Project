import numpy as np
import pandas as pd

#Loading data
df_test = pd.read_csv('./test_users.csv')
id_test = df_test['id']

#Removing id and date_first_booking
df_test = df_test.drop(['signup_method'], axis=1)
df_test = df_test.drop(['affiliate_channel'], axis=1)
df_test = df_test.drop(['gender'], axis=1)
df_test = df_test.drop(['signup_flow'], axis=1)
df_test = df_test.drop(['date_account_created'], axis=1)
df_test = df_test.drop(['timestamp_first_active'], axis=1)

#Filling none data
df_test = df_test.fillna(-1)

#Cleaning data: age
av = df_test.age.values
df_test['age'] = np.where(np.logical_or(av<12, av>80), -1, av)

ids = []
cities = []
for i in range(len(id_test)):
	ids.append(id_test[i])
	ids.append(id_test[i])
	if df_test["language"][i] == "en":
		cities.append("US")
		cities.append("NDF")
	else:
		cities.append("NDF")
		cities.append("other")


#Generate submission
print (len(ids),len(cities))
sub = pd.DataFrame(np.column_stack((ids, cities)), columns=['id', 'country'])
sub.to_csv('sub.csv',index=False)