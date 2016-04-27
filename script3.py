import numpy as np
import pandas as pd
import math
from sklearn.preprocessing import LabelEncoder

#http://machinelearningmastery.com/naive-bayes-classifier-scratch-python/
def mean(numbers):
	return sum(numbers)/float(len(numbers))
 
def stdev(numbers):
	avg = mean(numbers)
	variance = sum([pow(x-avg,2) for x in numbers])/float(len(numbers)-1)
	return math.sqrt(variance)

#gaussian
def gaussian(x, mean, stdev):
	exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
	return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent

def getClassProb(labels,classes):
	'''
	return P(Ci) as a list
	'''
	length=labels.shape[0]
	return [np.sum(labels==c)/float(length) for c in classes]

def getCondProb(t,className,X,labels):
	indices = [i for i,c in enumerate(labels) if c==className]
	length=len(indices)
	elements=[X[i] for i in indices]
	result=1
	for i in range(5):
		data=t[i]
		if isinstance(data,str):
			#categorical
			num = len([e for e in elements if e[i]==data])
			prob = num/float(length)
			result*=prob
		else:
			numbers=[e[i] for e in elements if math.isnan(e[i])==False]
			m=mean(numbers)
			std=stdev(numbers)
			prob=gaussian(data,m,std)
			result*=prob
	return result

train_data = pd.read_csv("train_users_2.csv")
test_data = pd.read_csv("test_users.csv")
labels = train_data['country_destination'].values
classes = ['US', 'FR', 'CA', 'GB', 'ES', 'IT', 'PT', 'NL','DE', 'AU', 'NDF', 'other']

drop = ['id','date_account_created','timestamp_first_active','date_first_booking','signup_flow','affiliate_channel','affiliate_provider','first_affiliate_tracked','first_device_type','first_browser']

piv_train = train_data.shape[0]
train_data = train_data.drop(drop, axis=1)
train_data = train_data.drop(['country_destination'], axis=1)

test_id = test_data['id']
test_data = test_data.drop(drop, axis=1)

df_all = pd.concat((train_data,test_data), axis=0, ignore_index=True)

vals = df_all.values
X = vals[:piv_train]
#le = LabelEncoder()
#y = le.fit_transform(labels)   
X_test = vals[piv_train:]

classProb = getClassProb(labels,classes)

prediction=[]
for x in X_test:
	condProb=[getCondProb(x,classes[i],X,labels)*classProb[i] for i in range(12)]
	index=condProb.index(max(condProb))
	print(classes[index])
	prediction.append(classes[index])

#https://www.kaggle.com/bibins/airbnb-recruiting-new-user-bookings/fisrtscript/run/111526
sample_submission = {}
sample_submission['id'] = test_id
sample_submission['country'] = prediction
s = pd.DataFrame.from_dict(sample_submission)
s.to_csv('sub.csv',index=False)
