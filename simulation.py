import numpy as np
import random
from sklearn.neural_network import MLPClassifier

num_of_customer = 100

#final table

#account balance (guassian)
table_balance = []
for i in range(num_of_customer):
	mu = random.random()*1000
	si = random.random()*100
	month_balance = np.random.normal(mu, si, 36)
	#print(month_balance)
	table_balance.append(list(month_balance))

#numbers of transaction (poisson)
table_transaction = []
for i in range(num_of_customer):
	lambda_p = random.random()*100
	month_transaction = np.random.poisson(lambda_p,36)
	table_transaction.append(list(month_transaction))

#number of times visiting ATM
table_ATM = []
for i in range(num_of_customer):
	lambda_p = random.random()*1
	ATM_times = np.random.poisson(lambda_p,36)
	table_ATM.append(list(ATM_times))

#the label of the success of the online channel
label_P_onlineChannel = []
for i in range(num_of_customer):
	if(sum(table_balance[i]) > (2500*36) or sum(table_transaction[i]) > 50*36):
		label_P_onlineChannel.append(1)
	else:
		label_P_onlineChannel.append(0)


#final table
result = np.hstack((np.array(table_balance),np.array(table_transaction)))
result = np.hstack((result,np.array(table_ATM)))
#print(result[0])

#apply neural network classifier
clf = MLPClassifier(solver='sgd', alpha=1e-3, hidden_layer_sizes=(100, 2), random_state=1)
clf = clf.fit(result, np.transpose(label_P_onlineChannel))
print(clf.score(result, np.transpose(label_P_onlineChannel)))




