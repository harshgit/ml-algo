from utils import commonutils as util
import math
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

#read data
spam_data = pd.read_csv('spambase.data',sep=",",header=None)
#randomize
spam_data = spam_data.sample(frac=1,random_state=0)
#test-train split
X = spam_data.iloc[:,:57].to_numpy()
Y = spam_data.iloc[:,57].to_numpy()
x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.33, random_state=0)

print(x_train.shape)
N = x_train.shape[0]
x_train_stan = util.standardize(x_train)
#standardize using train data
x_t_mean =np.mean(x_train, axis=0)
x_t_std = np.std(x_train, axis=0, ddof=1)
#use the mean and std from train to standardize test
x_test_stan = util.standardize_m_s(x_test,x_t_mean,x_t_std)


#train split on spam and not spam
#ind_spam = [i for i,row in enumerate(x_train_stan) if y_train[i] == 1]
ind_spam = []
ind_non_spam = []
for i,row in enumerate(x_train_stan):
    if(y_train[i] == 1):
        ind_spam.append(i)
    else:
        ind_non_spam.append(i)

x_spam = x_train_stan[ind_spam]
x_non_spam = x_train_stan[ind_non_spam]

print(f"spam {x_spam.shape}")
print(f"non spam {x_non_spam.shape}")




#Get the probabilities of test data using mean and std of
x_spam_g = util.convert_gaussian(x_spam,x_test)
x_non_spam_g = util.convert_gaussian(x_non_spam,x_test)

print(f"gaussian model {x_spam_g.shape}")
print(f"gaussian model {x_non_spam_g.shape}")



#Convert to probabilities by dividing by sum of all rows
p_x_spam = x_spam_g/np.sum(x_spam_g,axis=0)
p_x_non_spam = x_non_spam_g/np.sum(x_non_spam_g,axis=0)

#p_x_spam = x_spam_g
#p_x_non_spam = x_non_spam_g

#prior probabilities
p_spam = x_spam.shape[0]/N
p_non_spam = x_non_spam.shape[0]/N

print(f"spam prior {p_spam}")
print(f"non spam prior {p_non_spam}")


#Using naive baye's to find probability of test example being a spam or not spam
#p(spam|x)  = p(spam) *  p(x1|spam) * p(x2|spam) * .....p(x56|spam)
#p(Not-spam|x)  = p(Not-spam) *  p(x1|Not-spam) * p(x2|Not-spam) * .....p(x56|Not-spam)
#not log space
#p_test_spam = p_spam*np.prod(p_x_spam,axis=1)
#p_test_non_spam = p_non_spam*np.prod(p_x_non_spam,axis=1)

#log space
eps = 1e-100
#p_x_spam[p_x_spam<eps] = eps
p_x_spam_log = np.log((p_x_spam + eps))

#p_x_non_spam[p_x_non_spam<eps] = eps
p_x_non_spam_log = np.log((p_x_non_spam + eps))


p_test_spam = math.log(p_spam) + np.sum(p_x_spam_log, axis=1)
p_test_non_spam =  math.log(p_non_spam) + np.sum(p_x_non_spam_log, axis=1)

print(f"p_test_spam {p_test_spam.shape}")
print(f"p_test_non_spam {p_test_non_spam.shape}")



y_predicted = (p_test_spam > p_test_non_spam).astype(int)


print(y_predicted)
print(y_test)

#Metrics
precision,recall,f1,accuracy = util.get_metrics(y_test,y_predicted)
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1: {f1}")
print(f"Accuracy: {accuracy}")





