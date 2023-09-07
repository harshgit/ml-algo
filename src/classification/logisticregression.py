from utils import commonutils as util
from sklearn.model_selection import train_test_split
import numpy as np


def classify_spam(X):
    X[X>0.5] = 1
    X[X<0.5] = 0
    return X


#read data
spam_data = pd.read_csv('spambase.data',sep=",",header=None)
#randomize
spam_data = spam_data.sample(frac=1,random_state=0)
#test-train split
X = spam_data.iloc[:,:57].to_numpy()
Y = spam_data.iloc[:,57].to_numpy()
x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.33, random_state=0)

x_train_stan = util.standardize(x_train)
#add bias
x_train_stan = util.add_bias(x_train_stan)
#standardize using train data
x_t_mean =np.mean(x_train, axis=0)
x_t_std = np.std(x_train, axis=0, ddof=1)
#use the mean and std from train to standardize test
x_test_stan = util.standardize_m_s(x_test,x_t_mean,x_t_std)
x_test_stan = util.add_bias(x_test_stan)


#initialize thetas
theta = util.init_params((x_train_stan.shape[1],))

#start gradient descent/ascent
learning_rate = 0.01
N = X.shape[0]
i = 0
loss_prev =0
while True:
    grad = util.compute_gradient_logistic(x_train_stan,y_train,theta)
    theta = theta + learning_rate*(grad/N)
    loss = util.compute_logistic_cost(x_train_stan,y_train,theta)
    if((abs(loss_prev - loss) < 2**(-23)) or i>=1500):
        break
    i+=1
    loss_prev = loss
    
    
    
print(f"final thetas : {theta}")
print(f"Total Iterations: {i}")
print(f"Final Loss: {loss}")

#Classification
y_predicted = classify_spam(util.compute_logistic_output(x_test_stan,theta))


#Metrics
precision,recall,f1,accuracy = util.get_metrics(y_test,y_predicted)
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1: {f1}")
print(f"Accuracy: {accuracy}")