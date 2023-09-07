import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib as mpl
import math
from sklearn.model_selection import train_test_split
from utils import commonutils as util

eps = 1e-100

class Node():
    def __init__(self,index):
        self.branches ={}
        self.index = index
        
    def add_branch(self,key,value):
        self.branches[key]= value
        
    def get_all_branches(self):
        return self.branches
    
    def get_branch(self,key):
        return self.branches[key]
    
    def get_index(self):
        return self.index
        

def ID3(data,list_attribute,default):
    if(data.empty):
        return default
    elif(data["Y"].value_counts().shape[0] ==  1): #check if there is only one class left
        return data.iloc[0,-1]
    elif(len(list_attribute)==0):
        mode = list(data["Y"].value_counts().index)[0]
        return mode #mode
    
    else:
        
        list_attribute = get_sorted_feature_by_entropy(data, list_attribute)
        feature = list_attribute.pop()# choose the best feature
        if(feature == 'Y'):
            feature = list_attribute.pop()
        node = Node(feature) #create the node with least entropy
        data_g = data.groupby(feature)
        for i in range(2):
            if i in data_g.groups:
                data_i =  data_g.get_group(i)
                subtree = ID3(data_i, list_attribute, get_mode(data_i))
                node.add_branch(i,subtree)
            else:
                mode_rem = get_mode(data_g)
                mode_rem = list(data_g["Y"].value_counts().index)[0][1]
                node.add_branch(i,mode_rem) # add the mode of the training data going down
            
        return node
                      
        
    
def get_sorted_feature_by_entropy(data, list_attribute):
    entropy_list = []
    for col in list_attribute:
        e = get_entropy_spam(data,col)
        entropy_list.append((col,e))
    
    #sort by entropy decreasing
    e_s_list = [col for (col,e) in sorted(entropy_list,key=lambda x: x[1], reverse=True)]
    return e_s_list
    
def get_mode(df):
    return list(df["Y"].value_counts().index)[0]
    
    
def get_entropy_spam(df, column):
    #x1
    total_rows = df.shape[0]
    x1 = df.groupby(column)
    if 1 in x1.groups:
        x1_T = x1.get_group(1)
        #x1_T
        total_t = x1_T.shape[0]
        p_1= x1_T[x1_T['Y'] == 1].shape[0]/total_t
        p_0= x1_T[x1_T['Y'] == 0].shape[0]/total_t

        E1_T = -p_1*math.log2(p_1 + eps) + (-p_0*math.log2(p_0 + eps))
        P_T =  x1_T.shape[0]/total_rows
    else:
        return 0
        
    if 0 in x1.groups:
        #X1_F
        x1_F = x1.get_group(0)
        total_f = x1_F.shape[0]
        p_1= x1_F[x1_F['Y'] == 1].shape[0]/total_f
        p_0= x1_F[x1_F['Y'] == 0].shape[0]/total_f

        E1_F = -p_1*math.log2(p_1 + eps) + (-p_0*math.log2(p_0 + eps))
        P_F =  x1_F.shape[0]/total_rows
    else:
        return 0 

    E1  = P_T*E1_T + P_F*E1_F
    return E1

def classify_decision(node,x):
    
    while True:
        idx = node.get_index()
        branches = node.get_all_branches()
        child = branches[x[idx]]

        if child == 1 or child == 0:
            return child
        else:
            node = child

#read data
spam_data = pd.read_csv('spambase.data', header=None, sep=',')
spam_data.sample(frac=1,random_state=0)
X = spam_data.iloc[:,:57].to_numpy()
Y = spam_data.iloc[:,57].to_numpy()

x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.33, random_state=0)


x_train_stan = util.standardize(x_train)
#standardize using train data
x_t_mean =np.mean(x_train, axis=0)
x_t_std = np.std(x_train, axis=0, ddof=1)
#use the mean and std from train to standardize test
x_test_stan = util.standardize_m_s(x_test,x_t_mean,x_t_std)

### Binarize train and test to contain discrete values 1 and 0
# 1 for value > mean
# 0 for value < mean

x_train_stan = util.convert_binary_mean(x_train_stan)
x_test_stan = util.convert_binary_mean(x_test_stan)


#train split on spam and not spam
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


## Compute entropy w.r.t each feature and store in sorted list

df_train =  pd.DataFrame(x_train_stan)
df_train["Y"] = y_train
df_test = pd.DataFrame(x_test_stan)
df_test["Y"] = y_test

### Implement ID3 ##
tree = ID3(df_train,df_train.columns,get_mode(df_train))
print(tree)

y_pred = []
for idx,test in df_test.iterrows():
    y = classify_decision(tree,test)
    y_pred.append(y)
    
y_pred_arr = np.array(y_pred)

#Metrics
precision,recall,f1,accuracy = util.get_metrics(y_test,y_pred_arr)
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1: {f1}")
print(f"Accuracy: {accuracy}")