import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib as mpl
import math

from pprint import pprint
from matplotlib.colors import ListedColormap
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from scipy.special import logsumexp


#Common functions

eps = 1e-100

def standardize(x):
    mean = np.mean(x,axis=0)
    std = np.std(x,axis=0,ddof=1)
    return (x-mean)/std

def standardize_m_s(X,mean,std):
    return (X - mean)/std

def init_params(shape_tuple):
    return np.random.uniform(-1,1,shape_tuple)

def add_bias(x_input):
    arr_ones = np.ones((x_input.shape[0],1))
    X = np.hstack((arr_ones,x_input))
    return X

def sigmoid(x):
    sig = 1/(1 + np.exp(-x))
    #print(f"sigmoid shape: {sig.shape}")
    return sig

def compute_gradient_logistic(x,y,theta):
    #print(f"x-shape: {x.T.shape}")
    #print(f"y-shape: {y.shape}")
    grad = np.dot(x.T,y-sigmoid(x@theta))
    #print(f"grad-shape: {grad.shape}")
    return grad

def compute_logistic_cost(x,y,theta):
    #loss_t = y*ln(sigmoid(x@theta) + (1-y)*ln(1-sigmoid(x@theta)))
    return np.sum(y*np.log(sigmoid(x@theta)) + (1-y)*np.log((1-sigmoid(x@theta)) + eps))

def compute_logistic_output(x,theta):
    return sigmoid(x@theta)

#(1/(std * np.sqrt(2*np.pi))) * (np.e ** (-1 * (x-mean)**2 / (2 * std**2)))
def convert_gaussian(x_train, x_test):
    mean = np.mean(x_train,axis=0)
    std = np.std(x_train,axis=0,ddof=1)
    return gaussian(mean,std,x_test)
 

def gaussian(mean,std,x):
    return (1/(std * np.sqrt(2*np.pi))) * (np.e ** (-1 * (x-mean)**2 / (2 * std**2)))

def get_metrics(y,y_pred):
    total=0
    correct=0
    TP=0
    FP=0
    FN=0
    for (actual,pred) in zip(y,y_pred):
        total+=1
        if(actual==pred):
            correct+=1
            if(actual==1):
                TP+=1
        else:
            if(pred==1):
                FP+=1
            if(pred==0):
                FN+=1
                
    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
    f1 = 2*precision*recall/(precision+recall)
    accuracy = correct/total
    
    return (precision,recall,f1,accuracy)


def get_accuracy(y,y_pred):
    return np.sum((y==y_pred).astype(int))/y.shape[0]


def convert_binary_mean(x):
    mean = np.mean(x, axis=0)
    x_bin =  (x>mean).astype(int)
    return x_bin


def standardize_pd(df):
    return (df-df.mean())/df.std()

def standardize_pd_m_s(df,m,s):
    return (df-m)/s
    

    
    
    
    
    
    
    
    
