# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 23:21:21 2020

@author: ulyss
"""

import pandas
import numpy as np
from collections import Counter
from cvxopt import matrix, solvers

#Load of training sets

X_train_file = "./Xtr"

X_train_0 = pandas.read_csv(X_train_file + "0.csv")
X_train_1 = pandas.read_csv(X_train_file + "1.csv")
X_train_2 = pandas.read_csv(X_train_file + "2.csv")


Y_train_file = "./Ytr"

Y_train_0 = pandas.read_csv(Y_train_file + "0.csv")
Y_train_1 = pandas.read_csv(Y_train_file + "1.csv")
Y_train_2 = pandas.read_csv(Y_train_file + "2.csv")

#Change of label 0 into -1
for index, row in Y_train_0.iterrows():
    if row["Bound"] == 0:
        Y_train_0.at[index, "Bound"] = -1

for index, row in Y_train_1.iterrows():
    if row["Bound"] == 0:
        Y_train_1.at[index, "Bound"] = -1
        
for index, row in Y_train_2.iterrows():
    if row["Bound"] == 0:
        Y_train_2.at[index, "Bound"] = -1

Y_0 = np.c_[np.array(Y_train_0["Bound"])]
Y_1 = np.c_[np.array(Y_train_1["Bound"])]
Y_2 = np.c_[np.array(Y_train_2["Bound"])]

#Parameters
N = 2000
k_spectrum = 10
lamb = 0.1

def k_n(prot_A, prot_B, n):
    """
    Compute K(x_i,x_j) for a given k_spectrum
    
    -prot_A is a string for x_i
    -prot_B is a string for x_j
    -n is an int for k_spectrum
    """
    spectre_A = []
    for i in range(len(prot_A)-n+1):
        spectre_A.append(prot_A[i:i+n])
    
    spectre_B = []
    for i in range(len(prot_B)-n+1):
        spectre_B.append(prot_B[i:i+n])
    
    K_ = 0
    
    collection_A = Counter(spectre_A)
    collection_B = Counter(spectre_B)
    
    for key in collection_A.keys():
        if key in collection_B.keys():
            K_ += collection_A[key] * collection_B[key]
    
    return(K_)
  
def Gram(X_train, k):
    """
    return the Gram matrix for a training set and a k_spectrum
    """
    K = np.zeros((len(X_train),len(X_train)))
    for index1, row1 in X_train.iterrows():
        for index2, row2 in X_train.iterrows():
            seq1 = row1['seq']
            seq2 = row2['seq']
            K[index1][index2] = float(k_n(seq1, seq2, k))
    return(K)

##Allows to compute the Gram matrices and save them locally

#Gram_train_0 = Gram(X_train_0, k_spectrum)
#np.save('./K_0_10', Gram_train_0)
#Gram_train_1 = Gram(X_train_1, k_spectrum)
#np.save('./K_1_10', Gram_train_1)
#Gram_train_2 = Gram(X_train_2, k_spectrum)
#np.save('./K_2_10', Gram_train_2)

##Read the Gram matrices locally instead of computing them
Gram_train_0 = np.load("./K_0_10.npy")
Gram_train_1 = np.load("./K_1_10.npy")
Gram_train_2 = np.load("./K_2_10.npy")

def alpha(Gram, Y, Y_train, lamb):
    """
    compute the weight alpha of our dual SVM problem
    
    -Gram is a Gram matrice for a given training set
    -Y is a vector of label associated to the training set
    -Y_train is the dataframe with Id and labels
    -lamb is a float for lambda
    """
    P = Gram
    q = -Y
    q = q.astype('float')
    G = np.zeros((2*N,N))
    for k in range(len(np.array(Y_train["Bound"]))):
        G[k][k] = float(-np.array(Y_train["Bound"])[k])
        G[N+k][k] = float(np.array(Y_train["Bound"])[k])
    
    h = np.zeros((2*N,))
    for i in range(N):
        h[N+i] = 1/(2*lamb*N)
    
    P = matrix(P)
    q = matrix(np.transpose(q)[0])
    G = matrix(G)
    h = matrix(h)
    
    sol = solvers.qp(P, q, G, h)
    alpha = sol['x']
    return alpha

#We compute the weights for each training set
alpha_0 = np.array(alpha(Gram_train_0, Y_0, Y_train_0, lamb))
alpha_1 = np.array(alpha(Gram_train_1, Y_1, Y_train_1, lamb))
alpha_2 = np.array(alpha(Gram_train_2, Y_2, Y_train_2, lamb))

def pred(alpha, X_train, x, k):
    """
    prediction function returning a label 0 or 1
    
    -alpha is an array containing the weights
    -X_train is a dataframe with the training set
    -x is a string containing the sequence for which to find a label
    -k is an int for k_spectrum
    """
    f = 0
    for i in range(len(alpha)):
        f += alpha[i] * k_n(X_train["seq"][i], x, k)
    if f >= 0:
        return 1
    else: return 0

#Loading of the test set
X_test_file = "./Xte"
X_test_0 = pandas.read_csv(X_test_file + "0.csv")
X_test_1 = pandas.read_csv(X_test_file + "1.csv")
X_test_2 = pandas.read_csv(X_test_file + "2.csv")

#Creation of the dataframe of results
results = pandas.DataFrame(columns=["Id", "Bound"])

#Results for the first test set computed
for index, row in X_test_0.iterrows():
    seq = row["seq"]
    if index == 0:
        results.loc[0] = [row['Id'], pred(alpha_0, X_train_0, seq, k_spectrum)]
    else:
        results.loc[results.index.max() + 1] = [row['Id'], pred(alpha_0, X_train_0, seq, k_spectrum)]

print("results_0 done")

#Results for the second test set computed
for index, row in X_test_1.iterrows():
    seq = row["seq"]
    results.loc[results.index.max() + 1] = [row['Id'], pred(alpha_1, X_train_1, seq, k_spectrum)]

print("results_1 done")

#Results for the third test set computed
for index, row in X_test_2.iterrows():
    seq = row["seq"]
    results.loc[results.index.max() + 1] = [row['Id'], pred(alpha_2, X_train_2, seq, k_spectrum)]

print("results_2 done")

#Save the results in Yte.csv
results.to_csv("./Yte.csv", index=False)