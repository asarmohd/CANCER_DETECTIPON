# -*- coding: utf-8 -*-
"""
Created on Sun Aug  5 19:19:25 2018

@author: ma
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import time
import json

dataset = pd.read_csv("data.csv")

X = dataset.iloc[:, 2:].values
X = X[:,:-1]
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
