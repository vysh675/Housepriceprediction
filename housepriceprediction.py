

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

dataset = pd.read_csv('landpriceprediction.csv')


land=dataset.drop('price',axis='columns')

land

price=dataset.price


from sklearn import linear_model

model=linear_model.LinearRegression()

model.fit(land,price)


#predicting the land price

predictions=model.predict(land)
plt.scatter(price,predictions)


#visualizing the data

import seaborn as sns
sns.distplot((land),bins=50)

