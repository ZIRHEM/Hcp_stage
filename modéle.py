# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 20:14:39 2020

@author: usr
"""

import pyarabic.araby as araby
import pyarabic.number as number
import pandas as pd
import numpy as np

df=pd.read_excel(r"C:\Users\usr\Desktop\stage 2nd year\livrable\BD.xlsx")
df.iloc[4,4]
if df.iloc[4,4]=='ذر':
    print(4)

%matplotlib inline
import matplotlib.pyplot as plt
import seaborn; seaborn.set() # set plot style

plt.hist(df['الجنس'])
plt.title(u'الجنس')
plt.xlabel(u'نوع الجنس')
plt.ylabel(u'العدد');


print("ذهب الطالب الى المدرسة")
df['الجنس'][df.iloc[:,0]==2].count()

df['السن'][df.iloc[:,0]==1].mean()
plt.hist(df['السن'])
plt.title("l'âge")
plt.xlabel("l'age")
plt.ylabel("nombre");



df['تقدير الراتب الشهري بالدرهم '][df.iloc[:,15]==70000].count()
plt.hist(df['تقدير الراتب الشهري بالدرهم '])
plt.title("salaire")
plt.xlabel("salaire")
plt.ylabel("nombre");


np.count_nonzero(df.iloc[:,15] < 70000)

df['كيف تتقاضى راتبك '].unique()
plt.hist(df['كيف تتقاضى راتبك '])
plt.title("comment vous obtenez votre salaire")
plt.xlabel("méthode")
plt.ylabel("nombre");


np.sum((df.iloc[:,15] < 8000) & (df.iloc[:,4] == 1))
np.sum((df.iloc[:,15] <= 8000) & (df.iloc[:,15] >= 7000))

#
plt.scatter(df['السن'], df['تقدير الراتب الشهري بالدرهم ']);
plt.xlabel("age")
plt.ylabel("salaire");
# ces gens sont illogiques
np.sum((df.iloc[:,15] <= 70000) & (df.iloc[:,15] >= 40000) & (df.iloc[:,1] < 45))
df[(df.iloc[:,15] <= 70000) & (df.iloc[:,15] >= 40000) & (df.iloc[:,1] < 45)]
ind=df[(df.iloc[:,15] <= 70000) & (df.iloc[:,15] >= 40000) & (df.iloc[:,1] < 45)].index
# indices
df.index
df.keys()



# remarque
rng=rng = np.random.RandomState(42)
A = pd.DataFrame(rng.randint(0, 20, (2, 2)),
                 columns=list('AB'))
B = pd.DataFrame(rng.randint(0, 10, (3, 3)),
                 columns=list('BAC'))
fill = A.stack().mean()
A.add(B, fill_value=fill)
A.subtract(A['A'], axis=0)

# suite
df=df.drop(df[(df.iloc[:,15] == 70000) & (df.iloc[:,1] <= 50)].index)
df=df.drop(df[(df.iloc[:,15] == 60000) & (df.iloc[:,1] <= 40)].index)
df=df.drop(df[(df.iloc[:,15] < 60000) & (df.iloc[:,15] >= 40000) & (df.iloc[:,1] <= 35)].index)
df=df.drop(df[(df.iloc[:,15] < 40000) & (df.iloc[:,15] >= 25000) & (df.iloc[:,1] <= 30)].index)
df=df.drop(df[(df.iloc[:,1] < 23) & (df.iloc[:,15] > 10000)].index)
#U drop columns 10 11 12
L=[0,1,2,4,5,6,7,8,9,13,14,15]
df=df[df.columns[L]]




# Agregation
df['السن'].mean()
df['تقدير الراتب الشهري بالدرهم '].mean()
df.mean()
df.mean(axis='columns') # ca n'a pas de sens
a=df.describe()

# groupby
df.groupby(df.columns[0])[df.columns[15]].describe().unstack()
b=df.groupby(df.columns[5])[df.columns[15]].describe()
# filter
def filter_func(x):
    return x[df.columns[15]].mean() > 40000
c=df.groupby(df.columns[5]).filter(filter_func)

# transform
df.groupby(df.columns[0]).transform(lambda x: x - x.mean())
# pivot
df.groupby(df.columns[0])[[df.columns[4]]].count()
A=df.groupby([df.columns[0], df.columns[5]])[[df.columns[15]]].aggregate('mean').unstack()

age = pd.cut(df[df.columns[1]], [20, 40, 60])
B=df.pivot_table('تقدير الراتب الشهري بالدرهم ', index=[df.columns[0], age], columns=df.columns[4])


df.pivot_table('تقدير الراتب الشهري بالدرهم ', index=df.columns[1], columns=df.columns[0], aggfunc='mean').plot()
plt.ylabel("salaire en fonction de l'age chez les H & F");


# cutting outliers
quartiles = np.percentile(df[df.columns[15]], [25, 50, 75])
mu = quartiles[1]
sig = 0.74 * (quartiles[2] - quartiles[0])
births = df[(df[df.columns[15]] > mu - 5 * sig) & (df[df.columns[15]] < mu + 5 * sig)]

births.pivot_table('تقدير الراتب الشهري بالدرهم ', index=births.columns[1], columns=births.columns[0], aggfunc='mean').plot()
plt.ylabel("salaire en fonction de l'age chez les H & F sans outliers");
fig.savefig('my_figure.png')



# Example
import matplotlib.pyplot as plt
import numpy as np
rng = np.random.RandomState(42)
x = 10 * rng.rand(50)
y = 2 * x - 1 + rng.randn(50)
plt.scatter(x, y);

from sklearn.linear_model import LinearRegression
model = LinearRegression(fit_intercept=True)
model

X = x[:, np.newaxis]
X.shape
model.fit(X, y)
# exploring the model
model.coef_
model.intercept_
# predict
xfit = np.linspace(-1, 11)
Xfit = xfit[:, np.newaxis]
yfit = model.predict(Xfit)





#############" Linear Model
from sklearn import linear_model
reg = linear_model.LinearRegression()
reg.fit([[0, 0], [1, 1], [2, 2]], [0, 1, 2])
reg.coef_





# Encoding ordinal variables
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import column_or_1d

# original
le = LabelEncoder()
le.fit(['b', 'a', 'c', 'd' ])
le.classes_
le.transform(['a', 'b'])


# this solution didn't work
class MyLabelEncoder(LabelEncoder):
    def fit(self, y):
        y = column_or_1d(y, warn=True)
        self.classes_ = pd.Series(y).unique()
        return self

le = MyLabelEncoder()
le.fit(['b', 'a', 'c', 'd' ])
le.classes_
le.transform(['b', 'a', 'c', 'd'])


# so i used ordinalEncoder
import category_encoders as ce
one_hot_df=df.copy()
ordinal_cols_mapping = [{'col':one_hot_df.columns[9],'mapping':{0:0, 900:1, 1200:2}}]
encoder = ce.OrdinalEncoder(mapping = ordinal_cols_mapping, cols=one_hot_df.columns[9], return_df = True)  
one_hot_df = encoder.fit_transform(one_hot_df)


ordinal_cols_mapping = [{'col':one_hot_df.columns[10],'mapping':{0:0, 1000:1, 1500:2}}]
encoder = ce.OrdinalEncoder(mapping = ordinal_cols_mapping, cols=one_hot_df.columns[10], return_df = True)  
one_hot_df = encoder.fit_transform(one_hot_df)



ordinal_cols_mappigg = [{'col':one_hot_df.columns[4],'mapping':{0:0, 1:15, 2:2,3:7,
                                                                4:14,5:12,6:13,7:9,
                                                                8:8,9:4,10:11,11:10,
                                                                12:6,13:5,14:3,15:1}}]
encoder = ce.OrdinalEncoder(mapping = ordinal_cols_mappigg, cols=one_hot_df.columns[4], return_df = True)  
one_hot_df = encoder.fit_transform(one_hot_df)


# Feature engineering
# Encoding categorical variables:
one_hot_df = pd.get_dummies(one_hot_df, columns=[one_hot_df.columns[0],
                                         one_hot_df.columns[3],
                                         one_hot_df.columns[5],
                                         one_hot_df.columns[6],
                                         one_hot_df.columns[7],
                                         one_hot_df.columns[8]],
                                         drop_first=True)


# Réarranger les colonnes:


one_hot_df=one_hot_df[[one_hot_df.columns[6],
                       one_hot_df.columns[0],
                       one_hot_df.columns[1],
                       one_hot_df.columns[7],
                       one_hot_df.columns[2],
                       one_hot_df.columns[8],
                       one_hot_df.columns[9],
                       one_hot_df.columns[10],
                       one_hot_df.columns[11],
                       one_hot_df.columns[12],                      
                       one_hot_df.columns[13],
                       one_hot_df.columns[14],
                       one_hot_df.columns[15],
                       one_hot_df.columns[16],
                       one_hot_df.columns[17],                       
                       one_hot_df.columns[18],                     
                       one_hot_df.columns[19],
                       one_hot_df.columns[20],
                       one_hot_df.columns[21],
                       one_hot_df.columns[22],
                       one_hot_df.columns[23],
                       one_hot_df.columns[24],
                       one_hot_df.columns[25],
                       one_hot_df.columns[26],
                       one_hot_df.columns[27],
                       one_hot_df.columns[28],
                       one_hot_df.columns[29],
                       one_hot_df.columns[30],
                       one_hot_df.columns[31],
                       one_hot_df.columns[32],
                       one_hot_df.columns[3],
                       one_hot_df.columns[4],
                       one_hot_df.columns[5]]]

# target, features
Y=one_hot_df[one_hot_df.columns[32]]
X=one_hot_df.drop([one_hot_df.columns[32]], axis=1)

from sklearn.model_selection import train_test_split
# Split the data into training and testing sets
train_features, test_features, train_labels, test_labels = train_test_split(X, Y, test_size = 0.20, random_state = 42)

# The baseline predictions are the historical averages
baseline_preds = train_labels.median()
# Baseline errors, and display average baseline error
baseline_errors = abs(baseline_preds - test_labels)
print('Average baseline error: ', round(np.mean(baseline_errors), 2))
Average baseline error:  9111.8 DH.


# Import the model we are using
from sklearn.ensemble import RandomForestRegressor
# Instantiate model with 1000 decision trees
rf = RandomForestRegressor(n_estimators = 200, random_state = 42)
# Train the model on training data
rf.fit(train_features, train_labels);


# Use the forest's predict method on the test data
predictions = rf.predict(test_features)
# Calculate the absolute errors
errors = abs(predictions - test_labels)
# Print out the mean absolute error (mae)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')
Mean Absolute Error: 452.9 DH.



# Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / test_labels)
# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')
Accuracy: 89,32 %.

# calculate R2
t=(np.mean((predictions - test_labels)**2))/(np.mean((test_labels-np.mean(test_labels))**2))
print(1-t)
R2 : 0.9978825866091922


prediction = rf.predict(train_features)
# Calculate the absolute errors
error = abs(prediction - train_labels)
# Print out the mean absolute error (mae)
print('Mean Absolute Error:', round(np.mean(error), 2), 'degrees.')
Mean Absolute Error: 175.83 DH.
# Calculate mean absolute percentage error (MAPE)
Mape = 100 * (error / train_labels)
# Calculate and display accuracy
accuracy = 100 - np.mean(Mape)
print('Accuracy:', round(accuracy, 2), '%.')
Accuracy:  95.87%.
# calculate R2
t=(np.mean((prediction - train_labels)**2))/(np.mean((train_labels-np.mean(train_labels))**2))
print(1-t)
R2 : 0.999686





# Import tools needed for visualization
from sklearn.tree import export_graphviz
import pydot
# Pull out one tree from the forest
tree = rf.estimators_[5]
# Export the image to a dot file
feature_list = A
export_graphviz(tree, out_file = 'tree.dot', feature_names = feature_list, rounded = True, precision = 1)
# Use dot file to create a graph
(graph, ) = pydot.graph_from_dot_file('tree.dot')
# Write graph to a png file
graph.write_png('tree.png')




#creating and training a model
#serializing our model to a file called model.pkl
import pickle
pickle.dump(rf, open("model.pkl","wb"))














































train_features, test_features, train_labels, test_labels

mu=np.mean(train_features[train_features.columns[1]])
sd=np.std(train_features[train_features.columns[1]])
train_features[train_features.columns[1]]=(train_features[train_features.columns[1]]-mu)/sd

m=np.mean(test_features[test_features.columns[1]])
s=np.std(test_features[test_features.columns[1]])
test_features[test_features.columns[1]]=(test_features[test_features.columns[1]]-m)/s


# use neural network
import keras
from keras.layers import Dense
from keras.models import Sequential
from keras import backend
from matplotlib import pyplot

def R2(y_true, y_pred):	
    return 1- (backend.mean(backend.square(y_pred - y_true), axis=-1)/backend.mean(backend.square(backend.mean(y_true) - y_true), axis=-1))       

n_cols = X.shape[1]
input_shape = (n_cols,)
# Specify the model
model = Sequential()
model.add(Dense(350, activation='relu', input_shape = input_shape))
model.add(Dense(200, activation='relu'))
model.add(Dense(150, activation='relu'))
model.add(Dense(1))
# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse', 'mae','mape', R2])
# train model
history = model.fit(train_features, train_labels, validation_data=(test_features,test_labels), epochs=1000, batch_size=1200)
# plot metrics
pyplot.plot(history.history['val_mean_squared_error'])
pyplot.plot(history.history['val_mean_absolute_error'])
pyplot.plot(history.history['val_mean_absolute_percentage_error'])
pyplot.plot(history.history['val_R2'])
pyplot.show()

