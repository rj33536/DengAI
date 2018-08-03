from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from sklearn.linear_model import LogisticRegression,LinearRegression
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt



def get_err(train_X,train_y,val_X,val_y,leaf):
    my_model=RandomForestClassifier(max_leaf_nodes=leaf)
    my_model.fit(train_X,train_y)
    predicted=my_model.predict(val_X)
    return (mean_absolute_error(val_y,predicted))

#getting some data
train=pd.read_csv("train.csv")
testdata=pd.read_csv("test.csv")
train_label=pd.read_csv("train_label.csv")
print(train.columns)
print(train.shape)
print(train_label.columns)
print(train_label.shape)
y=train_label.total_cases
prediction_col=['city', 'year', 'weekofyear','ndvi_ne', 'ndvi_nw',
       'ndvi_se', 'ndvi_sw', 'precipitation_amt_mm', 'reanalysis_air_temp_k',
       'reanalysis_avg_temp_k', 'reanalysis_dew_point_temp_k',
       'reanalysis_max_air_temp_k', 'reanalysis_min_air_temp_k',
       'reanalysis_precip_amt_kg_per_m2',
       'reanalysis_relative_humidity_percent', 'reanalysis_sat_precip_amt_mm',
       'reanalysis_specific_humidity_g_per_kg', 'reanalysis_tdtr_k',
       'station_avg_temp_c', 'station_diur_temp_rng_c', 'station_max_temp_c',
       'station_min_temp_c', 'station_precip_mm']
X=train[prediction_col]
h1='reanalysis_tdtr_k'


#here comes data manipulation/data preprocessing

col_to_be_normalized=['year','weekofyear','precipitation_amt_mm','reanalysis_dew_point_temp_k',
                      'reanalysis_precip_amt_kg_per_m2','station_precip_mm']
print(X.describe())
#plt.hist(X[h1][~np.isnan(X[h1])] ,bins=100)
plt.hist(X.year[y>10],bins=20)
plt.show()
del train
del train_label
#plt.show()
col_normalized=[]
for c in col_to_be_normalized:
    col_normalized.append(c+'log')
X=pd.get_dummies(X)
X[col_normalized]=np.log(X[col_to_be_normalized]+5)
#X=(X-np.mean(X))/np.std(X)
my_imputer=Imputer()
X=my_imputer.fit_transform(X)
model=DecisionTreeClassifier()
X=pd.DataFrame(X)
print(X.describe())
trainX,val_X,trainy,val_y=train_test_split(X,y,random_state=0)

model.fit(trainX,trainy)
test=testdata[prediction_col]
test=pd.get_dummies(test)
test[col_normalized]=np.log(test[col_to_be_normalized]+5)
#test=(test-np.mean(test))/np.std(test)
print(X.shape)
test=my_imputer.fit_transform(test)
test=pd.DataFrame(test)
print(test.describe())
predicted=model.predict(val_X)

#model optimization
min_err=100
leaf=50
for i in range(50,500):
    err=get_err(trainX,trainy,val_X,val_y,i)
    if err<min_err:
        min_err=err
        leaf=i


print(f'leaf should be %d with min err = %f',leaf,min_err)


predicted=np.absolute(np.array(predicted).round(0))
print(predicted)
#print(val_y)
print(mean_absolute_error(val_y,predicted))
#my_submission = pd.DataFrame({'city': testdata.city, 'year': testdata.year,"weekofyear":testdata.weekofyear, 'total_cases':predicted})
#my_submission.to_csv('submission.csv', index=False)
