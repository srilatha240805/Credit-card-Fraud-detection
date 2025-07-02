import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
import joblib
data=pd.read_csv("credit_card.csv")
print(data)
x=data.drop(columns=["slno","last","first","lat","long","city_pop"])                             
# Replace infinity with NaN
x.replace([np.inf, -np.inf], np.nan, inplace=True)
# Drop or fill NaN values
x.dropna(inplace=True)  # or use df.fillna(0) or df.fillna(df.mean())
# Ensure correct data type
#x = x.astype('float64')
x=pd.get_dummies(x)
print(x.shape)          
y=data["is_fraud"]
print(y.shape)
y = y[x.index]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
model=LogisticRegression()
model.fit(x_train,y_train)
y_predict=model.predict(x_test)
ac1=accuracy_score(y_predict,y_test)
print(" Logisticregression Accuracy:",ac1*100)
r=RandomForestClassifier()
r.fit(x_train,y_train)
y_predict=r.predict(x_test)
ac2=accuracy_score(y_predict,y_test)
print("RanodomForest acuracy:",ac2*100)
l=LinearRegression()
l.fit(x_train,y_train)
y_predict=model.predict(x_test)
ac3=accuracy_score(y_predict,y_test)
print("LinearRegression accuracy:",ac3*100)
svm = SVC()
svm.fit(x_train, y_train)
y_predict=svm.predict(x_test)
ac4=accuracy_score(y_predict,y_test)
print("svm accuracy:",ac4*100)
m=GaussianNB()
m.fit(x_train, y_train)
y_predict=m.predict(x_test)
ac5=accuracy_score(y_predict,y_test)
print("Naive bayes accuracy:",ac5*100)
print(confusion_matrix(y_test, y_predict))
print(classification_report(y_test, y_predict, zero_division=1))
joblib.dump(r,"My_credit.h5")
