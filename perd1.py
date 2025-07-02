import pandas as pd
import joblib
model=joblib.load("My_credit.h5")
f=model.feature_names_in_
a=input("enter the trans_date_trans_time:\n")
b=input("Enter cc_num:\n")
g=input("enter category:\n")
c=float(input("transaction amount:\n"))
x=input("Enter gender:\n")
y_s=input("Enter street:\n")
s=input("enter state:\n")
t=input("Enter city:\n")
v=input("Enter dob:\n")
z=input("Enter zip :\n")
w=input("Enter job:\n")
y_t=input("enter transaction number:\n")
h=input("Enter merch_lat:\n")
i=input("enter merch_long:\n")
d=({"trans_date_trans_time":"a","cc_num":"b","category":"g","AMT_TRANS":c,"gender":"x","street":"y_s","state":"s","city":"t",
    "dob":"v","zip":"z","job":"w","trans_num":"y_t","merch_lat":"h","merch_long":"i"})
d=pd.DataFrame([d])
d=pd.get_dummies(d)
d=d.reindex(columns=f,fill_value=0)
p=model.predict(d)
if p==0:
    print("NotFraudulent transaction")
else:
    print("Fraudulent transaction")
