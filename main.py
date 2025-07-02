from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import joblib
model=joblib.load("My_credit.h5")
f=model.feature_names_in_
app = Flask(__name__)
@app.route('/')
def homepage():
    return render_template('homepage.html')


@app.route('/add', methods=['POST'])
def add():
    try:
        creditcardnumber=str(request.form['cc_number'])
        transactionamount = float(request.form['trans_amount'])
        category = str(request.form['category'])
        gender=str(request.form['gender'])
        street=str(request.form['street'])
        city=str(request.form['city'])
        state=str(request.form['street'])
        dateofbirth=str(request.form['dateofbirth'])
        job=str(request.form['job'])
        transactionnumber=str(request.form['trans_number'])
        d=({"cc_num":creditcardnumber,"category":category,"AMT_TRANS":transactionamount,
            "gender":gender,"street":street,"state":state,"city":city,
         "dob":dateofbirth,"job":job,"trans_num":transactionnumber})
        d=pd.DataFrame([d])
        d=pd.get_dummies(d)
        d=d.reindex(columns=f,fill_value=0)
        p=model.predict(d)
        if p==0:
            result="NotFraudulent transaction"
        else:
            result="Fraudulent transaction"
        
    except ValueError:
        result = "Invalid input! Please enter numbers only."
    
    return render_template('result.html', result=result)

@app.route('/predict')
def index():
    return render_template('index.html')
@app.route('/result',methods=['GET'])
def result():
    result = request.args.get('result')
    print(result)
    return render_template('result.html', result="fraud" if result==1 else "not fraud")

if __name__ == '__main__':
    app.run(debug=True)


