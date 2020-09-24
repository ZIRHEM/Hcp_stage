# -*- coding: utf-8 -*-
L=['الجنس', 'السن', 'هل لديك شهادة التكوين المهني ?',
       'هل لديك مسؤولية في المؤسسة التي تزاول بها عملك',
       'ما هي هذه المسؤولية ', 'الاستمرارية في العمل ', 'هل لك عقدة عمل ',
       'هل تابعت خلال 12 شهرا الأخيرة تكوينا ',
       'ماهي الفئة الاكتر ملاءمة مع العمل الذي تقوم به ',
       'قيمة التعويضات العائلية في الشهر', 'قيمة تعويضات أخرى مرتبطة بالشغل']
import numpy as np
from flask import Flask, request, jsonify, render_template, url_for
from feature_eng import ismail
import pickle
import pandas as pd
import csv


app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
@app.route('/index')
def home():
    return render_template('index.html')

@app.route('/predict', methods = ['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    if request.method == 'POST':
        to_predict_list = request.form.to_dict()
        to_predict_list=list(to_predict_list.values())  
        to_predict_list=handling(to_predict_list)
        result = ValuePredictor(to_predict_list)       
        return render_template("predict.html", prediction='le salaire de la personne questionnée est de DH {}'.format(result))    

@app.route('/data', methods=['POST'])
def data():
    if request.method == 'POST':
        f=request.form['csvfile']
        data=[]
        with open(f) as file:
            csvfile=csv.reader(file)
            c=0
            for row in csvfile:
                c=c+1
                if c>1: 
                    
                    Z=row[0].split(';')
                    Z=[int(z) for z in Z]
                    data.append(Z)        
        data=pd.DataFrame(data, columns=L)
        test_features=ismail(data)
        loaded_model = pickle.load(open("model.pkl","rb"))
        test_labels = loaded_model.predict(test_features)
        data['تقدير الراتب الشهري بالدرهم ']=test_labels
        return render_template("data.html", data = data.to_html()) 
def handling(to_predict_list):
    A=to_predict_list[0]
    for i in range(1,len(to_predict_list)):
        A=A+','+to_predict_list[i]
    A=A.split(',')
    A=[float(a) for a in A]
    return A
def ValuePredictor(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(1,-1)
    loaded_model = pickle.load(open("model.pkl","rb"))
    result = loaded_model.predict(to_predict)
    return result[0]

if __name__ == "__main__":
    app.run(debug=True)












