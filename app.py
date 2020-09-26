# -*- coding: utf-8 -*-
L=['الجنس', 'السن', 'هل لديك شهادة التكوين المهني ?',
       'هل لديك مسؤولية في المؤسسة التي تزاول بها عملك',
       'ما هي هذه المسؤولية ', 'الاستمرارية في العمل ', 'هل لك عقدة عمل ',
       'هل تابعت خلال 12 شهرا الأخيرة تكوينا ',
       'ماهي الفئة الاكتر ملاءمة مع العمل الذي تقوم به ',
       'قيمة التعويضات العائلية في الشهر', 'قيمة تعويضات أخرى مرتبطة بالشغل']
import numpy as np
from flask import Flask, request, jsonify, render_template, url_for
import pickle
import pandas as pd
import csv
import category_encoders as ce


app = Flask(__name__)
loaded_model = pickle.load(open("model.pkl","rb"))


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

def handling(to_predict_list):
    A=to_predict_list[0]
    for i in range(1,len(to_predict_list)):
        A=A+','+to_predict_list[i]
    A=A.split(',')
    A=[float(a) for a in A]
    return A

def ValuePredictor(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(1,-1)
    #loaded_model = pickle.load(open("model.pkl","rb"))
    result = loaded_model.predict(to_predict)
    return result[0]


@app.route('/data', methods=['GET','POST'])
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
        #loaded_model = pickle.load(open("model.pkl","rb"))
        test_labels = loaded_model.predict(test_features)
        data['تقدير الراتب الشهري بالدرهم ']=test_labels
        return render_template("data.html", data = data.to_html()) 




def ismail(one_hot_df):

    ordinal_cols_mapping = [{'col':one_hot_df.columns[9],'mapping':{0:0, 900:1, 1200:2}}]
    encoder = ce.OrdinalEncoder(mapping = ordinal_cols_mapping, cols=one_hot_df.columns[9], return_df = True)  
    one_hot_df = encoder.fit_transform(one_hot_df)
        
        
    ordinal_cols_mappingg = [{'col':one_hot_df.columns[10],'mapping':{0:0, 1000:1, 1500:2}}]
    encoder = ce.OrdinalEncoder(mapping = ordinal_cols_mappingg, cols=one_hot_df.columns[10], return_df = True)  
    one_hot_df = encoder.fit_transform(one_hot_df)
        
        
        
    ordinal_cols_mappigg = [{'col':one_hot_df.columns[4],'mapping':{0:0, 1:15, 2:2,3:7,
                                                                    4:14,5:12,6:13,7:9,
                                                                    8:8,9:4,10:11,11:10,
                                                                    12:6,13:5,14:3,15:1}}]
    encoder = ce.OrdinalEncoder(mapping = ordinal_cols_mappigg, cols=one_hot_df.columns[4], return_df = True)  
    one_hot_df = encoder.fit_transform(one_hot_df)
        
    ordinal_cols_mappingg = [{'col':one_hot_df.columns[0],'mapping':{1:0, 2:1}}]
    encoder = ce.OrdinalEncoder(mapping = ordinal_cols_mappingg, cols=one_hot_df.columns[0], return_df = True)  
    one_hot_df = encoder.fit_transform(one_hot_df)
    
    ordinal_cols_mappingg = [{'col':one_hot_df.columns[3],'mapping':{1:0, 2:1}}]
    encoder = ce.OrdinalEncoder(mapping = ordinal_cols_mappingg, cols=one_hot_df.columns[3], return_df = True)  
    one_hot_df = encoder.fit_transform(one_hot_df)
    
    A=np.eye(6)[:,1:]
    Q=one_hot_df[one_hot_df.columns[5]].apply(lambda x:A[x-1])
    for i in range(5):
        one_hot_df['q'+str(i)]=[Q[j][i] for j in range(len(Q))]
    
    Q=one_hot_df[one_hot_df.columns[6]].apply(lambda x:A[x-1])
    for i in range(5):
        one_hot_df['r'+str(i)]=[Q[j][i] for j in range(len(Q))]
    
    ordinal_cols_mappingg = [{'col':one_hot_df.columns[7],'mapping':{1:0, 2:1}}]
    encoder = ce.OrdinalEncoder(mapping = ordinal_cols_mappingg, cols=one_hot_df.columns[7], return_df = True)  
    one_hot_df = encoder.fit_transform(one_hot_df)
    
    B=np.eye(15)[:,1:]
    Q=one_hot_df[one_hot_df.columns[8]].apply(lambda x:B[x-1])
    for i in range(14):
        one_hot_df['s'+str(i)]=[Q[j][i] for j in range(len(Q))]
        
    one_hot_df=one_hot_df.drop([one_hot_df.columns[5],one_hot_df.columns[6],one_hot_df.columns[8]], axis=1)  
    
    one_hot_df=one_hot_df[[one_hot_df.columns[0],
                       one_hot_df.columns[1],                       
                       one_hot_df.columns[2],
                       one_hot_df.columns[3],
                       one_hot_df.columns[4],
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
                       one_hot_df.columns[5],                      
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
                       one_hot_df.columns[6],
                       one_hot_df.columns[7]]]
    return one_hot_df


if __name__ == "__main__":
    app.run(debug=True)












