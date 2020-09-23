# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 22:54:42 2020

@author: usr
"""
import pandas as pd
import category_encoders as ce
import numpy as np
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