#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 16:26:22 2020

@author: nataliakokareva
"""

import joblib
from pymystem3 import Mystem
import re

class SentimentClassifier(object):
    def __init__(self):
        self.pipe = joblib.load("sentiment_review_classifier.pkl")
        self.classes_dict = {0: "Отзыв отрицательный", 1: "Отзыв положительный", -1: "ошибка предсказания"}


    def predict_text(self, text):
        try:
            return self.pipe.predict([text])[0]#,\
                   
        except:
            print("ошибка предсказания")
            return -1#, 0.8

    def predict_list(self, list_of_texts):
        try:
            return self.pipe.predict(list_of_texts)
                   
        except:
            print('ошибка предсказания')
            return None

    def get_prediction_message(self, text):
        prediction = self.predict_text(text)
        class_prediction = prediction#[0]
        #prediction_probability = prediction[1]
        return self.classes_dict[class_prediction]
    

        

def lemmatize(text):
    m = Mystem()
    lem = []
    clean_text = re.sub('[^A-Za-z0-9А-Яа-я]+', ' ', text)
    tkn = clean_text.split()
    for t in tkn:
        lem.append(m.lemmatize(t))
    sent = []  
    for word in lem:
        sent.append(word[0])
    
    return ' '.join(sent)

