import nltk
from nltk.corpus import stopwords
import time
import re
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
from sklearn.model_selection import cross_val_predict

stopwordsNLTK = nltk.corpus.stopwords.words('portuguese')
stopwordsNLTK.append('vou')
stopwordsNLTK.append('tão')

dataset = pd.read_csv('base_teste.csv',error_bad_lines=False)
print(dataset.count())
tweets = dataset['Text'].values
classes = dataset['Classificacao'].values
classes_limpas = []
str = 'duhasudas'
cont = 0
# for word in dataset['Classificacao'].values:
#         if type(word) != type(str):
#             print(type(word))
#             print(cont)
#         cont = cont+1
# for word in dataset['Classificacao'].values:
#         if word.count('\tneutro') >=1:
#             print(cont)
#         cont = cont+1
# print(np.unique(classes))
# print(tweets[1423])
#alegria,tristeza,medo,raiva,surpresa,
# for x in classes:
#
#     classes_limpas.append(x.replace('\t', "").strip().lower())

# print(np.unique(classes_limpas))
# print(len(classes_limpas))

def remove_number(word_list):
    processed_word_list = []

    for word in word_list:
        limpo = " ".join(word.strip() for word in re.split('1|2|3|4|5|6|7|8|9|0',word))
        result = ''.join([i for i in limpo if not i.isdigit()])

        processed_word_list.append(result)

    return processed_word_list

def remove_hashtag(word_list):
    processed_word_list = []

    for word in word_list:
        limpo = " ".join(word.strip() for word in re.split('#|_|@|-|',word))
        result = ''.join([i for i in limpo if not i.isdigit()])

        processed_word_list.append(result)

    return processed_word_list
def remove_url(word_list):
    processed_word_list = []

    for word in word_list:
        limpo = re.sub(r'^https?:\/\/.*[\r\n]*','',word,flags=re.MULTILINE)
        processed_word_list.append(limpo)

    return processed_word_list
#Diminui as palvras ao seu radical
def aplica_stemmer(word_list):
        stemmer = nltk.stem.RSLPStemmer()
        processed_word_list = []
        for word in word_list:
                comStemming = [stemmer.stem(p) for p in word.split()]
                retorno = ' '.join(comStemming)
                processed_word_list.append(retorno)
        return processed_word_list

def remove_stopwords(word_list):
    processed_word_list = []
    for word in word_list:
        word = word.lower()  # in case they arenet all lower cased
        semStop = [p for p in word.split() if p not in stopwordsNLTK]
        retorno = ' '.join(semStop)
        processed_word_list.append(retorno)
    return processed_word_list

tweets_limpos = remove_number(tweets)
tweets_limpos1 = remove_url(tweets_limpos)
tweets_sem_stop_word = remove_stopwords(tweets)
tweets_com_stemmer = aplica_stemmer(tweets)
#analyzer="word"
#stop_words=stopwordsNLTK
#ngram_range=(1,2)
vectorizer = TfidfVectorizer(encoding='utf-8',strip_accents='ascii',ngram_range=(1,2))
freq_tweets = vectorizer.fit_transform(tweets_com_stemmer)
#print(freq_tweets)
#modelo = LogisticRegression()
#print(tweets_sem_stop_word)
modelo = MultinomialNB()
modelo.fit(freq_tweets,classes)
resultados = cross_val_predict(modelo,freq_tweets,classes,cv=10)
resultados_limpos = []
# for x in resultados:
#     resultados_limpos.append(x.strip().lower())

print(metrics.accuracy_score(classes,resultados))
sentimento=['Alegria','Medo','Desprezo','Tristeza','Raiva','Desgosto']
print(metrics.classification_report(classes,resultados,labels=np.unique(resultados)))
# print(resultados)
# print(classes_limpas)
print(pd.crosstab(classes,resultados,rownames=['Real'],colnames=['Predito'],margins=True),)
# print(len(resultados))
# print(len(classes_limpas))