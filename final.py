import pandas as pd
import gensim
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb
import re
from snowballstemmer import stemmer
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from gensim.models import Word2Vec
import random
from googletrans import Translator
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix,accuracy_score, classification_report

model = gensim.models.Word2Vec.load("models/full_grams_cbow_100_twitter.mdl")
def clean_text(text):
    search = ["أ","إ","آ","ة","_","-","/",".","،"," و "," يا ",'"',"ـ","'","ى","\\",'\n', '\t','&quot;','?','؟','!']
    replace = ["ا","ا","ا","ه"," "," ","","",""," و"," يا","","","","ي","",' ', ' ',' ',' ? ',' ؟ ',' ! ']  
    p_tashkeel = re.compile(r'[\u0617-\u061A\u064B-\u0652]')
    text = re.sub(p_tashkeel,"", text)
    p_longation = re.compile(r'(.)\1+')
    subst = r"\1\1"
    text = re.sub(p_longation, subst, text)
    text = text.replace('وو', 'و')
    text = text.replace('يي', 'ي')
    text = text.replace('اا', 'ا')
    
    for i in range(0, len(search)):
        text = text.replace(search[i], replace[i])

    text = text.strip()
    return text
    
data= pd.read_csv(r"short-answer-scoring-ar.csv", encoding="UTF-16") 
# print(data.head())
data=data.dropna()
# # print(data['EssayText'])
arabic_diacritics = re.compile(""" ّ    | # Tashdid
                             َ    | # Fatha
                             ً    | # Tanwin Fath
                             ُ    | # Damma
                             ٌ    | # Tanwin Damm
                             ِ    | # Kasra
                             ٍ    | # Tanwin Kasr
                             ْ    | # Sukun
                             ـ     # Tatwil/Kashida
                         """, re.VERBOSE)
def remove_diacritics(text):
    text = re.sub(arabic_diacritics, '', str(text))
    return text

def remove_repeating_char(text):
    # return re.sub(r'(.)\1+', r'\1', text)     # keep only 1 repeat
    return re.sub(r'(.)\1+', r'\1\1', str(text))  # keep 2 repeat

def stemming(text):
    snowball = stemmer("arabic")
    return snowball.stemWord(text)

def process_text(text, grams=False):
    clean_text = remove_diacritics(text)
    clean_text = remove_repeating_char(clean_text)
    clean_text=stemming(clean_text)
    return clean_text
    # if grams is False:
    #     return clean_text.split()
    # else:
    #     tokens = clean_text.split()
    #     grams = list(window(tokens))
    #     grams = [' '.join(g) for g in grams]
    #     grams = grams + tokens
    #     return grams

# data['EssayText'] = data['EssayText'].apply(process_text)
# print(data.head(5))

# def synonym_replacement(sequence,augment,synonym):
#     words = word_tokenize(sequence)
#     new_sentences=[]
#     for j in range(augment):
#         random_word=random.choice(words)
#         token=clean_text(random_word).replace(" ","_")
#         if token in model.wv:
#          for i in range(synonym):
#              most_similar=model.wv.most_similar(token,topn=synonym)
#              for term, score in most_similar:
#                 term = clean_text(term).replace(" ", "_")
#                 if term != token:
#                     output=sequence.replace(random_word,term)
#                     new_sentences.append(output)
#     return new_sentences

# def drop_duplicates(sentence):
#     aug_syn=synonym_replacement(sentence, 2, 1)
#     mylist = list(dict.fromkeys(aug_syn))
#     return mylist

# def back_translate(sequence):
#     languages = ['en', 'fr', 'th', 'tr', 'ur', 'ru', 'bg', 'de', 'ar', 'zh-cn', 'hi',
#                  'sw', 'vi', 'es', 'el']
#     translator = Translator()
#     org_lang = translator.detect(sequence).lang
#     random_lang = np.random.choice([lang for lang in languages if lang is not org_lang])
#     if org_lang in languages:
#         random_lang = np.random.choice([lang for lang in languages if lang is not org_lang])
#         translated = translator.translate(sequence, dest = random_lang).text
#         for x in range(4):
#             random_lang = np.random.choice([lang for lang in languages if lang is not org_lang])
#             t = translator.translate(translated, dest = random_lang).text
#             translated=t
#         translated_back = translator.translate(translated, dest = org_lang).text
#         output_sequence = translated_back            
#     else:
#         output_sequence = sequence
#     return output_sequence

# data['EssayTextAugmented'] = (data.apply(lambda x: drop_duplicates(x.EssayText), axis=1)) 
# # print(data['EssayTextAugmented'])

# data.to_csv(r'C:\Users\Lenovo\Desktop\Bachelor\TakeTwo\augmented.csv')

df= pd.read_csv(r"augmented.csv", encoding="utf-8") 
df.dropna()
# # print(df.head())
# df['BackTranslation']=(df.apply(lambda x: back_translate(x.EssayText), axis=1)) 
# print(df['BackTranslation'])

feature = df.EssayTextAugmented
target = df.Score1

# splitting into train and tests
X_train, X_test, Y_train, Y_test = train_test_split(feature, target, test_size =.15, random_state=100)

#  make pipeline
pipe = make_pipeline(TfidfVectorizer(),
                    RandomForestClassifier())

param_grid = {'randomforestclassifier__n_estimators':[10, 100, 1000],
             'randomforestclassifier__max_features':['auto']}

rf_model = GridSearchCV(pipe, param_grid, cv=5)
rf_model.fit(X_train,Y_train)

# # make prediction and print accuracy
prediction = rf_model.predict(X_test)
print(f"Accuracy score is {accuracy_score(Y_test, prediction):.2f}")
print(classification_report(Y_test, prediction))

