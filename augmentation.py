import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import string
import re
from nltk.corpus import stopwords
from snowballstemmer import stemmer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix,accuracy_score, classification_report

import gensim
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
import spacy
from gensim.models import Word2Vec
import random

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
# token = clean_text(u'ابو تريكه').replace(" ", "_")
# print(model.wv.most_similar(token))

data= pd.read_csv(r"short-answer-scoring-ar.csv", encoding="UTF-16") 
# print(data.head())
# print(data.sample(5))
data.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)
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
    return re.sub(r'(.)\1+', r'\1\1', text)  # keep 2 repeat

def stemming(text):
    snowball = stemmer("arabic")
    return snowball.stemWord(text)

def process_text(text, grams=False):
    clean_text = remove_diacritics(text)
    clean_text = remove_repeating_char(clean_text)
    clean_text=stemming(clean_text)
    if grams is False:
        return clean_text.split()
    else:
        tokens = clean_text.split()
        grams = list(window(tokens))
        grams = [' '.join(g) for g in grams]
        grams = grams + tokens
        return grams

data['EssayText'] = data['EssayText'].apply(process_text)
# print(data.head(5))

data['EssayTextString'] = [' '.join(map(str, l)) for l in data['EssayText']]
print(data.head(5))


# # def tokenization_s(sentences): # same can be achieved for words tokens    --tokenize words in the sentence
# #     s_new = []
# #     for sent in (sentences[:][0]): #For NumpY = sentences[:]
# #         s_token = sent_tokenize(sent)
# #         if s_token != '':
# #             s_new.append(s_token)
# #     return s_new
# # print(tokenization_s(data['EssayText']))

# # print(data['EssayText'])

# # feature = data.EssayTextString
# # target = data.Score1

# # # splitting into train and tests
# # X_train, X_test, Y_train, Y_test = train_test_split(feature, target, test_size =.2, random_state=100)

# # #  make pipeline
# # pipe = make_pipeline(TfidfVectorizer(),
# #                     RandomForestClassifier())

# # param_grid = {'randomforestclassifier__n_estimators':[10, 100, 1000],
# #              'randomforestclassifier__max_features':['auto']}

# # rf_model = GridSearchCV(pipe, param_grid, cv=5)
# # rf_model.fit(X_train,Y_train)

# # # # make prediction and print accuracy
# # prediction = rf_model.predict(X_test)
# # print(f"Accuracy score is {accuracy_score(Y_test, prediction):.2f}")
# # print(classification_report(Y_test, prediction))

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
#     # aug_syn=synonym_replacement("الصحة اهم احب البحر", 2, 2)
#     aug_syn=synonym_replacement(sentence, 2, 2)
#     mylist = list(dict.fromkeys(aug_syn))
#     return mylist
# # data['EssayTextAugmented']=data['EssayText'].apply(drop_duplicates())

# for i, row in data.iterrows():
#     data['EssayTextAugmented']=drop_duplicates(f"{row['EssayTextString']}")
#     # print(f"{row['EssayTextString']}")
#     print(f"Index: {i}")

    # print(sentence)
# data.head(5)
# def data_augment_synonym_replacement(data, column='EssayText'):
#   generated_data = pd.DataFrame([], columns=data.columns)
#   for index in data.index:
#     text_to_augment = data[column][index]
#     for generated_sentence in drop_duplicates(text_to_augment):
#       new_entry =  data.loc[[index]]
#       new_entry[column] = generated_sentence
#       generated_data=generated_data.append(new_entry)

#   generated_data_df = generated_data.drop_duplicates()
    # augmented_data= pd.concat([data.loc[:],generated_data_df], ignore_index=True)
    # return augmented_data   

# data_augment_synonym_replacement(data)
# print(data[['EssayText']].iloc[1])

# # # df['column'] = df['column'].astype('|S') # which will by default set the length to the max len it encounters
# # sentence=data[['EssayTextString']].iloc[1]
# # x= str(sentence)
# # print(drop_duplicates(x))
# # # drop_duplicates("الصحة اهم احب البحر")


# def replace_text(text):
#     return ' '.join([get_synonym_list(word) for word in text.split(' ')])  

# data['EssayTextAugmented'] = data['EssayText'].apply(lambda x: drop_duplicates(x))
# data['EssayTextString','EssayTextAugmented'].head(5)



