# -*- coding: utf-8 -*-
"""objective5_entity_function_version_2.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1kSmm_pFHqj93rV7IJvaTzedPtAKOIXJD
"""

import numpy as np
import pandas as pd
from copy import deepcopy

#from keras.preprocessing.text import Tokenizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline

import nltk
nltk.download('punkt')
nltk.download('stopwords')

from nltk.tokenize import word_tokenize

import warnings
warnings.filterwarnings('ignore')

import spacy.cli
spacy.cli.download("en_core_web_lg")

import spacy
from spacy import displacy
from collections import Counter
import en_core_web_lg
nlp = en_core_web_lg.load()

from google.colab import drive
drive.mount('/content/drive')

root_path = 'drive/MyDrive/comment_based_summarization/'  #change dir to your project folder

with open(root_path+'topics_all.txt') as f:
    lines = f.readlines()

files_list = []
for i in range(len(lines)):
  files_list.append(lines[i].replace('\n',''))

for i in range(len(files_list)):
  print('Topic '+str(i+1)+' : '+str(files_list[i]))

path = 'drive/MyDrive/vae/'  #change dir to your project folder
comment_folder = path + 'text_comments'

document_folder = path + 'text_documents'

doc = nlp("Apple acquired Zoom in China on Wednesday 6th May 2020.\
This news has made Apple and Google stock jump by 5% on Dow Jones Index in the \
United States of America")
print([(X.text, X.label_) for X in doc.ents ])

type(doc.ents)

print([(X, X.ent_iob_, X.ent_type_) for X in doc])

labels = [x.label_ for x in doc.ents]
Counter(labels)

def calculate_entity(file_name):
    
    print('Calculation begins')

    document = open(document_folder + '/' + file_name + '.sent', "r")
    doc_sent = document.readlines()

    comment = open(comment_folder + '/' + file_name + '.cmt', "r")
    comment_sent = comment.readlines()
    
    ################## Finding named entites present in each document sentence and storing them ####################
    doc_entity_list = []
    doc_sentences = ''
    for i in range(len(doc_sent)):
        doc_sentences += doc_sent[i][1:-2]
        temp_doc_entity = nlp(doc_sent[i][1:-2])
        
        sent_entities = []
        
        for entity in temp_doc_entity.ents:
          sent_entities.append(entity.text)

        doc_entity_list.append(sent_entities)

    ################## Finding named entites present in each comment sentence and storing them ####################
    comment_sentences = ''
    print('Length of comment sentences: '+str(len(comment_sent)))
    for i in range(len(comment_sent)):
      comment_sentences += comment_sent[i][1:-2]
        
    
    #print('document sentences:')
    #print(doc_sentences)
     
    ################## Finding all the unique entities present in the document and comments #####################
    ################## and also finding common entities between both documents and comments #################### 
    
    doc_entities = nlp(doc_sentences)

    document_entities = []

    for entity in doc_entities.ents:
      document_entities.append(entity.text)

    document_entity_set = set(document_entities)

    com_entities = nlp(comment_sentences)

    comment_entities = []

    for entity in com_entities.ents:
      comment_entities.append(entity.text)

    comment_entity_set = set(comment_entities)

    common_entity_set = document_entity_set.intersection(comment_entity_set)
    
    common_entity_list = list(common_entity_set)

    length_of_unique_entity = len(common_entity_list)
    print('Number of entities present in intersection of document and comments are: '+str(length_of_unique_entity))
    ################## Computing weight of each document sentence ################################
    weight = np.zeros(len(doc_sent))

    for i in range(len(doc_entity_list)):
      frequency = []
      doc_sentence_entity = doc_entity_list[i]
      
      for j in range(len(doc_sentence_entity)):
        word_frequency = 1 # initial weight given to each entity
        for k in range(len(common_entity_list)):
          if doc_sentence_entity[j] in common_entity_list[k]:
              word_frequency += 2 # calculating frequency of each word
        frequency.append(word_frequency)
      print('Frequency of ' + str(i+1) + ' sentence: '+str(sum(frequency)))  
      weight[i] = sum(frequency)/ length_of_unique_entity

    ################## Saving the outputs of 5 objectives ########################################

    final_path = 'drive/MyDrive/comment_based_summarization/objective_5_version_2/'  #change dir to your project folder
    df = pd.DataFrame(data = weight, columns = ['entity_weight'])
    df.to_csv(final_path + file_name + '.csv', index=False)
    #file = open(final_path + file_name + '.entities', 'w')
    #file.write(final_result)
    #file.close()

i =0
print('\n ******************************* \n Topic No.: '+str(i+1)+'\n Topic Name: '+str(files_list[i]))
calculate_entity(files_list[i])
print('Finished!')

#i =15
#print('\n ******************************* \n Topic No.: '+str(i+1)+'\n Topic Name: '+str(files_list[i]))
#calculate_weight(files_list[i])
#print('Finished!')

for i in range(len(files_list)):
    print('\n ******************************* \n Topic No.: '+str(i+1)+'\n Topic Name: '+str(files_list[i]))
    calculate_entity(files_list[i])
    print('Finished!')
