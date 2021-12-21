# -*- coding: utf-8 -*-
"""objective_3_density_based_score.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Tn5BHhV_EA8taJ91_--2J4wmRJ2Cw146
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

#dir_folder = root_path+'important_results/'

from nltk.corpus import stopwords
stop_words = set(stopwords.words("english"))
import string
import re

def preprocess_sentences(x):
    #Decontraction
    # specific
    x = re.sub(r"won't", "will not", str(x))
    x = re.sub(r"can\'t", "can not", str(x))
    
    # general
    x = re.sub(r"n\'t", " not", str(x))
    x = re.sub(r"\'re", " are", str(x))
    x = re.sub(r"\'s", " is", str(x))
    x = re.sub(r"\'d", " would", str(x))
    x = re.sub(r"\'ll", " will", str(x))
    x = re.sub(r"\'t", " not", str(x))
    x = re.sub(r"\'ve", " have", str(x))
    x = re.sub(r"\'m", " am", str(x))
    
    # Cleaning the urls
    x = re.sub(r'https?://\S+|www\.\S+', '', str(x))

    # Cleaning the html elements
    x = re.sub(r'<.*?>', '', str(x))

    #Removing Special Characters
    x = x.replace('\\r', ' ')
    x = x.replace('\\n', ' ')
    x = x.replace('\\"', ' ')
    x = re.sub('[^A-Za-z0-9]+', ' ', str(x))
    
    # Cleaning the whitespaces
    x = re.sub(r'\s+', ' ', str(x)).strip()
    
    #Removing Punctuation and converting to lower case
    sentence=x.translate(str.maketrans('', '', string.punctuation)).lower()
    word_tokens=sentence.split(' ')
    updated_sentence=""
    #Removing Stopwords
    for w in word_tokens:
        if w not in stop_words:
            updated_sentence+=w+" "
    return updated_sentence

def calculate_dbs(folder_name):
  
  print('Calculation begins')

  full_document_df=pd.read_csv(root_path + 'processed_documents/' + str(folder_name) + '/' + str(folder_name) + '.csv')

  cleaned_doc_df=pd.read_csv(str(root_path) + 'important_results/' + str(folder_name) + '/' + str(folder_name) +'_reader_attention.csv')
  
  wmd_df=pd.read_csv(str(root_path) + 'important_results/' + str(folder_name) + '/' + str(folder_name) +'_word_mover_distance.csv')
  
  complete_document_df = pd.read_csv(str(root_path) + 'important_results/' + str(folder_name) + '/' + str(folder_name) +'_document_para.csv')
  Y = full_document_df.document_sentence
  X = cleaned_doc_df.clean_document_sentences

  #complete_document_df = pd.read_csv(str(dir_folder) + str(folder_name) + '/' + str(folder_name) +'_document_para.csv')

  ################################# distance(w(k),w(k+1)) #########################################
  print('Computing distance(w(k),w(k+1))')
  
  empty_sentence = np.zeros( ( len(X),), dtype=int ) # contains 0 or 1 , 1 means the sentence is empty 0 means not empty
  sentence_distance = []

  for i in range(len(X)):
    if str(X[i]) != 'nan':
      #print('sentence : '+str(i+1))
      clean_doc = word_tokenize(X[i])

      unclean_doc = preprocess_sentences(Y[i])

      index1 = 0
      index2 = 0
      temp_dist = []
  
      for w in range(len(clean_doc)-1):
        first_word = clean_doc[w]
        second_word = clean_doc[w+1]
        index1 = index2

        #print('pairs:' +str(pairs))
        for k in range(index1,len(unclean_doc)):
          if first_word == unclean_doc[k]:
            index1 = k

          if second_word == unclean_doc[k]:
            index2 = k
            break
    
        temp_dist.append(index2 - index1-1)
  
      sentence_distance.append(temp_dist)
      empty_sentence[i] = 0
    
    else:
      sentence_distance.append(0)
      empty_sentence[i] = 1

  print('distance(w(k),w(k+1)) complete')
  ################################################################################

  useful_comments = list(wmd_df.comment_sentences)
  useful_comments_scores = np.array(wmd_df['useful_wmd'], dtype=int)

  ##############################################################################
  print('Computing sentence score')
  sentence_score = [] 
  for i in range(len(X)):
    if str(X[i]) != 'nan':
      word_score = []
      words = word_tokenize(X[i])
      for j in range(len(words)):
        temp_word_score = 0
        for k in range(len(useful_comments)):
          if words[j] in useful_comments[k]:
            temp_word_score += useful_comments_scores[k]
        word_score.append(temp_word_score)
  
      sentence_score.append(word_score) 
    else:
      sentence_score.append([0])   
  print('Complete sentence score')
  ############################################################################# 
  complete_sentence = list(complete_document_df['document'])
  words_in_sentence = word_tokenize(complete_sentence[0])
  vocabulary_set = set(words_in_sentence)
  vocabulary = list(vocabulary_set)

  print('Size of corpus: '+str(len(words_in_sentence)))
  print('Size of vocabulary: '+str(len(vocabulary)))
  corpus = []

  for i in range(len(X)):
      #print('\n***************************** '+str(i+1) + '**********************************')
      #print('Input : '+str(cleaned_doc_df['clean_document_sentences'][i]))
      #print('Length : '+str(len(cleaned_doc_df['clean_document_sentences'][i])))
      if str(X[i]) != 'nan':
        corpus.append(X[i])

  pipe = Pipeline([('count', CountVectorizer(vocabulary=vocabulary)),('tfid', TfidfTransformer())]).fit(corpus)
  pipe['count'].transform(corpus).toarray()

  sentence_tfidf_score = []
  for i in range(len(corpus)):
      word_tfidf_score = []

      temp_word_tokenizer = word_tokenize(corpus[i])

      for j in range(len(temp_word_tokenizer)):
          try:
            word_index = vocabulary.index(temp_word_tokenizer[j])
            word_tfidf_score.append(pipe['tfid'].idf_[word_index])
          except ValueError:
            word_tfidf_score.append(0)
      sentence_tfidf_score.append(word_tfidf_score) 

  ##############################################################################
  beta = 2

  sentence_weight = []
  for i in range(len(sentence_tfidf_score)):
    word_weight = []

    for j in range(min(len(sentence_score[i]),len(sentence_tfidf_score[i]))):
      calculate_weight = (sentence_tfidf_score[i][j] + (beta*sentence_score[i][j]) )/(1 + beta)
      word_weight.append(calculate_weight)

    sentence_weight.append(word_weight) 

  #############################################################################
  final_sentence_score = np.zeros((len(sentence_weight),1))

  for i in range(len(sentence_weight)):
    final_word_sum = 0
    K = len(sentence_weight[i])
    for j in range(K-1):
      final_word_sum += (sentence_weight[i][j] + sentence_weight[i][j+1])/( pow(sentence_distance[i][j],2) + 1)  

    #print('sentence : '+str(i+1))
    #print('K : '+str(K))
    if K == 1:
      final_sentence_score[i] = (K*final_word_sum)
    else:
      final_sentence_score[i] = (K*final_word_sum)/(K-1)    

  ############################################################################

  index = 0
  dsa_score = np.zeros((len(empty_sentence),1))

  for i in range(len(empty_sentence)):
      if empty_sentence[i] == 0:
        dsa_score[i] = final_sentence_score[index]
        index += 1
      else:
        dsa_score[i] = 0   
  #############################################################################

  df = pd.DataFrame(data= dsa_score,columns = ['dbs_score'])

  df.to_csv(str(root_path) + 'important_results/' + str(folder_name) + '/' + str(folder_name) +'_sentence_score.csv', index=False)

i =7
print('\n ******************************* \n Topic No.: '+str(i+1)+'\n Topic Name: '+str(files_list[i]))
calculate_dbs(files_list[i])
print('Finished!')

for i in range(len(files_list)):
    print('\n ******************************* \n Topic No.: '+str(i+1)+'\n Topic Name: '+str(files_list[i]))
    calculate_dbs(files_list[i])
    print('Finished!')

