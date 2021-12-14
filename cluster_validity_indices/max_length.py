import numpy as np
import pandas as pd
import nltk
"""
     TWEET LENGTH ARRAY::
         @length_tweet_arr : 1-D array storing length of ith tweet at ith index 
"""

def cal_tweets_length(text):
    num_tweet=len(text)
    length_tweet_arr = np.zeros(num_tweet)
    for i in range(0,num_tweet):
        #print(text[i])
        tokenized_text = nltk.word_tokenize(text[i])
        length_tweet_arr[i] = len(tokenized_text)
        #break
    return length_tweet_arr

""" ENDS """

#
# clean_text_data=[]
# filepath = '/home/saini/PycharmProjects/Microblog_Text_summ/data_set/ensemble-summarization-intellisys-2018-dataset/input_datasets/clean_tweets/clean_hangupit_tweets.txt'
# with open(filepath) as fp:
#    for cnt, line in enumerate(fp):
#        clean_text_data.append(line)
#
# a=cal_tweets_length(clean_text_data)
# #print a[0], a[3]
# file1=open('/home/saini/PycharmProjects/Microblog_Text_summ/preprocessing/MAX_tfidf_SCORE/hangup_max_tweet_length.txt','w')
# counter=0
# for i in range(len(a)):
#     file1.write(str(i)+ '\t'+str(a[i])+'\n')
#     counter+=1
# print(counter)
# file1.close()
# print a
