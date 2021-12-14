"""
    TF-IDF :
        Max td-idf sum of each tweet
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def cal_tf_idf(text):
    num_tweet = len(text)
    # tf_idf_matrix = np.zeros(shape=(num_tweet,num_tweet))
    print("number of tweets : ", num_tweet)
    tf_idf_sum = np.zeros(num_tweet)
    tfidf = TfidfVectorizer(analyzer='word', ngram_range=(1, 1), min_df=0, stop_words='english')
    tfs = tfidf.fit_transform(text[i] for i in range(0, num_tweet))
    feature_names = tfidf.get_feature_names()
    corpus_index = [i for i in range(0, num_tweet)]
    df = pd.DataFrame(tfs.T.todense(), index=feature_names, columns=corpus_index)
    #print(df.head())
    #print(feature_names), len(feature_names)

    for i in range(0, num_tweet):
        tf_idf_sum[i] = sum(df[i])
    return tf_idf_sum


"""  ENDS  """

#
# clean_text_data=[]
# filepath = '/home/saini/PycharmProjects/Microblog_Text_summ/data_set/ensemble-summarization-intellisys-2018-dataset/input_datasets/clean_tweets/clean_hangupit_tweets.txt'
# with open(filepath) as fp:
#    for cnt, line in enumerate(fp):
#        clean_text_data.append(line)
#
# a=cal_tf_idf(clean_text_data)
# #print a[0], a[3]
# file1=open('/home/saini/PycharmProjects/Microblog_Text_summ/preprocessing/MAX_tfidf_SCORE/hangup_max_tfidf_score.txt','w')
# counter=0
# for i in range(len(a)):
#     file1.write(str(i)+ '\t'+str(a[i])+'\n')
#     counter+=1
# print(counter)
# file1.close()