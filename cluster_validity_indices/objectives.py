import numpy as np
import pandas as pd

"""
      TWEET LENGTH ARRAY::
      @length_tweet_arr : 1-D array storing length of ith tweet at ith index 
"""

def cal_tweets_length(text):
    num_tweet=len(text)
    length_tweet_arr = np.zeros(num_tweet)
    for i in range(0,num_tweet):
        length_tweet_arr[i] = len(text[i])
    return length_tweet_arr

""" ENDS """



"""  
    TF-IDF :
    Max td-idf sum of each tweet
"""
def cal_tf_idf(text):
    num_tweet=len(text)
    #tf_idf_matrix = np.zeros(shape=(num_tweet,num_tweet))
    tf_idf_sum = np.zeros(num_tweet)
    
    from sklearn.feature_extraction.text import TfidfVectorizer
    tfidf = TfidfVectorizer(analyzer='word', ngram_range=(1,1), min_df = 0, stop_words = 'english')
    
    tfs = tfidf.fit_transform(text[i] for i in range(0,num_tweet))
    feature_names = tfidf.get_feature_names()
    corpus_index = [i for i in range(0,num_tweet)]
    df=pd.DataFrame(tfs.T.todense(),index=feature_names,columns=corpus_index)
    
    for i in range(0,num_tweet):
        tf_idf_sum[i]=sum(df[i])
    return tf_idf_sum

"""  ENDS  """
