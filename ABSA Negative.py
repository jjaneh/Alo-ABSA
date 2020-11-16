import os
import tweepy as tw
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import collections
import nltk
from nltk.corpus import stopwords
import re
from textblob import TextBlob
from nltk import word_tokenize
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem.wordnet import WordNetLemmatizer 
import stanfordnlp
stanfordnlp.download('en')
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS 


consumer_key= ''
consumer_secret= 'uFX7mMlo4TKccQBsDsIscWhBnx3S2gOtIzYTVd4E7pzn3Wsewc'
access_token= '1294858760561688576-N1Q1ChvtP40BhlIyU1c3LjrGraNc4G'
access_token_secret= 'sdYNwgEVeXNLA4KmRSqpum8DJYz2cyHaQqkQ9hUsRK5ID'

try:
  auth = tw.OAuthHandler(consumer_key, consumer_secret)
  auth.set_access_token(access_token, access_token_secret)
  api = tw.API(auth, wait_on_rate_limit=True)
except:
  print("Authentication Failed")

twitr = pd.read_csv("part.csv", header=None, encoding='mac_roman')
twitr.columns = ['url','date','content','id','username']

extract_tid = lambda x: x["url"].split("/")[-1]
twitr['id'] = twitr.apply(extract_tid, axis=1)

tweets = twitr['content'].to_list()

tweets=np.unique(tweets)
tweets=tweets.tolist()

#Preprocessing the tweets
cleaned=[]
def preprocess_tweet(text):
  #url
  cleaned= " ".join(re.sub("([^0-9A-Za-z \t])|(\w+:\/\/\S+)", "", text).split())
  #lowercase
  cleaned= "".join(cleaned.lower())
  #usernames
  cleaned= "".join(re.sub('@[^\s]+', '', cleaned))
  #hashtag
  cleaned= "".join(re.sub('#([^\s]+)', '', cleaned))
  #remove stopwords
  word_tokens = word_tokenize(cleaned) 
  cleaned= " ".join(word for word in word_tokens if word not in stopwords.words('english'))
  return cleaned

map_object = map(preprocess_tweet, tweets)
cleaned_tweets = list(map_object)

twitr['content']=twitr['content'].apply(preprocess_tweet)

#Sentiment Analysis
sentiment_objects = [TextBlob(tweet) for tweet in cleaned_tweets]
sentiment_values = [[ str(tweet), tweet.sentiment.polarity] for tweet in sentiment_objects]
sentiment_df = pd.DataFrame(sentiment_values, columns=["tweet","polarity"])

fig, ax = plt.subplots(figsize=(15, 10))

# Plotting histogram of the polarity values
sentiment_df.hist(bins=[-1, -0.75, -0.5, -0.25, 0.25, 0.5, 0.75, 1],
             ax=ax,
             color="grey")

plt.title("Sentiments from Tweets on Covid19")
plt.show()

def sentiment(polarity):
    if (polarity < 0):
        return 'negative'
    elif (polarity == 0):
        return 'neutral'
    else:
        return 'positive'
sentiment_df['sentiment'] = sentiment_df['polarity'].apply(sentiment)

sentiment_df.sentiment.value_counts().plot(kind='pie', autopct='%5.0f%%', colors=["red", "yellow", "blue"])

#Splitting the dataframe for negative tweets
negative_df=sentiment_df.loc[sentiment_df['sentiment'] == 'negative']
negative_list = negative_df['tweet'].to_list()

#Aspect term extraction
new_list=[]
for line in negative_list:
    txt_list = nltk.word_tokenize(line)
    taggedList = nltk.pos_tag(txt_list)
    new_list.append(taggedList)
    
newwordList = []
flag = 0
for j in new_list:
  for i in range(0,(len(j)-1)):
    if (new_list[i][1]=="NN" and new_list[i+1][1]=="NN"):
        newwordList.append(new_list[i][0]+new_list[i+1][0])
        flag=1
    else:
        if (flag==1):

            flag=0
            continue
        newwordList.append(new_list[i][0])
        if (i==len(new_list)-2):
            newwordList.append(new_list[i+1][0])
finaltxt = '\n '.join(' '.join(word) for word in newwordList)

nlp = stanfordnlp.Pipeline(pos_batch_size=4096)
doc = nlp(finaltxt)
dep_node = []
for dep_edge in doc.sentences[0].dependencies:
    dep_node.append([dep_edge[2].text, dep_edge[0].index, dep_edge[1]])
for i in range(0, len(dep_node)):
    if (int(dep_node[i][1]) != 0):
        dep_node[i][1] = newwordList[(int(dep_node[i][1]) - 1)]

featureList = []
totalfeatureList=[]
categories = []
categoriesList=[]
for j in new_list:
  for i in j:
      if(i[1]=='JJ' or i[1]=='NN' or i[1]=='JJR' or i[1]=='NNS' or i[1]=='RB'):
          featureList.append(list(i))
          totalfeatureList.append(list(i)) # This list will store all the features for every sentence
          categories.append(i[0])

fcluster = []
for i in featureList:
    filist = []
    for j in dep_node:
        if((j[0]==i[0] or j[1]==i[0]) and (j[2] in ["nsubj", "acl:relcl", "obj", "dobj", "agent", "advmod", "amod", "neg", "prep_of", "acomp", "xcomp", "compound"])):
            if(j[0]==i[0]):
                filist.append(j[1])
            else:
                filist.append(j[0])
    fcluster.append([i[0], filist])

finalcluster = []
dic = {}
for i in featureList:
    dic[i[0]] = i[1]
for i in fcluster:
    if(dic[i[0]]=="NN"):
        finalcluster.append(i)

aspect_terms_negative=list(zip(*finalcluster))[0]

unique_string=(" ").join(aspect_terms_negative)
wordcloud = WordCloud(width = 1500, height = 1000).generate(unique_string)
plt.figure(figsize=(18,12))
plt.imshow(wordcloud)
plt.axis("off")
plt.savefig("your_file_name"+".png", bbox_inches='tight')
plt.show()
plt.close()
