# -*- coding: utf-8 -*-
"""
Created on Fri Dec  3 16:50:12 2021

@author: itsna
"""

# In[1]:


import requests
from bs4 import BeautifulSoup 
import csv
import pandas as pd
import numpy as np
import os
from nltk.stem.porter import PorterStemmer
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
nltk.download('punkt')


# In[4]:


def call_crawler(url, highest_iter):
    list_q = [url] 
    num_count = 0
    OP_list=[]  #list to append OPs
    while(list_q!=[] and num_count < highest_iter):
        cycle_start = list_q.pop(0)    
        print("Accessing " + cycle_start)
        
        #Reading HTML text and parsing it so that data can be extracted
        snippet = requests.get(cycle_start)
        snippet_str = snippet.text            
        bf_soup = BeautifulSoup(snippet_str, "html.parser")  
           
        #extracting information from div tag
        outpt=bf_soup.find_all("div", head="result-container")

        for publications in outpt:
            dct_record={}
            #Topic of publication
            Topic_publications = publications.find("h3", head="Topic")
            #link of the publication
            link_publications = publications.find("a", head="link")
            #date of publication published
            date_publications=publications.find("span", head="date")
            
            #printing extracted information
            print("Topic of the Publication: ",Topic_publications.text)
            print("Link to the publication: ",link_publications.get('href'))
            print("publications were published on: ",date_publications.text)

            #reading CU authors' details
            Name_of_author= []   #initialising list to store names of authors
            #to store links to authors' pureportal profile
            Link_of_author=[]
            
            for author in publications.find_all("a", head="link person"):
                #fetching author's profile links
                portal_link = author.get('href')
                #fetching author's name
                name=author.string
                Name_of_author.append(name)
                Link_of_author.append(portal_link)
                print(name)
                print(portal_link)

            #writing all the fetched details about each publication in dictionary
            dct_record['Name of Publication']=Topic_publications.text
            dct_record['Publication Link']= link_publications.get('href')
            dct_record['Date of Publish']= date_publications.text
            dct_record['CU Author']=Name_of_author
            dct_record['Pureportal Profile Link']=Link_of_author
            
            #appending all rows in list
            OP_list.append(dct_record)
        
        #searching link for next page and storing in list_que so that it can be traversed next        
        link_of_page=bf_soup.find("a", head="nextLink")
        #It will store links till last page only
        if(link_of_page!= None):
            #getting link to next page
            goto_next=link_of_page.get('href')
            
            #Normalisation
            link_base = "https://pureportal.coventry.ac.uk"
            goto_nextpage_link= link_base+goto_next
            print(goto_nextpage_link)
            
            #appending link to next page in list_que
            list_q.append(goto_nextpage_link) 

        #writing information to csv rec_file
        rec_file = open('outpt.csv', 'w', encoding="utf-8")
        attribute_col=['Name of Publication','Publication Link','Date of Publish','CU Author','Pureportal Profile Link']
        add_field = csv.DictWriter(rec_file, fieldnames=attribute_col)
        add_field.writeheader()
        for dct_record in OP_list:
            add_field.writerow(dct_record)
        rec_file.close()
        
        num_count+= 1
        
    print("Crawling completed ") 
        


# In[5]:



#Calling crawler function
func_crawl=input("Do you want to run a crawler (y/n): ").lower()
if (func_crawl=='y'):
    call_crawler('https://pureportal.coventry.ac.uk/en/organisations/school-of-life-sciences/publications/',17)
else:
    if(os.path.isfile('outpt.csv')):
        df=pd.read_csv("outpt.csv", names=['Name of Publication','Publication Link','Date of Publish',
                                            'CU Author','Pureportal Profile Link'],
                                       encoding= 'unicode_escape') 
        print("Search started")
    else: 
        print("No Crawler output exists, you need to run crawler first!")


# In[6]:


#Reading csv file and storing contents in dataframe
df=pd.read_csv("outpt.csv")

print("No. of publications found: ", df.count())
df.head()


# In[7]:


#Pre-processing
import re
import string
cleaned_Topic = []
for Topic in df['Name of Publication']:
    # Removing Unicode
    filtered_document = re.sub(r'[^\x00-\x7F]+', '', Topic)
    # Removing Mentions
    filtered_document = re.sub(r'@\w+', '', filtered_document)
    # Converting into lowercase
    filtered_document = filtered_document.lower()
    # Removing punctuations
    filtered_document = re.sub(r'[%s]' % re.escape(string.punctuation), '', filtered_document)
    #removing stop words
    stringendo = re.compile(r'\b(' + r'|'.join(stopwords.words('english')) + r')\b\s*')
    text = stringendo.sub('', filtered_document)
    cleaned_Topic.append(text)
    
print(cleaned_Topic)


# In[8]:


Topic_list=[]
for i in cleaned_Topic:
    token_words=word_tokenize(i)
    stem_words=[]
    for word in token_words:
        #Stemming and removing stop words
        if(word.isalpha()):
            stemmer = PorterStemmer()
            stem_words.append(stemmer.stem(word))
            query=' '.join(stem_words)
    Topic_list.append(query)

print(Topic_list)


# In[9]:


#Tfidf model
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(Topic_list)
features = vectorizer.get_feature_names()
X = vectors.todense()
dense_list = X.tolist()
tf_idf = pd.DataFrame(dense_list, columns=features)
# print the full sparse matrix
tf_idf


# In[11]:



#Query processor and Indexer
find_string=input('enter a string: ')
#Pre-processing input string
find_string=find_string.lower()
#tokenization
token_words=word_tokenize(find_string)
stem_words=[]
for word in token_words:
    #Stemming and removing stop words, punctualtion
    if word not in stopwords.words("english"):
        if(word.isalpha()):
            stem_words.append(stemmer.stem(word))
            query=' '.join(stem_words)
filtered_query=query.split()
print(filtered_query)    #Preprocessed find_ string

match=list()

for i in range(len(Topic_list)):
    matches=0
    for j in range(len(filtered_query)):
        try:
            t=filtered_query[j]
            matches+=tf_idf.iloc[i][t]
        except KeyError:
            matches+=0
    match.insert(i,matches)

boolean = all(element == 0 for element in match)

if(boolean):
    print("No related research publications found.")
    
else:
    a=( sorted( [(x,i) for (i,x) in enumerate(match)], reverse=True ) [:5])
    pd.set_option('display.max_colwidth', None)
    for i in range(len(a)):
        max_index=a[i][1]
        print(df.iloc[max_index])
        print("***********************************************")


# In[12]:


import feedparser
news=[]
#extracting news from BBC website using RSS feed
Politics = feedparser.parse("http://feeds.bbci.co.uk/news/politics/rss.xml")
Education = feedparser.parse("http://feeds.bbci.co.uk/news/education/rss.xml")
Science = feedparser.parse("http://feeds.bbci.co.uk/news/science_and_environment/rss.xml")

for i in Politics.entries:
    pol= i.summary
    news.append(pol)

for i in Education.entries:
    edu= i.summary
    news.append(edu)

for i in Science.entries:
    sci= i.summary
    news.append(sci)

print("No of news headlines retrieved: ", len(news))
print(news)


# In[13]:


#Preprocessing
ps = PorterStemmer()
filtered_news = []
for i in news:
    tokens = word_tokenize(i.lower())
    n = ""
    for w in tokens:
        if w not in stopwords.words("english"):
            if (w.isalpha()):
                n += ps.stem(w) + " "
    filtered_news.append(n)
print(filtered_news)


# In[14]:


from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(filtered_news)
print(X.todense())


# In[15]:


from sklearn.cluster import KMeans
K = 8
cluster_model = KMeans(n_clusters=K)
cluster_model.fit(X)

print(cluster_model.labels_)


# In[16]:


#Taking input from user
query = input("Enter a string: ")

#pre-processing user input
tokens = word_tokenize(query)
x = ""
for w in tokens:
    if w not in stopwords.words("english"):
        x += ps.stem(w) + " "
print(x)   #preprocessed user query

#predicting cluster
Y = vectorizer.transform([x])
prediction = cluster_model.predict(Y)
print("cluster is: ", prediction)

