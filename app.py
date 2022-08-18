#import packages 
import streamlit as st # data web app development
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import re
import time  
import plotly.express as px  
import matplotlib.pyplot as plt
#import arabic_ner
from wordcloud import WordCloud
from arabic_reshaper import arabic_reshaper
from bidi.algorithm import get_display
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.tokenize import word_tokenize
import ClusterTransformer.ClusterTransformer as ctrans
import os

@st.cache
#define a required  functions:




#  function to split_text by special characters 
def split_text(text):
  
  if ' - ' in text:
    return text.split('-',1)

  elif '/' in text:
    return text.rsplit('/',1)

  elif '|' in text:
    return text.rsplit('|',1)
  
  else:
    return [text,0]



# add app title
st.set_page_config(
    page_title="ماذا يقول عنك محرك البحث قوقل",

    page_icon="✅",

    layout="wide",
)



# def function to get source site name from given link 
def get_source_site_name(site_link):

  site_link[2].split('.')[-2]
  return site_link[2].split('.')[-2]



# def function to clean non Arabic letters and spicial characters   
def get_text_preprocessing(text):

  #clean non Arabic letters and spicial characters  
  text = re.sub('([@A-Za-z0-9_ـــــــــــــ]+)|[^\w\s]|#|http\S+', '', text) # cleaning up

  return " ".join(text.split())



# join text list to one  
def get_cleaned_text(text):
    return ' '.join(text.tolist())



#plot cluster

def remove_stopword_withtokenize_for_clusters(text):
  text_tokens = word_tokenize(text)
  tokens_without_sw = [word for word in text_tokens if not word in stopwords_list]
  return ' '.join(tokens_without_sw)


def generate_wordcloud(df, cluster_num):

  df_grouped_by_cluster = df[df['Cluster'] == cluster_num]
  df_grouped_by_cluster['text_without_stopword'] = [remove_stopword_withtokenize_for_clusters(text) for text in df_grouped_by_cluster['Text'] ] 
  
  text = df_grouped_by_cluster['text_without_stopword']

  text = [''.join(sentence) for sentence in text]
  text = ' '.join(text)
  reshaped_text = arabic_reshaper.reshape(text)
  arabic_text = get_display(reshaped_text)
  wordcloud = WordCloud(font_path = 'arial.ttf',width=700, height=300, background_color="white").generate(arabic_text)
  return wordcloud

# print title
st.title('لوحة بيانات لنمذجة الفورية لنتائج بحث قوقل ')


# print markdown
st. markdown("""
______________________________________________________________________________
""")


# to remove any warning coming on streamlit web app page
st.set_option('deprecation.showPyplotGlobalUse', False)

# print header
st.header("ماذا يقول عنك محرك البحث قوقل")

# print header
st.sidebar.header("اكتب الكلمات")

# text_input to write a query words 
query = st.sidebar.text_input('  اضف/ـي كلمات البحث , مثال :شركة ثقة لخدمات الأعمال ')

if query:
    query = query 

else :
  # add a default  value
    query = "شركة ثقة لخدمات الأعمال"



# text_input س 
st.sidebar.header("اكتب عدد النتائج")
num_of_results = st.sidebar.text_input("حدد/ـي عدد النتائج التي ترغب في نمذجتها")

if query:
    num_of_results = num_of_results

     # add a default  value
else :
    num_of_results = 10



# formalizing  google search query    
search = query.replace(' ', '+')
url = (f"https://www.google.com/search?q={search}&num={num_of_results}")
requests_results = requests.get(url)



# requests search results using BeautifulSoup
soup_link = BeautifulSoup(requests_results.content, "html.parser")

links = soup_link.find_all("a")

title_link_list = []
# structure search results 
for link in links:

    link_href = link.get('href')
    if "url?q=" in link_href and not "webcache" in link_href:
      title = link.find_all('h3')


      #filter search results 
      if len(title) > 0:

          title_list = []
          title_list.append(title[0].getText())
          title_list.append(link.get('href').split("?q=")[1].split("&sa=U")[0])
          title_link_list.append(title_list)



# add result to dataframe
df = pd.DataFrame(title_link_list, columns =['title', 'link']) 



# text preprocessing 
# remove special characters 
df['title'] = df['title'].str.replace('.', '')

#remove extra spaces  
df['title'].str.strip()

# extract information from title and link data :
# call function to split_text
df['splited_title'] =[split_text(text) for text in df['title']]

# call function to split_text
df['sub_title'] = [splited_title[0] for splited_title in df['splited_title']]

df['source_name'] = [splited_title[-1] for splited_title in df['splited_title']]

# get source_site_name from link
df['sub_link'] = df['link'].str.split('/', 3)

#apply get_text_preprocessing on sub title
df['cleaned_title'] = [get_text_preprocessing(text) for text in df['sub_title']]

# remove no values in cleaned_title

df['title_length'] = df.cleaned_title.str.len()

df = df[df.title_length > 1]

# dataframe filter
#df = df[df["cleaned_title"] == title_filter] 


cleaned_text = get_cleaned_text(df['cleaned_title'])
# drop unwanted columns
df =df.drop(columns=['splited_title', 'sub_link' , 'title_length' ])



##################################### NER ##################################


# def text_to_ner_model_line(text):
#   text = arabic_ner.get_ner(text)
#   return get_entity_key_value(text)
# def get_entity_key_value(text):
#   key__value_list_outer = []
#   for ner_ in text[0]:
#     for key , value in ner_.items():
#       if '-' in value:
#         #key_value_list_inner = [key,value]
#         key__value_list_outer.append(key)

#   return key__value_list_outer

# df['entity_list'] = [text_to_ner_model_line(text) for text in df['cleaned_title']]



############################################################################

# create a charts
st.markdown("عناوين نتيجة البحث ")

# create a charts cleaned_title
fig1 = px.bar(df, x="cleaned_title")
st.write(fig1)

# create a charts source_name
st.markdown("اسماء المواقع لنتيجة البحث ")
fig2 = px.bar(df[df['source_name'] != 0], x="source_name")
st.write(fig2)



# print  line 
st. markdown("""______________________________________________________________________________""")



# # remove stop wordsin order to modeling text.
stopwords_list = stopwords.words('arabic')

# stop TOKENIZERS_PARALLELISM
os.environ["TOKENIZERS_PARALLELISM"] = "false"

#Declare the model
cr=ctrans.ClusterTransformer()

#Transformer model name from from Huggingface pretrained library
model_name='albert-base-v1'

# creating an input list of sentences using cleaned_title as cluster feed 
li_sentence = list(df['cleaned_title'])

# declare models hyperparameters:
# batch size for running model inference
batch_size=2

# maximum sequence length for transformer to enable truncation
max_seq_length=64

# if enabled will return the embeddings in numpy ,else will keep in torch.Tensor
convert_to_numpy=False

# if set to True will enable normalization of embeddings
normalize_embeddings=False

# this is used for neighborhood_detection method and determines the minimum number of entries in each cluster
neighborhood_min_size=1

# this is used for neighborhood_detection method and determines the cutoff cosine similarity score to cluster the embeddings
cutoff_threshold=0.83

# hyperparameter for kmeans_detection method signifying nnumber of iterations for convergence
kmeans_max_iter=100

# hyperparameter for kmeans_detection method signifying random initial state
kmeans_random_state=42

# hyperparameter for kmeans_detection method signifying number of cluster
kmeans_no_clusters=3



# call the methods:
# creating the embeddings by running inference through Transformer library to returns a torch.Tensor containing the embeddings
embeddings=cr.model_inference(li_sentence,batch_size,model_name,max_seq_length,normalize_embeddings,convert_to_numpy)

# agglomerative clustering from the embeddings created from the model_inference method to returns a dictionary
output_dict=cr.neighborhood_detection(li_sentence,embeddings,cutoff_threshold,neighborhood_min_size)

# Kmeans clustering from the embeddings created from the model_inference method to returns a dictionary
output_kmeans_dict=cr.kmeans_detection(li_sentence,embeddings,kmeans_no_clusters,kmeans_max_iter,kmeans_random_state)

#  convert result to dataframe
neighborhood_detection_df=cr.convert_to_df(output_dict)
kmeans_df=cr.convert_to_df(output_kmeans_dict)



# extract a unique clusters
kmeans_clusters_list = kmeans_df.Cluster.unique()

neighborhood_detection_clusters_list = neighborhood_detection_df.Cluster.unique()
print('neighborhood_detection_clusters_list', neighborhood_detection_clusters_list)



# plot cluster result 
# print markdown 
st. markdown("""___ تجميع كلمات من  عناوين نتائج البحث بناء على تشابة سياقها ___""")



# insert containers laid out as side-by-side columns
col1, col2, col3 = st.columns(3)

for  index , cluster_num in enumerate(kmeans_clusters_list):

  print('if index == 0 :', index)

  # group df based on cluster filter :
  wordcloud_result =generate_wordcloud(kmeans_df,cluster_num)
   


  #plot cluster result 
  if index == 0 :
    
    with col1:
        st.caption(f'Topic {index+1} Words :\n ')
        st.image(wordcloud_result.to_array())
        
  if index == 1 :

    with col2:
        st.caption(f'Topic {index+1} Words :\n ')
        st.image(wordcloud_result.to_array())

  if index == 2 :

    with col3:
        st.caption(f'Topic {index+1} Words :\n ')
        st.image(wordcloud_result.to_array())




# print line
st. markdown("""________________________________________________""")

st. markdown("""** جميع الحقوق محفوظة@امل بن عيسى **""")
