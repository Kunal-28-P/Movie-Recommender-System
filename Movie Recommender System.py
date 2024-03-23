#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import ast


# In[5]:


movies=pd.read_csv(r"D:\movie_recommender_system\tmdb_5000_movies.csv")
credit=pd.read_csv(r"D:\movie_recommender_system\tmdb_5000_credits.csv")


# In[6]:


movies.head()


# In[7]:


credit.head()


# In[9]:


movies.shape


# In[10]:


credit.shape


# In[11]:


movies.info()


# In[13]:


credit.info()


# In[14]:


movies.describe()


# In[15]:


credit.describe()


# In[16]:


movies.isnull().sum()


# In[17]:


credit.isnull().sum()


# In[21]:


movies = movies.merge(credit, on='title')
movies.head(1)


# In[22]:


movies.shape


# In[23]:


#useful features
#genres
#id
#keywords
#title
#overview
#cast
#crew
movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]
movies.head()


# In[26]:


movies.isnull().sum()


# In[25]:


movies.dropna(inplace=True)


# In[27]:


movies.duplicated().sum()


# In[28]:


movies.iloc[0].genres


# In[29]:


#make this data '[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {"id": 878, "name": "Science Fiction"}]'
# to ['Action', 'Adventure', 'Fantasy', 'Science Fiction']
#In this we will pass the given Dictionary and append it in list 
def convert(obj):
    L = []
    for i in ast.literal_eval(obj):  #ast.literal_eval(obj)  we used this function because our name is in string it cant convert to list so this function help us to convert it
       L.append(i['name'])
    return L
movies['genres'] = movies['genres'].apply(convert)
movies.head()


# In[30]:


movies['keywords'] = movies['keywords'].apply(convert)
movies.head()


# In[31]:


movies['cast'][0]


# In[32]:


#to get the top 3 actor
def convert2(obj):
    L = []
    counter = 0
    for i in ast.literal_eval(obj):  
        if counter != 3:
            L.append(i['name'])
            counter += 1
        else:
            break
    return L

movies['cast'] = movies['cast'].apply(convert2)
movies.head()


# In[33]:


movies['crew'][0]


# In[34]:


#Getting the Director Name of Movie
def fetch_director(obj):
    L = []
    for i in ast.literal_eval(obj):  
        if i['job'] == 'Director':
            L.append(i['name'])
            break
    return L



# In[35]:


movies['crew'] = movies['crew'].apply(fetch_director)
movies.head()


# In[36]:


movies['overview'][0]


# In[37]:


#convert String to list so we can Easily merge
movies['overview'] = movies['overview'].apply(lambda x:x.split())
movies.head()


# In[38]:


#Remove Spaces from the features cause then it can be Problematic for the Recommender system
movies['genres'] = movies['genres'].apply(lambda x:[i.replace(" ", "") for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x:[i.replace(" ", "") for i in x])
movies['cast'] = movies['cast'].apply(lambda x:[i.replace(" ", "") for i in x])
movies['crew'] = movies['crew'].apply(lambda x:[i.replace(" ", "") for i in x])
movies.head()


# In[39]:


movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast']  + movies['crew']
movies.head()


# In[40]:


new_df = movies[['movie_id', 'title', 'tags']]
new_df.head()


# In[41]:


#Convert the list tag to string
new_df['tags'] = new_df['tags'].apply(lambda x:" ".join(x))
new_df.head()


# In[42]:


new_df['tags'][0]


# In[43]:


#convert into lowercase as suggested
new_df['tags'] = new_df['tags'].apply(lambda x:x.lower())
new_df.head()


# In[44]:


#Now we have the tags all we needed now how we can derive the similarity between different tags of movie for suggestion
#So, guess what! there is the conept called Text-Vectorization  in which text are converted to the vectors 
#In this frameWork there is technique called bags of word which works like now we have the around 5000 movies and their tags we will then merge all the tags and get the most frequent words
#now lets assume we have 5000 words we will create the dataset where it will count how many times the that 5000 times words are repeated and it will created the data of about 5000,5000 shape 
# this data is then plot in 5000 Dimensional and for particular movie there is 1 vector now if we select that movie we will see which is the closet vectors and which ever movies vectors is closet we suggest that accordingly
#and this doesnt consider the stopwords like, a, are, I, is, etc,
#so for this there is library in sklearn which is CountVector


# In[45]:


import nltk
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()


# In[48]:


#for example
ps.stem('loving')


# In[49]:


ps.stem('having')


# In[50]:


def stem(text):
    y = []
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)


# In[51]:


new_df['tags'] = new_df['tags'].apply(stem)
new_df


# In[52]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 5000, stop_words = 'english')


# In[53]:


vectors = cv.fit_transform(new_df['tags']).toarray()
vectors[0] #its for first movie


# In[54]:


vectors.shape


# In[55]:


#these are most frequent words
cv.get_feature_names_out()


# In[56]:


from sklearn.metrics.pairwise import cosine_similarity
similarity = cosine_similarity(vectors)
similarity[0]   

# this is the similarity between the first movie with other


# In[57]:


similarity.shape


# In[58]:


def recommend(movie):
    movie_index=new_df[new_df['title'] == movie].index[0]  #fetching the index of that movie 
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x:x[1])[1:6]
    for i in movies_list:
        print(new_df.iloc[i[0]].title)


# In[66]:


recommand_movie = input('Movie to watch: ')
print('')
print("Movies you may like:")
recommend(recommand_movie)


# In[ ]:




