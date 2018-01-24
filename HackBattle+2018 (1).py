
# coding: utf-8

# In[1]:


import pandas as pd
df = pd.read_csv("fake_or_real_news.csv")
df.shape
df.shapedf = df.set_index("Unnamed: 0")
y = df.label 
df.drop("label", axis=1)


# In[4]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
df = pd.read_csv("fake_or_real_news.csv")
df.shapedf = df.set_index("Unnamed: 0")
df.head()
y = df.label 
df.drop("label", axis=1)
X_train, X_test, y_train, y_test = train_test_split(df['text'], y, test_size=0.6339, random_state=53)
count_vectorizer = CountVectorizer(stop_words='english')
count_train = count_vectorizer.fit_transform(X_train)
count_test = count_vectorizer.transform(X_test)
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
tfidf_train = tfidf_vectorizer.fit_transform(X_train) 
tfidf_test = tfidf_vectorizer.transform(X_test)

