
# coding: utf-8

# In[1]:


import pandas as pd
df = pd.read_csv("fake_or_real_news.csv")
df.shape
df.shapedf = df.set_index("Unnamed: 0")
y = df.label 
df.drop("label", axis=1)


# In[3]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
df = pd.read_csv("fake_or_real_news.csv")
df.shapedf = df.set_index("Unnamed: 0")
df.head()
y = df.label 
df.drop("label", axis=1)
X_train, X_test, y_train, y_test = train_test_split(df['text'], y, test_size=0.6339, random_state=53)
count_vectorizer = CountVectorizer(stop_words='english')
count_train = count_vectorizer.fit_transform(X_train)
count_test = count_vectorizer.transform(X_test)
count_vectorizer

