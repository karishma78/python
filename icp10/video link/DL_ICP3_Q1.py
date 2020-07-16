#!/usr/bin/env python
# coding: utf-8

# In[2]:


from keras.models import Sequential
from keras import layers
from keras.preprocessing.text import Tokenizer
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


# In[3]:


df = pd.read_csv('imdb-master.csv',encoding='latin-1')
df.head(5)


# # Dropping Unnecessary labelfrom dataset
# 

# In[4]:


df = df[df['label']!='unsup']
sentences = df['review'].values
y = df['label'].values


# #Tokenizing data

# In[5]:


tokenizer = Tokenizer(num_words=2000)


# 
# #Getting the vocabulary of data
# 

# In[6]:


sentences = tokenizer.texts_to_matrix(sentences)


# Label Encoder

# In[7]:


le = preprocessing.LabelEncoder()
y = le.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(sentences, y, test_size=0.25, random_state=1000)


# In[8]:


model = Sequential()
model.add(layers.Dense(300,input_dim= 2000, activation='relu'))
model.add(layers.Dense(2, activation='sigmoid')) #changing number of neuron to 2 as we have only two labels Pos and Neg
model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['acc'])
history=model.fit(X_train,y_train, epochs=5, verbose=True, validation_data=(X_test,y_test), batch_size=256)


# In[9]:


test_loss, test_acc = model.evaluate(X_test, y_test)
print("Accurracy",test_acc)


# In[ ]:




