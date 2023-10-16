#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split


# In[3]:


df_init = pd.read_csv("Tweetsss.csv")


# In[5]:


df_init.head()


# In[7]:


df = df_init[['airline_sentiment', 'text']].copy()


# In[8]:


df.head()


# In[9]:


sentiment_counts = df['airline_sentiment'].value_counts()

plt.figure(figsize=(8, 6))
ax = sns.histplot(df['airline_sentiment'], bins=3, color='skyblue', discrete=True)
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.title('Distribution of Airline Sentiments')

for p in ax.patches:
    ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', fontsize=12, xytext=(0, 10), textcoords='offset points')

plt.xticks()
plt.show()


# In[10]:


sentiment_counts = df['airline_sentiment'].value_counts()
plt.figure(figsize=(8, 8))
plt.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=140)
plt.title('Distribution of Airline Sentiments')
plt.axis('equal') 
plt.show()


# In[12]:


target_map = {'positive': 1, 'negative': 0, 'neutral': 2}

df['airline_sentiment'] = df['airline_sentiment'].str.strip()

df['target'] = df['airline_sentiment'].map(target_map)


# In[13]:


df.head()


# In[14]:


df_train, df_test = train_test_split(df)


# In[15]:


df_train.head()


# In[16]:


vectorizer = TfidfVectorizer(max_features=2000)
x_train = vectorizer.fit_transform(df_train['text'])
x_test = vectorizer.transform(df_test['text'])
y_train = df_train['target']
y_test = df_test['target']


# In[17]:


model = LogisticRegression(max_iter=500)
model.fit(x_train, y_train)


# In[18]:


train_accuracy = model.score(x_train, y_train)
test_accuracy = model.score(x_test, y_test)
print('Train accuracy: ', train_accuracy)
print('Test accuracy: ', test_accuracy)


# In[19]:


Pr_train = model.predict_proba(x_train)
Pr_test = model.predict_proba(x_test)
train_auc = roc_auc_score(y_train, Pr_train, multi_class='ovo')
test_auc = roc_auc_score(y_test, Pr_test, multi_class='ovo')
print('Train AUC: ', train_auc)
print('Test AUC: ', test_auc)


# In[20]:


P_train = model.predict(x_train)
P_test = model.predict(x_test)


# In[21]:


cm_train = confusion_matrix(y_train, P_train, normalize='true')
cm_train


# In[22]:


plt.figure(figsize=(8, 6))
sns.set(font_scale=1.2)
heatmap = sns.heatmap(cm_train, annot=True, fmt='.2%', cmap='Blues', cbar=False,
            xticklabels=['Predicted Negative', 'Predicted Positive', 'Neutral'],
            yticklabels=['Actual Negative', 'Actual Positive', 'Neutral'])
heatmap.xaxis.set_ticks_position('top')  # Move x-axis labels to the top
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Normalized Confusion Matrix for the Train Set')
plt.show()


# In[23]:


cm_test = confusion_matrix(y_test, P_test, normalize='true')
cm_test


# In[24]:


plt.figure(figsize=(8, 6))
sns.set(font_scale=1.2)
heatmap = sns.heatmap(cm_test, annot=True, fmt='.2%', cmap='Blues', cbar=False,
            xticklabels=['Predicted Negative', 'Predicted Positive', 'Neutral'],
            yticklabels=['Actual Negative', 'Actual Positive', 'Neutral'])
heatmap.xaxis.set_ticks_position('top')  # Move x-axis labels to the top
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Normalized Confusion Matrix for the Test Set')
plt.show()


# In[25]:


binary_target_list = [target_map['positive'], target_map['negative']]
df_b_train = df_train[df_train['target'].isin(binary_target_list)]
df_b_test = df_test[df_test['target'].isin(binary_target_list)]


# In[26]:


x_train = vectorizer.fit_transform(df_b_train['text'])
x_test = vectorizer.transform(df_b_test['text'])
y_train = df_b_train['target']
y_test = df_b_test['target']


# In[27]:


model = LogisticRegression(max_iter=500)
model.fit(x_train, y_train)
binary_train_accuracy = model.score(x_train, y_train)
binary_test_accuracy = model.score(x_test, y_test)
print('Binary Train accuracy: ', binary_train_accuracy)
print('Binary Test accuracy: ', binary_test_accuracy)


# In[28]:


Pr_train = model.predict_proba(x_train)[:, 1]
Pr_test = model.predict_proba(x_test)[:, 1]
binary_train_auc = roc_auc_score(y_train, Pr_train)
binary_test_auc = roc_auc_score(y_test, Pr_test)
print('Binary Train AUC: ', binary_train_auc)
print('Binary Test AUC: ', binary_test_auc)


# In[29]:


plt.hist(model.coef_[0], bins=40)
plt.xlabel('Feature Weight')
plt.ylabel('Frequency')
plt.title('Distribution of Feature Weights')
plt.show()


# In[30]:


word_index_map = vectorizer.vocabulary_


# In[31]:


threshold = 2


# In[32]:


print('Most Positive Words')
for word, index in word_index_map.items():
    weight = model.coef_[0][index]
    if weight > threshold:
        print(word, weight)


# In[33]:


print('Most Negative Words')
for word, index in word_index_map.items():
    weight = model.coef_[0][index]
    if weight < -threshold:
        print(word, weight)


# In[ ]:




