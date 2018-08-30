#Coding: utf-8 
#Note: Some of the cells in this notebook are computationally expensive. To reduce runtime, 
#    this notebook is using a subset of the data.
#Case Study: Sentiment Analysis
#Data Prep
import pandas as pd
import numpy as np
import ImproveAccureacy_funcitons as fn

# Read in the data
df = pd.read_csv('Amazon_Unlocked_Mobile.csv', error_bad_lines=False)
# Sample the data to speed up computation
# Comment out this line to match with lecture
df = df.sample(frac = 1, random_state = 10)
df.head()
# Drop missing values
df.dropna(inplace=True)
# Remove any 'neutral' ratings equal to 3
df = df[df['Rating'] != 3]
# Encode 4s and 5s as 1 (rated positively)
# Encode 1s and 2s as 0 (rated poorly)
df['Positively Rated'] = np.where(df['Rating'] > 3, 1, 0)
df.head(10)
from sklearn.model_selection import train_test_split
# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(df['Reviews'], 
                                                    df['Positively Rated'])
# CountVectorizer
print('CountVectorizer method')
from sklearn.feature_extraction.text import CountVectorizer
# Fit the CountVectorizer to the training data
vect = CountVectorizer().fit(X_train)
vect.get_feature_names()[::2000]
# transform the documents in the training data to a document-term matrix
X_train_vectorized = vect.transform(X_train)
from sklearn.linear_model import LogisticRegression
# Train the model
model = LogisticRegression()
model.fit(X_train_vectorized, y_train)
from sklearn.metrics import roc_auc_score
# Predict the transformed test documents
Accuracy = roc_auc_score(y_test, model.predict(vect.transform(X_test)))
print('ACC: ', Accuracy)
fn.print_pie(Accuracy)
## Tfidf
print('TfidfVectorizer method')
from sklearn.feature_extraction.text import TfidfVectorizer
# Fit the TfidfVectorizer to the training data specifiying a minimum document frequency of 5
vect = TfidfVectorizer(min_df=5).fit(X_train)
X_train_vectorized = vect.transform(X_train)
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train_vectorized, y_train)
from sklearn.metrics import roc_auc_score
Accuracy = roc_auc_score(y_test, model.predict(vect.transform(X_test)))
print('ACC: ', Accuracy)
fn.print_pie(Accuracy)
print(model.predict(vect.transform(['not an issue, phone is working',
                                    'an issue, phone is not working'])))

## n-grams
# Fit the CountVectorizer to the training data specifiying a minimum 
# document frequency of 5 and extracting 1-grams and 2-grams
print('n-grams method')
vect = CountVectorizer(min_df=5, ngram_range=(1,2)).fit(X_train)
X_train_vectorized = vect.transform(X_train)
model = LogisticRegression()
model.fit(X_train_vectorized, y_train)
Accuracy = roc_auc_score(y_test, model.predict(vect.transform(X_test)))
print('ACC: ', Accuracy)
fn.print_pie(Accuracy)
print(model.predict(vect.transform(['not an issue, phone is working',
                                    'an issue, phone is not working'])))








