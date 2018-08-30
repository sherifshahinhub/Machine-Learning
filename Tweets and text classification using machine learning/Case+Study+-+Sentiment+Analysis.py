#Coding: utf-8 
#Note: Some of the cells in this notebook are computationally expensive. To reduce runtime, 
#    this notebook is using a subset of the data.
#Case Study: Sentiment Analysis
#Data Prep
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read in the data
df = pd.read_csv('Amazon_Unlocked_Mobile.csv')
# Sample the data to speed up computation
# Comment out this line to match with lecture
df = df.sample(frac = 0.1, random_state = 10)
df.head()
# Drop missing values
df.dropna(inplace=True)
# Remove any 'neutral' ratings equal to 3
df = df[df['Rating'] != 3]
# Encode 4s and 5s as 1 (rated positively)
# Encode 1s and 2s as 0 (rated poorly)
df['Positively Rated'] = np.where(df['Rating'] > 3, 1, 0)
df.head(10)
# Most ratings are positive
df['Positively Rated'].mean()
from sklearn.model_selection import train_test_split
# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(df['Reviews'], 
                                                    df['Positively Rated'], 
                                                    random_state=0)
print('X_train first entry:\n\n', X_train.iloc[0])
print('\n\nX_train shape: ', X_train.shape)
# # CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer
# Fit the CountVectorizer to the training data
vect = CountVectorizer().fit(X_train)
vect.get_feature_names()[::2000]
print('len(vect.get_feature_names()): {}'.format(len(vect.get_feature_names())))
# transform the documents in the training data to a document-term matrix
X_train_vectorized = vect.transform(X_train)
X_train_vectorized


from sklearn.linear_model import LogisticRegression
# Train the model
model = LogisticRegression()
model.fit(X_train_vectorized, y_train)
from sklearn.metrics import roc_auc_score
# Predict the transformed test documents
predictions = model.predict(vect.transform(X_test))
print('AUC: ', roc_auc_score(y_test, predictions))
# get the feature names as numpy array
feature_names = np.array(vect.get_feature_names())
# Sort the coefficients from the model
sorted_coef_index = model.coef_[0].argsort()
# Find the 10 smallest and 10 largest coefficients
# The 10 largest coefficients are being indexed using [:-11:-1] 
# so the list returned is in order of largest to smallest
#print('Smallest Coefs:\n{}\n'.format(feature_names[sorted_coef_index[:10]]))
#print('Largest Coefs: \n{}'.format(feature_names[sorted_coef_index[:-11:-1]]))
## Tfidf

from sklearn.feature_extraction.text import TfidfVectorizer
# Fit the TfidfVectorizer to the training data specifiying a minimum document frequency of 5
vect = TfidfVectorizer(min_df=5).fit(X_train)
len(vect.get_feature_names())
X_train_vectorized = vect.transform(X_train)
model = LogisticRegression()
model.fit(X_train_vectorized, y_train)
predictions = model.predict(vect.transform(X_test))
print('ACC: ', roc_auc_score(y_test, predictions))
feature_names = np.array(vect.get_feature_names())
sorted_tfidf_index = X_train_vectorized.max(0).toarray()[0].argsort()
#print('Smallest tfidf:\n{}\n'.format(feature_names[sorted_tfidf_index[:10]]))
#print('Largest tfidf: \n{}'.format(feature_names[sorted_tfidf_index[:-11:-1]]))
sorted_coef_index = model.coef_[0].argsort()
#print('Smallest Coefs:\n{}\n'.format(feature_names[sorted_coef_index[:10]]))
#print('Largest Coefs: \n{}'.format(feature_names[sorted_coef_index[:-11:-1]]))
# These reviews are treated the same by our current model
print(model.predict(vect.transform(['not an issue, phone is working',
                                    'an issue, phone is not working'])))
## n-grams
# Fit the CountVectorizer to the training data specifiying a minimum 
# document frequency of 5 and extracting 1-grams and 2-grams
vect = CountVectorizer(min_df=5, ngram_range=(1,2)).fit(X_train)
X_train_vectorized = vect.transform(X_train)
len(vect.get_feature_names())
model = LogisticRegression()
model.fit(X_train_vectorized, y_train)
predictions = model.predict(vect.transform(X_test))
print('AUC: ', roc_auc_score(y_test, predictions))
feature_names = np.array(vect.get_feature_names())
sorted_coef_index = model.coef_[0].argsort()
#print('Smallest Coefs:\n{}\n'.format(feature_names[sorted_coef_index[:10]]))
#print('Largest Coefs: \n{}'.format(feature_names[sorted_coef_index[:-11:-1]]))
# These reviews are now correctly identified
print(model.predict(vect.transform(['not an issue, phone is working',
                                    'an issue, phone is not working'])))
"""
"""
data_size = 0
from sklearn.feature_extraction.text import CountVectorizer  
correct_preictions = 0
false_preictions = 0
test_size = 20000
counter = 0
with open('pos.txt', encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i <= data_size:
            continue
        elif i >= data_size + test_size:
            break
        else :
            counter = counter + 1
            prediction = model.predict(vect.transform([line]))
            if prediction >= 0.5:
                correct_preictions = correct_preictions + 1
            else:
                false_preictions = false_preictions + 1
            #print('. {0}'.format(i-test_size, prediction))

with open('neg.txt', encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i <= data_size:
            continue
        elif i >= data_size + test_size:
            break
        else:
            counter = counter + 1
            prediction = model.predict(vect.transform([line]))
            if prediction < 0.5:
                correct_preictions = correct_preictions + 1
            else:
                false_preictions = false_preictions + 1
            #print('.. {0} -'.format(i-test_size, prediction))

print('correct_preictions: {0} | false_preictions: {1}'.format(correct_preictions, false_preictions))
total_size = correct_preictions + false_preictions
print('Accuracy: {0}'.format(correct_preictions/counter))
import matplotlib.pyplot as plt
labels = ['Correct predictions', 'False predictions']
sizes = [correct_preictions, false_preictions]
colors = ['yellowgreen','lightcoral']
patches, texts = plt.pie(sizes, colors=colors, shadow=True, startangle=90)
plt.legend(patches, labels, loc="best")
plt.axis('equal')
plt.tight_layout()
plt.show()










