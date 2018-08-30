#populate datasets
y = []
data_x = []
data_size = 1000
with open('very_large_dataset_pos_neg/pos.txt') as f:
    for i, line in enumerate(f):
        if i >= data_size:
            break
        data_x.append(line)
        y.append(1)
print('. pos done')    
with open('very_large_dataset_pos_neg/neg.txt') as f:
    for i, line in enumerate(f):
        if i >= data_size:
            break
        data_x.append(line)
        y.append(0)
print('. neg done')

from sklearn.utils import shuffle
data_x, y = shuffle(data_x, y)    

#convertion text to feasures
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
x = vectorizer.fit_transform(data_x).toarray()

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x , y , test_size = 0.2)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(penalty='l2', dual=True, tol=0.0001, 
                           C=0.75, fit_intercept=True, intercept_scaling=1.0)
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
import numpy as np
diagonal_sum = np.trace(np.asarray(cm))
total_sum = np.sum(cm)
print('Accuracy_Logistic_Regression: {}'.format(diagonal_sum/total_sum))













