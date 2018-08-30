#populate datasets
y = []
data_x = []
#space 0
#politics 1
#electronics 2
#baseball 3

with open('new_papers\space.txt') as f:
    for i, line in enumerate(f):
        data_x.append(line)
        y.append(0)
        
with open('new_papers\politics_guns.txt') as f:
    for i, line in enumerate(f):
        data_x.append(line)
        y.append(1)
with open('new_papers\electronics.txt') as f:
    for i, line in enumerate(f):
        data_x.append(line)
        y.append(2)
        
with open('new_papers\baseball.txt') as f:
    for i, line in enumerate(f):
        data_x.append(line)
        y.append(3)
        
from sklearn.utils import shuffle
data_x, y = shuffle(data_x, y)    

#convertion text to feasures
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
x = vectorizer.fit_transform(data_x).toarray()

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x , y , test_size = 0.1)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
import numpy as np
diagonal_sum = np.trace(np.asarray(cm))
total_sum = np.sum(cm)
print('Accuracy_Logistic_Regression: {}'.format(diagonal_sum/total_sum))











