#ANN
#populate datasets
y = []
data_x = []
with open('neg.txt') as f:
    for i, line in enumerate(f):
        if i >= 4000:
            break
        data_x.append(line)
        y.append(0)
        
with open('pos.txt') as f:
    for i, line in enumerate(f):
        if i >= 4000:
            break
        data_x.append(line)
        y.append(1)
        
from sklearn.utils import shuffle
data_x, y = shuffle(data_x, y)    
#convertion text to feasures
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
import numpy as np
x = vectorizer.fit_transform(data_x).toarray()
#Free some space
data_x = None
#splitting the dataset into training set and test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.1)
#making ANN
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
#initializing ANN
classifier = Sequential()
#Adding input layer and the first hidden layer
classifier.add(Dense(output_dim=int(len(x[0])/2), init='uniform', activation='relu', input_dim=len(x[0])))
classifier.add(Dropout(p=0.15))
#Adding the second hidden layer
classifier.add(Dense(output_dim=int(len(x[0])/2), init='uniform', activation='relu'))
classifier.add(Dropout(p=0.15))
#Adding the output layer
classifier.add(Dense(output_dim=1, init='uniform', activation='sigmoid'))
#compiling the ANN
classifier.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
#Fitting the ANN
classifier.fit(np.array(x_train), np.array(y_train) , batch_size=15)
# serialize model to JSON
model_json = classifier.to_json()
with open("ANN_classifier.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
classifier.save_weights("classifier.h5")
print("Saved ANN_classifier to disk")

y_pred = classifier.predict(x_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
import numpy as np
diagonal_sum = np.trace(np.asarray(cm))
total_sum = np.sum(cm)
print('Accuracy_ANN_classifier: {}'.format(diagonal_sum/total_sum))








