#populate datasets
y = []
data_x = []
with open('wide_data\bussiness.txt') as f:
    for line in f:
        data_x.append(line)
        y.append(-2)
with open('wide_data\entertainment.txt') as f:
    for line in f:
        data_x.append(line)
        y.append(-1)
with open('wide_data\health.txt') as f:
    for line in f:
        data_x.append(line)
        y.append(0)
with open('wide_data\politics.txt') as f:
    for line in f:
        data_x.append(line)
        y.append(1)
with open('wide_data\sports.txt') as f:
    for line in f:
        data_x.append(line)
        y.append(2)
with open('wide_data\technology.txt') as f:
    for line in f:
        data_x.append(line)
        y.append(3)
        
from sklearn.utils import shuffle
data_x, y = shuffle(data_x, y, random_state = 0)    
#convertion text to feasures
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
import numpy as np
x = vectorizer.fit_transform(data_x).toarray()
#Free some space
data_x = None
#splitting the dataset into training set and test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)
x, y = None
#making ANN
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
#initializing ANN
classifier = Sequential()
#Adding input layer and the first hidden layer
classifier.add(Dense(output_dim=int(len(x[0])/2), init='uniform', activation='relu', input_dim=len(x[0])))
classifier.add(Dropout(p=0.1))
#Adding the second hidden layer
classifier.add(Dense(output_dim=int(len(x[0])/2), init='uniform', activation='relu'))
classifier.add(Dropout(p=0.1))
#Adding the output layer
classifier.add(Dense(output_dim=6, init='uniform', activation='softmax'))
#compiling the ANN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#Fitting the ANN
classifier.fit(np.array(x_train), np.array(y_train) ,nb_epoch=2)
# serialize model to JSON
model_json = classifier.to_json()
with open("wide_classifier.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
classifier.save_weights("wide_classifier.h5")
print("Saved classifier to disk")
predict = 'the products of this company is good !'
print('Positive' if classifier.predict(vectorizer.transform([predict]).toarray()) > 0.5 else 'Negative')







