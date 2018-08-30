#populate datasets
y = []
data_x = []

with open('neg.txt') as f:
    for i, line in enumerate(f):
        if i >= 1000:
            break
        data_x.append(line)
        y.append(0)
        
with open('pos.txt') as f:
    for i, line in enumerate(f):
        if i >= 1000:
            break
        data_x.append(line)
        y.append(1)
        

#from sklearn.utils import shuffle
#data_x, y = shuffle(data_x, y, random_state = 0)
       
#convertion text to feasures
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
x = vectorizer.fit_transform(data_x).toarray()
#splitting the dataset into training set and test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3)

#Improving the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(output_dim=int(len(x[0])/2), init='uniform', activation='relu', input_dim=len(x[0])))
    classifier.add(Dropout(p=0.15))
    classifier.add(Dense(output_dim=int(len(x[0])/2), init='uniform', activation='relu'))
    classifier.add(Dropout(p=0.15))
    classifier.add(Dense(output_dim=1, init='uniform', activation='sigmoid'))
    classifier.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn=build_classifier)
parameters = {
              'batch_size':[1, 10],
              'epochs': [1, 3],
              'optimizer': ['adam', 'rmsprop']
             }
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10)
grid_search = grid_search.fit(x_train, y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_
print('best_parameters={0} , best_accuracy={1}'.format(best_parameters, best_accuracy))
# serialize model to JSON
model_json = classifier.to_json()
with open("best_classifier.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
classifier.save_weights("best_classifier.h5")
print("Saved best_classifier to disk")

from sklearn.feature_extraction.text import CountVectorizer  
correct_preictions = 0
false_preictions = 0
with open('pos_testing.txt') as f:
    for i, line in enumerate(f):
        if i >= 250:
            break
        if classifier.predict(vectorizer.transform([line]).toarray()) >= 0.5:
            correct_preictions = correct_preictions + 1
        else:
            false_preictions = false_preictions + 1
        print('. {0}'.format(i))

with open('neg_testing.txt') as f:
    for i, line in enumerate(f):
        if i >= 250:
            break
        if classifier.predict(vectorizer.transform([line]).toarray()) < 0.5:
            correct_preictions = correct_preictions + 1
        else:
            false_preictions = false_preictions + 1
        print('.. {0}'.format(i))

total_size = correct_preictions + false_preictions
print('Accuracy: {0}'.format(correct_preictions/total_size))
import matplotlib.pyplot as plt
labels = ['Correct predictions', 'False predictions']
sizes = [correct_preictions, false_preictions]
colors = ['yellowgreen','lightcoral']
patches, texts = plt.pie(sizes, colors=colors, shadow=True, startangle=90)
plt.legend(patches, labels, loc="best")
plt.axis('equal')
plt.tight_layout()
plt.show()





















