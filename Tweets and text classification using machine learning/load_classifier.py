#populate datasets
data_x = []
with open('neg.txt') as f:
    i = 0
    for line in f:
        data_x.append(line)
        
with open('pos.txt') as f:
    for line in f:
        data_x.append(line)
        
from sklearn.utils import shuffle
data_x = shuffle(data_x, random_state = 0)
#convertion text to feasures
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
vectorizer.fit_transform(data_x).toarray()
#Free some space
data_x = None
from keras.models import model_from_json
# load json and create model
jason_file_name = 'classifier'
json_file = open("{0}.json".format(jason_file_name) , 'r')
readed_classifier = json_file.read()
json_file.close()
classifier = model_from_json(readed_classifier)
# load weights into new model
classifier.load_weights("{0}.h5".format(jason_file_name))
print("Loaded classifier from disk")
from sklearn.feature_extraction.text import CountVectorizer  
correct_preictions = 0
false_preictions = 0
with open('pos_testing.txt') as f:
    for i, line in enumerate(f):
        if classifier.predict(vectorizer.transform([line]).toarray()) >= 0.5:
            correct_preictions = correct_preictions + 1
        else:
            false_preictions = false_preictions + 1
        print('. {0}'.format(i))

with open('neg_testing.txt') as f:
    for i, line in enumerate(f):
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





















