#Naive_bayes
#populate datasets
y = []
data_x = []
data_size = 10000
with open('very_large_dataset_pos_neg/pos.txt', encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i >= data_size:
            break
        data_x.append(line)
        y.append(1)
print('. pos done')    
with open('very_large_dataset_pos_neg/neg.txt', encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i >= data_size:
            break
        data_x.append(line)
        y.append(0)
print('. neg done')
          

#convertion text to feasures
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
x = vectorizer.fit_transform(data_x).toarray()

from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
pred = classifier.fit(x, y)

from sklearn.feature_extraction.text import CountVectorizer  
correct_preictions = 0
false_preictions = 0
test_size = 2000
counter = 0
with open('very_large_dataset_pos_neg/pos.txt', encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i <= data_size:
            continue
        elif i >= data_size + test_size:
            break
        else :
            counter = counter + 1
            prediction = pred.predict(vectorizer.transform([line]).toarray())
            if prediction >= 0.5:
                correct_preictions = correct_preictions + 1
            else:
                false_preictions = false_preictions + 1
            #print('. {0}'.format(i-test_size, prediction))

with open('very_large_dataset_pos_neg/neg.txt', encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i <= data_size:
            continue
        elif i >= data_size + test_size:
            break
        else:
            counter = counter + 1
            prediction = pred.predict(vectorizer.transform([line]).toarray())
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











