from textblob.classifiers import NaiveBayesClassifier
from textblob import TextBlob

x = []
y = []
# health 0
# politics 1
file = open("data/health.txt", encoding='utf-8')
for line in file :
    x.append(line[50:])
    y.append(0)
file.close()


print('1-DONE --health.txt')

file = open("politics.txt", encoding='utf-8')
for line in file:
    x.append(line)
    y.append(1)
file.close()
print('2-DONE politics.txt')


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x , y , test_size = 0.2)

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
#cl = NaiveBayesClassifier(x_train + y_train)
print('DONE --NaiveBayesClassifier')
# Compute accuracy
print("Accuracy: {0}".format(cl.accuracy(x_test + y_test)))

# Show 5 most informative features
cl.show_informative_features(5)

# Classify some text
print(cl.classify("Dr. driving Patrick explores medicine"))  # "health"
print(cl.classify("researchers discovers a cure"))  # "health"
print(cl.classify("Health is good"))  # "health"
print(cl.classify("I feel good"))   # "health"
print(cl.classify("Heart attack !"))   # "health"
print(cl.classify("Agami Hospital is not bad"))   # "health"


print(cl.classify("President Trump"))   # "politiccs"
print(cl.classify("Republican party"))   # "politiccs"
print(cl.classify("Obama democratic"))   # "politiccs"
