#Data Prep
import pandas as pd
import ImproveAccureacy_funcitons as fn

# Read in the data
business = pd.read_csv('classify_tweets_by_classes/business.csv', encoding = 'unicode_escape', error_bad_lines=False)
business.head()
# Drop missing values
business.dropna(inplace=True)

entertainment = pd.read_csv('classify_tweets_by_classes/entertainment.csv', encoding = 'unicode_escape', error_bad_lines=False)
entertainment.head()
# Drop missing values
entertainment.dropna(inplace=True)

health = pd.read_csv('classify_tweets_by_classes/health.csv', encoding = 'unicode_escape', error_bad_lines=False)
health.head()
# Drop missing values
health.dropna(inplace=True)

politics = pd.read_csv('classify_tweets_by_classes/politics.csv', encoding = 'unicode_escape', error_bad_lines=False)
politics.head()
# Drop missing values
politics.dropna(inplace=True)

sports = pd.read_csv('classify_tweets_by_classes/sports.csv', encoding = 'unicode_escape', error_bad_lines=False)
sports.head()
# Drop missing values
sports.dropna(inplace=True)

technology = pd.read_csv('classify_tweets_by_classes/technology.csv', encoding = 'unicode_escape', error_bad_lines=False)
technology.head()
# Drop missing values
technology.dropna(inplace=True)



from sklearn.model_selection import train_test_split
# Split data into training and test sets
business_X_train,business_X_test, business_y_train, business_y_test = train_test_split(business['data'],business['res'])
entertainment_X_train,entertainment_X_test, entertainment_y_train, entertainment_y_test = train_test_split(entertainment['data'],entertainment['res'])
health_X_train, health_X_test, health_y_train, health_y_test = train_test_split(health['data'],health['res'])
politics_X_train, politics_X_test, politics_y_train, politics_y_test = train_test_split(politics['data'],politics['res'])
sports_X_train, sports_X_test, sports_y_train, sports_y_test = train_test_split(sports['data'],sports['res'])
technology_X_train, technology_X_test, technology_y_train, technology_y_test = train_test_split(technology['data'],technology['res'])

## n-grams
# Fit the CountVectorizer to the training data specifiying a minimum 
# document frequency of 5 and extracting 1-grams and 2-grams
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
min_df = 2

business_vect = CountVectorizer(min_df=min_df, ngram_range=(1,2)).fit(business_X_train)
business_X_train_vectorized = business_vect.transform(business_X_train)
business_model = LogisticRegression()
business_model.fit(business_X_train_vectorized, business_y_train)
Accuracy = roc_auc_score(business_y_test, business_model.predict(business_vect.transform(business_X_test)))
print('business ACC: ', Accuracy)
fn.print_pie(Accuracy)

entertainment_vect = CountVectorizer(min_df=min_df, ngram_range=(1,2)).fit(entertainment_X_train)
entertainment_X_train_vectorized = entertainment_vect.transform(entertainment_X_train)
entertainment_model = LogisticRegression()
entertainment_model.fit(entertainment_X_train_vectorized, entertainment_y_train)
Accuracy = roc_auc_score(entertainment_y_test, entertainment_model.predict(entertainment_vect.transform(entertainment_X_test)))
print('entertainment ACC: ', Accuracy)
fn.print_pie(Accuracy)

health_vect = CountVectorizer(min_df=min_df, ngram_range=(1,2)).fit(health_X_train)
health_X_train_vectorized = health_vect.transform(health_X_train)
health_model = LogisticRegression()
health_model.fit(health_X_train_vectorized, health_y_train)
Accuracy = roc_auc_score(health_y_test, health_model.predict(health_vect.transform(health_X_test)))
print('health ACC: ', Accuracy)
fn.print_pie(Accuracy)

politics_vect = CountVectorizer(min_df=min_df, ngram_range=(1,2)).fit(politics_X_train)
politics_X_train_vectorized = politics_vect.transform(politics_X_train)
politics_model = LogisticRegression()
politics_model.fit(politics_X_train_vectorized, politics_y_train)
Accuracy = roc_auc_score(politics_y_test, politics_model.predict(politics_vect.transform(politics_X_test)))
print('politics ACC: ', Accuracy)
fn.print_pie(Accuracy)

sports_vect = CountVectorizer(min_df=min_df, ngram_range=(1,2)).fit(sports_X_train)
sports_X_train_vectorized = sports_vect.transform(sports_X_train)
sports_model = LogisticRegression()
sports_model.fit(sports_X_train_vectorized, sports_y_train)
Accuracy = roc_auc_score(sports_y_test, sports_model.predict(sports_vect.transform(sports_X_test)))
print('sports ACC: ', Accuracy)
fn.print_pie(Accuracy)

technology_vect = CountVectorizer(min_df=min_df, ngram_range=(1,2)).fit(technology_X_train)
technology_X_train_vectorized = technology_vect.transform(technology_X_train)
technology_model = LogisticRegression()
technology_model.fit(technology_X_train_vectorized, technology_y_train)
Accuracy = roc_auc_score(technology_y_test, technology_model.predict(technology_vect.transform(technology_X_test)))
print('technology ACC: ', Accuracy)
fn.print_pie(Accuracy)

#business 0 , entertainment 1 , health 2 , politics 3 , sports 4 , technology 5
classes_table = ['business', 'entertainment', 'health', 'politics', 'sports', 'technology']
predict_tweet = 'The republican party elections will held in this month'
prediction_table = []
prediction_table.append(business_model.predict_proba(business_vect.transform([predict_tweet]))[0][1])
prediction_table.append(entertainment_model.predict_proba(entertainment_vect.transform([predict_tweet]))[0][1])
prediction_table.append(health_model.predict_proba(health_vect.transform([predict_tweet]))[0][1])
prediction_table.append(politics_model.predict_proba(politics_vect.transform([predict_tweet]))[0][1])
prediction_table.append(sports_model.predict_proba(sports_vect.transform([predict_tweet]))[0][1])
prediction_table.append(technology_model.predict_proba(technology_vect.transform([predict_tweet]))[0][1])
print(prediction_table)
print(classes_table[prediction_table.index(max(prediction_table))])
fn.print_bars(prediction_table)





























