from sklearn.datasets import fetch_20newsgroups

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from pprint import pprint
twenty_train1 = fetch_20newsgroups(subset='train', categories=None, shuffle=True)

pprint(list(twenty_train1.target_names))

categories = ['alt.atheism', 'soc.religion.christian',
              'sci.electronics', 'sci.crypt']

twenty_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True)

pprint(list(twenty_train.target_names))



tfidf_Vect = TfidfVectorizer()
X_train_tfidf = tfidf_Vect.fit_transform(twenty_train.data)
clf = SVC()
clf.fit(X_train_tfidf, twenty_train.target)
twenty_test = fetch_20newsgroups(subset='test', categories=categories, shuffle=True)
X_test_tfidf = tfidf_Vect.transform(twenty_test.data)
predicted = clf.predict(X_test_tfidf)
score = round(clf.score(X_test_tfidf, twenty_test.target) * 100, 2)
print(score)

tfidf_Vect = TfidfVectorizer(ngram_range=(1,2))
X_train_tfidf = tfidf_Vect.fit_transform(twenty_train.data)
clf = SVC()
clf.fit(X_train_tfidf, twenty_train.target)
twenty_test = fetch_20newsgroups(subset='test', categories=categories, shuffle=True)
X_test_tfidf = tfidf_Vect.transform(twenty_test.data)
predicted = clf.predict(X_test_tfidf)
score = round(clf.score(X_test_tfidf, twenty_test.target) * 100, 2)
print('the score for Bigram:',score)

tfidf_Vect = TfidfVectorizer(stop_words='english')
X_train_tfidf = tfidf_Vect.fit_transform(twenty_train.data)
clf = SVC()
clf.fit(X_train_tfidf, twenty_train.target)
twenty_test = fetch_20newsgroups(subset='test', categories=categories, shuffle=True)
X_test_tfidf = tfidf_Vect.transform(twenty_test.data)
score = round(clf.score(X_test_tfidf, twenty_test.target) * 100, 2)
print('the score after setting Stop Words:',score)