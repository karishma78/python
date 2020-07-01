import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
from nltk.util import ngrams
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer
from nltk.stem import SnowballStemmer
from nltk import ne_chunk
from collections import Counter
ps = PorterStemmer()
ls=LancasterStemmer()
ws=SnowballStemmer("english")

lemmatizer = WordNetLemmatizer()
text = open('input.txt', encoding="utf8").read()

word_tokens =word_tokenize(text)
sent_tokens = sent_tokenize(text)
print("Word tokens:",word_tokens)
print("\nSentence tokens:",sent_tokens)

trigrams = ngrams(word_tokens,3)
print("\nTrigrams: ",list(trigrams))

lemmatized_output = ' '.join([lemmatizer.lemmatize(w) for w in word_tokens])
print("\nLemmatization:\n",lemmatized_output)

stemmed_output = ' '.join([ps.stem(w) for w in word_tokens])
print("\nStemming:\n",stemmed_output)

n_pos = nltk.pos_tag(word_tokens)
print("\nParts of Speech :", n_pos)

noe = ne_chunk(n_pos)
print("\nNamed Entity Recognition :", noe)

lsstemmed_output = ' '.join([ls.stem(w) for w in word_tokens])
print("\nLSStemming:\n",lsstemmed_output)

wsstemmed_output = ' '.join([ws.stem(w) for w in word_tokens])
print("\nSBStemming:\n",wsstemmed_output)