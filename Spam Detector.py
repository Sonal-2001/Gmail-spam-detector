# Importing Required Libraries
import pandas as pd
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score


# Loading Data
dt = pd.read_csv("spam.csv", encoding='latin1')
dt["spam"] = dt["type"].map({"spam" : 1, "ham" : 0}).astype(int)


# Tokenisation
dt["text"] = dt["text"].apply(lambda x: x.split())


# Stemming
porter = SnowballStemmer("english", ignore_stopwords=False)
dt["text"] = dt["text"].apply(lambda x: [porter.stem(word) for word in x])


# Lemmitization
lemmatizer = WordNetLemmatizer()
dt["text"] = dt["text"].apply(lambda x : [lemmatizer.lemmatize(word, pos="a") for word in x])


# Stopword Removal
stop_words = stopwords.words('english')
dt["text"] = dt["text"].apply(lambda x : [word for word in x if not word in stop_words]).apply(' '.join)


# Transforming Text Data into TF-IDF Vectors
tfidf = TfidfVectorizer()
y = dt.spam.values
x = tfidf.fit_transform(dt["text"])
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state = 1, test_size = 0.2, shuffle = False)


# Classification using Linear SVC
model = LinearSVC()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
acc_model = accuracy_score(y_pred, y_test)*100
print("accuracy :", acc_model)