import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
import nltk
import re
nltk.download('stopwords')
from nltk.corpus import stopwords

# Load CSV
df = pd.read_csv("data/training.1600000.processed.noemoticon.csv", 
                 encoding='latin-1', 
                 header=None)

# Keep only the target and text columns
df = df[[0, 5]]
df.columns = ['target', 'text']

# Map target 0 -> negative, 4 -> positive
df['target'] = df['target'].map({0: 0, 4: 1})  # 0 = negative, 1 = positive

print(df.head())

stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#','', text)
    text = re.sub(r'[^A-Za-z\s]', '', text)
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

df['clean_text'] = df['text'].apply(clean_text)

X = df['clean_text']
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = MultinomialNB()
model.fit(X_train_vec, y_train)

y_pred = model.predict(X_test_vec)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

def predict_sentiment(tweet):
    tweet_clean = clean_text(tweet)
    vec = vectorizer.transform([tweet_clean])
    pred = model.predict(vec)[0]
    return "Positive" if pred == 1 else "Negative"

print(predict_sentiment("I love this new phone!"))
print(predict_sentiment("This is the worst movie ever."))
