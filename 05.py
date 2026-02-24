import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.stem import WordNetLemmatizer
import re

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

raw_text = "Last week I visited Delhi for a tech conference and honestly it was a mix of excitement and chaos. The event was held on 12th January 2026 at the Grand Convention Center near Connaught Place. I met several developers from Bangalore, Mumbai and even a team from Singapore. One of the keynote speakers, Dr. Arjun Mehta, spoke about Artificial Intelligence and how small startups are leveraging machine learning for healthcare innovation."

cleaned_data = re.sub(r'[^\w\s]', '', raw_text).lower()
tokens = word_tokenize(cleaned_data)
stop_words = set(stopwords.words('english'))
filtered_tokens = [w for w in tokens if w not in stop_words]
lemmatizer = WordNetLemmatizer()
lemmatized_tokens = [lemmatizer.lemmatize(w) for w in filtered_tokens]
fdist = FreqDist(lemmatized_tokens)
print("Original tokens:", tokens)
print("Filtered and Lemmatized tokens:", lemmatized_tokens)
print("Most common words:", fdist.most_common(3))

