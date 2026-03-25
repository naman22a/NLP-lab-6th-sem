import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

text = "The striped bats are hanging on their feet for best"

words = word_tokenize(text)

lemmatizer = WordNetLemmatizer()

lemmatized_words = []
for word in words:
    lemmatized_word = lemmatizer.lemmatize(word)
    lemmatized_words.append(lemmatized_word)

print("Original Words:")
print(words)

print("\nLemmatized Words:")
print(lemmatized_words)
