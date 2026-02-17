import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')

text = "NLTK is a powerful library for natural language processing."
words = word_tokenize(text)
pos_tags = pos_tag(words)

print(pos_tags)

new_text = "This is some text."
tokens = word_tokenize(new_text)
print(nltk.pos_tag(tokens))