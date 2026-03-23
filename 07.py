import nltk
from nltk.wsd import lesk
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

sentence = "He went to the bank to deposit money"

tokens = word_tokenize(sentence)

word = "bank"

sense = lesk(tokens, word)

print("Sentence:", sentence)
print("Target Word:", word)
print("Predicted Sense:", sense)
print("Definition:", sense.definition())
print("Examples:", sense.examples())
