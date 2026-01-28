import nltk
nltk.download('wordnet')

from nltk.corpus import wordnet as wn

dog_synsets = wn.synsets('dog')
print(dog_synsets)

flower_synsets = wn.synsets('flower')
print(flower_synsets)
