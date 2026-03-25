import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker_tab')
nltk.download('words')

text = "Barack Obama was born in Hawaii and served as the President of the United States."

tokens = word_tokenize(text)

pos_tags = pos_tag(tokens)

named_entities = ne_chunk(pos_tags)

print(named_entities)
