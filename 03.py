from nltk.tokenize import TreebankWordTokenizer

tokenizer = TreebankWordTokenizer()

text = "This is some text."

tokens = tokenizer.tokenize(text)

print(tokens)