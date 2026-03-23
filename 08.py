import nltk
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import sent_tokenize, word_tokenize
from collections import defaultdict
import string

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

text = """
Natural language processing is a field of artificial intelligence that focuses on the interaction
between computers and human language. It enables machines to understand, interpret and generate
human language. Applications of natural language processing include machine translation,
chatbots, text summarization and sentiment analysis. Text summarization helps users understand
large documents quickly by extracting the most important information.
"""

sentences = sent_tokenize(text)

stop_words = set(stopwords.words("english"))

word_freq = defaultdict(int)

for sentence in sentences:
    words = word_tokenize(sentence.lower())
    for word in words:
        if word not in stop_words and word not in string.punctuation:
            word_freq[word] += 1

for word in list(word_freq.keys()):
    synsets = wordnet.synsets(word)
    for syn in synsets:
        for lemma in syn.lemmas():
            synonym = lemma.name().lower()
            if synonym in word_freq:
                word_freq[word] += 1

sentence_scores = defaultdict(int)

for sentence in sentences:
    words = word_tokenize(sentence.lower())
    for word in words:
        if word in word_freq:
            sentence_scores[sentence] += word_freq[word]

summary_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)[:2]

summary = " ".join(summary_sentences)

print("Original Text:\n", text)
print("\nSummary:\n", summary)
