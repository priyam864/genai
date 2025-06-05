#pgm 3
!pip install gensim nltk

from gensim.models import Word2Vec
import nltk
from nltk.tokenize import word_tokenize

# Download necessary tokenizer data
nltk.download('punkt')

# Define medical-related corpus
corpus = [
    "A patient with diabetes requires regular insulin injections.",
    "Medical professionals recommend exercise for heart health.",
    "Doctors use MRI scans to diagnose brain disorders.",
    "Antibiotics help fight bacterial infections but not viral infections.",
    "The surgeon performed a complex cardiac surgery successfully.",
    "Doctors and nurses work together to treat patients.",
    "A doctor specializes in diagnosing and treating diseases."
]

# Tokenize each sentence
tokenized_corpus = [word_tokenize(sentence.lower()) for sentence in corpus]

# Train the Word2Vec model
model = Word2Vec(
    sentences=tokenized_corpus,
    vector_size=100,
    window=3,
    min_count=1,
    workers=4,
    sg=1  # Skip-gram model
)

# Get top 5 similar words to "doctor"
similar_words = model.wv.most_similar("doctor", topn=5)

# Print similar words
print("Top 5 words similar to 'doctor':")
for word, score in similar_words:
    print(f"{word}: {score:.4f}")
