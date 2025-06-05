#pgm2
!pip install gensim

import gensim.downloader as api
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np

# Load GloVe model
model = api.load("glove-wiki-gigaword-50")

# Define technology-related words
words = ["computer", "internet", "software", "hardware", "disk", "robot", "data", 
         "network", "cloud", "algorithm"]

# Get word vectors
word_vectors = np.array([model[word] for word in words])

# Apply PCA for dimensionality reduction
pca = PCA(n_components=2)
reduced_vectors = pca.fit_transform(word_vectors)

# Plot the words in 2D space
plt.figure(figsize=(8, 6))
for i, word in enumerate(words):
    plt.scatter(reduced_vectors[i, 0], reduced_vectors[i, 1])
    plt.annotate(word, (reduced_vectors[i, 0], reduced_vectors[i, 1]))

plt.title("PCA Visualization of Word Embeddings (Technology Domain)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.grid(True)
plt.show()

# Show top 5 similar words to a given word
input_word = "computer"
similar_words = model.most_similar(input_word, topn=5)
print("Top 5 similar words are:\n")
for word, similarity in similar_words:
    print(f"{word}: {similarity:.2f}")
