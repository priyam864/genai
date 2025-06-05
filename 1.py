#pgm 1
!pip install gensim

import gensim.downloader as api

# Load the pre-trained GloVe model
model = api.load("glove-wiki-gigaword-50")

# Define words for vector arithmetic
word1 = "king"
word2 = "man"
word3 = "woman"

# Perform vector arithmetic
result_vector = model[word1] - model[word2] + model[word3]

# Find the top 5 most similar words to the result vector
predicted_word = model.most_similar([result_vector], topn=5)

# Print the results
print("Top 5 similar words:\n")
for word, similarity in predicted_word:
    print(f"{word}: {similarity:.2f}")

# Display the second most similar word as the result
print(f"\nResult of '{word1} - {word2} + {word3}' is: {predicted_word[1][0]}")
