#pgm 4
!pip install gensim cohere

import os
import gensim.downloader as api
import cohere

# Initialize Cohere client (replace key with a secure method in production)
co = cohere.Client("3JE437qflNp5M7TbvoQXMmeDuRrRWgylGCq09x29")

# Load the GloVe model
model = api.load("glove-wiki-gigaword-50")

# Define input word
input_word = "technology"

# Get top 5 similar words
similar_words = model.most_similar(input_word, topn=5)

# Display similar words
print("Top 5 similar words are:\n")
for word, similarity in similar_words:
    print(f"{word}: {similarity:.2f}")

# Build enriched prompt
related_terms = [input_word] + [w[0] for w in similar_words]
enriched_prompt = (
    "Explain the impact of " + ", ".join(related_terms[:-1]) +
    f", and {related_terms[-1]} on society."
)

# Function to generate response from Cohere
def generate_response(prompt):
    response = co.generate(
        model="command",
        prompt=prompt,
        max_tokens=150
    )
    return response.generations[0].text.strip()

# Generate and display original and enriched responses
original_prompt = "Explain the impact of technology on society."
original_response = generate_response(original_prompt)
enriched_response = generate_response(enriched_prompt)

print("\nOriginal Prompt Response:")
print(original_response)

print("\nEnriched Prompt Response:")
print(enriched_response)
