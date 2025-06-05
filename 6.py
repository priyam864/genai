#pgm 6
!pip install transformers torch

# Load the Hugging Face sentiment analysis pipeline for easy implementation.
from transformers import pipeline

# Load the sentiment analysis pipeline
sentiment_pipeline = pipeline("sentiment-analysis")

# Example sentences
sentences = [
    "I love this product! It works perfectly.",
    "This is the worst experience I've ever had.",
    "The weather is nice today.",
    "I feel so frustrated with this service."
]

# Analyze sentiment for each sentence
results = sentiment_pipeline(sentences)

# Print the results
for sentence, result in zip(sentences, results):
    print(f"Sentence: {sentence}")
    print(f"Sentiment: {result['label']}, Confidence:{result['score']:.4f}")
    print()
