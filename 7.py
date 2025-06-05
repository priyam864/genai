#pgm 7
!pip install transformers torch

from transformers import pipeline

# Initialize summarization pipeline with T5-small
summarizer = pipeline("summarization", model="t5-small", tokenizer="t5-small")

# Input passage to summarize
passage = """
Machine learning is a subset of artificial intelligence that focuses on 
training algorithms to make predictions or decisions without being 
explicitly programmed. It is widely used in industries such as healthcare,
where it helps in early diagnosis, in finance for fraud detection, and in
retail for customer behavior prediction. As data continues to grow, machine
learning models are becoming increasingly important for extracting 
meaningful patterns and making data-driven decisions.
"""

# Generate summary
summary = summarizer(passage, max_length=30, min_length=10, do_sample=False)

# Print summary
print("Summary:")
print(summary[0]['summary_text'])
