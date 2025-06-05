#pgm 8
!pip install langchain langchain-community cohere google-auth 
!pip google-auth-oauthlib google-auth-httplib2 googleapiclient
import os
from cohere import Client
from langchain.prompts import PromptTemplate

# Replace with your actual Cohere API key
co = Client("RGFlw7ccpgRpgZZKOX6Z4vM2vaIpSAmuf7LIUWgI")

# Simulated text (replace with your own file content if needed)
text_document = """
Machine learning is a subset of artificial intelligence that focuses on training
algorithms to make predictions. It is widely used in industries like healthcare,
finance, and retail.
"""

# Define prompt template
template = """
You are an expert summarizer. Summarize the following text in a concise manner:
Text: {text}
Summary:
"""

# Create prompt using LangChain
prompt_template = PromptTemplate(
    input_variables=["text"],
    template=template
)

# Format the prompt with text
formatted_prompt = prompt_template.format(text=text_document)

# Generate summary from Cohere
response = co.generate(
    model="command",
    prompt=formatted_prompt,
    max_tokens=50
)

# Output the result
print("Summary:")
print(response.generations[0].text.strip())
