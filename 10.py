#pgm10
!pip install cohere pypdf ipywidgets

import cohere
from pypdf import PdfReader

# Initialize Cohere client (replace with your API key)
co = cohere.Client("RGFlw7ccpgRpgZZKOX6Z4vM2vaIpSAmuf7LIUWgI")

# Load and extract text from the Indian Penal Code PDF file
file_path = r"C:\Users\Priyam\OneDrive\Desktop\ipc.pdf"  # Make sure this file is in your working directory
reader = PdfReader(file_path)

text = ""
for page in reader.pages:
    text += page.extract_text()

def ask_ipc(question: str):
    prompt = f"""
You are a legal assistant specialized in the Indian Penal Code (IPC).
Use the following content to answer the user's question:
{text[:10000]}

User Question: {question}

Respond with a relevant IPC section (if any) and a clear explanation.
"""
    response = co.generate(
        model="command-r-plus",
        prompt=prompt,
        max_tokens=300,
        temperature=0.3,
    )
    answer = response.generations[0].text.strip()
    print("\nAnswer:\n" + answer)

# Get user question and respond
user_input = input("Ask your IPC question: ")
ask_ipc(user_input)
