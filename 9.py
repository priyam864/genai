#pgm 9
!pip install wikipedia cohere pydantic
!pip install wikipedia cohere pydantic

from typing import List
from pydantic import BaseModel
import wikipedia
import cohere
import json

# Initialize Cohere client
co = cohere.Client("3JE437qflNp5M7TbvoQXMmeDuRrRWgylGCq09x29")

# Define the schema using Pydantic
class InstitutionDetails(BaseModel):
    institution_name: str
    founded_year: int
    founded_by: str
    no_of_employees: str
    branches: List[str]
    courses: List[str]

# Format for the expected JSON structure
details_format = {
    "institution_name": "str",
    "founded_year": "int",
    "founded_by": "str",
    "no_of_employees": "str",
    "branches": "List[str]",
    "courses": "List[str]",
}

# Function to fetch Wikipedia summary
def fetch_wikipedia_summary(name: str) -> str:
    return wikipedia.summary(name, sentences=4)

# Function to format the summary using Cohere into InstitutionDetails
def format_wiki_summary(summary):
    response = co.generate(
        prompt=f"""
        Summarize the following text: ||{summary}||
        in this json format {details_format}
        """
    )
    text = response.generations[0].text.strip()
    start = text.find("{")
    end = text.rfind("}") + 1
    json_str = text[start:end]
    data = json.loads(json_str)
    return InstitutionDetails(**data)

# Main function to get institution details
def get_institution_details() -> InstitutionDetails:
    name = input("Enter institution name: ")
    summary = fetch_wikipedia_summary(name)
    return format_wiki_summary(summary)

# Calling the function and displaying the result
institution = get_institution_details()
print("Institution Name: ", institution.institution_name)
print("Founded in: ", institution.founded_year)
print("Founded by: ", institution.founded_by)
print("Employee count: ", institution.no_of_employees)
print("Departments: ", institution.branches)
print("Courses Offered: ", institution.courses)
