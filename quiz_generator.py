import os
import json
import re
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

def generate_quiz(text: str, num_questions: int = 5) -> list[dict]:
    llm = ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.3-70b-versatile",
        temperature=0.7
    )

    prompt = PromptTemplate(
        input_variables=["num_questions", "text"],
        template="""Create {num_questions} multiple choice questions from the study material below.
Return ONLY a valid JSON array. No explanation, no markdown, no extra text. Just the JSON array.

Format:
[
  {{
    "question": "What is ...?",
    "options": ["A) ...", "B) ...", "C) ...", "D) ..."],
    "answer": "A",
    "explanation": "Because ..."
  }}
]

Study Material:
{text}

JSON:"""
    )

    chain = prompt | llm | StrOutputParser()
    raw = chain.invoke({
        "num_questions": num_questions,
        "text": text[:3000]
    }).strip()

    raw = re.sub(r"```json|```", "", raw).strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r'\[.*\]', raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                return []
        return []