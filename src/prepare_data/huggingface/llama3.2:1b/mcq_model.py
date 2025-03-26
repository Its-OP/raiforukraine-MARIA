from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from config import TOGETHER_API_KEY, DEFAULT_PROMPT
from openai import OpenAI
import os

class MCQQuestion(BaseModel):
    question: str = Field(description="The multiple-choice question")
    option_a: str = Field(description="The first answer option labeled 'A'")
    option_b: str = Field(description="The second answer option labeled 'B'")
    option_c: str = Field(description="The third answer option labeled 'C'")
    option_d: str = Field(description="The fourth answer option labeled 'D'")
    correct_option: str = Field(description="This consists only a letter of the correct option")

mcq_parser = JsonOutputParser(pydantic_object=MCQQuestion)

def create_prompt_chain(prompt_text=DEFAULT_PROMPT):
    prompt_template = PromptTemplate(
        template="{prompt}.\n{format_instructions}\n{query}\n",
        input_variables=["query"],
        partial_variables={
            "prompt": prompt_text,
            "format_instructions": mcq_parser.get_format_instructions(),
        },
    )
    
    # Using OpenAI client with OpenRouter
    model = ChatOpenAI(
        model="meta-llama/Llama-3.2-1B-Instruct",
        temperature=0.5,
        api_key=os.environ.get("HF_API_KEY"),
        base_url="https://router.huggingface.co/nebius/v1"
    )
    
    chain = prompt_template | model | mcq_parser
    return chain
