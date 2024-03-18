#!pip install langsmith 
#!pip install langchain 
#!pip install python-dotenv 
from langchain_community.llms import CTransformers
from dotenv import load_dotenv
import os
from langsmith import Client

load_dotenv()

local_llm = "C:/Users/saahila/Desktop/zephyr-7b-beta.Q5_K_M.gguf"

config = {
    'max_new_tokens' : 1024,
    'repetition_penalty' : 1.1,
    'temperature' : 0.2,
    'top_k' : 50,
    'top_p' : 0.9,
    'stream' : True,
    'threads' : int(os.cpu_count()/2)
} 

llm_init = CTransformers(
    model=local_llm,
    model_type = "mistral",
)

query = 'Who is WWE Champion'
result = llm_init(query)

print(result)

client = Client()

## Create a Dataset
dataset_name = f"Dataset_01"

dataset = client.create_dataset(
    dataset_name,
    description="An example dataset documentation",
)

client.create_examples(
    inputs=[{"input": query}],
    outputs=[{"output": result}],
    dataset_id=dataset.id,
)