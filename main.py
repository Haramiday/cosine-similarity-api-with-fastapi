from typing import Union
import numpy as np
from numpy.linalg import norm
from fastapi import FastAPI, Request
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Access environment variables
open_key = os.environ['API_KEY']

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


# @app.get("/items/{item_id}")
# def read_item(item_id: int, q: Union[str, None] = None):
#     return {"item_id": item_id, "q": q}


#Open ai
from openai import OpenAI
client = OpenAI(api_key = open_key) 

def get_embedding(text, model="text-embedding-3-large"):
   text = text.replace("\n", " ")
   return client.embeddings.create(input = [text], model=model).data[0].embedding

def generate_feedback(question, teacher, student, score, total):
  prompt = "Given the question: {}, my lecturer's answer was {} while the my answer was {}. I scored {} over {} with reference to my lecturer's question and answer. In one paragraph, tell me why my lecturer gave me that score".format(question, teacher, student, score, total)
  response = client.chat.completions.create(
  model="gpt-3.5-turbo-1106",
  temperature=0.2,  # Lower temperature for less randomness
  top_p=1,          # Full range of token probabilities
  max_tokens=150,   # Adjust based on your needs
  n=1,              # Number of completions to generate
  stop=None,
  messages=[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": prompt}
    ]
  )
  return response.choices[0].message.content

def cosine_similarity(teacher,student):
    A = get_embedding(teacher, model='text-embedding-3-large')
    B = get_embedding(student, model='text-embedding-3-large')
    cosine = np.dot(A,B)/(norm(A)*norm(B))
    return cosine

@app.post("/predict")
async def get_prediction(request: Request):
    message = await request.json()
    result = {}
    for id in message:
        score = cosine_similarity(message[id]["teacher"],message[id]["student"])
        aggregrate = round(score * message[id]["total"])
        feedback = generate_feedback(message[id]["question"], message[id]["teacher"],message[id]["student"], aggregrate, message[id]["total"])
        result[id] = { "feedback": feedback , "score":round(score,2)}
    return result #await request.json()
