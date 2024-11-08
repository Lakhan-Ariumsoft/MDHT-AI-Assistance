from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import uvicorn
import time
from openai import OpenAI
import requests
from dotenv import load_dotenv
import json
import re
from typing import Dict, Any 

load_dotenv()
client = OpenAI()
app = FastAPI()


thread_cache = {}
client.api_key = os.getenv("OPENAI_API_KEY")


def extractData(apiResponse):

        diseases_data = apiResponse.get("diseases", [])
        extracted_data = {}

        # Iterate over each disease in the diseases list
        for disease in diseases_data:
            ds_name = disease.get("ds_name")
            symptoms_list = disease.get("symptoms", [])
            
            # Extract symptoms if present
            symptoms = [
                f"{symptom.get('title')} score {symptom.get('value')}"
                for symptom in symptoms_list
                if 'title' in symptom and 'value' in symptom
            ]
            
            # Add disease and symptoms to extracted data if disease name exists
            if ds_name:
                extracted_data[ds_name] = symptoms

        print("Extracted Data" ,extracted_data)
        return extracted_data

# Function to interact with assistant and get a response for each prompt
def getAssistantResponse(prompt, assistant_id, vector_store_id, max_retries=10, retry_delay=2):
    # responses = []
    # Iterate through each prompt and get a response
    thread_id = thread_cache.get(assistant_id)
    
    # Check if thread exists; if completed, reset it
    if thread_id:
        try:
            run_response = client.beta.threads.get(thread_id=thread_id)
            if run_response["status"] == "completed":
                thread_cache.pop(assistant_id, None)
                thread_id = None
        except Exception:
            thread_cache.pop(assistant_id, None)
            thread_id = None
    
    # Create a new thread if no valid one exists
    if not thread_id:
        response = client.beta.threads.create_and_run(
            instructions="Strictly Give me a response in the following JSON format:"
                        "- Summary: summary of my health condition in 30 words."
                        "- Suggested medications: (top 3)."
                        "- Risk Profile: High Risk, Medium Risk, or Low Risk."
                        "- Immediate consultation needed: Yes or No.",
            assistant_id=assistant_id,
            thread={
                "messages": [{"role": "user", "content": prompt}],
                "tool_resources": {"file_search": {"vector_store_ids": vector_store_id}},
            },
            extra_headers={"OpenAI-Beta": "assistants=v2"} 
        )
        thread_id = response.thread_id
        thread_cache[assistant_id] = thread_id
        run_id = response.id
    else:
        # Use the existing thread
        run_response = client.beta.threads.runs.create(
            thread_id=thread_id, messages=[{"role": "user", "content": prompt}]
        )
        run_id = run_response.id

    # Retry loop to check for completion
    retries = 0
    while retries < max_retries:
        run_status = client.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run_id)
        if run_status.status == "completed":
            # Retrieve the last message in the thread
            thread_messages = client.beta.threads.messages.list(thread_id=thread_id, limit=5, order="desc")
            responses  = thread_messages.data[0].content[0].text.value
            # responses.append(response_content)
            break  # Exit retry loop on success
        time.sleep(retry_delay)
        retries += 1
    else:
        responses = ("The assistant did not respond in time for this prompt. Please try again.")
    
    return responses


# Function to generate prompts for each disease
def generatePrompts(extracted_data):
    prompts = []
    for ds_name, symptoms in extracted_data.items():
        if not ds_name or not symptoms:
            continue

        symptoms_text = ', '.join(symptoms[:-1]) + f" and {symptoms[-1]}" if len(symptoms) > 1 else symptoms[0]
        
        # Generate the prompt
        prompt = (
            f"I am suffering from {ds_name} disease with these symptoms: {symptoms_text}. "
            "Please provide me with the proper medication and advice on this condition."
        )
        
        prompts.append(prompt)
    for prompt in prompts:
        print(prompt)
    return prompts

# Function to retrieve data and format as a prompt
def getDataToPrompt(token , apiUrl):
    headers = {
        'token': token,
        'Content-Type': 'application/json'
    }
    response = requests.get(apiUrl, headers=headers)
    if response.status_code == 200:
        response = response.json()
        
        data = extractData(response)
        prompts = generatePrompts(data)

        return prompts
    else:
        raise HTTPException(status_code=500, detail="Failed to retrieve data")


class RequestPayload(BaseModel):
    jsonResponse: Dict[str, Any]


class AIPayload(BaseModel):
    prompt: str
    vectorStoreID: list[str]
    AssistantID: str

class ConvertJson(BaseModel):
    AIinsights : str


@app.post("/getAIinsights/")
async def fetch_and_respond(payload: AIPayload):
    try:
        prompt = payload.prompt
        vectorStoreID = payload.vectorStoreID
        AssistantID = payload.AssistantID


        AI_insights = getAssistantResponse(prompt ,AssistantID ,  vectorStoreID)
        
        print("Assistant:", AI_insights)

        return  {"AI Insights":AI_insights}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@app.post("/convertToJson/")
async def convert_to_json(payload :ConvertJson):
    # Check if the response is empty
    assistant_response = payload.AIinsights

    if not assistant_response.strip():
        print("Error: assistant_response is empty.")
        return {"error": "Empty response"}

    # Initialize an empty dictionary to hold the parsed data
    response_json = {
        "summary": None,
        "medications": [],
        "risk profile": None,
        "consultation_needed": None
    }

    # Define regex patterns to match each part of the response
    summary_pattern = r"(?i)summary[:\-\s\*]*(.*?)(?=\n|$)"
    medications_pattern = r"(?i)suggested\s+medications[:\-\s\*]*(.*?)(?=risk|immediate|$)"
    # risk_pattern = r"(?i)risk\s+profile[:\-\s\*]*(.*?)(?=\n|$)
    risk_pattern = r"(?i)(risk\s+profile|risk)[:\-\s\*]*(.*?)(?=\n|$)"
    consultation_pattern = r"(?i)immediate\s+consultation\s+needed[:\-\s\*]*(.*?)(?=\n|$)"

    # Extract summary
    summary_match = re.search(summary_pattern, assistant_response, re.DOTALL)
    if summary_match:
        response_json["summary"] = summary_match.group(1).strip()

    # Extract medications as list items, handling bullet points
    medications_match = re.search(medications_pattern, assistant_response, re.DOTALL)
    if medications_match:
        medications_text = medications_match.group(1).strip()
        # Split medications by line breaks or numbers if they are listed as bullet points or numbered
        medications = re.split(r'\n|\d+\.', medications_text)
        response_json["medications"] = [med.strip() for med in medications if med.strip()]

    # Extract risk
    risk_match = re.search(risk_pattern, assistant_response, re.DOTALL)
    if risk_match:
        response_json["risk profile"] = risk_match.group(1).strip()

    # Extract consultation needed
    consultation_match = re.search(consultation_pattern, assistant_response, re.DOTALL)
    if consultation_match:
        response_json["consultation_needed"] = consultation_match.group(1).strip()

    return response_json 


@app.post("/getPrompts/")
async def getPromptsdata(payload: RequestPayload):
    try:

        jsonResponse = payload.jsonResponse
        data = extractData(jsonResponse)

        if not data:
            return {"error": "No data extracted from jsonResponse"}

        prompts = generatePrompts(data)

        print("Prompt",prompts)
        return {"Prompts": prompts}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def healthCheck():
    try:
        return {"server":"running"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))





# code to test the API 

# import requests

# # Define the URL of the FastAPI endpoint
# url = 'http://localhost:8000/aiResponse/'

# # Create the payload as a dictionary
# payload = {
#     "patientID": "6438ebb7cd65404046541494",
#     "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJsb2dpbl9pZCI6IjY0MzhlYmI3Y2Q2NTQwNDA0NjU0MTQ5NCIsIm5hbWUiOiJWaXNoYWwgUGhpcmtvaiIsIm1vYmlsZSI6Iis5MTcyNzY1MTY1NTgiLCJlbWFpbCI6InZpc2hhbHBoaXJrb2oxM0BnbWFpbC5jb20iLCJ0eXBlIjoicGF0aWVudCIsImlwIjoiOjpmZmZmOjEyNy4wLjAuMSIsImlhdCI6MTczMDE4MTYzNCwiZXhwIjoxNzMwMjY4MDM0fQ.eNHG4OU4tnbPoqnbzjPLom6v4toswTxAFpbxQZ9Sma4",
#     # "mdhtApiUrl": "https://www.mdhealthtrak.com/api/v2/get-patient-ds?patientId=6438ebb7cd65404046541494&recordType=0",
#     "loginID": "login_id",
#     "vectorStoreID": ["vs_YlS1nkk93AP1She2LVcYF0BA"],
#     "AssistantID": "asst_a7Q3ShPG1POthBVVJm9v7Bjw"
# }

# # Send a POST request to the FastAPI endpoint
# try:
#     response = requests.post(url, json=payload)
    
#     # Check if the request was successful
#     if response.status_code == 200:
#         # Parse the JSON response
#         data = response.json()
#         print("Response from AI:", data)
#     else:
#         print("Failed to call API:", response.status_code, response.text)
# except Exception as e:
#     print("Error occurred:", str(e))
