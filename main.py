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

    try:
        diseases_data = apiResponse.get("diseases", [])
        extracted_data = {}
        # Iterate over each disease in the diseases list
        for disease in diseases_data:
            ds_name = disease.get("ds_name")
            symptoms_list = disease.get("symptoms", [])
            
            # Extract symptoms if present
            symptoms = [
                f"{symptom.get('title')} score {round(symptom.get('value'),2)}/10"
                for symptom in symptoms_list
                if 'title' in symptom and 'value' in symptom
            ]
            
            # Add disease and symptoms to extracted data if disease name exists
            if ds_name:
                extracted_data[ds_name] = symptoms
        print("Extracted Data" ,extracted_data)
        return extracted_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Function to interact with assistant and get a response for each prompt
def getAssistantResponse(prompt, assistant_id, vector_store_id, max_retries=10, retry_delay=2):
    try:
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
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Function to generate prompts for each disease
def generatePrompts(extracted_data):
    try:
        prompts = []
        for ds_name, symptoms in extracted_data.items():
            if not ds_name or not symptoms:
                continue

            symptoms_text = ', '.join(symptoms[:-1]) + f" and {symptoms[-1]}" if len(symptoms) > 1 else symptoms[0]
            
            # Generate the prompt
            prompt = (
                f'''I am suffering from {ds_name} disease with these symptoms: {symptoms_text}. Scored on a scale of 0 (lowest) to 10 (highest).Please provide me with the proper medication and advice on this condition.'''
            )
            
            prompts.append(prompt)
        for prompt in prompts:
            print(prompt)
        return prompts
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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

    try:
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

        def clean_text(text):
            return re.sub(r'[^A-Za-z0-9\s]', '', text).strip()

        # Define regex patterns to match each part of the response
        summary_pattern = r"(?i)summary[:\-\s\*#]*([^\*\#]*?)(?=\n|$)"
        medications_pattern = r"(?i)suggested\s+medications[:\-\s\*#]*([\s\S]*?)(?=###|risk|immediate|$)"
        risk_pattern = r"(?i)(risk\s+profile|risk)[:\-\s\*#]*([^\*\#]*?)(?=\n|$)"
        consultation_pattern = r"(?i)immediate\s+consultation\s+needed[:\-\s\*#]*([^\*\#]*?)(?=\n|$)"


        # Extract summary
        summary_match = re.search(summary_pattern, assistant_response, re.DOTALL)
        if summary_match:
            response_json["summary"] = clean_text(summary_match.group(1).strip())

        # Extract medications as list items, handling bullet points
        medications_match = re.search(medications_pattern, assistant_response, re.DOTALL)
        if medications_match:
            medications_text = medications_match.group(1).strip()
            medications = re.split(r'\s*\d+\.\s*|\n|,\s*', medications_text)
            response_json["medications"] = [clean_text(med) for med in medications if clean_text(med)]

        # Extract and validate risk profile
        risk_match = re.search(risk_pattern, assistant_response, re.DOTALL)
        if risk_match:
            risk_value = clean_text(risk_match.group(2)).lower()
            if "low" in risk_value:
                response_json["risk profile"] = "Low Risk"
            elif "moderate" in risk_value:
                response_json["risk profile"] = "Moderate Risk"
            elif "high" in risk_value:
                response_json["risk profile"] = "High Risk"
            elif "medium" in risk_value:
                response_json["risk profile"] = "Medium Risk"
            else:
                response_json["risk profile"] = "None"
                

        # Extract and validate consultation needed (Yes/No only)
        consultation_match = re.search(consultation_pattern, assistant_response, re.DOTALL)
        if consultation_match:
            consultation_value = clean_text(consultation_match.group(1)).lower()
            if "yes" in consultation_value:
                response_json["consultation_needed"] = "Yes"
            elif "no" in consultation_value:
                response_json["consultation_needed"] = "No"

        return response_json 
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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
