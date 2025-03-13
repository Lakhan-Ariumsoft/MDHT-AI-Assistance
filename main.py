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
from datetime import datetime, timedelta ,timezone

load_dotenv()
client = OpenAI()
app = FastAPI()


thread_cache = {}
client.api_key = os.getenv("OPENAI_API_KEY")



def extractData(apiResponse):
    try:
        # Retrieve resident details
        resident = apiResponse.get("resident", {})
        name = resident.get("name", "Unknown")
        age = resident.get("age", "Unknown")
        gender = resident.get("gender", "Unknown")

        diseases_data = apiResponse.get("diseases", [])
        extracted_data = []
        common_symptoms = {}
        current_date = datetime.now(timezone.utc)
        cutoff_date = current_date - timedelta(days=15)  # Past 15 days

        # Iterate over each disease in the diseases list
        for disease in diseases_data:
            ds_name = disease.get("ds_name")
            records = disease.get("records", [])

            disease_details = []

            for record in records:
                record_date_str = record.get("updatedAt")
                if not record_date_str:
                    continue

                # Parse the ISO 8601 date format with UTC timezone
                try:
                    record_date = datetime.fromisoformat(record_date_str.replace("Z", "+00:00"))
                except ValueError:
                    continue

                # Compare against the cutoff date
                if record_date >= cutoff_date:
                    symptoms_list = record.get("symptoms", [])
                    log_time = record_date.strftime("%I:%M %p")
                    log_date = record_date.strftime("%d %B %Y")

                    symptoms = {
                        symptom.get("title"): round(symptom.get("value"), 2)
                        for symptom in symptoms_list
                        if symptom.get('value', 0) > 0
                    }

                    # Store the detailed log for the disease
                    if symptoms:
                        disease_details.append({
                            "date": log_date,
                            "time": log_time,
                            "symptoms": symptoms
                        })

                        # Add to common symptoms
                        for title, value in symptoms.items():
                            if title not in common_symptoms:
                                common_symptoms[title] = []
                            common_symptoms[title].append((log_date, log_time, value))

            if disease_details:
                extracted_data.append((ds_name, disease_details))

        # If no data found within the past 15 days
        if not extracted_data:
            return (
                f"Personal Information: Name: {name}, Age: {age}, Gender: {gender}.\n"
                f"There is no disease or symptom added recently within the past 15 days."
            )

        # Construct the prompt for available data
        prompt = f"Personal Information: Name: {name}, Age: {age}, Gender: {gender}.\nMedical History and Symptoms:"

        for idx, (ds_name, details) in enumerate(extracted_data, 1):
            prompt += f"\n{idx}. {ds_name}, Date of Diagnosis: {details[0]['date']} at {details[0]['time']} with multiple symptom logs."
            for log in details:
                log_date = log["date"]
                log_time = log["time"]
                symptoms = ", ".join([f"{key}: {value}/10" for key, value in log["symptoms"].items()])
                prompt += f"\nSymptom Log at {log_date}, {log_time}: {symptoms}."

        # Add common symptoms over time
        if common_symptoms:
            prompt += "\n\nCommon Symptoms Logged Over Time:"
            for title, occurrences in common_symptoms.items():
                for log_date, log_time, value in occurrences:
                    prompt += f"\n{log_date}, {log_time}: {title}: {value}/10."

        # Add the request
        prompt += "\n\nRequest: Provide guidance or recommendations for medication based on the above symptoms and conditions."

        return prompt
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
        

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
                instructions='''Response Format Restriction: Always provide insights in the exact JSON format as shown below, and do not include any additional information or explanation.\n
                {
                    "Summary": "overview of current condition in 200 words strictly without repeating the same and also do not repeat the scores",
                    "AI-Recommended Next Steps": Provide a minimum of 3 and a maximum of 10 recommendations, formatted as a bullet-point list.,
                }
                Data Analysis Rules:
                Summary:strictly avoid the scores/values into response, Generate an overview of the patient's condition in no more than 200 words by analyzing the progression and recorded symptoms and disease diagnosis  give a generic response over this The summary should be written in a way that is clear and easy for the patient to understand.
                Strict Adherence: Do not deviate from the JSON format. Exclude extra commentary, footnotes, or references.
                "AI-Recommended Next Steps": Search on web where what is recommended by web for the given disease and what should be avoided and strictly Minimum 3 recommendations maximum upto 10 recommendations all in bullets points in a list,

                Input Expectation: Assume the input will include a dataset with the following structure:
                Diseases: Name and diagnosis date.
                Symptoms: Name, severity, and time of recording.
                Medications (optional): If included, review previous medications to avoid redundancy.
                Example Response:{
                    "Summary": "The patient exhibits a progressive decline in symptom severity, indicating moderate deterioration over the past six months.",
                    "AI-Recommended Next Steps": [
                        "Follow a structured treatment plan tailored to disease progression.",
                        "Ensure regular follow-ups with specialists to monitor condition changes.",
                        "Implement lifestyle modifications to improve overall health."
                    ]
                }

                Failure Scenario: If the input data is insufficient or unclear, respond with:{
                    "Summary": "Insufficient data to provide an accurate overview.",
                    "AI-Recommended Next Steps:" : "Insufficient data to provide an accurate overview."
                    
                }''',
                assistant_id=assistant_id,
                thread={
                    "messages": [{"role": "user", "content": prompt}],
                    # "tool_resources": {"file_search": {"vector_store_ids": vector_store_id}},
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


# Update the model to directly reflect the JSON structure
class DiseaseRecord(BaseModel):
    recordName: str
    _id: str
    updatedAt: str
    symptoms: list[Dict[str, Any]]
    status: str

class Disease(BaseModel):
    disease_id: str
    ds_name: str
    updatedAt: str
    records: list[DiseaseRecord]
    highValueSymptoms: list[Any]

class Resident(BaseModel):
    name: str
    gender: str
    age: int

class RequestPayload(BaseModel):
    message: str
    resident: Resident
    diseases: list[Disease]



class AIPayload(BaseModel):
    prompt: str
    vectorStoreID: list[str]
    AssistantID: str

class ConvertJson(BaseModel):
    ai_insights : str


@app.post("/getAIinsights/")
async def fetch_and_respond(payload: AIPayload):
    try:
        prompt = payload.prompt
        vectorStoreID = payload.vectorStoreID
        AssistantID = payload.AssistantID


        AI_insights = getAssistantResponse(prompt ,AssistantID ,  vectorStoreID)
        
        print("Assistant:", AI_insights)

        return  {"ai_insights":AI_insights}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
import re
from fastapi import HTTPException

@app.post("/convertToJson/")
async def convert_to_json(payload: ConvertJson):
    try:
        # Parse the JSON string inside 'ai_insights'
        try:
            insights_data = json.loads(payload.ai_insights)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON format in 'ai_insights'")

        # Extract summary and recommendations
        response_json = {
            "summary": insights_data.get("Summary", "").strip(),
            "AI-Recommended Next Steps": insights_data.get("AI-Recommended Next Steps", [])
        }

        return response_json

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
    


@app.post("/getPrompts/")
async def getPromptsdata(payload: RequestPayload):
    try:

        prompts = extractData(payload.dict())

        if not prompts:
            return {"error": "No data extracted from jsonResponse"}

        print("Prompt",prompts) 
        return {"prompt": prompts}
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
