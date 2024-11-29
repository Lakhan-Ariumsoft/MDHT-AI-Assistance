from fastapi import FastAPI, HTTPException, Depends , status
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
from dotenv import load_dotenv
from fastapi.security import OAuth2PasswordBearer
import jwt 
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional
import secrets
import datetime 

load_dotenv()

# Load environment variables
SECRET_KEY = secrets.token_urlsafe(32)  # Secure secret key for signing JWT
ALGORITHM = os.getenv("ALGORITHM", "HS256")  # Default algorithm
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 30))  # Default expiration time
USERNAME = os.getenv("USERNAME")
PASSWORD = os.getenv("PASSWORD")

# Initialize FastAPI app
app = FastAPI()

client = OpenAI()

thread_cache = {}
client.api_key = os.getenv("OPENAI_API_KEY")
# Security headers
security = HTTPBearer()


oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = verify_token(token)
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
        return username
    except Exception:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
    

def create_access_token(data: Dict):
    # Add expiration time
    to_encode = data.copy()
    expire = datetime.datetime.utcnow() + datetime.timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    
    # Create JWT token
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(token: str):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise Exception("Token has expired")
    except jwt.JWTError:
        raise Exception("Invalid token")


class LoginRequest(BaseModel):
    username: str
    password: str

@app.post("/token")
def login(request: LoginRequest):
    # Check if username and password match (you could add more validation)
    if request.username != os.getenv("USERNAME") or request.password != os.getenv("PASSWORD"):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
    
    # Create token and send back
    access_token = create_access_token(data={"sub": request.username})
    return {"access_token": access_token, "token_type": "bearer"}


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
                    "Summary": "overview of current condition in 50 words",
                    "Suggested Medications": ["medication1", "medication2", "medication3"],
                    "Risk Profile": "High Risk/Medium Risk/Low Risk",
                    "Immediate Consultation Needed": "Yes/No/May be"
                }
                Data Analysis Rules:
                Summary: Generate an overview of the patient's condition in no more than 50 words by analyzing the progression and timeline of recorded symptoms and disease diagnosis.
                Suggested Medications: Recommend up to 3–5 medications based on the analyzed disease and symptoms. Prioritize common and effective treatments for the condition.
                Risk Profile: Determine the patient's health risk as either "High Risk," "Medium Risk," or "Low Risk" based on the historical progression of symptoms and diagnosis date.
                Immediate Consultation: Indicate whether the patient needs an immediate consultation based on symptom severity and risk profile using "Yes," "No," or "May be."
                Strict Adherence: Do not deviate from the JSON format. Exclude extra commentary, footnotes, or references.
                Input Expectation: Assume the input will include a dataset with the following structure:
                Diseases: Name and diagnosis date.
                Symptoms: Name, severity, and time of recording.
                Medications (optional): If included, review previous medications to avoid redundancy.
                Example Response:{
                    "Summary": "The patient shows a progressive trend in symptom severity, indicating moderate deterioration over 6 months.",
                    "Suggested Medications": ["Medication A", "Medication B", "Medication C"],
                    "Risk Profile": "Medium Risk",
                    "Immediate Consultation Needed": "May be"
                }
                Failure Scenario: If the input data is insufficient or unclear, respond with:{
                    "Summary": "Insufficient data to provide an accurate overview.",
                    "Suggested Medications": [],
                    "Risk Profile": "Low Risk",
                    "Immediate Consultation Needed": "No"
                }''',
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
async def fetch_and_respond(payload: AIPayload , _: dict = Depends(get_current_user) ):
    try:
        prompt = payload.prompt
        vectorStoreID = payload.vectorStoreID
        AssistantID = payload.AssistantID


        AI_insights = getAssistantResponse(prompt ,AssistantID ,  vectorStoreID)
        
        print("Assistant:", AI_insights)

        return  {"ai_insights":AI_insights}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/convertToJson/")
async def convert_to_json(payload: ConvertJson ,  _: dict = Depends(get_current_user) ):
    try:
        # Check if the response is empty
        assistant_response = payload.ai_insights
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
async def getPromptsdata(payload: RequestPayload ,  _: dict = Depends(get_current_user)):
    try:
        print("User authenticated successfully")

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