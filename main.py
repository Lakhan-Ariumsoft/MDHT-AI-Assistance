from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import uvicorn
import time
from openai import OpenAI
import requests
from dotenv import load_dotenv
load_dotenv()
client = OpenAI()
app = FastAPI()


thread_cache = {}
client.api_key = os.getenv("OPENAI_API_KEY")


# Login URL and external data URL
login_url = 'https://www.mdhealthtrak.com/api/v2/userLogin'


# Function to get login token and ID
def getLoginToken():
    payload = {
        "user": "vishalphirkoj13@gmail.com",
        "ccode": "+91_IN",
        "logintype": "email",
        "password": "123456789@V"
    }
    response = requests.post(login_url, json=payload)
    if response.status_code == 200:
        response_json = response.json()
        token = response_json['data']['token']
        login_id = response_json['data']['login_id']
        return token, login_id
    else:
        raise HTTPException(status_code=500, detail="Failed to log in")
    

def extractData(apiResponse):
    # Access the data list
    diseases_data = apiResponse.get("data", [])
    # Dictionary to store the extracted data
    extracted_data = {}

    # Iterate over each disease in the data list
    for disease in diseases_data:
        # Get the ds_name for the current disease
        ds_name = disease.get("ds_name")
        # Get the list of titles from baseline_data's symptoms
        titles = [symptom.get("title") for symptom in disease.get("baseline_data", {}).get("symptoms", [])]

        # Add the titles to the extracted_data dictionary under the ds_name key
        if ds_name:
            extracted_data[ds_name] = titles
    print(extracted_data)
    return extracted_data

# Function to generate prompts for each disease
def generatePrompts(extracted_data):
    prompts = []
    for ds_name, symptoms in extracted_data.items():
        # Join symptoms with commas, with the last one separated by 'and' for readability
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
def get_data_and_convert_to_prompt(token , apiUrl):
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


# Function to interact with assistant and get a response for each prompt
def get_assistant_response(prompts, assistant_id, vector_store_id, max_retries=10, retry_delay=2):
    responses = []
    
    # Iterate through each prompt and get a response
    for prompt in prompts:
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
                instructions="Give me a response in the following format:\n"
                            "- Summary of my health condition in 30 words.\n"
                            "- Suggested medications (top 3).\n"
                            "- Risk assessment: High Risk, Medium Risk, or Low Risk.\n"
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
                response_content = thread_messages.data[0].content[0].text.value
                responses.append(response_content)
                break  # Exit retry loop on success
            time.sleep(retry_delay)
            retries += 1
        else:
            responses.append("The assistant did not respond in time for this prompt. Please try again.")
    
    return responses


class RequestPayload(BaseModel):
    patientID: str
    token: str
    # mdhtApiUrl: str
    loginID: str
    vectorStoreID: list[str]
    AssistantID: str


@app.post("/aiResponse/")
async def fetch_and_respond(payload: RequestPayload):
    try:
        # Access data from the payload
        token = payload.token
        login_id = payload.loginID
        patientID = payload.patientID
        # mdhtApiUrl = payload.mdhtApiUrl
        vectorStoreID = payload.vectorStoreID
        AssistantID = payload.AssistantID

        apiUrl = f"https://www.mdhealthtrak.com/api/v2/get-patient-ds?patientId={patientID}&recordType=0"

        # Step 2: Fetch data and convert it to a prompt
        prompt = get_data_and_convert_to_prompt(token, apiUrl)

        assistant_response = get_assistant_response(prompt ,AssistantID ,  vectorStoreID)
        print("Assistant:", assistant_response)

        
        return {"assistant_response": assistant_response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def health_check():
    return {"server":"running"}






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
