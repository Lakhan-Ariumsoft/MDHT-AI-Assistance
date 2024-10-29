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


# Function to retrieve data and format as a prompt
def get_data_and_convert_to_prompt(token , apiUrl):
    headers = {
        'token': token,
        'Content-Type': 'application/json'
    }
    response = requests.get(apiUrl, headers=headers)
    if response.status_code == 200:
        result = response.json()
        # Create a prompt based on the retrieved data
        prompt = f"Based on this symptoms: {result}, please provide medical recommendations."
        return prompt
    else:
        raise HTTPException(status_code=500, detail="Failed to retrieve data")


# Function to interact with assistant and get a response
def get_assistant_response(prompt, assistant_id, vectorStoreID , max_retries=10, retry_delay=2):
    threadID = thread_cache.get(assistant_id)
    # Check if thread exists; if completed, reset it
    if threadID:
        try:
            run_response = client.beta.threads.get(thread_id=threadID)
            if run_response["status"] == "completed":
                thread_cache.pop(assistant_id, None)
                threadID = None
        except Exception as e:
            thread_cache.pop(assistant_id, None)
            threadID = None

    # If no valid thread, create a new one
    if not threadID:
        response = client.beta.threads.create_and_run(
            instructions="Give me a response in following format. Summary of my health condition in 30 words.Give the response in bullet points Suggested medications top 3. Looking at my situation, what do you think about my existing situation. Rate it among three: High Risk, Medium Risk and Low Risk. Next line, provide whether I need immediate consultation from doctor or no in just one word which is Yes or No",
            assistant_id=assistant_id,
            thread={
                "messages": [{"role": "user", "content": prompt}],
                "tool_resources": {"file_search": {"vector_store_ids": vectorStoreID}},
            },
            extra_headers={"OpenAI-Beta": "assistants=v2"}
        )
        threadID = response.thread_id
        thread_cache[assistant_id] = threadID
        run_id = response.id
    else:
        # Use existing thread if available
        run_response = client.beta.threads.runs.create(
            thread_id=threadID, messages=[{"role": "user", "content": prompt}]
        )
        run_id = run_response.id

    retries = 0
    while retries < max_retries:
        run_status = client.beta.threads.runs.retrieve(thread_id=threadID, run_id=run_id)
        if run_status.status == "completed":
            thread_messages = client.beta.threads.messages.list(thread_id=threadID, limit=5, order="desc")
            return thread_messages.data[0].content[0].text.value
        time.sleep(retry_delay)
        retries += 1

    return "The assistant did not respond in time. Please try again."

class RequestPayload(BaseModel):
    patientID: str
    token: str
    # mdhtApiUrl: str
    loginID: str
    vectorStoreID: list[str]
    AssistantID: str

@app.get("/")
async def health_check():
    return {"server":"running"}

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

        # Step 3: Use the prompt to get a response from OpenAI's assistant
        assistant_response = get_assistant_response(prompt ,AssistantID ,  vectorStoreID)
        print("Assistant RESPONSE:", assistant_response)
        
        return {"assistant_response": assistant_response}
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

