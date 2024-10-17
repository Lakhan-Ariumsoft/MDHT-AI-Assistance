from fastapi import FastAPI, UploadFile, File, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from werkzeug.utils import secure_filename
import os
import uvicorn
import time
from openai import OpenAI
from contextlib import asynccontextmanager
client = OpenAI()

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Processing startup function")
    create_assistant()
    create_thread()
    # Yield allows the app to start receiving requests
    yield
    # Shutdown: Clean up resources or save state here
    print("Devices cleared")

# Initialize FastAPI app with lifespan
app = FastAPI(lifespan=lifespan)
ALLOWED_EXTENSIONS = {"txt", "pdf", "png", "jpg", "jpeg", "gif", "csv"}
assistant_id = ""
thread_id = ""
chat_history = [{"role": "system", "content": "You are a helpful assistant."}]
templates = Jinja2Templates(directory="templates")


class DeleteFileRequest(BaseModel):
    fileId: str


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
from fastapi.staticfiles import StaticFiles

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    global assistant_id
    
    if file.filename == "":
        raise HTTPException(status_code=400, detail="No selected file")
    
    if allowed_file(file.filename):
        filename = secure_filename(file.filename)

        # Upload the file and add it to the Assistant
        uploaded_file = client.files.create(file=file.file, purpose="assistants")
        assistant_files = client.beta.assistants.files.list(assistant_id=assistant_id)

        file_ids = [file.id for file in assistant_files.data]
        file_ids.append(uploaded_file.id)

        client.beta.assistants.update(
            assistant_id,
            file_ids=file_ids,
        )
        return {"success": True, "message": "File uploaded successfully", "filename": filename}
    raise HTTPException(status_code=400, detail="File type not allowed")


@app.get("/get_ids")
async def get_ids():
    return {"assistant_id": assistant_id, "thread_id": thread_id}


@app.get("/get_messages")
async def get_messages():
    global thread_id
    if thread_id != "":
        thread_messages = client.beta.threads.messages.list(thread_id, order="asc")
        messages = [{"role": msg.role, "content": msg.content[0].text.value} for msg in thread_messages.data]
        return {"success": True, "messages": messages}
    return {"success": False, "message": "No thread ID"}


@app.post("/delete_files")
async def delete_files(request: DeleteFileRequest):
    global assistant_id
    deleted_assistant_file = client.beta.assistants.files.delete(
        assistant_id=assistant_id, file_id=request.fileId
    )
    if deleted_assistant_file.deleted:
        return {"success": True, "message": "File deleted!"}
    else:
        raise HTTPException(status_code=400, detail="File failed to be deleted")


@app.get("/get_files")
async def get_files():
    global assistant_id
    assistant_files = client.beta.assistants.files.list(assistant_id=assistant_id)

    files_list = [{"id": file.id, "object": file.object, "created_at": file.created_at} for file in assistant_files.data]
    return {"assistant_files": files_list}


def create_assistant():
    global assistant_id
    if assistant_id == "":
        my_assistant = client.beta.assistants.create(
            instructions="You are a helpful assistant. If asked about math or computing problems, write and run code to answer the question.",
            name="MyQuickstartAssistant",
            model="gpt-3.5-turbo",
            tools=[{"type": "code_interpreter"}],
        )
        assistant_id = my_assistant.id
    else:
        my_assistant = client.beta.assistants.retrieve(assistant_id)
        assistant_id = my_assistant.id
    return my_assistant


def create_thread():
    global thread_id
    if thread_id == "":
        thread = client.beta.threads.create()
        thread_id = thread.id
    else:
        thread = client.beta.threads.retrieve(thread_id)
        thread_id = thread.id
    return thread


@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index_old.html", {"request": request, "chat_history": chat_history})


@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    content = data["message"]
    chat_history.append({"role": "user", "content": content})

    # Send the message to the assistant
    message_params = {"thread_id": thread_id, "role": "user", "content": content}

    thread_message = client.beta.threads.messages.create(**message_params)

    # Run the assistant
    run = client.beta.threads.runs.create(thread_id=thread_id, assistant_id=assistant_id)

    while run.status != "completed":
        time.sleep(0.5)
        run = client.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run.id)

    response = client.beta.threads.messages.list(thread_id).data[0]

    text_content = None
    for content in response.content:
        if content.type == "text":
            text_content = content.text.value
            break

    if text_content:
        chat_history.append({"role": "assistant", "content": text_content})
        return {"success": True, "message": text_content}
    else:
        return {"success": False, "message": "No text content found"}


@app.post("/reset")
async def reset_chat():
    global chat_history
    chat_history = [{"role": "system", "content": "You are a helpful assistant."}]

    global thread_id
    thread_id = ""
    create_thread()
    return {"success": True}



if __name__ == '__main__':
    uvicorn.run("main:app", host='127.0.0.1', port=8000,
                log_level="info", reload=True)
    print("running")






















# import os
# from fastapi import FastAPI, UploadFile, File, Form, HTTPException
# from fastapi.responses import HTMLResponse
# from fastapi.templating import Jinja2Templates
# from fastapi import Request
# from dotenv import load_dotenv
# import openai
# import shutil

# load_dotenv()

# # Load OpenAI API Key from .env
# openai.api_key = os.getenv("OPENAI_API_KEY")

# app = FastAPI()

# # Set up templates
# templates = Jinja2Templates(directory="templates")

# # Directory to store uploaded files
# UPLOAD_DIRECTORY = "uploaded_files"
# os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)

# # Store chat history in-memory
# chat_history = []

# @app.get("/", response_class=HTMLResponse)
# async def read_root(request: Request):
#     return templates.TemplateResponse("index.html", {"request": request})

# @app.post("/uploadfiles/")
# async def upload_files(files: list[UploadFile] = File(...)):
#     for file in files:
#         file_location = f"{UPLOAD_DIRECTORY}/{file.filename}"
#         with open(file_location, "wb") as f:
#             shutil.copyfileobj(file.file, f)
#     return {"info": f"Uploaded {len(files)} files"}

# @app.delete("/deletefiles/{filename}")
# async def delete_file(filename: str):
#     file_path = os.path.join(UPLOAD_DIRECTORY, filename)
#     if os.path.exists(file_path):
#         os.remove(file_path)
#         return {"info": f"File '{filename}' deleted"}
#     raise HTTPException(status_code=404, detail="File not found")

# @app.post("/chat/")
# async def chat(message: str = Form(...)):
#     messages = []
#     for filename in os.listdir(UPLOAD_DIRECTORY):
#         file_path = os.path.join(UPLOAD_DIRECTORY, filename)
#         try:
#             with open(file_path, "r", encoding="utf-8") as f:
#                 messages.append(f.read())
#         except UnicodeDecodeError:
#             # Fallback: try reading the file as binary and decoding to handle different encodings
#             with open(file_path, "rb") as f:
#                 content = f.read()
#                 try:
#                     messages.append(content.decode("utf-8"))
#                 except UnicodeDecodeError:
#                     messages.append("Could not decode file: " + filename)

#     messages.append(message)

#     response = openai.ChatCompletion.create(
#         model="gpt-3.5-turbo",
#         messages=[{"role": "user", "content": msg} for msg in messages],
#     )
    
#     # Store the message and response in chat history
#     chat_history.append({"user": message, "bot": response.choices[0].message['content']})
#     return {"response": response.choices[0].message['content'], "chat_history": chat_history}




# import os
# from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Request
# from fastapi.responses import HTMLResponse
# from fastapi.staticfiles import StaticFiles
# from fastapi.templating import Jinja2Templates
# from typing import List
# import shutil
# import openai
# from dotenv import load_dotenv

# # Load environment variables from .env file
# load_dotenv()

# # Set your OpenAI API key
# openai.api_key = os.getenv("OPENAI_API_KEY")

# app = FastAPI()

# # Setup templates and static files
# templates = Jinja2Templates(directory="templates")
# app.mount("/static", StaticFiles(directory="static"), name="static")

# # Set the directory for uploaded files
# UPLOAD_DIRECTORY = "uploads"
# os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)

# @app.get("/", response_class=HTMLResponse)
# async def read_root(request: Request):
#     return templates.TemplateResponse("index.html", {"request": request})

# @app.post("/uploadfiles/")
# async def upload_files(files: List[UploadFile] = File(...)):
#     file_names = []
#     for file in files:
#         file_location = os.path.join(UPLOAD_DIRECTORY, file.filename)
#         with open(file_location, "wb+") as file_object:
#             shutil.copyfileobj(file.file, file_object)
#         file_names.append(file.filename)
#     return {"info": f"Successfully uploaded {len(files)} file(s): {', '.join(file_names)}"}

# @app.delete("/deletefiles/{filename}")
# async def delete_file(filename: str):
#     file_location = os.path.join(UPLOAD_DIRECTORY, filename)
#     if os.path.isfile(file_location):
#         os.remove(file_location)
#         return {"info": f"File '{filename}' deleted successfully!"}
#     raise HTTPException(status_code=404, detail="File not found")

# @app.get("/uploadedfiles/")
# async def get_uploaded_files():
#     files = os.listdir(UPLOAD_DIRECTORY)
#     return files

# @app.post("/chat/")
# async def chat(message: str = Form(...)):
#     try:
#         # Load all the messages from the uploaded files
#         messages = []
#         for filename in os.listdir(UPLOAD_DIRECTORY):
#             file_path = os.path.join(UPLOAD_DIRECTORY, filename)

#             # Attempt to read the file as text
#             try:
#                 with open(file_path, "r", encoding="utf-8") as f:
#                     messages.append({"role": "user", "content": f.read()})
#             except UnicodeDecodeError:
#                 # Skip files that are not valid UTF-8 text files
#                 print(f"Warning: Skipped non-text file '{filename}'")

#         # Add the user message to the conversation
#         messages.append({"role": "user", "content": message})

#         # Create the OpenAI API call with formatted messages
#         response = openai.ChatCompletion.create(
#             model="gpt-3.5-turbo",
#             messages=messages,
#         )
        
#         # Extract the assistant's response
#         bot_message = response['choices'][0]['message']['content']
#         return {"response": bot_message}
    
#     except Exception as e:
#         # Log the error for debugging purposes
#         print(f"Error occurred: {str(e)}")
#         raise HTTPException(status_code=500, detail=str(e))

# # To run the app, use the command: uvicorn main:app --reload
