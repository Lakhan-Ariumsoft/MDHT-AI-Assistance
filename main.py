from fastapi import FastAPI, UploadFile, File, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel 
from werkzeug.utils import secure_filename
import os

from dotenv import load_dotenv
load_dotenv()

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
    uvicorn.run("main:app", host='0.0.0.0', port=8000,
                log_level="info", reload=True)
    print("running")