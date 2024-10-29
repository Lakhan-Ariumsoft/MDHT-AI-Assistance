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
from fastapi.staticfiles import StaticFiles
import shutil
from datetime import datetime
# from pinecone import Pinecone, ServerlessSpec

client = OpenAI()


UPLOAD_DIRECTORY = "uploads"
if not os.path.exists(UPLOAD_DIRECTORY):
    os.makedirs(UPLOAD_DIRECTORY)


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

app.mount("/static", StaticFiles(directory="static"), name="static")

client.api_key = os.getenv("OPENAI_API_KEY")

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    if file.filename == "":
        raise HTTPException(status_code=400, detail="No selected file")
    
    if allowed_file(file.filename):
        # Rename file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        original_filename = secure_filename(file.filename)
        file_extension = original_filename.rsplit('.', 1)[1]
        new_filename = f"{original_filename.rsplit('.', 1)[0]}_{timestamp}.{file_extension}"
        
        # Save file to the uploads folder
        file_location = os.path.join(UPLOAD_DIRECTORY, new_filename)
        with open(file_location, "wb") as buffer:
            buffer.write(await file.read())

        # Upload renamed file to OpenAI
        with open(file_location, "rb") as f:
            uploaded_file = client.files.create(file=f, purpose="assistants")

        # Retrieve current files and update the assistant's file list
        assistant_files = client.beta.assistants.files.list(assistant_id=assistant_id)
        # file_ids = [file["id"] for file in assistant_files["data"]]
        file_ids = [file.id for file in assistant_files.data]
        file_ids.append(uploaded_file.id)
        client.beta.assistants.update(
            assistant_id,
            file_ids=file_ids,
        )

        print(":::::::::::",client.beta)

        # client.beta.v
        # Optionally, create a vector store for the uploaded file if required by your assistant
        # vector_store = await client.beta.vectorStores.create({"name": "MyVectorStore"})

        # Upload the file to the vector store
        # await client.beta.vectorStores.fileBatches.upload_and_poll(vector_store.id, {"files": [uploaded_file.id]})

        # client.vectorStores.create(file=uploaded_file["id"])

        return {"success": True, "message": "File uploaded successfully", "filename": new_filename}
    
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
    print("::::::::::::",request)
    # if os.path.exists(file_path):
    # os.remove(file_path)
    # return {"success": True, "message": f"File {filename} deleted successfully"}
    
    if deleted_assistant_file.deleted:
        return {"success": True, "message": "File deleted!"}
    else:
        raise HTTPException(status_code=400, detail="File failed to be deleted")
    

@app.get("/uploadedfiles/")
async def get_uploaded_files():
    files = os.listdir(UPLOAD_DIRECTORY)
    return files

    #dont remove need for reference 
# @app.get("/get_files")
# async def get_files():
#     files = os.listdir(UPLOAD_DIRECTORY)
#     # return files
#     print(files)
#     global assistant_id
#     assistant_files = client.beta.assistants.files.list(assistant_id=assistant_id)
#     files_list = [{"id": file.id, "object": file.object, "created_at": file.created_at ,"File::":file} for file in assistant_files.data]
#     print(files_list)
#     return {"assistant_files": files_list}


@app.get("/get_files")
async def get_files():
    # Get files from the local uploads directory
    local_files = os.listdir(UPLOAD_DIRECTORY)

    global assistant_id
    # Fetch files uploaded to OpenAI's assistant
    assistant_files = client.beta.assistants.files.list(assistant_id=assistant_id)

    # Create a list combining local file names and OpenAI file details
    files_list = [
        {
            "id": file.id,
            "object": file.object,
            "created_at": file.created_at,
            "filename": file.filename if hasattr(file, 'filename') else None  # Check if filename is available
        }
        for file in assistant_files.data
    ]
    print("::::::::files_listttL::::::", files_list)
    # Combine the filenames from local storage
    for local_file in local_files:
        files_list.append({"filename": local_file})

    print(files_list)  # For debugging purposes
    return {"assistant_files": files_list}


def create_assistant():
    global assistant_id
    if assistant_id == "":
        my_assistant = client.beta.assistants.create(
            instructions="You are a helpful assistant. give the answers only from the uploaded data limit the response into 100 words and give everything in bullet points.",
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
    chat_history = [{"role": "system", "content": "You are a helpful assistant.limit the response into 100 words and give everything in bullet points"}]
    global thread_id
    thread_id = ""
    create_thread()
    return {"success": True}

if __name__ == '__main__':
    uvicorn.run("main:app", host='127.0.0.1', port=8000,
                log_level="info", reload=True)





# Reference and working code below dont remove

# # Set up OpenAI API key
# client.api_key = os.getenv("OPENAI_API_KEY")
# # Initialize Pinecone with API key and environment details
# pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
# index_name = "quickstart"  # Replace with your actual Pinecone index name

# # Check if the index exists; create if not
# if index_name not in pc.list_indexes().names():
#     pc.create_index(
#         name=index_name,
#         dimension=1536,  # Dimension for `text-embedding-ada-002`
#         metric='cosine',
#         spec=ServerlessSpec(
#             cloud='aws',  # Change to 'gcp' if using Google Cloud
#             region='us-east-1'  # Replace with a region that your plan supports
#         )
#     )

# index = pc.Index(index_name)
# print("Index initialized:", index)

# # Function to generate embeddings using OpenAI
# def generate_embeddings(text: str):
#     response = client.Embedding.create(input=text, model="text-embedding-ada-002")
#     return response['data'][0]['embedding']

# # Endpoint to upload files and process them
# @app.post("/upload")
# async def upload_file(file: UploadFile = File(...)):    
#     try:
#         # Read and decode file content
#         contents = await file.read()
#         decoded_text = contents.decode("utf-8")  # assuming the file is in UTF-8 encoding
        
#         # Generate embeddings
#         embeddings = generate_embeddings(decoded_text)
        
#         # Create a unique ID for the file
#         file_id = f"{file.filename}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
        
#         # Save embedding to Pinecone
#         index.upsert(vectors=[{"id": file_id, "values": embeddings}])
        
#         # Optional: Save file locally
#         save_path = f"uploads/{file_id}.txt"
#         os.makedirs("uploads", exist_ok=True)  # Create uploads directory if not exists
#         with open(save_path, "w") as f:
#             f.write(decoded_text)
        
#         return {"status": "success", "file_id": file_id, "message": "File uploaded and vector created."}

#     except Exception as e:
#         return {"status": "error", "message": str(e)}



# @app.post("/upload")    
# async def upload_file(file: UploadFile = File(...)):
#     global assistant_id
    
#     if file.filename == "":
#         raise HTTPException(status_code=400, detail="No selected file")
    
#     # file_names = []
#     # for file in files:
#     #     file_location = os.path.join(UPLOAD_DIRECTORY, file.filename)
#     #     with open(file_location, "wb+") as file_object:
#     #         shutil.copyfileobj(file.file, file_object)
#     #     file_names.append(file.filename)
#     # print(file_names)
    
#     if allowed_file(file.filename):

#          # Rename file with timestamp
#         timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
#         original_filename = secure_filename(file.filename)
#         file_extension = original_filename.rsplit('.', 1)[1]
#         new_filename = f"{original_filename.rsplit('.', 1)[0]}_{timestamp}.{file_extension}"
        
#         # Save file to the uploads folder
#         file_location = os.path.join(UPLOAD_DIRECTORY, new_filename)
#         with open(file_location, "wb") as buffer:
#             buffer.write(await file.read())

#         filename = secure_filename(file.filename)
#         # Upload the file and add it to the Assistant
#         uploaded_file = client.files.create(file=file.file, purpose="assistants")
#         assistant_files = client.beta.assistants.files.list(assistant_id=assistant_id)
#         file_ids = [file.id for file in assistant_files.data]
#         file_ids.append(uploaded_file.id)
#         client.beta.assistants.update(
#             assistant_id,
#             file_ids=file_ids,
#         )

        # client.VectorStore.create(file=uploaded_file["id"])

#         return {"success": True, "message": "File uploaded successfully", "filename": filename}
#     raise HTTPException(status_code=400, detail="File type not allowed")

