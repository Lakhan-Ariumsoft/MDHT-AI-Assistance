from openai import OpenAI
from time import sleep

client = OpenAI()

thread_cache = {}


VECTOR_STORE = ["vs_WfjquKMnYL0kaqWjHrVQOwua"]
ASSISTANT = "asst_bD4qJtf5kwPv57JUkQOOr2ae"

def get_assistant_response(prompt, assistant_id=ASSISTANT, max_retries=10, retry_delay=2):

    thread_id = thread_cache.get(assistant_id)
    
    if thread_id:
    
        try:
            run_response = client.beta.threads.get(thread_id=thread_id)
            
        
            if run_response["status"] == "completed":
                thread_cache.pop(assistant_id, None)
                thread_id = None
        except Exception as e:
        
            print(f"Error retrieving thread: {e}")
            thread_cache.pop(assistant_id, None)
            thread_id = None


    if not thread_id:
        response = client.beta.threads.create_and_run(
            assistant_id=assistant_id,
            thread={
                "messages": [{"role": "user", "content": prompt}],
                "tool_resources": {
                    "file_search": {
                        "vector_store_ids": VECTOR_STORE
                        }
                        }
            },
            extra_headers= {"OpenAI-Beta": "assistants=v2"}
        )
        thread_id = response.thread_id
        thread_cache[assistant_id] = thread_id
        run_id = response.id
    else:
    
        run_response = client.beta.threads.runs.create(thread_id=thread_id, messages=[{"role": "user", "content": prompt}])
        run_id = run_response.id
    retries = 0
    while retries < max_retries:
        run_status = client.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run_id)
    
        if run_status.status == "completed":
            thread_messages = client.beta.threads.messages.list(thread_id=thread_id,limit=5,order='desc')
            return thread_messages.data[0].content[0].text.value
        sleep(retry_delay)
        retries += 1
    return "The assistant did not respond in time. Please try again."

prompt = "Let me about russian ukranie war in terms of global loss"
response = get_assistant_response(prompt)
print(response)