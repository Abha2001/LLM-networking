import ollama

def speak(transcript, message, model="llava-mistral", temperature=0.6):
    llama_messages = []
    for m in transcript:
        llama_messages.append({
            "role": m[0],
            "content": m[1]
        })
    
    message['role'] = 'user'
    llama_messages.append(message)

    print(f"[DEBUG] Calling ollama.chat with model={model}, {len(llama_messages)} messages", flush=True)

    response = ollama.chat(
        model=model,
        messages=llama_messages,
        options={
            "temperature": temperature
        }
    )

    return response['message']['content'].lower().strip()

    