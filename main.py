from QR_Codes.qrcode import QRCodeHandler
from rag import * 
from config import *
from sound import TTS, Recorder
from rich.console import Console
import ollama
import os

console = Console()
embed_m = EmbeddingM()
text_simi = TextSimilarity("pg16.txt")
recorder = Recorder(sample_rate=16000, channels=1)
qr_handler = QRCodeHandler()  # Create an instance of QRCodeHandler

def initialize_embeddings():
    paragraphs = []
    embeddings = []

    # อ่านไฟล์ทั้งหมดในโฟลเดอร์ `upload/`
    files = [f for f in os.listdir("upload") if f.endswith(".txt")]
    for filename in files:
        print(f"Processing {filename}...")
        text_simi = TextSimilarity(filename)
        file_paragraphs = text_simi.parse_file()
        file_paragraphs = [
            "".join(file_paragraphs[i : i + 2]) for i in range(0, len(file_paragraphs), 2)
        ]
        file_embeddings = embed_m.get_embeddings(filename, EMBEDDING_MODEL, file_paragraphs)
        paragraphs.extend(file_paragraphs)
        embeddings.extend(file_embeddings)

    return paragraphs, embeddings

def handle_chat(paragraphs, embeddings):
    """
    Handle a single chat interaction with the user.
    """
    select = input("Enter 1 if select Record or enter 0 for type -> ")

    if select == '1':
        recorder.run()
        prompt = recorder.faster_whisper()
        print(prompt)
    else:
        prompt = input("What do you want to know? -> ")

    # qr_handler.handle_qr_code(prompt)
           
            
    prompt_embedding = ollama.embeddings(model=EMBEDDING_MODEL, prompt=prompt)["embedding"]
    most_similar_chunks = text_simi.find_most_similar(prompt_embedding, embeddings)[:5]

    response = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
                + "\n".join(paragraphs[item[1]] for item in most_similar_chunks),
            },
            {"role": "user", "content": prompt},
        ],
    )
    print("\n")
    # text_res = response['message']['content']
    qr_handler.find_qr_in_response(response['message']['content'])
    # TTS(TTS_MODEL, response['message']['content'])
    console.print(f"[cyan]{response['message']['content']}[/cyan]")

def main():
    
    paragraphs, embeddings = initialize_embeddings()  # Load once at the start
    while True:
        handle_chat(paragraphs, embeddings)
        continue_chat = input("Do you want to continue chatting? (yes/no): ").lower()
        if continue_chat != 'yes':
            break

if __name__ == "__main__":
    main()
