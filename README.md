# icare-關於「icare愛健康」這個頻道，我先用爬蟲系統把該頻道上的影片抓下來，然後再用whisper把影片轉成語音檔，接下來寫出一個問答系統，來回答有關影片中的問題，以下是問答系統的程式碼：
import gradio as gr
import chromadb
import ollama
import os

def read_text_files(folder_path):
    dialogues = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, "r", encoding="utf-8") as file:
                lines = file.readlines()
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                if ":" in line and len(line.split(":")) >= 2:
                    _, content = line.split(":", 1)
                    dialogues.append(content.strip())
                else:
                    dialogues.append(line)
    return dialogues

def setup_database(folder_path):
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection(name="dialogues")
    dialogues = read_text_files(folder_path)
    existing_data = collection.get()
    existing_ids = existing_data.get("ids", [])
    if existing_ids:
        collection.delete(ids=existing_ids)
    for idx, content in enumerate(dialogues):
        try:
            response = ollama.embeddings(model="mxbai-embed-large", prompt=content)
            collection.add(ids=[str(idx)], embeddings=[response["embedding"]], documents=[content])
        except Exception as e:
            print(f"建立向量時發生錯誤：{e}, 內容：{content}")
    return collection

def handle_user_input(user_input, collection, history):
    try:
        response = ollama.embeddings(prompt=user_input, model="mxbai-embed-large")
        results = collection.query(query_embeddings=[response["embedding"]], n_results=3)
        if results["documents"]:
            context = "\n".join([" ".join(doc) for doc in results["documents"]])
            prompt = f"你是一個根據提供的資訊回答問題的助理。請仔細閱讀以下資訊並回答使用者的問題。如果問題的答案不在提供的資訊中，請回答「檢索不到此問題，無法回答」。\n\n提供的資訊：\n{context}\n\n之前的對話：\n"
            for turn in history:
                prompt += f"{turn['role']}: {turn['content']}\n"
            prompt += f"使用者問題：{user_input}\n請用中文回答。"
            output = ollama.generate(model="ycchen/breeze-7b-instruct-v1_0", prompt=prompt)
            return output["response"]
        else:
            return "檢索不到此問題，無法回答"
    except Exception as e:
        return f"處理輸入時發生錯誤：{e}"

def main():
    folder_path = "C:/Users/Serena Li/Desktop/實驗室/team"
    collection = setup_database(folder_path)

    def predict(query, history):
        response = handle_user_input(query, collection, history)
        history.append({'role': 'user', 'content': query})
        history.append({'role': 'assistant', 'content': response})
        return history, history, ""

    with gr.Blocks() as demo:
        gr.Markdown("<h1 style='text-align: center;'>我的LLM+RAG本地知識問答</h1>")
        with gr.Column():
            chatbot = gr.Chatbot(label="聊天記錄", type='messages')
            input_box = gr.Textbox(label="請輸入問題...", placeholder="請輸入問題...")
        state = gr.State([])

        input_box.submit(predict, [input_box, state], [chatbot, state, input_box])

    demo.launch()

if __name__ == "__main__":
    main()

