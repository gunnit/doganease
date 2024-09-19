import os
import requests
import tempfile
import json
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.schema import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain.chat_models.base import BaseChatModel
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.schema.output import ChatGeneration, ChatResult
from typing import Any, List, Mapping, Optional
import gradio as gr

class RegoloAI(BaseChatModel):
    regolo_api_key: str
    model: str = "llama3.1:70b-instruct-q8_0"

    @property
    def _llm_type(self) -> str:
        return "regolo-ai"

    def _call(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {self.regolo_api_key}"
        }

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": self._get_role(message),
                    "content": message.content
                }
                for message in messages
            ]
        }

        response = requests.post(
            "https://api.regolo.ai/v1/chat/completions",
            headers=headers,
            json=payload
        )
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        response = self._call(messages, stop, run_manager, **kwargs)
        return ChatResult(generations=[ChatGeneration(text=response, message=AIMessage(content=response))])

    def _get_role(self, message: BaseMessage) -> str:
        if isinstance(message, HumanMessage):
            return "user"
        elif isinstance(message, AIMessage):
            return "assistant"
        elif isinstance(message, SystemMessage):
            return "system"
        else:
            raise ValueError(f"Got unknown type {message}")

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"model": self.model}

def set_regolo_api_key(api_key):
    os.environ["REGOLO_API_KEY"] = api_key

# Function to download and read PDF content
def get_pdf_text(pdf_url):
    response = requests.get(pdf_url)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(response.content)
        temp_file_path = temp_file.name

    text = ""
    with open(temp_file_path, "rb") as file:
        pdf_reader = PdfReader(file)
        for page in pdf_reader.pages:
            text += page.extract_text()

    os.unlink(temp_file_path)
    return text

# List of PDF URLs
pdf_urls = [
    "https://www.sace.it/docs/default-source/e2e/guide-doganali/sace_guide_doganali_usa.pdf?sfvrsn=687004b9_1",
    "https://www.sace.it/docs/default-source/e2e/guide-doganali/sace_guide_doganali_russia.pdf?sfvrsn=667004b9_1",
    "https://www.sace.it/docs/default-source/e2e/guide-doganali/sace_guide_doganali_giappone.pdf?sfvrsn=b47304b9_1",
    "https://www.sace.it/docs/default-source/e2e/guide-doganali/sace_guide_doganali_vietnam.pdf?sfvrsn=ad7304b9_1",
    "https://www.sace.it/docs/default-source/e2e/guide-doganali/sace_guide_doganali_sudafrica_rev.pdf?sfvrsn=8a3c30b9_2",
]

def initialize_rag_system():
    # Extract text from all PDFs
    all_text = ""
    for url in pdf_urls:
        all_text += get_pdf_text(url)

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(all_text)

    # Create embeddings and vector store
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma.from_texts(chunks, embeddings)

    # Create a conversational chain
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    qa = ConversationalRetrievalChain.from_llm(
        RegoloAI(regolo_api_key=os.environ["REGOLO_API_KEY"]),
        vectorstore.as_retriever(),
        memory=memory
    )

    return qa

# Global variable to store the QA system
qa_system = None

def chat_with_pdf_data(message, history):
    global qa_system
    if qa_system is None:
        return "Errore: Sistema RAG non inizializzato. Inserisci prima la tua chiave API."
    
    result = qa_system.invoke({"question": message})
    return result['answer']

def init_system(api_key):
    global qa_system
    set_regolo_api_key(api_key)
    try:
        qa_system = initialize_rag_system()
        return "Sistema Doganease inizializzato con successo. Puoi iniziare a fare domande sulle procedure doganali per USA, Russia, Giappone, Vietnam e Sudafrica."
    except Exception as e:
        return f"Si è verificato un errore: {str(e)}"

# Updated Gradio interface for Doganease
import base64

def doganease_interface():
    # Read and encode the logo image
    with open("doganease.png", "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()

    with gr.Blocks(css="""
        footer {visibility: hidden}
        .logo-container {text-align: center; margin-bottom: 20px;}
        .logo {max-width: 200px; height: auto;}
        .main-header {font-size: 3em; color: #2c3e50; text-align: center; margin-bottom: 20px;}
        .sub-header {font-size: 1.8em; color: #34495e; margin-top: 15px; margin-bottom: 10px;}
        .country-flags {font-size: 1.4em; text-align: center; margin-bottom: 20px;}
        .custom-button {
            background: linear-gradient(135deg, #3498db, #2980b9);
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            font-size: 1.1em;
            transition: all 0.3s ease;
        }
        .custom-button:hover {
            background: linear-gradient(135deg, #2980b9, #3498db);
            transform: translateY(-2px);
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .api-input {border: 2px solid #3498db; border-radius: 5px; font-size: 1.1em;}
        .chat-area {border: 1px solid #bdc3c7; border-radius: 10px; padding: 15px;}
        .info-text {font-size: 1.2em; line-height: 1.6;}
    """) as demo:
        gr.HTML(
            f"""
            <div class="logo-container">
                <img src="data:image/png;base64,{encoded_string}" alt="Doganease Logo" class="logo">
            </div>
            """
        )
        
        gr.Markdown(
            """
            # 🛃 Doganease: Il tuo Assistente Doganale Intelligente
            """, elem_classes=["main-header"]
        )
        
        with gr.Tabs():
            with gr.TabItem("Inizializzazione"):
                gr.Markdown(
                    """
                    ### Inizializza Doganease
                    Inserisci la tua chiave API Regolo per iniziare ad utilizzare Doganease.
                    """, elem_classes=["sub-header"]
                )
                with gr.Row():
                    api_key_input = gr.Textbox(label="Chiave API Regolo", type="password", value="", elem_classes=["api-input"])
                    init_button = gr.Button("Inizializza", elem_classes=["custom-button"])
                init_output = gr.Textbox(label="Stato di Inizializzazione", elem_classes=["api-input"])

            with gr.TabItem("Chat con Doganease"):
                gr.Markdown(
                    """
                    ### Chat con Doganease
                    Fai le tue domande sulle procedure doganali per i seguenti paesi:
                    """, elem_classes=["sub-header"]
                )
                gr.Markdown(
                    """
                    🇺🇸 USA | 🇷🇺 Russia | 🇯🇵 Giappone | 🇻🇳 Vietnam | 🇿🇦 Sudafrica
                    """, elem_classes=["country-flags"]
                )
                with gr.Column(elem_classes=["chat-area"]):
                    chatbot = gr.Chatbot(label="Conversazione con Doganease")
                    msg = gr.Textbox(
                        label="La tua domanda",
                        placeholder="Es: Quali sono i documenti necessari per esportare in USA?",
                        elem_classes=["api-input"]
                    )
                    with gr.Row():
                        submit_btn = gr.Button("Invia", elem_classes=["custom-button"])
                        clear_btn = gr.Button("Pulisci Chat", elem_classes=["custom-button"])

            with gr.TabItem("Informazioni"):
                gr.Markdown(
                    """
                    ### Doganease: Il tuo partner per l'espansione internazionale
                    
                    Doganease è disponibile 24/7 per aiutarti a navigare le complessità delle normative doganali,
                    riducendo errori e ritardi nelle spedizioni. Con Doganease, la tua azienda può espandersi sui
                    mercati internazionali con sicurezza e efficienza.
                    
                    #### Caratteristiche principali:
                    - 📚 Accesso rapido alle informazioni doganali
                    - 🌐 Supporto per operazioni di import/export
                    - ⏱️ Risposte in tempo reale alle tue domande
                    - 💼 Riduzione di costi e tempi nelle pratiche doganali
                    
                    #### Paesi attualmente supportati:
                    - 🇺🇸 Stati Uniti d'America (USA)
                    - 🇷🇺 Russia
                    - 🇯🇵 Giappone
                    - 🇻🇳 Vietnam
                    - 🇿🇦 Sudafrica
                    
                    Stiamo lavorando per espandere continuamente la nostra copertura a nuovi paesi e mercati.
                    Ogni risposta include le fonti delle informazioni per garantire affidabilità e tracciabilità.
                    """, elem_classes=["info-text"]
                )

        def user(user_message, history):
            return "", history + [[user_message, None]]

        def bot(history):
            user_message = history[-1][0]
            bot_message = chat_with_pdf_data(user_message, history)
            history[-1][1] = bot_message
            return history

        msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
            bot, chatbot, chatbot
        )
        submit_btn.click(user, [msg, chatbot], [msg, chatbot], queue=False).then(
            bot, chatbot, chatbot
        )
        clear_btn.click(lambda: None, None, chatbot, queue=False)
        init_button.click(init_system, inputs=[api_key_input], outputs=[init_output])

    return demo

if __name__ == "__main__":
    demo = doganease_interface()
    demo.launch(share=True)