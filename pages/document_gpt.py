
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.document_loaders import UnstructuredFileLoader, UnstructuredPowerPointLoader
from langchain.embeddings import CacheBackedEmbeddings, OpenAIEmbeddings
from langchain.storage import LocalFileStore
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.callbacks.base import BaseCallbackHandler
from pydantic import ValidationError

import streamlit as st

st.set_page_config(
    page_title="Ask me anything",
    page_icon="📃",
)


class ChatCallbackHandler(BaseCallbackHandler):
    message = ""

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)

openai_api_key = ""




if "messages" not in st.session_state:
    st.session_state["messages"] = []


@st.cache_data(show_spinner="Embedding files...")
def embed_file(file):
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
    cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredPowerPointLoader(file_path=file_path) if file.name.endswith(".pptx") else  UnstructuredFileLoader(file_path=file_path) 
    docs = loader.load_and_split(text_splitter=splitter)
    try:
        if openai_api_key == "":
            st.error("API KEY를 입력해주세요.")
            return None
        embeddings = OpenAIEmbeddings(api_key=openai_api_key)
        cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
        vectorstore = FAISS.from_documents(docs, cached_embeddings)
        retriever = vectorstore.as_retriever()
        print("성공")
        return retriever
    except ValidationError as e:
        print("에러 나니?")
        st.error("LLM을 초기화하는 동안 오류가 발생했습니다. API KEY를 확인해보세요.")
        return None
    


def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)


def save_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})


def paint_history():
    for message in st.session_state["messages"]:
        send_message(
            message["message"],
            message["role"],
            save=False,
        )


def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
        Answer the question using ONLY the following context. IF you don't know the answer, just say you don't know. DON'T make anything up.

        Context: {context}
        """,
        ),
        ("human", "{question}"),
    ]
)


st.title("무엇이든 물어보세요!")

st.caption("파일을 업로드하면 파일 내용을 기반으로 답변을 제공합니다.")
st.caption("지원하는 파일 형식: .txt, .pdf, .docx, .pptx")

file = None
with st.sidebar:
    api_key = st.text_input("API Key")
    error_msg = st.error("API Key를 입력해주세요.")
    if api_key != "":
        llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.1,
        streaming=True,
        callbacks=[
            ChatCallbackHandler(),
        ],
        openai_api_key=api_key,
        )
        openai_api_key=api_key
        error_msg.empty()
        file = st.file_uploader(
            "Upload a .txt .pdf or .docx file",
            type=["pdf", "txt", "docx", "pptx"],
        )

if file:
    retriever = embed_file(file)
    if retriever is not None:
        send_message("주신 파일 잘 읽었습니다. 무엇이든 물어보세요!", "ai", save=False)
        paint_history()

        message = st.chat_input("무엇이든 물어보세요!")

        if message:
            send_message(message, "human")

            chain = (
                {
                    "context": retriever | RunnableLambda(format_docs),
                    "question": RunnablePassthrough(),
                }
                | prompt
                | llm
            )
            with st.chat_message("ai"):
                response = chain.invoke(message)
    else:
        st.session_state["messages"] = []

else:
    st.session_state["messages"] = []
