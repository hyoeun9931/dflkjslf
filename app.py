import os
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory


# 1. 환경변수 로드
load_dotenv("../data/.env")
openai_api_key = os.getenv("OPENAI_API_KEY")

pdf_path = "../data/2024_KB_부동산_보고서_최종.pdf"
faiss_path = "./faiss_db"


# 2. PDF 처리
@st.cache_resource
def process_pdf():
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    splits = text_splitter.split_documents(documents)
    return splits


# 3. 벡터DB 생성 또는 로드
@st.cache_resource
def initialize_vectorstore():
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=openai_api_key
    )

    # 기존 FAISS DB가 있으면 로드
    if os.path.exists(faiss_path):
        try:
            vectorstore = FAISS.load_local(
                faiss_path,
                embeddings,
                allow_dangerous_deserialization=True
            )
            return vectorstore
        except Exception:
            pass

    # 없으면 새로 생성
    splits = process_pdf()
    vectorstore = FAISS.from_documents(splits, embeddings)
    vectorstore.save_local(faiss_path)
    return vectorstore


# 4. 문서 포맷
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# 5. 체인 초기화
@st.cache_resource
def initialize_chain():
    vectorstore = initialize_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    template = """
당신은 KB 부동산 보고서 전문가입니다.
다음 정보를 바탕으로 사용자의 질문에 답변하세요.
모르는 내용은 추측하지 말고, 문서에서 확인 가능한 내용만 답변하세요.

컨텍스트:
{context}
"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", template),
        ("placeholder", "{chat_history}"),
        ("human", "{question}")
    ])

    model = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=openai_api_key
    )

    base_chain = (
        RunnablePassthrough.assign(
            context=lambda x: format_docs(retriever.invoke(x["question"]))
        )
        | prompt
        | model
        | StrOutputParser()
    )

    return base_chain


# 6. 세션 메모리
if "history_store" not in st.session_state:
    st.session_state.history_store = {}

def get_session_history(session_id: str):
    if session_id not in st.session_state.history_store:
        st.session_state.history_store[session_id] = ChatMessageHistory()
    return st.session_state.history_store[session_id]


# 7. UI
def main():
    st.set_page_config(page_title="KB 부동산 보고서 챗봇", page_icon="🏠")
    st.title("KB 부동산 보고서 AI 어드바이저")
    st.caption("2024 KB 부동산 보고서를 기반으로 답변합니다.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_prompt = st.chat_input("부동산 관련 질문을 입력하세요")

    if user_prompt:
        with st.chat_message("user"):
            st.markdown(user_prompt)
        st.session_state.messages.append({"role": "user", "content": user_prompt})

        base_chain = initialize_chain()
        chain_with_memory = RunnableWithMessageHistory(
            base_chain,
            get_session_history,
            input_messages_key="question",
            history_messages_key="chat_history",
        )

        with st.chat_message("assistant"):
            with st.spinner("답변 생성 중..."):
                response = chain_with_memory.invoke(
                    {"question": user_prompt},
                    config={"configurable": {"session_id": "streamlit_session"}}
                )
                st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()