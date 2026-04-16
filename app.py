# app.py
import os
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

# ── API 키 로드 ──────────────────────────────────────────
# Streamlit Cloud: st.secrets 우선 / 로컬: data/.env fallback
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# OpenAI API 키를 코드에 직접 입력해서 사용합니다.
api_key="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"

# Streamlit Cloud는 STREAMLIT_SHARING_MODE 환경변수를 자동으로 설정함
IS_CLOUD = "STREAMLIT_SHARING_MODE" in os.environ

# PDF 처리 함수
@st.cache_resource
import os
from pathlib import Path

def process_pdf():
    # 현재 실행 중인 파일(app.py)의 위치를 기준으로 절대 경로를 생성합니다.
    current_dir = Path(__file__).parent if "__file__" in locals() else Path.cwd()
    
    # 'data' 폴더 안의 PDF 파일 경로 설정
    file_path = current_dir / "data" / "2024_KB_부동산_보고서_최종.pdf"
    
    # 디버깅: 파일이 실제로 존재하는지 확인 (로그에 출력됨)
    if not file_path.exists():
        print(f"ERROR: 파일을 찾을 수 없습니다 -> {file_path}")
        raise FileNotFoundError(f"PDF 파일을 찾을 수 없습니다: {file_path}")
# 벡터 스토어 초기화
FAISS_INDEX_PATH = os.path.join(BASE_DIR, "faiss_index")

@st.cache_resource
def initialize_vectorstore():
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    if IS_CLOUD:
        # Streamlit Cloud: 파일 저장이 휘발성이므로 항상 메모리에 새로 생성
        chunks = process_pdf()
        vectorstore = FAISS.from_documents(chunks, embeddings)
    elif os.path.exists(FAISS_INDEX_PATH):
        # 로컬: 저장된 인덱스가 있으면 로드
        vectorstore = FAISS.load_local(
            FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True
        )
    else:
        # 로컬: 인덱스 없으면 새로 생성 후 저장
        chunks = process_pdf()
        vectorstore = FAISS.from_documents(chunks, embeddings)
        vectorstore.save_local(FAISS_INDEX_PATH)
    return vectorstore

# 체인 초기화
@st.cache_resource
def initialize_chain():
    vectorstore = initialize_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    template = """당신은 KB 부동산 보고서 전문가입니다. 다음 정보를 바탕으로 사용자의 질문에 답변해주세요.

    컨텍스트: {context}
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", template),
        ("placeholder", "{chat_history}"),
        ("human", "{question}")
    ])

    model = ChatOpenAI(model_name="gpt-4o-mini", temperature=0, openai_api_key=api_key)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    base_chain = (
        RunnablePassthrough.assign(
            context=lambda x: format_docs(retriever.invoke(x["question"]))
        )
        | prompt
        | model
        | StrOutputParser()
    )

    # 최근 4번의 발화(human + assistant 각 2쌍 = 메시지 4개)만 유지하는 히스토리 래퍼
    def get_trimmed_history(session_id: str) -> ChatMessageHistory:
        if "chat_history_store" not in st.session_state:
            st.session_state.chat_history_store = ChatMessageHistory()
        history = st.session_state.chat_history_store
        # 최근 4개의 메시지만 유지 (human 2 + assistant 2)
        if len(history.messages) > 4:
            history.messages = history.messages[-4:]
        return history

    return RunnableWithMessageHistory(
        base_chain,
        get_trimmed_history,
        input_messages_key="question",
        history_messages_key="chat_history",
    )

# Streamlit UI
def main():
    st.set_page_config(page_title="KB 부동산 보고서 챗봇", page_icon="🏠")
    st.title("🏠 KB 부동산 보고서 AI 어드바이저")
    st.caption("2024 KB 부동산 보고서 기반 질의응답 시스템")

    # 세션 상태 초기화
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 채팅 기록 표시
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 사용자 입력 처리
    if prompt := st.chat_input("부동산 관련 질문을 입력하세요"):
        # 사용자 메시지 표시
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # 체인 초기화
        chain = initialize_chain()

        # AI 응답 생성
        with st.chat_message("assistant"):
            with st.spinner("답변 생성 중..."):
                response = chain.invoke(
                    {"question": prompt},
                    {"configurable": {"session_id": "streamlit_session"}}
                )
                st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
