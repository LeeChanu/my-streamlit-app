
import streamlit as st
import os
import random
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain_pymupdf4llm import PyMuPDF4LLMLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from operator import itemgetter

# 페이지 설정
st.set_page_config(page_title="AIVLE School 학습 멘토 도우미", page_icon="🚀", layout="wide")

# 폴더 설정 및 상수 정의
CHROMA_DIR = "chroma_db"
COLLECTION = "aivle_docs"

# CSS 스타일링으로 UI 개선
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 5px;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #4B5563;
        text-align: center;
        margin-bottom: 25px;
    }
    .recommend-box {
        border: 1px solid #E5E7EB;
        border-radius: 10px;
        padding: 12px;
        background-color: #F9FAFB;
        margin-bottom: 10px;
    }
    .stButton>button {
        border-radius: 8px;
    }
    .egg-box {
        text-align: center;
        padding: 10px;
        background-color: #FEF3C7;
        border-radius: 12px;
        border: 1px dashed #F59E0B;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# --- 1. 벡터 스토어 & 리트리버 설정 ---
@st.cache_resource
def get_retriever():
    embedding = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma(
        collection_name=COLLECTION,
        embedding_function=embedding,
        persist_directory=CHROMA_DIR
    )
    return vectorstore

# 임시 업로드 파일 처리용
def process_uploaded_file(uploaded_file):
    with open("temp_uploaded.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    loader = PyMuPDF4LLMLoader("temp_uploaded.pdf")
    docs = loader.load()
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    
    vectorstore = get_retriever()
    vectorstore.add_documents(chunks)
    os.remove("temp_uploaded.pdf")
    st.success(f"📄 {uploaded_file.name} 문서가 성공적으로 학습 데이터에 추가되었습니다!")

# --- 2. RAG 파이프라인 구성 ---
@st.cache_resource
def get_chain():
    vectorstore = get_retriever()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3.0})
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # 시스템 메시지 고도화 (에이블스쿨의 열정과 도전을 이끌어내는 인격 부여)
    s_msg = (
        "당신은 KT AIVLE School의 열정적이고 전문적인 학습 멘토입니다.\n"
        "주어진 [문서] 내용을 바탕으로 성실하고 명확하게 답변해 주세요.\n"
        "답변 끝에는 에이블러(AIVLEer)가 성장을 위해 적극적으로 도전하고 한걸음 나아갈 수 있도록 격려와 자신감을 북돋아 주는 따뜻한 멘토링 메시지를 항상 한두 줄 덧붙여 주세요.\n\n"
        "[문서]\n{context}"
    )
    h_msg = "{input}"
    
    prompt = ChatPromptTemplate.from_messages([("system", s_msg), ("human", h_msg)])
    llm = ChatOpenAI(model="gpt-4o-mini", streaming=True, temperature=0.3)
    
    rag_chain = (
        {
            "context": itemgetter("input") | retriever | format_docs,
            "input": itemgetter("input")
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain

chain = get_chain()

# --- 3. 세션 상태 초기화 ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "user_query" not in st.session_state:
    st.session_state.user_query = ""

# --- 4. 레이아웃 & 사이드바 구성 ---
# 로고 출력 (시작 화면 상단)
st.markdown("<center><img src='https://raw.githubusercontent.com/Jangrae/img/master/rag_title.png' width='450'/></center>", unsafe_allow_html=True)
st.markdown("<div class='main-title'>AIVLE School AI 학습 멘토 도우미</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>🚀 에이블러의 지식 충전소! 궁금한 모든 것을 질문하고 성장해 보세요!</div>", unsafe_allow_html=True)

# 사이드바 영역
with st.sidebar:
    st.image("https://raw.githubusercontent.com/Jangrae/img/master/rag_title.png", use_container_width=True)
    st.title("🛠️ 에이블 서포트 센터")
    
    # 대화 기록 관리 및 대화 삭제 기능
    st.subheader("💬 대화 관리")
    if st.button("🗑️ 이전 대화 내용 초기화", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
        
    st.markdown("---")
    
    # 파일 업로드 기능 추가 (PDF 학습)
    st.subheader("📂 나만의 학습 백서 추가")
    uploaded_file = st.file_uploader("PDF 파일을 추가하여 질문 대상 범위를 넓혀보세요.", type=["pdf"])
    if uploaded_file is not None:
        if st.button("💡 문서 학습 시키기", use_container_width=True):
            process_uploaded_file(uploaded_file)
            
    st.markdown("---")
    
    # 교육 커리큘럼 로드맵 가이드 제공
    st.subheader("📅 AIVLE 핵심 커리큘럼")
    st.info("1. AI 분석 & 파이썬 기초\n2. 머신러닝 & 딥러닝 핵심 실습\n3. DX 컨설팅 & 빅프로젝트 실전과정")

# --- 5. 이스터 에그 (포춘쿠키 기능) ---
st.markdown("### 🥠 오늘 나의 성장 한마디")
col_egg1, col_egg2 = st.columns([1, 4])
with col_egg1:
    click_cookie = st.button("🥠 포춘쿠키 열기")
with col_egg2:
    if click_cookie:
        fortunes = [
            "실패는 다음 단계로 나아가기 위한 정교한 튜닝 과정일 뿐입니다! 힘내세요!",
            "지금 풀리지 않는 버그는 내일의 가장 든든한 기술 자산이 됩니다. 포기하지 마세요!",
            "에이블스쿨의 협업 경험은 훗날 현업에서 가장 빛나는 협업 마인드가 됩니다!",
            "1만 시간의 법칙보다 값진 것은, 6개월간 매일 동료들과 치열하게 치달리는 몰입의 경험입니다.",
            "어려운 과제가 찾아왔다면, 당신이 그만큼 크게 도약할 준비가 되었다는 우주의 시그널입니다!"
        ]
        st.markdown(f"<div class='egg-box'>✨ **행운의 조언** : {random.choice(fortunes)}</div>", unsafe_allow_html=True)

st.markdown("---")

# --- 6. 추천 질문 목록 ---
st.markdown("#### 💡 자주 묻는 질문 (클릭 시 자동 입력)")
recommends = [
    "AIVLE School의 전반적인 공부 방법과 꿀팁을 가르쳐줘!",
    "선배들이 추천하는 자기계발 방법은 무엇이 있나요?",
    "프로젝트 협업을 잘 조율하고 이끄는 방법이 궁금해.",
    "비전공자인데 진도를 잘 따라가려면 어떻게 해야 할까요?"
]

col1, col2 = st.columns(2)
with col1:
    if st.button(f"📌 {recommends[0]}", use_container_width=True):
        st.session_state.user_query = recommends[0]
    if st.button(f"📌 {recommends[1]}", use_container_width=True):
        st.session_state.user_query = recommends[1]
with col2:
    if st.button(f"📌 {recommends[2]}", use_container_width=True):
        st.session_state.user_query = recommends[2]
    if st.button(f"📌 {recommends[3]}", use_container_width=True):
        st.session_state.user_query = recommends[3]

st.markdown("---")

# --- 7. 대화 인터페이스 렌더링 ---
# 세션 기반 이전 메시지 렌더링
for message in st.session_state.messages:
    role = "user" if isinstance(message, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(message.content)

# 사용자 메시지 전송 및 답변 생성
user_input = st.chat_input("질문할 내용을 입력해 주세요.")
if st.session_state.user_query:
    user_input = st.session_state.user_query
    st.session_state.user_query = "" # 사용 후 초기화

if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append(HumanMessage(content=user_input))
    
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        response_stream = chain.stream({"input": user_input})
        full_response = st.write_stream(response_stream)
    
    st.session_state.messages.append(AIMessage(content=full_response))
