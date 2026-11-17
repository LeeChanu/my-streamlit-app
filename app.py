
import streamlit as st
import os
import random
import shutil
import tempfile
from pathlib import Path
from uuid import uuid4
from copy import deepcopy
from PIL import Image
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain_pymupdf4llm import PyMuPDF4LLMLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from operator import itemgetter

# 페이지 설정
st.set_page_config(page_title="AIVLE School 학습 도우미", page_icon="💬", layout="wide")

# 폴더 설정 및 상수 정의
CHROMA_DIR = "chroma_db"
COLLECTION = "aivle_docs"

# 입력창 내부 왼쪽 일체형 클립 UI를 구현하는 CSS
st.markdown("""
<style>
    /* 대화창 영역 최적화 */
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 6rem;
        max-width: 850px;
    }
    
    /* 포춘쿠키 박스 스타일 */
    .egg-box {
        text-align: center;
        padding: 12px;
        background-color: #F3F4F6;
        border-radius: 8px;
        border: 1px solid #E5E7EB;
        font-size: 0.9rem;
        color: #374151;
        margin-top: 10px;
    }
    @keyframes floatCookie {
        0% { transform: translateY(0px) rotate(0deg); }
        50% { transform: translateY(-15px) rotate(5deg); }
        100% { transform: translateY(0px) rotate(0deg); }
    }
    .floating-cookie {
        font-size: 3.5rem;
        text-align: center;
        display: block;
        cursor: pointer;
        animation: floatCookie 3s ease-in-out infinite;
        user-select: none;
        margin: 15px 0;
    }
    
    /* 세로형 커리큘럼 타임라인 블록 */
    .timeline-card {
        background-color: #F9FAFB;
        border-left: 4px solid #1E3A8A;
        padding: 10px 14px;
        margin-bottom: 10px;
        border-radius: 0 8px 8px 0;
    }
    .timeline-title {
        font-weight: bold;
        color: #1E3A8A;
        font-size: 0.85rem;
        margin-bottom: 3px;
    }
    .timeline-desc {
        font-size: 0.8rem;
        color: #4B5563;
        line-height: 1.3;
    }

    /* 첨부 아이콘은 st.chat_input의 기본 업로드 버튼을 사용합니다. */
    [data-testid="stChatInput"] button[aria-label="Upload a file"] svg { display: none; }
    [data-testid="stChatInput"] button[aria-label="Upload a file"]::after { content: "📎"; font-size: 20px; line-height: 1; }

    .block-container { padding-top: 4rem; }
    .welcome-title { text-align: center; color: #1E3A8A; margin: 8vh auto 2rem; font-size: clamp(1.5rem, 3vw, 2.2rem); line-height: 1.5; overflow-wrap: anywhere; }
    .st-key-fortune_cookie button { border: 0; background: transparent; box-shadow: none; padding: 18px; min-height: 100px; animation: floatCookie 3s ease-in-out infinite; }
    .st-key-fortune_cookie button p { font-size: 3.5rem; line-height: 1.5; }
    .st-key-fortune_cookie button:hover { background: transparent; border: 0; }

    /* 하단 입력칸을 대화 본문 폭에 맞춰 가운데 정렬 */
    [data-testid="stBottomBlockContainer"] {
        max-width: 850px;
        margin-inline: auto;
        padding-inline: 1rem;
        box-sizing: border-box;
    }
    [data-testid="stChatInput"] {
        border: 1px solid #D1D5DB;
        border-radius: 24px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }
    [data-testid="stChatInput"]:focus-within {
        border-color: #14B8A6;
        box-shadow: 0 0 0 2px rgba(20,184,166,0.12);
    }
    [data-testid="stChatInputTextArea"] {
        padding-block: 12px;
    }
</style>
""", unsafe_allow_html=True)

# --- 1. 벡터 스토어 & 리트리버 설정 ---
def get_session_db_dir():
    if "writable_chroma_dir" not in st.session_state:
        runtime_dir = Path(tempfile.mkdtemp(prefix="aivle_chroma_")) / "chroma_db"
        source_dir = Path(CHROMA_DIR)
        if source_dir.is_dir():
            shutil.copytree(source_dir, runtime_dir, copy_function=shutil.copyfile)
        else:
            runtime_dir.mkdir(parents=True)
        st.session_state.writable_chroma_dir = str(runtime_dir)
    return st.session_state.writable_chroma_dir

@st.cache_resource
def get_retriever(db_dir):
    embedding = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma(
        collection_name=COLLECTION,
        embedding_function=embedding,
        persist_directory=db_dir
    )
    return vectorstore

# 업로드 파일과 DB를 접속 세션별 쓰기 가능한 임시 경로에서 처리
def process_uploaded_file(uploaded_file):
    with tempfile.TemporaryDirectory(prefix="aivle_pdf_") as upload_dir:
        pdf_path = Path(upload_dir) / "uploaded.pdf"
        pdf_path.write_bytes(uploaded_file.getbuffer())
        loader = PyMuPDF4LLMLoader(str(pdf_path))
        docs = loader.load()
        for doc in docs:
            doc.metadata["source"] = uploaded_file.name
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        chunks = splitter.split_documents(docs)
        if not chunks:
            raise ValueError("PDF에서 텍스트를 추출하지 못했습니다. 스캔 문서는 OCR 처리가 필요합니다.")
        vectorstore = get_retriever(get_session_db_dir())
        vectorstore.add_documents(chunks)
    st.success(f"📄 {uploaded_file.name} 반영 완료! 이 문서에 대해 질문해 주세요.")

# --- 2. RAG 파이프라인 구성 ---
@st.cache_resource
def get_chain(db_dir):
    vectorstore = get_retriever(db_dir)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    s_msg = (
        "당신은 KT AIVLE School의 전문적이고 다정한 인공지능 학습 멘토입니다.\n"
        "주어진 [문서]를 바탕으로 군더더기 없이 핵심 중심의 친절하고 명확한 답변을 제시해 주세요.\n"
        "답변의 마지막 줄에는 항상 에이블러(AIVLEer)의 성장을 열정적으로 응원하는 격려와 긍정의 멘토링 코멘트를 잊지 말고 작성해 주세요.\n\n"
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

# 체인 초기화 안전하게 감싸기
try:
    chain = get_chain(get_session_db_dir())
except Exception as e:
    st.error(f"체인 로딩 에러: {str(e)}")
    chain = None

# --- 3. 세션 상태 초기화 ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "user_query" not in st.session_state: 
    st.session_state.user_query = ""

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "active_chat_id" not in st.session_state:
    st.session_state.active_chat_id = None

def save_current_chat():
    if not st.session_state.messages:
        return
    if st.session_state.active_chat_id is None:
        st.session_state.active_chat_id = uuid4().hex
    title = next((m.content for m in st.session_state.messages if isinstance(m, HumanMessage)), "새 대화")
    entry = {"id": st.session_state.active_chat_id, "title": title[:32], "messages": deepcopy(st.session_state.messages)}
    others = [c for c in st.session_state.chat_history if c["id"] != entry["id"]]
    st.session_state.chat_history = [entry, *others][:3]

# --- 4. 사이드바 구성 ---
with st.sidebar:
    # 투명 배경 로고 적용
    logo_path = "투명 배경 AIVLE 로고 복원.png"
    if os.path.exists(logo_path):
        try:
            logo_img = Image.open(logo_path)
            st.image(logo_img, use_container_width=True)
        except Exception as e:
            st.warning("로고 이미지를 불러오지 못했습니다.")
    else:
        st.image("https://raw.githubusercontent.com/Jangrae/img/master/rag_title.png", use_container_width=True)
    
    st.title("AIVLE School")

    # 🥠 사이드바 포춘쿠키 이스터에그 배치
    st.subheader("🥠 오늘의 성장 행운")

    if st.button("🥠", key="fortune_cookie", help="쿠키를 눌러 오늘의 한마디 보기", use_container_width=True):
        fortunes = [
            "실패는 다음 단계로 나아가기 위한 정교한 튜닝 과정일 뿐입니다! 힘내세요!",
            "지금 풀리지 않는 버그는 내일의 가장 든든한 기술 자산이 됩니다!",
            "에이블스쿨의 협업 경험은 훗날 현업에서 가장 빛나는 무기가 됩니다.",
            "치열한 하루하루의 몰입이 쌓여 가장 완성도 높은 내일을 만들어냅니다.",
            "어려운 과제가 찾아왔다면 한 걸음 더 크게 도약할 때가 된 것입니다!"
        ]
        st.markdown(f"<div class='egg-box'>✨ {random.choice(fortunes)}</div>", unsafe_allow_html=True)

    st.markdown("---")

    st.subheader("💬 최근 대화")
    if st.button("＋ 새 대화", use_container_width=True):
        save_current_chat()
        st.session_state.messages = []
        st.session_state.user_query = ""
        st.session_state.active_chat_id = None
        st.rerun()
    if not st.session_state.chat_history:
        st.caption("대화를 시작하면 최근 3개가 여기에 저장돼요.")
    for chat in st.session_state.chat_history:
        label = ("💬 " if chat["id"] == st.session_state.active_chat_id else "🕘 ") + chat["title"]
        if st.button(label, key="restore_" + chat["id"], use_container_width=True, help="이 대화 이어보기"):
            save_current_chat()
            st.session_state.messages = deepcopy(chat["messages"])
            st.session_state.active_chat_id = chat["id"]
            st.session_state.user_query = ""
            st.rerun()
    st.markdown("---")

    # 📅 세로형 타임라인식 카드 커리큘럼 가이드 (가로 폭을 절대 침범하지 않음)
    st.subheader("📅 AIVLE 커리큘럼")
    st.markdown("""
    <div class="timeline-card">
        <div class="timeline-title">Step 1. AI 분석 & 파이썬 기초</div>
        <div class="timeline-desc">데이터 분석을 위한 파이썬 라이브러리와 통계의 기초를 학습합니다.</div>
    </div>
    <div class="timeline-card">
        <div class="timeline-title">Step 2. 머신러닝 & 딥러닝 실전</div>
        <div class="timeline-desc">다양한 알고리즘 설계와 딥러닝 실전 모델 구현 역량을 체득합니다.</div>
    </div>
    <div class="timeline-card">
        <div class="timeline-title">Step 3. DX 컨설팅 & 빅프로젝트</div>
        <div class="timeline-desc">실제 산업 도메인의 문제를 비즈니스 관점에서 정의하고 해결합니다.</div>
    </div>
    """, unsafe_allow_html=True)

# --- 5. 메인화면: GPT 스타일의 채팅창 단독 배치 ---
welcome = st.empty()
if not st.session_state.messages:
    with welcome.container():
        st.markdown('<h1 class="welcome-title">💬 AIVLE School 학습 도우미</h1>', unsafe_allow_html=True)
        # 추천 질문 목록
        recommends = [
            "AIVLE School 공부 방법 추천해줘!",
            "선배들의 자기계발 꿀팁은 무엇인가요?",
            "효과적인 프로젝트 협업 노하우는?",
            "비전공자를 위한 학습 전략은?"
        ]
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button(f"💡 {recommends[0]}", use_container_width=True):
                st.session_state.user_query = recommends[0]
            if st.button(f"💡 {recommends[1]}", use_container_width=True):
                st.session_state.user_query = recommends[1]
        with col2:
            if st.button(f"💡 {recommends[2]}", use_container_width=True):
                st.session_state.user_query = recommends[2]
            if st.button(f"💡 {recommends[3]}", use_container_width=True):
                st.session_state.user_query = recommends[3]
        
        st.markdown("---")

# --- 6. 대화 인터페이스 렌더링 ---
for message in st.session_state.messages:
    role = "user" if isinstance(message, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(message.content)

# --- 7. 입력바 안쪽 왼쪽의 기본 첨부 아이콘 ---
chat_submission = st.chat_input(
    "질문할 내용을 입력해 주세요.",
    accept_file=True,
    file_type=["pdf"],
    key="aivle_chat_input",
)
user_input = ""
if chat_submission is not None:
    user_input = chat_submission.text
    for uploaded_file in chat_submission.files:
        try:
            with st.spinner("첨부한 PDF를 반영하고 있습니다..."):
                process_uploaded_file(uploaded_file)
        except Exception as e:
            st.error(f"PDF 처리에 실패했습니다: {e}")
            st.stop()

if st.session_state.user_query:
    user_input = st.session_state.user_query
    st.session_state.user_query = ""

if user_input and chain is not None:
    welcome.empty()
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append(HumanMessage(content=user_input))

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        response_stream = chain.stream({"input": user_input})
        full_response = st.write_stream(response_stream)

    st.session_state.messages.append(AIMessage(content=full_response))
    save_current_chat()
    st.rerun()
