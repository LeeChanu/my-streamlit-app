
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
st.set_page_config(page_title="AIVLE School 학습 도우미", page_icon="💬", layout="wide")

# 폴더 설정 및 상수 정의
CHROMA_DIR = "chroma_db"
COLLECTION = "aivle_docs"

# GPT 느낌의 모던하고 심플한 CSS 스타일링 + 떠다니는 포춘쿠키 애니메이션
st.markdown("""
<style>
    /* 대화창 상단 여백 제거 및 너비 제한 */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
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
    /* 둥둥 떠다니는 애니메이션 정의 */
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
    .floating-cookie:hover {
        filter: drop-shadow(0px 8px 12px rgba(245, 158, 11, 0.4));
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
    st.success(f"📄 {uploaded_file.name}가 성공적으로 동기화되었습니다!")

# --- 2. RAG 파이프라인 구성 ---
@st.cache_resource
def get_chain():
    vectorstore = get_retriever()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})  # 3.0을 정수 3으로 수정하여 ValueError 해결

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
    chain = get_chain()
except Exception as e:
    st.error(f"체인 로딩 에러: {str(e)}")
    chain = None

# --- 3. 세션 상태 초기화 ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "user_query" not in st.session_state:
    st.session_state.user_query = ""

# --- 4. 사이드바 구성 (기타 기능은 모두 사이드바 내부로 숨김) ---
with st.sidebar:
    st.image("https://raw.githubusercontent.com/Jangrae/img/master/rag_title.png", use_container_width=True)
    st.title("AIVLE School")

    # 🥠 사이드바 포춘쿠키 이스터에그 배치
    st.subheader("🥠 오늘의 성장 행운")
    st.markdown("<span class='floating-cookie'>🥠</span>", unsafe_allow_html=True)

    if st.button("둥둥 떠다니는 포춘쿠키 잡아라!", use_container_width=True):
        fortunes = [
            "실패는 다음 단계로 나아가기 위한 정교한 튜닝 과정일 뿐입니다! 힘내세요!",
            "지금 풀리지 않는 버그는 내일의 가장 든든한 기술 자산이 됩니다!",
            "에이블스쿨의 협업 경험은 훗날 현업에서 가장 빛나는 무기가 됩니다.",
            "치열한 하루하루의 몰입이 쌓여 가장 완성도 높은 내일을 만들어냅니다.",
            "어려운 과제가 찾아왔다면 한 걸음 더 크게 도약할 때가 된 것입니다!"
        ]
        st.markdown(f"<div class='egg-box'>✨ <b>조언:</b> {random.choice(fortunes)}</div>", unsafe_allow_html=True)

    st.markdown("---")

    # 대화 삭제
    st.subheader("💬 대화 관리")
    if st.button("이전 대화 내용 초기화", use_container_width=True):
        st.session_state.messages = []
        st.session_state.user_query = ""
        st.rerun()

    st.markdown("---")

    # 파일 업로드
    st.subheader("📂 나만의 문서 추가")
    uploaded_file = st.file_uploader("추가할 PDF 백서를 업로드하세요.", type=["pdf"])
    if uploaded_file is not None:
        if st.button("학습 데이터에 동기화", use_container_width=True):
            process_uploaded_file(uploaded_file)

    st.markdown("---")

    # 커리큘럼
    st.subheader("📅 AIVLE 커리큘럼")
    st.caption("1단계: AI 분석 & 파이썬 기초\n\n2단계: 머신러닝 & 딥러닝 실전\n\n3단계: DX 컨설팅 & 빅프로젝트")

# --- 5. 메인화면: GPT 스타일의 채팅창만 단독 배치 ---
st.markdown("<h3 style='text-align: center; color: #1E3A8A;'>💬 AIVLE School 학습 도우미</h3>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #6B7280; font-size: 0.95rem; margin-bottom: 30px;'>무엇이든 질문해 보세요. 학습 백서 기반으로 신속하게 답변해 드립니다.</p>", unsafe_allow_html=True)

# 추천 질문 목록 (대화 시작을 유도하는 미니 버튼 리스트)
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

# 사용자 입력 감지 및 처리 우선순위 지정
user_input = st.chat_input("질문할 내용을 입력해 주세요.")

if st.session_state.user_query:
    user_input = st.session_state.user_query
    st.session_state.user_query = ""

if user_input and chain is not None:
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append(HumanMessage(content=user_input))

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        response_stream = chain.stream({"input": user_input})
        full_response = st.write_stream(response_stream)

    st.session_state.messages.append(AIMessage(content=full_response))
    st.rerun()
