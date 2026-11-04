
import os
import re
import json
import base64
import hashlib
import smtplib
from email.message import EmailMessage
from typing import Literal

import streamlit as st
from openai import OpenAI
from pydantic import BaseModel, Field

# ============================================================
# 1. 기본 설정
# ============================================================

st.set_page_config(
    page_title="이찬우의 개인비서",
    page_icon="🦦",
    layout="wide",
    initial_sidebar_state="expanded"
)

def get_secret(name: str, default: str = "") -> str:
    try:
        value = st.secrets.get(name, None)
        if value:
            return value
    except Exception:
        pass
    return os.getenv(name, default)

API_KEY = get_secret("OPENAI_API_KEY")

if not API_KEY:
    st.error("OPENAI_API_KEY가 설정되지 않았습니다.")
    st.stop()

client = OpenAI(api_key=API_KEY)

MODEL = "gpt-4o-mini"
TEMPERATURE = 0

# ============================================================
# 2. 수달 캐릭터 SVG
# ============================================================

OTTER_SVG = """
<svg xmlns="http://www.w3.org/2000/svg" width="360" height="360" viewBox="0 0 360 360">
  <defs>
    <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#eefbff"/>
      <stop offset="100%" stop-color="#dff4ff"/>
    </linearGradient>
    <linearGradient id="fur" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#b97a50"/>
      <stop offset="100%" stop-color="#8e5b39"/>
    </linearGradient>
    <linearGradient id="belly" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#ffe8cf"/>
      <stop offset="100%" stop-color="#f4d6b7"/>
    </linearGradient>
  </defs>

  <rect x="18" y="18" width="324" height="324" rx="26" fill="url(#bg)" stroke="#addff2" stroke-width="3"/>

  <!-- 말풍선 -->
  <rect x="208" y="32" width="108" height="58" rx="16" fill="white" stroke="#c9e7f2" stroke-width="2"/>
  <polygon points="216,84 200,96 209,76" fill="white" stroke="#c9e7f2" stroke-width="2"/>
  <text x="262" y="56" text-anchor="middle" font-size="15" font-family="Arial" fill="#3f5160">안녕하세요!</text>
  <text x="262" y="74" text-anchor="middle" font-size="12" font-family="Arial" fill="#617384">에이블이에요</text>

  <!-- 꼬리 -->
  <ellipse cx="110" cy="282" rx="28" ry="54" fill="#865435" transform="rotate(-20 110 282)"/>

  <!-- 몸 -->
  <ellipse cx="168" cy="218" rx="76" ry="96" fill="url(#fur)"/>
  <ellipse cx="168" cy="235" rx="50" ry="66" fill="url(#belly)"/>

  <!-- 팔 -->
  <ellipse cx="98" cy="220" rx="22" ry="50" fill="url(#fur)" transform="rotate(20 98 220)"/>
  <ellipse cx="238" cy="220" rx="22" ry="50" fill="url(#fur)" transform="rotate(-18 238 220)"/>

  <!-- 다리 -->
  <ellipse cx="132" cy="305" rx="24" ry="16" fill="#74492e"/>
  <ellipse cx="204" cy="305" rx="24" ry="16" fill="#74492e"/>

  <!-- 클립보드 -->
  <rect x="230" y="184" width="48" height="78" rx="10" fill="#ffffff" stroke="#8ed2ea" stroke-width="3"/>
  <rect x="244" y="178" width="18" height="10" rx="4" fill="#8ed2ea"/>
  <line x1="239" y1="204" x2="269" y2="204" stroke="#9dbccc" stroke-width="3"/>
  <line x1="239" y1="216" x2="269" y2="216" stroke="#9dbccc" stroke-width="3"/>
  <line x1="239" y1="228" x2="265" y2="228" stroke="#9dbccc" stroke-width="3"/>
  <line x1="239" y1="240" x2="261" y2="240" stroke="#9dbccc" stroke-width="3"/>

  <!-- 머리 -->
  <circle cx="168" cy="110" r="70" fill="url(#fur)"/>

  <!-- 귀 -->
  <circle cx="118" cy="63" r="16" fill="#8e5b39"/>
  <circle cx="218" cy="63" r="16" fill="#8e5b39"/>
  <circle cx="118" cy="63" r="8" fill="#f6d7bf"/>
  <circle cx="218" cy="63" r="8" fill="#f6d7bf"/>

  <!-- 얼굴 -->
  <ellipse cx="168" cy="122" rx="44" ry="36" fill="#f8dfc9"/>
  <circle cx="144" cy="107" r="8" fill="#222"/>
  <circle cx="192" cy="107" r="8" fill="#222"/>
  <circle cx="141" cy="104" r="2" fill="white"/>
  <circle cx="189" cy="104" r="2" fill="white"/>
  <ellipse cx="168" cy="121" rx="8" ry="6" fill="#4a3428"/>
  <path d="M154 134 Q168 147 182 134" fill="none" stroke="#4a3428" stroke-width="4" stroke-linecap="round"/>
  <circle cx="136" cy="123" r="4" fill="#f3b8ae" opacity="0.85"/>
  <circle cx="200" cy="123" r="4" fill="#f3b8ae" opacity="0.85"/>

  <!-- 배지 -->
  <rect x="145" y="184" width="46" height="22" rx="11" fill="#86d7ef"/>
  <text x="168" y="199" text-anchor="middle" font-size="12" font-family="Arial" fill="#24576a">AIBLE</text>
</svg>
"""

OTTER_IMAGE = "data:image/svg+xml;base64," + base64.b64encode(OTTER_SVG.encode("utf-8")).decode("utf-8")

# ============================================================
# 3. 시스템 프롬프트
# ============================================================

SYS_ROLE = """
[정체성]
당신은 이찬우의 개인비서 '에이블'입니다.
당신은 회의록 분석, 업무 정리, 음성 대화, 이메일 작성 및 발송 지원을 담당하는 전문 AI 비서입니다.

[업무 원칙]
1. 회의록 기반으로 정확하게 응답합니다.
2. 회의의 논의사항, 제안사항, 최종 결정사항을 구분합니다.
3. 사용자 업무를 우선적으로 정리합니다.
4. 회의록에 없는 사실은 만들지 않습니다.
5. 담당자, 마감일, 수치, 이메일 주소를 임의 생성하지 않습니다.
6. 이메일은 초안 작성과 실제 발송을 구분합니다.
7. 실제 발송은 사용자 승인 후에만 가능하다고 가정합니다.
8. 멀티턴 대화에서 이전 문맥을 활용합니다.

[회의록 분석 기준]
- 회의 주제
- 핵심 논의 내용
- 최종 결정사항
- 사용자 담당 업무
- 업무별 상세 설명
- 명시된 마감일
- 지시자
- 추가 확인 사항
- 근거 발언 요지

[요청 의도 분류]
- meeting_summary: 회의 요약
- my_tasks: 사용자 담당 업무
- meeting_qa: 회의 관련 질문
- general_qa: 일반 질문
- email_draft: 이메일 초안 작성
- email_send: 실제 이메일 발송 요청
- clarification: 추가 정보 확인

[이메일 규칙]
- '메일 작성해줘'는 초안입니다.
- '메일 보내줘', '이메일 발송해줘'는 발송 요청입니다.
- 수신자가 없으면 확인 질문을 합니다.
- 발송 결과는 외부 도구 결과 기준으로 판단합니다.

[응답 스타일]
- 자연스럽고 전문적인 한국어를 사용합니다.
- 핵심 결론을 먼저 제시합니다.
- 필요하면 번호나 항목으로 정리합니다.
- voice_answer는 2~5문장 정도로 짧고 자연스럽게 작성합니다.
"""

# ============================================================
# 4. 구조화 응답 스키마
# ============================================================

class EmailInfo(BaseModel):
    recipient: str = Field(description="수신자 이메일. 없으면 빈 문자열")
    subject: str = Field(description="이메일 제목. 없으면 빈 문자열")
    body: str = Field(description="이메일 본문. 없으면 빈 문자열")

class AIResult(BaseModel):
    intent: Literal[
        "meeting_summary",
        "my_tasks",
        "meeting_qa",
        "general_qa",
        "email_draft",
        "email_send",
        "clarification"
    ]
    answer: str = Field(description="화면에 표시할 상세 답변")
    voice_answer: str = Field(description="자연스러운 음성 답변")
    needs_clarification: bool
    clarification_question: str
    email: EmailInfo

# ============================================================
# 5. 이메일 도구
# ============================================================

EMAIL_TOOL = {
    "type": "function",
    "name": "send_email",
    "description": "사용자가 승인한 이메일을 발송합니다.",
    "parameters": {
        "type": "object",
        "properties": {
            "to": {"type": "string"},
            "subject": {"type": "string"},
            "body": {"type": "string"}
        },
        "required": ["to", "subject", "body"],
        "additionalProperties": False
    },
    "strict": True
}

def valid_email(address: str) -> bool:
    return bool(re.fullmatch(r"[^\s@]+@[^\s@]+\.[^\s@]+", address or ""))

def send_email(to: str, subject: str, body: str):
    gmail = get_secret("GMAIL_ADDRESS")
    password = get_secret("GMAIL_APP_PASSWORD")

    if not gmail or not password:
        return False, "Gmail 계정 또는 앱 비밀번호가 설정되지 않았습니다."

    if not valid_email(to):
        return False, "수신자 이메일 주소 형식이 올바르지 않습니다."

    msg = EmailMessage()
    msg["From"] = gmail
    msg["To"] = to
    msg["Subject"] = subject
    msg.set_content(body)

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=25) as smtp:
            smtp.login(gmail, password)
            smtp.send_message(msg)
        return True, "이메일이 정상적으로 발송되었습니다."
    except smtplib.SMTPAuthenticationError:
        return False, "Gmail 인증에 실패했습니다. 앱 비밀번호를 확인해주세요."
    except Exception as e:
        return False, f"이메일 발송 실패: {type(e).__name__}"

def execute_email_tool(email: dict):
    approved = {
        "to": email["recipient"],
        "subject": email["subject"],
        "body": email["body"]
    }

    response = client.responses.create(
        model=MODEL,
        temperature=TEMPERATURE,
        input=[
            {
                "role": "system",
                "content": (
                    "사용자가 아래 이메일을 최종 승인했습니다. "
                    "send_email 도구를 정확히 한 번 호출하고 "
                    "값을 수정하지 마세요."
                )
            },
            {
                "role": "user",
                "content": json.dumps(approved, ensure_ascii=False)
            }
        ],
        tools=[EMAIL_TOOL],
        tool_choice={"type": "function", "name": "send_email"}
    )

    calls = [
        item for item in response.output
        if item.type == "function_call" and item.name == "send_email"
    ]

    if len(calls) != 1:
        return False, "이메일 도구 호출에 실패했습니다."

    try:
        args = json.loads(calls[0].arguments)
    except Exception:
        return False, "이메일 도구 인자 분석에 실패했습니다."

    if args != approved:
        return False, "승인된 내용과 달라 발송을 중단했습니다."

    return send_email(args["to"], args["subject"], args["body"])

# ============================================================
# 6. 상태 초기화
# ============================================================

DEFAULTS = {
    "messages": [],
    "meeting_text": "",
    "meeting_hash": None,
    "meeting_name": "",
    "pending_email": None,
    "sent_mail_ids": set(),
    "last_answer": "",
    "last_audio": None,
    "last_stt": "",
    "stt_pending": None,
    "stt_editor": "",
    "analysis": None,
    "autoplay_pending": False,
    "notice": "",
    "answer_style": "기본"
}

for k, v in DEFAULTS.items():
    if k not in st.session_state:
        if isinstance(v, (list, dict, set)):
            st.session_state[k] = v.copy()
        else:
            st.session_state[k] = v

def reset_conversation():
    st.session_state.messages = []
    st.session_state.pending_email = None
    st.session_state.last_answer = ""
    st.session_state.last_audio = None
    st.session_state.last_stt = ""
    st.session_state.stt_pending = None
    st.session_state.stt_editor = ""
    st.session_state.analysis = None
    st.session_state.autoplay_pending = False
    st.session_state.notice = ""

# ============================================================
# 7. STT / TTS
# ============================================================

STT_HINT = """
이 음성은 이찬우가 개인비서 에이블에게 요청하는 한국어 업무 명령입니다.
주요 용어:
에이블, 이찬우, 회의록, 회의 요약, 담당 업무, 마감일,
이메일, 메일 발송, 발송 승인, 데이터 분석, 마케팅 전략, KPI.
"""

def speech_to_text(audio):
    response = client.audio.transcriptions.create(
        model="whisper-1",
        file=("recording.wav", audio.getvalue(), "audio/wav"),
        language="ko",
        temperature=0,
        prompt=STT_HINT
    )
    return response.text.strip()

def text_to_speech(text: str):
    response = client.audio.speech.create(
        model="tts-1",
        voice="alloy",
        input=text[:4000]
    )
    return response.content

# ============================================================
# 8. 프롬프트 / 분석
# ============================================================

def build_system_prompt():
    style = st.session_state.answer_style
    style_guide = {
        "간결": "답변은 핵심 위주로 짧고 간결하게 작성합니다.",
        "기본": "답변은 핵심과 필요한 설명을 균형 있게 작성합니다.",
        "상세": "답변은 항목별로 자세히 작성하되 지나치게 장황하지 않게 정리합니다."
    }.get(style, "답변은 균형 있게 작성합니다.")

    system_content = SYS_ROLE + "\n\n[추가 스타일 지침]\n" + style_guide

    if st.session_state.meeting_text:
        system_content += (
            "\n\n[현재 회의록]\n<meeting_record>\n"
            + st.session_state.meeting_text +
            "\n</meeting_record>"
        )
    else:
        system_content += "\n\n현재 업로드된 회의록이 없습니다. 회의록 내용을 추측하지 마세요."

    if st.session_state.pending_email:
        system_content += (
            "\n\n[미완료 이메일 요청]\n"
            + json.dumps(st.session_state.pending_email, ensure_ascii=False)
        )

    return system_content

def analyze_request(question: str):
    input_messages = [{"role": "system", "content": build_system_prompt()}]
    input_messages.extend(st.session_state.messages[-20:])
    input_messages.append({"role": "user", "content": question})

    response = client.responses.parse(
        model=MODEL,
        temperature=TEMPERATURE,
        input=input_messages,
        text_format=AIResult
    )

    result = response.output_parsed
    if result is None:
        raise RuntimeError("AI 응답 생성에 실패했습니다.")
    return result

# ============================================================
# 9. 질문 처리
# ============================================================

def process_question(question: str, enable_tts: bool = True):
    question = question.strip()
    if not question:
        st.warning("질문을 입력해주세요.")
        return

    with st.spinner("에이블이 요청을 분석하고 있습니다..."):
        result = analyze_request(question)

    answer = result.answer
    previous_mail = st.session_state.pending_email

    if result.intent == "email_send":
        mail = result.email.model_dump()

        if previous_mail:
            for key in ["recipient", "subject", "body"]:
                if not mail.get(key) and previous_mail.get(key):
                    mail[key] = previous_mail[key]

        st.session_state.pending_email = mail

        if not valid_email(mail.get("recipient", "")):
            answer = result.clarification_question or "수신자 이메일 주소를 알려주세요."
        elif not mail.get("subject", "").strip() or not mail.get("body", "").strip():
            answer = "이메일 제목과 본문을 확인해주세요."
        else:
            answer = "이메일 발송 준비가 완료되었습니다. 아래에서 내용을 확인하고 승인해주세요."

    elif result.intent == "clarification" and previous_mail:
        pass
    else:
        st.session_state.pending_email = None

    st.session_state.messages.extend([
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer}
    ])
    st.session_state.messages = st.session_state.messages[-20:]

    st.session_state.last_answer = answer
    st.session_state.analysis = result.model_dump()
    st.session_state.last_audio = None
    st.session_state.autoplay_pending = False

    if enable_tts:
        voice_text = answer if result.intent == "email_send" else (result.voice_answer or answer)
        try:
            with st.spinner("음성 답변을 생성하고 있습니다..."):
                st.session_state.last_audio = text_to_speech(voice_text)
            st.session_state.autoplay_pending = True
        except Exception:
            st.session_state.notice = "텍스트 답변은 생성됐지만 음성 생성에 실패했습니다."

# ============================================================
# 10. UI 스타일
# ============================================================

st.markdown("""
<style>
.block-container {
    max-width: 1180px;
    padding-top: 1.2rem;
    padding-bottom: 3rem;
}
.hero-box {
    background: linear-gradient(135deg, #f8feff, #eefbff);
    border: 1px solid #d7eef7;
    border-radius: 20px;
    padding: 20px;
}
.hero-title {
    font-size: 44px;
    font-weight: 800;
    color: #183244;
    line-height: 1.2;
}
.hero-sub {
    font-size: 17px;
    color: #607283;
    margin-top: 8px;
}
.badge {
    display: inline-block;
    padding: 6px 12px;
    border-radius: 999px;
    background: #eaf7ff;
    color: #2a647e;
    font-size: 13px;
    margin-right: 8px;
    margin-top: 10px;
}
.chat-shell {
    background: #fbfdff;
    border: 1px solid #e3eef5;
    border-radius: 18px;
    padding: 16px;
}
.otter-chat-header {
    display: flex;
    align-items: center;
    gap: 16px;
    background: #f7fcff;
    border: 1px solid #dbeef6;
    border-radius: 16px;
    padding: 14px 16px;
    margin-bottom: 14px;
}
.otter-avatar {
    width: 74px;
    height: 74px;
    min-width: 74px;
    border-radius: 18px;
    border: 1px solid #cae7f2;
    background: #eaf7ff;
    display: flex;
    align-items: center;
    justify-content: center;
}
.otter-bubble-title {
    font-size: 22px;
    font-weight: 700;
    color: #183244;
}
.otter-bubble-sub {
    font-size: 14px;
    color: #637584;
    margin-top: 4px;
}
.side-status-note {
    font-size: 13px;
    color: #677888;
}
.stButton > button {
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# 11. 사이드바
# ============================================================

with st.sidebar:
    st.title("🦦 에이블 설정")

    st.markdown("### 📌 상태")

    with st.expander("📄 회의록 상태", expanded=True):
        uploaded_file = st.file_uploader("회의록 TXT 업로드", type=["txt"])

        if uploaded_file is not None:
            raw = uploaded_file.getvalue()
            current_hash = hashlib.sha256(raw).hexdigest()

            if current_hash != st.session_state.meeting_hash:
                try:
                    text = raw.decode("utf-8-sig")
                    st.session_state.meeting_text = text
                    st.session_state.meeting_hash = current_hash
                    st.session_state.meeting_name = uploaded_file.name
                    reset_conversation()
                    st.success("새 회의록을 등록했습니다.")
                except UnicodeDecodeError:
                    st.error("UTF-8 형식의 TXT 파일을 사용해주세요.")
                    st.stop()

        if st.session_state.meeting_text:
            st.success("회의록 준비 완료")
            st.write(f"파일명: **{st.session_state.meeting_name}**")
            with st.expander("회의록 원문 보기"):
                st.text(st.session_state.meeting_text[:12000])

            if st.button("회의록 제거", use_container_width=True):
                st.session_state.meeting_text = ""
                st.session_state.meeting_hash = None
                st.session_state.meeting_name = ""
                reset_conversation()
                st.rerun()
        else:
            st.info("현재 업로드된 회의록이 없습니다.")

    with st.expander("💬 대화 기억", expanded=False):
        st.write(f"현재 저장된 질문·답변 쌍: **{len(st.session_state.messages)//2} / 10**")
        st.markdown('<div class="side-status-note">최근 10개의 질문·답변 쌍만 유지합니다.</div>', unsafe_allow_html=True)

    with st.expander("🧠 응답 설정", expanded=False):
        st.session_state.answer_style = st.radio(
            "응답 방식 선택",
            ["간결", "기본", "상세"],
            index=["간결", "기본", "상세"].index(st.session_state.answer_style)
        )
        st.write(f"모델: **{MODEL}**")
        st.write("Temperature: **0**")
        st.write("Structured Outputs: **적용**")

    with st.expander("📧 이메일 상태", expanded=False):
        mail_ready = bool(get_secret("GMAIL_ADDRESS") and get_secret("GMAIL_APP_PASSWORD"))
        st.write("이메일 기능:", "사용 가능 ✅" if mail_ready else "설정 필요 ⚠️")
        st.markdown('<div class="side-status-note">실제 발송은 사용자 승인 후 진행됩니다.</div>', unsafe_allow_html=True)

    st.divider()

    st.markdown("### 🔊 음성 설정")
    enable_tts = st.toggle("AI 음성 답변 사용", value=True)
    auto_play = st.toggle("답변 즉시 자동재생", value=True)
    st.caption("브라우저 정책에 따라 자동재생이 차단될 수 있습니다.")

    st.divider()

    if st.button("🗑️ 대화 초기화", use_container_width=True):
        reset_conversation()
        st.rerun()

    if st.session_state.messages:
        chat_text = "\n\n".join(
            ("사용자: " if m["role"] == "user" else "에이블: ") + m["content"]
            for m in st.session_state.messages
        )
        st.download_button(
            "📥 대화 내역 다운로드",
            data=chat_text,
            file_name="aivle_history.txt",
            mime="text/plain",
            use_container_width=True
        )

# ============================================================
# 12. 메인 헤더
# ============================================================

col_img, col_head = st.columns([1, 2.2])

with col_img:
    st.markdown(
        f"""
        <div class="hero-box" style="display:flex; justify-content:center; align-items:center; min-height:300px;">
            <img src="{OTTER_IMAGE}" width="250">
        </div>
        """,
        unsafe_allow_html=True
    )

with col_head:
    st.markdown('<div class="hero-box">', unsafe_allow_html=True)
    st.markdown('<div class="hero-title">🦦 에이블 | 이찬우의 개인비서</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">회의록 분석 · 담당 업무 정리 · 음성 인식 · 이메일 작성 및 발송 지원</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <span class="badge">회의 요약</span>
        <span class="badge">업무 추출</span>
        <span class="badge">STT / TTS</span>
        <span class="badge">이메일 발송</span>
        <span class="badge">멀티턴 대화</span>
        """,
        unsafe_allow_html=True
    )
    st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# 13. 추천 명령
# ============================================================

st.divider()

with st.expander("💡 추천 명령 버튼", expanded=True):
    recommended_commands = [
        ("🎤 회의 핵심 요약", "오늘 회의의 핵심 내용을 요약해줘."),
        ("📋 내 담당 업무", "내가 맡은 업무와 마감일을 자세히 알려줘."),
        ("✅ 최종 결정사항", "이번 회의에서 최종 확정된 사항만 정리해줘."),
        ("📧 이메일 초안", "오늘 회의 요약을 이메일 초안으로 작성해줘."),
        ("📨 이메일 발송", "오늘 회의 핵심 내용을 이메일로 보내줘."),
        ("🔎 업무 상세 분석", "내 담당 업무를 근거와 함께 자세히 설명해줘.")
    ]

    cols = st.columns(2)
    for i, (label, command) in enumerate(recommended_commands):
        with cols[i % 2]:
            if st.button(label, key=f"quick_{i}", use_container_width=True):
                try:
                    process_question(command, enable_tts=enable_tts)
                except Exception as e:
                    st.error(f"명령 실행 오류: {e}")

# ============================================================
# 14. 채팅형 대화 영역
# ============================================================

st.divider()
st.subheader("💬 에이블과 대화하기")

st.markdown('<div class="chat-shell">', unsafe_allow_html=True)

st.markdown(
    f"""
    <div class="otter-chat-header">
        <div class="otter-avatar">
            <img src="{OTTER_IMAGE}" width="60">
        </div>
        <div>
            <div class="otter-bubble-title">에이블과 대화 중</div>
            <div class="otter-bubble-sub">음성 또는 텍스트로 말을 걸면 회의 요약, 업무 정리, 이메일 작성까지 도와드릴게요.</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

if not st.session_state.messages:
    st.info("회의록을 업로드한 뒤 추천 명령 버튼을 누르거나, 아래에서 텍스트/음성으로 질문해보세요.")

for message in st.session_state.messages:
    if message["role"] == "user":
        with st.chat_message("user", avatar="🙂"):
            st.markdown(message["content"])
    else:
        with st.chat_message("assistant", avatar="🦦"):
            st.markdown(message["content"])

st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# 15. 텍스트 입력
# ============================================================

user_prompt = st.chat_input("에이블에게 메시지를 보내보세요...")

if user_prompt:
    try:
        process_question(user_prompt.strip(), enable_tts=enable_tts)
        st.rerun()
    except Exception as e:
        st.error(f"AI 응답 오류: {e}")

# ============================================================
# 16. 음성 입력
# ============================================================

with st.expander("🎙️ 음성으로 말하기", expanded=False):
    st.caption("녹음 후 음성 인식 결과를 확인하고 수정한 뒤 실행할 수 있습니다.")
    audio_value = st.audio_input("무엇을 도와드릴까요?")
    if audio_value is not None:
        if st.button("🎧 음성 인식 실행", use_container_width=True):
            try:
                with st.spinner("음성을 인식하고 있습니다..."):
                    transcript = speech_to_text(audio_value)
                st.session_state.last_stt = transcript
                st.session_state.stt_pending = transcript
                st.session_state.stt_editor = transcript
                st.rerun()
            except Exception as e:
                st.error(f"음성 인식 오류: {e}")

# ============================================================
# 17. STT 결과 수정
# ============================================================

if st.session_state.stt_pending is not None:
    st.divider()
    with st.container(border=True):
        st.subheader("🎤 음성 인식 결과 확인")
        st.info("인식된 문장을 확인하고, 잘못된 내용이 있으면 수정한 뒤 에이블에게 보내세요.")

        corrected_text = st.text_area(
            "인식된 문장",
            key="stt_editor",
            height=120
        )

        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("✅ 이 문장으로 대화하기", type="primary", use_container_width=True):
                if corrected_text.strip():
                    try:
                        st.session_state.stt_pending = None
                        process_question(corrected_text, enable_tts=enable_tts)
                        st.rerun()
                    except Exception as e:
                        st.error(f"AI 처리 오류: {e}")
                else:
                    st.warning("인식된 문장을 입력해주세요.")

        with col_b:
            if st.button("❌ 취소", use_container_width=True):
                st.session_state.stt_pending = None
                st.rerun()

# ============================================================
# 18. TTS 자동재생
# ============================================================

if st.session_state.last_audio:
    st.markdown("#### 🔊 에이블 음성 답변")
    should_autoplay = auto_play and st.session_state.autoplay_pending
    st.audio(st.session_state.last_audio, format="audio/mp3", autoplay=should_autoplay)
    st.session_state.autoplay_pending = False

if st.session_state.notice:
    st.warning(st.session_state.notice)
    st.session_state.notice = ""

# ============================================================
# 19. 이메일 승인
# ============================================================

if st.session_state.pending_email:
    email = st.session_state.pending_email

    st.divider()
    with st.container(border=True):
        st.subheader("📧 이메일 발송 확인")
        st.caption("아래 내용을 확인하고 승인하면 실제 이메일이 발송됩니다.")

        with st.form("email_approval_form"):
            recipient = st.text_input("수신자", value=email.get("recipient", ""))
            subject = st.text_input("제목", value=email.get("subject", ""))
            body = st.text_area("본문", value=email.get("body", ""), height=240)

            send_col, cancel_col = st.columns(2)
            with send_col:
                send_submit = st.form_submit_button("✅ 이메일 발송 승인", type="primary", use_container_width=True)
            with cancel_col:
                cancel_submit = st.form_submit_button("❌ 취소", use_container_width=True)

        if cancel_submit:
            st.session_state.pending_email = None
            st.rerun()

        if send_submit:
            approved_mail = {
                "recipient": recipient.strip(),
                "subject": subject.strip(),
                "body": body.strip()
            }

            if not valid_email(approved_mail["recipient"]):
                st.error("올바른 수신자 이메일 주소를 입력해주세요.")
            elif not approved_mail["subject"] or not approved_mail["body"]:
                st.error("제목과 본문을 모두 입력해주세요.")
            else:
                mail_id = hashlib.sha256(
                    json.dumps(approved_mail, sort_keys=True, ensure_ascii=False).encode()
                ).hexdigest()

                if mail_id in st.session_state.sent_mail_ids:
                    st.warning("현재 세션에서 이미 발송한 동일한 이메일입니다.")
                else:
                    with st.spinner("이메일을 발송하고 있습니다..."):
                        try:
                            success, message = execute_email_tool(approved_mail)
                            if success:
                                st.session_state.sent_mail_ids.add(mail_id)
                                st.session_state.pending_email = None
                                st.success(message)
                            else:
                                st.error(message)
                        except Exception as e:
                            st.error(f"이메일 처리 오류: {type(e).__name__}")

# ============================================================
# 20. 분석 결과 / 다운로드
# ============================================================

if st.session_state.analysis:
    with st.expander("🔎 AI 구조화 분석 결과 보기", expanded=False):
        st.json(st.session_state.analysis)

if st.session_state.last_answer:
    st.download_button(
        "📥 마지막 답변 다운로드",
        data=st.session_state.last_answer,
        file_name="last_answer.txt",
        mime="text/plain"
    )

st.divider()
st.caption("에이블 AI 개인비서 | Whisper STT · GPT 응답 · TTS · 이메일 Tool Calling · 멀티턴 대화")
