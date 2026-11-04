
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
    page_title="에이블 | 김 대리의 AI 개인비서",
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
<svg xmlns="http://www.w3.org/2000/svg" width="320" height="360" viewBox="0 0 320 360">
  <defs>
    <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#eef9ff"/>
      <stop offset="100%" stop-color="#dff5ff"/>
    </linearGradient>
    <linearGradient id="fur" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#b9784f"/>
      <stop offset="100%" stop-color="#8f5b39"/>
    </linearGradient>
    <linearGradient id="belly" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#ffe8cf"/>
      <stop offset="100%" stop-color="#f4d6b8"/>
    </linearGradient>
  </defs>

  <rect x="10" y="10" width="300" height="340" rx="28" fill="url(#bg)" stroke="#b5e7f6" stroke-width="3"/>
  
  <!-- bubble -->
  <rect x="154" y="34" width="126" height="74" rx="18" fill="white" stroke="#b5ddeb" stroke-width="2"/>
  <polygon points="168,95 152,106 159,84" fill="white" stroke="#b5ddeb" stroke-width="2"/>
  <text x="217" y="61" text-anchor="middle" font-size="16" font-family="Arial" fill="#3a4b57">안녕하세요!</text>
  <text x="217" y="84" text-anchor="middle" font-size="14" font-family="Arial" fill="#5c6d77">저는 에이블🦦</text>

  <!-- tail -->
  <ellipse cx="98" cy="286" rx="28" ry="52" fill="#8a5737" transform="rotate(-18 98 286)"/>

  <!-- body -->
  <ellipse cx="150" cy="218" rx="72" ry="88" fill="url(#fur)"/>
  <ellipse cx="150" cy="234" rx="48" ry="62" fill="url(#belly)"/>

  <!-- feet -->
  <ellipse cx="116" cy="304" rx="23" ry="15" fill="#73492f"/>
  <ellipse cx="184" cy="304" rx="23" ry="15" fill="#73492f"/>

  <!-- arms -->
  <ellipse cx="88" cy="220" rx="22" ry="48" fill="url(#fur)" transform="rotate(24 88 220)"/>
  <ellipse cx="212" cy="220" rx="22" ry="48" fill="url(#fur)" transform="rotate(-24 212 220)"/>

  <!-- clipboard -->
  <rect x="203" y="193" width="54" height="74" rx="10" fill="#f8fbff" stroke="#93d2e8" stroke-width="3"/>
  <rect x="220" y="186" width="19" height="10" rx="4" fill="#93d2e8"/>
  <line x1="213" y1="211" x2="246" y2="211" stroke="#8cb8c8" stroke-width="3"/>
  <line x1="213" y1="223" x2="246" y2="223" stroke="#8cb8c8" stroke-width="3"/>
  <line x1="213" y1="235" x2="241" y2="235" stroke="#8cb8c8" stroke-width="3"/>
  <line x1="213" y1="247" x2="236" y2="247" stroke="#8cb8c8" stroke-width="3"/>

  <!-- head -->
  <circle cx="150" cy="106" r="66" fill="url(#fur)"/>

  <!-- ears -->
  <circle cx="103" cy="63" r="16" fill="#8f5b39"/>
  <circle cx="197" cy="63" r="16" fill="#8f5b39"/>
  <circle cx="103" cy="63" r="8" fill="#f7d8bf"/>
  <circle cx="197" cy="63" r="8" fill="#f7d8bf"/>

  <!-- face -->
  <ellipse cx="150" cy="118" rx="42" ry="34" fill="#f6dcc4"/>
  <circle cx="126" cy="103" r="8" fill="#222"/>
  <circle cx="174" cy="103" r="8" fill="#222"/>
  <circle cx="123" cy="100" r="2.2" fill="white"/>
  <circle cx="171" cy="100" r="2.2" fill="white"/>
  <ellipse cx="150" cy="118" rx="8" ry="6" fill="#4a3428"/>
  <path d="M136 130 Q150 143 164 130" fill="none" stroke="#4a3428" stroke-width="4" stroke-linecap="round"/>
  <circle cx="118" cy="120" r="4" fill="#f2b5ab" opacity="0.8"/>
  <circle cx="182" cy="120" r="4" fill="#f2b5ab" opacity="0.8"/>

  <!-- chest badge -->
  <rect x="123" y="183" width="54" height="22" rx="11" fill="#87d7ef"/>
  <text x="150" y="198" text-anchor="middle" font-size="12" font-family="Arial" fill="#1f5060">AIBLE</text>
</svg>
"""

OTTER_IMAGE = "data:image/svg+xml;base64," + base64.b64encode(OTTER_SVG.encode("utf-8")).decode("utf-8")

# ============================================================
# 3. 시스템 프롬프트
# ============================================================

SYS_ROLE = """
[정체성]
당신은 김 대리의 전담 AI 개인비서 '에이블'입니다.
당신은 회의 내용을 분석하고, 김 대리의 업무를 정리하며,
필요할 경우 이메일 초안을 작성하고 발송을 돕는 전문 AI 비서입니다.

[핵심 목표]
1. 회의 주제와 핵심 내용을 정확히 요약합니다.
2. 회의의 논의사항, 제안사항, 최종 결정사항을 구분합니다.
3. 김 대리에게 명시적으로 배정된 업무를 추출합니다.
4. 담당 업무의 구체적인 내용, 마감일, 지시자, 근거를 정리합니다.
5. 이메일 초안을 작성하고, 사용자가 발송을 요청한 경우에만 발송용 정보를 구성합니다.
6. 멀티턴 대화에서 이전 문맥을 활용합니다.

[정확성 원칙]
- 회의록에 없는 사실은 만들지 않습니다.
- 담당자, 마감일, 수치, 이메일 주소를 임의로 생성하지 않습니다.
- 김 대리의 업무와 다른 사람의 업무를 혼동하지 않습니다.
- 정보가 불충분하면 '회의록에서 확인되지 않습니다'라고 답합니다.
- 사용자의 질문에 직접적으로 먼저 답하고, 필요한 경우만 보충 설명합니다.

[회의록 분석 기준]
회의록을 분석할 때 다음을 고려합니다.
1. 회의 주제
2. 핵심 논의 내용
3. 최종 결정사항
4. 김 대리의 담당 업무
5. 업무별 상세 설명
6. 명시된 마감일
7. 지시자
8. 추가 확인사항
9. 근거 발언 요지

[요청 의도 분류]
다음 중 하나로 분류합니다.
- meeting_summary: 회의 요약
- my_tasks: 김 대리 담당 업무
- meeting_qa: 회의 관련 질문
- general_qa: 일반 질문
- email_draft: 이메일 초안 작성
- email_send: 실제 이메일 발송 요청
- clarification: 추가 정보 확인

[이메일 규칙]
- '메일 작성해줘'는 초안 생성입니다.
- '메일 보내줘', '이메일 발송해줘'는 실제 발송 요청입니다.
- 발송 요청이 아니면 절대로 발송이라고 판단하지 않습니다.
- 수신자가 없으면 확인 질문을 합니다.
- 사용자가 후속 대화에서 수신자만 알려주면 직전 이메일 요청과 연결할 수 있습니다.
- 메일 내용은 회의록과 사용자 요청에 근거해 작성합니다.

[답변 스타일]
- 자연스럽고 전문적인 한국어를 사용합니다.
- 핵심 결론을 먼저 제시합니다.
- 필요하면 번호 목록으로 정리합니다.
- 음성 답변은 2~5문장 정도로 간결하게 작성합니다.
- 화면용 답변(answer)과 음성용 답변(voice_answer)은 같은 사실만 포함해야 합니다.
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
    answer: str = Field(description="사용자에게 보여줄 상세 답변")
    voice_answer: str = Field(description="자연스럽고 짧은 음성 답변")
    needs_clarification: bool
    clarification_question: str
    email: EmailInfo

# ============================================================
# 5. 이메일 전송 도구
# ============================================================

EMAIL_TOOL = {
    "type": "function",
    "name": "send_email",
    "description": "사용자가 최종 승인한 이메일을 발송합니다.",
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
                    "send_email 도구를 정확히 한 번 호출하고, "
                    "수신자/제목/본문 값을 수정하지 마세요."
                )
            },
            {
                "role": "user",
                "content": json.dumps(approved, ensure_ascii=False)
            }
        ],
        tools=[EMAIL_TOOL],
        tool_choice={
            "type": "function",
            "name": "send_email"
        }
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
        return False, "승인된 이메일 내용과 달라 발송을 중단했습니다."

    return send_email(args["to"], args["subject"], args["body"])

# ============================================================
# 6. Session State 초기화
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
이 음성은 김 대리가 AI 개인비서 에이블에게 업무를 요청하는 한국어 음성입니다.
주요 용어:
에이블, 김 대리, 김대리, 회의록, 회의 요약, 담당 업무, 마감일,
이메일, 메일 발송, 발송 승인, 데이터 분석, 마케팅 전략, KPI, 신제품.
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
# 8. LLM 분석
# ============================================================

def build_system_prompt() -> str:
    style = st.session_state.answer_style
    style_guide = {
        "간결": "답변은 핵심 위주로 짧고 간결하게 작성합니다.",
        "기본": "답변은 핵심과 필요한 설명을 균형 있게 작성합니다.",
        "상세": "답변은 항목별로 구체적으로 작성하되 장황하지 않게 정리합니다."
    }.get(style, "답변은 균형 있게 작성합니다.")

    system_content = SYS_ROLE + "\n\n[추가 스타일 지침]\n" + style_guide

    if st.session_state.meeting_text:
        system_content += (
            "\n\n[현재 회의록]\n"
            "<meeting_record>\n"
            + st.session_state.meeting_text +
            "\n</meeting_record>"
        )
    else:
        system_content += "\n\n현재 업로드된 회의록이 없습니다. 회의록 내용을 추측하지 마세요."

    if st.session_state.pending_email:
        system_content += (
            "\n\n[미완료 이메일 요청 참고]\n"
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
        raise RuntimeError("AI 구조화 응답 생성에 실패했습니다.")
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
            answer = "이메일 발송 준비가 완료되었습니다. 아래 승인 화면에서 내용을 확인해주세요."

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
    max-width: 1150px;
    padding-top: 1.2rem;
    padding-bottom: 3rem;
}
.status-card {
    background: #f8fbff;
    border: 1px solid #d9edf7;
    border-radius: 16px;
    padding: 18px 18px;
    min-height: 108px;
}
.status-title {
    color: #5b7083;
    font-size: 14px;
    margin-bottom: 8px;
}
.status-value {
    color: #122433;
    font-size: 26px;
    font-weight: 700;
    margin-bottom: 6px;
}
.status-sub {
    color: #6e7d8a;
    font-size: 13px;
}
.hero-box {
    background: linear-gradient(135deg, #f8feff, #eefbff);
    border: 1px solid #d8eef6;
    border-radius: 20px;
    padding: 20px;
}
.otter-name {
    font-size: 48px;
    font-weight: 800;
    color: #183042;
    line-height: 1.2;
}
.otter-sub {
    font-size: 17px;
    color: #617382;
    margin-top: 8px;
}
.helper-badge {
    display: inline-block;
    padding: 6px 12px;
    background: #e6f8ff;
    border-radius: 999px;
    color: #23627d;
    font-size: 13px;
    margin-right: 6px;
    margin-bottom: 8px;
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
    st.title("⚙️ 에이블 설정")

    st.subheader("📁 회의록 관리")
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
        st.caption(f"파일명: {st.session_state.meeting_name}")

        with st.expander("회의록 원문 보기"):
            st.text(st.session_state.meeting_text[:12000])

        if st.button("회의록 제거", use_container_width=True):
            st.session_state.meeting_text = ""
            st.session_state.meeting_hash = None
            st.session_state.meeting_name = ""
            reset_conversation()
            st.rerun()

    st.divider()

    st.subheader("🧠 답변 스타일")
    st.session_state.answer_style = st.radio(
        "응답 방식 선택",
        ["간결", "기본", "상세"],
        index=["간결", "기본", "상세"].index(st.session_state.answer_style)
    )

    st.divider()

    st.subheader("🔊 음성 설정")
    enable_tts = st.toggle("AI 음성 답변 사용", value=True)
    auto_play = st.toggle("답변 즉시 자동재생", value=True)
    st.caption("브라우저 정책에 따라 자동재생이 차단될 수 있습니다.")

    st.divider()

    st.subheader("📧 이메일 상태")
    mail_ready = bool(get_secret("GMAIL_ADDRESS") and get_secret("GMAIL_APP_PASSWORD"))
    st.write("이메일 기능:", "사용 가능 ✅" if mail_ready else "설정 필요 ⚠️")

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
# 12. 헤더
# ============================================================

col_img, col_head = st.columns([1, 2.2])

with col_img:
    st.markdown(
        f'<div class="hero-box" style="text-align:center;"><img src="{OTTER_IMAGE}" width="250"></div>',
        unsafe_allow_html=True
    )

with col_head:
    st.markdown('<div class="hero-box">', unsafe_allow_html=True)
    st.markdown('<div class="otter-name">🦦 에이블 | 김 대리의 AI 개인비서</div>', unsafe_allow_html=True)
    st.markdown('<div class="otter-sub">회의록 분석 · 담당 업무 추출 · 음성 인식 · 이메일 작성 및 발송</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div style="margin-top:14px;">
            <span class="helper-badge">회의 요약</span>
            <span class="helper-badge">김 대리 업무 추출</span>
            <span class="helper-badge">STT / TTS</span>
            <span class="helper-badge">이메일 발송</span>
            <span class="helper-badge">멀티턴 대화</span>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown('</div>', unsafe_allow_html=True)

st.divider()

# ============================================================
# 13. 상태 카드
# ============================================================

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown(
        f"""
        <div class="status-card">
            <div class="status-title">회의록 상태</div>
            <div class="status-value">{"등록 완료" if st.session_state.meeting_text else "미등록"}</div>
            <div class="status-sub">{st.session_state.meeting_name if st.session_state.meeting_name else "TXT 파일을 업로드하세요."}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with c2:
    st.markdown(
        f"""
        <div class="status-card">
            <div class="status-title">대화 기억</div>
            <div class="status-value">{len(st.session_state.messages)//2} / 10</div>
            <div class="status-sub">최근 10개의 질문·답변 쌍을 유지합니다.</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with c3:
    st.markdown(
        f"""
        <div class="status-card">
            <div class="status-title">AI 응답 설정</div>
            <div class="status-value">{st.session_state.answer_style}</div>
            <div class="status-sub">Temperature 0 · Structured Outputs 적용</div>
        </div>
        """,
        unsafe_allow_html=True
    )

# ============================================================
# 14. 추천 명령 버튼
# ============================================================

st.divider()

with st.expander("💡 추천 명령 버튼", expanded=True):
    recommended_commands = [
        ("🎤 회의 핵심 요약", "오늘 회의의 핵심 내용을 요약해줘."),
        ("📋 김 대리 담당 업무", "김 대리에게 배정된 업무와 마감일을 자세히 알려줘."),
        ("✅ 최종 결정사항", "이번 회의에서 최종 확정된 사항만 정리해줘."),
        ("📧 이메일 초안", "오늘 회의 요약을 이메일 초안으로 작성해줘."),
        ("📨 이메일 발송", "오늘 회의 핵심 내용을 이메일로 보내줘."),
        ("🔎 업무 상세 분석", "김 대리의 업무를 근거와 함께 자세히 설명해줘.")
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
# 15. 입력 영역
# ============================================================

st.divider()
st.subheader("🎙️ 에이블에게 업무 요청")

with st.form("task_form", clear_on_submit=True):
    audio_value = st.audio_input("무엇을 도와드릴까요?")
    text_input = st.text_input(
        "또는 텍스트로 질문하세요",
        placeholder="예: 오늘 회의에서 내가 맡은 업무 알려줘"
    )
    submit = st.form_submit_button("🚀 실행", use_container_width=True)

if submit:
    if audio_value is not None:
        try:
            with st.spinner("음성을 인식하고 있습니다..."):
                transcript = speech_to_text(audio_value)
            st.session_state.last_stt = transcript
            st.session_state.stt_pending = transcript
            st.session_state.stt_editor = transcript
            st.rerun()
        except Exception as e:
            st.error(f"음성 인식 오류: {e}")

    elif text_input.strip():
        try:
            process_question(text_input.strip(), enable_tts=enable_tts)
        except Exception as e:
            st.error(f"AI 응답 오류: {e}")
    else:
        st.warning("음성을 녹음하거나 텍스트 질문을 입력해주세요.")

# ============================================================
# 16. STT 결과 수정
# ============================================================

if st.session_state.stt_pending is not None:
    st.divider()
    with st.container(border=True):
        st.subheader("🎤 음성 인식 결과 확인")
        st.info("인식된 문장을 확인하고, 잘못된 내용이 있으면 수정한 뒤 실행하세요.")

        corrected_text = st.text_area(
            "인식된 문장",
            key="stt_editor",
            height=120
        )

        a, b = st.columns(2)
        with a:
            if st.button("✅ 이 문장으로 AI 실행", type="primary", use_container_width=True):
                if corrected_text.strip():
                    try:
                        st.session_state.stt_pending = None
                        process_question(corrected_text, enable_tts=enable_tts)
                        st.rerun()
                    except Exception as e:
                        st.error(f"AI 처리 오류: {e}")
                else:
                    st.warning("인식된 문장을 입력해주세요.")

        with b:
            if st.button("❌ 음성 명령 취소", use_container_width=True):
                st.session_state.stt_pending = None
                st.rerun()

# ============================================================
# 17. 대화 히스토리
# ============================================================

st.divider()
st.subheader("💬 AI 개인비서 대화")

if not st.session_state.messages:
    st.info("회의록을 업로드한 뒤 추천 명령 버튼을 누르거나, 음성/텍스트로 질문해보세요.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ============================================================
# 18. TTS 즉시 재생
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
# 19. 이메일 승인 UI
# ============================================================

if st.session_state.pending_email:
    email = st.session_state.pending_email

    st.divider()
    with st.container(border=True):
        st.subheader("📧 이메일 발송 확인")
        st.caption("아래 내용을 검토한 뒤 승인하면 실제 이메일이 발송됩니다.")

        with st.form("email_approval_form"):
            recipient = st.text_input("수신자", value=email.get("recipient", ""))
            subject = st.text_input("제목", value=email.get("subject", ""))
            body = st.text_area("본문", value=email.get("body", ""), height=240)

            col_send, col_cancel = st.columns(2)
            with col_send:
                send_submit = st.form_submit_button("✅ 이메일 발송 승인", type="primary", use_container_width=True)
            with col_cancel:
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
    with st.expander("🔎 AI 구조화 분석 결과 보기"):
        st.json(st.session_state.analysis)

if st.session_state.last_answer:
    st.download_button(
        "📥 마지막 답변 다운로드",
        data=st.session_state.last_answer,
        file_name="aivle_last_answer.txt",
        mime="text/plain"
    )

st.divider()
st.caption("에이블 AI 개인비서 | Whisper STT · GPT 응답 · TTS · 이메일 Tool Calling · 멀티턴 대화")
