"""StudBud: upload a book PDF, get explanations, quizzes and answers.

Run:  streamlit run app.py
"""

import base64
import io
import json
import os

import anthropic
import streamlit as st
from pypdf import PdfReader, PdfWriter

MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-5-5")
MAX_PAGES_PER_REQUEST = 20  # keeps cost and size under control

GRADES = [
    "Grade 1-3", "Grade 4-6", "Grade 7-9", "Grade 10-12",
    "University / College", "Adult learner",
]
LANGUAGES = [
    "English", "Arabic", "Spanish", "French", "German", "Portuguese",
    "Turkish", "Hindi", "Urdu", "Indonesian", "Russian",
    "Chinese (Simplified)", "Japanese", "Korean", "Italian",
]
RTL_LANGUAGES = {"Arabic", "Urdu"}


# ---------- helpers ----------

def slice_pdf(file_bytes: bytes, start: int, end: int) -> bytes:
    """Return a new PDF containing pages start..end (1-indexed, inclusive)."""
    reader = PdfReader(io.BytesIO(file_bytes))
    writer = PdfWriter()
    for i in range(start - 1, end):
        writer.add_page(reader.pages[i])
    out = io.BytesIO()
    writer.write(out)
    return out.getvalue()


def build_system_prompt(grade: str, language: str, country: str, subject: str) -> str:
    parts = [
        "You are StudBud, a patient, friendly AI tutor. If asked your name, say you are StudBud.",
        f"The student's level is: {grade}.",
        f"Write ALL of your output in {language}, even if the book is in another language.",
        "Base everything strictly on the pages provided. If something is not in the pages, say so.",
        "Write math with plain readable notation. Describe diagrams and charts in words when they matter.",
    ]
    if country.strip():
        parts.append(
            f"The student studies in {country.strip()}. Use familiar examples and units for that place."
        )
    if subject.strip():
        parts.append(f"The subject is: {subject.strip()}.")
    else:
        parts.append("Work out the subject from the pages and teach in a way that suits it.")
    return " ".join(parts)


def pdf_block(pdf_bytes: bytes) -> dict:
    return {
        "type": "document",
        "source": {
            "type": "base64",
            "media_type": "application/pdf",
            "data": base64.standard_b64encode(pdf_bytes).decode("utf-8"),
        },
    }


def ask_claude(client, system: str, pdf_bytes: bytes, history: list, max_tokens: int = 3000) -> str:
    """history: list of {"role": "user"|"assistant", "content": str}.
    The PDF is attached to the first user message."""
    messages = []
    for i, turn in enumerate(history):
        if i == 0:
            content = [pdf_block(pdf_bytes), {"type": "text", "text": turn["content"]}]
        else:
            content = turn["content"]
        messages.append({"role": turn["role"], "content": content})

    response = client.messages.create(
        model=MODEL,
        max_tokens=max_tokens,
        system=system,
        messages=messages,
    )
    return "".join(b.text for b in response.content if b.type == "text").strip()


def parse_quiz(raw: str) -> list:
    """Extract and validate a JSON quiz from the model's reply."""
    start, end = raw.find("["), raw.rfind("]")
    if start == -1 or end == -1 or end < start:
        raise ValueError("No JSON list found in the reply.")
    data = json.loads(raw[start:end + 1])
    quiz = []
    for q in data:
        options = q["options"]
        idx = int(q["answer_index"])
        if not isinstance(options, list) or len(options) < 2 or not 0 <= idx < len(options):
            raise ValueError("A question had invalid options or answer.")
        quiz.append({
            "question": str(q["question"]),
            "options": [str(o) for o in options],
            "answer_index": idx,
            "explanation": str(q.get("explanation", "")),
        })
    if not quiz:
        raise ValueError("The quiz was empty.")
    return quiz


def rtl_wrap(text: str, language: str) -> str:
    if language in RTL_LANGUAGES:
        return f'<div dir="rtl" style="text-align:right">\n\n{text}\n\n</div>'
    return text


# ---------- page ----------

st.set_page_config(page_title="StudBud", page_icon="🎓", layout="wide")
st.title("🎓 StudBud")
st.caption("Upload a few pages of your textbook. Get explanations, quizzes and answers.")

with st.sidebar:
    st.header("About you")
    grade = st.selectbox("Your level", GRADES, index=2)
    language = st.selectbox("Explain in", LANGUAGES)
    country = st.text_input("Country / curriculum (optional)", placeholder="e.g. Egypt, Mexico, UK GCSE")
    subject = st.text_input("Subject (optional)", placeholder="Leave empty to auto-detect")
    st.divider()
    api_key = st.text_input(
        "Anthropic API key",
        type="password",
        value=os.getenv("ANTHROPIC_API_KEY", ""),
        help="Get one at console.anthropic.com. Never share it.",
    )

uploaded = st.file_uploader("Upload your book (PDF)", type=["pdf"])

if not uploaded:
    st.info("Upload a PDF to begin.")
    st.stop()

file_bytes = uploaded.getvalue()
try:
    total_pages = len(PdfReader(io.BytesIO(file_bytes)).pages)
except Exception:
    st.error("Could not read this PDF. It may be corrupted or password-protected.")
    st.stop()

st.write(f"**{uploaded.name}**: {total_pages} pages")

col1, col2 = st.columns(2)
start_page = col1.number_input("From page", min_value=1, max_value=total_pages, value=1, step=1)
end_default = min(total_pages, int(start_page) + 4)
end_page = col2.number_input(
    "To page", min_value=1, max_value=total_pages, value=end_default, step=1
)

start_page, end_page = int(start_page), int(end_page)
if end_page < start_page:
    st.error("'To page' must be the same as or after 'From page'.")
    st.stop()
if end_page - start_page + 1 > MAX_PAGES_PER_REQUEST:
    st.error(f"Please choose at most {MAX_PAGES_PER_REQUEST} pages at a time.")
    st.stop()
if not api_key:
    st.warning("Enter your Anthropic API key in the sidebar.")
    st.stop()

client = anthropic.Anthropic(api_key=api_key)
system_prompt = build_system_prompt(grade, language, country, subject)

# Reset saved results when the selected pages change
selection_key = (uploaded.name, len(file_bytes), start_page, end_page)
if st.session_state.get("selection_key") != selection_key:
    st.session_state.selection_key = selection_key
    st.session_state.explanation = None
    st.session_state.quiz = None
    st.session_state.quiz_result = None
    st.session_state.chat = []

try:
    pdf_slice = slice_pdf(file_bytes, start_page, end_page)
except Exception as e:
    st.error(f"Could not extract those pages: {e}")
    st.stop()

tab_explain, tab_quiz, tab_chat = st.tabs(["Explain", "Quiz", "Ask a question"])

# --- Explain ---
with tab_explain:
    if st.button("Explain these pages", type="primary"):
        with st.spinner("Reading and explaining..."):
            try:
                st.session_state.explanation = ask_claude(
                    client, system_prompt, pdf_slice,
                    [{"role": "user", "content": (
                        "Explain these pages to me step by step. Start with a short summary, "
                        "then explain the key ideas simply with examples, and end with "
                        "a list of the most important terms."
                    )}],
                )
            except anthropic.APIError as e:
                st.error(f"API error: {e}")
    if st.session_state.explanation:
        st.markdown(rtl_wrap(st.session_state.explanation, language), unsafe_allow_html=True)

# --- Quiz ---
with tab_quiz:
    num_q = st.slider("Number of questions", 3, 10, 5)
    if st.button("Make a quiz"):
        with st.spinner("Writing questions..."):
            try:
                raw = ask_claude(
                    client, system_prompt, pdf_slice,
                    [{"role": "user", "content": (
                        f"Write {num_q} multiple-choice questions testing understanding of these pages. "
                        "Reply with ONLY a JSON list, no other text and no code fences. "
                        'Each item must look like: {"question": "...", "options": ["...", "...", "...", "..."], '
                        '"answer_index": 0, "explanation": "..."}. '
                        "answer_index is the zero-based position of the correct option. "
                        "Use exactly 4 options and vary which position is correct."
                    )}],
                )
                st.session_state.quiz = parse_quiz(raw)
                st.session_state.quiz_result = None
            except anthropic.APIError as e:
                st.error(f"API error: {e}")
            except (ValueError, KeyError, TypeError, json.JSONDecodeError):
                st.session_state.quiz = None
                st.error("The quiz came back in a broken format. Please click 'Make a quiz' again.")

    quiz = st.session_state.quiz
    if quiz:
        with st.form("quiz_form"):
            picks = []
            for i, q in enumerate(quiz):
                picks.append(st.radio(
                    f"{i + 1}. {q['question']}",
                    options=list(range(len(q["options"]))),
                    format_func=lambda k, opts=q["options"]: opts[k],
                    index=None,
                    key=f"q_{i}",
                ))
            submitted = st.form_submit_button("Check answers")
        if submitted:
            st.session_state.quiz_result = picks

        result = st.session_state.quiz_result
        if result is not None:
            score = sum(1 for p, q in zip(result, quiz) if p == q["answer_index"])
            st.subheader(f"Score: {score} / {len(quiz)}")
            for i, (p, q) in enumerate(zip(result, quiz)):
                correct_text = q["options"][q["answer_index"]]
                if p == q["answer_index"]:
                    st.success(f"{i + 1}. Correct. {q['explanation']}")
                elif p is None:
                    st.warning(f"{i + 1}. Not answered. Correct answer: {correct_text}. {q['explanation']}")
                else:
                    st.error(f"{i + 1}. Wrong. Correct answer: {correct_text}. {q['explanation']}")

# --- Chat ---
with tab_chat:
    for turn in st.session_state.chat:
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])

    question = st.chat_input("Ask anything about these pages")
    if question:
        st.session_state.chat.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    answer = ask_claude(client, system_prompt, pdf_slice, st.session_state.chat)
                    st.markdown(answer)
                    st.session_state.chat.append({"role": "assistant", "content": answer})
                except anthropic.APIError as e:
                    st.error(f"API error: {e}")
                    st.session_state.chat.pop()  # keep roles alternating
