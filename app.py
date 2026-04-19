import streamlit as st
from rag_engine import extract_text, build_vector_store, get_rag_chain, summarize_text
from quiz_generator import generate_quiz
from email_scheduler import start_scheduler

st.set_page_config(
    page_title="AI Study Assistant",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 AI Study Assistant")
st.caption("Upload your notes → Get summaries, quizzes, and a daily email quiz — powered by LLaMA 3")

# Session state
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "raw_text" not in st.session_state:
    st.session_state.raw_text = ""
if "quiz" not in st.session_state:
    st.session_state.quiz = []
if "answers" not in st.session_state:
    st.session_state.answers = {}
if "scheduler" not in st.session_state:
    st.session_state.scheduler = None

# Sidebar: Upload
with st.sidebar:
    st.header("Upload Study Material")
    uploaded = st.file_uploader("PDF or text file", type=["pdf"])

    if uploaded and st.button("Process Document"):
        with st.spinner("Extracting text and building vector store..."):
            text = extract_text(uploaded)
            st.session_state.raw_text = text
            st.session_state.vector_store = build_vector_store(text)
        st.success(f"Processed! ({len(text):,} characters)")

    st.markdown("---")
    st.header("Daily Email Quiz")
    email = st.text_input("Your email", placeholder="you@email.com")
    time_hour = st.slider("Send at (hour)", 6, 22, 8)
    if st.button("Start Daily Quiz"):
        if st.session_state.raw_text and email:
            st.session_state.scheduler = start_scheduler(
                email, st.session_state.raw_text, time_hour
            )
            st.success(f"Scheduled daily at {time_hour}:00!")
        else:
            st.error("Upload a document and enter email first.")

# Main tabs
tab1, tab2, tab3 = st.tabs(["Chat with Notes", "Quiz Mode", "Summarize"])

# Tab 1: RAG Chat
with tab1:
    st.subheader("Ask questions about your notes")
    if st.session_state.vector_store is None:
        st.info("Upload and process a document first.")
    else:
        question = st.text_input("Your question", placeholder="What is gradient descent?")
        if question:
            with st.spinner("Searching your notes..."):
                chain = get_rag_chain(st.session_state.vector_store)
                result = chain.invoke(question)
            st.markdown("**Answer:**")
            st.write(result)

# Tab 2: Interactive Quiz
with tab2:
    st.subheader("Test your knowledge")
    if st.session_state.raw_text == "":
        st.info("Upload and process a document first.")
    else:
        col1, col2 = st.columns([2, 1])
        with col1:
            num_q = st.slider("Number of questions", 3, 10, 5)
        with col2:
            if st.button("Generate Quiz", use_container_width=True):
                with st.spinner("Generating quiz with LLaMA 3..."):
                    st.session_state.quiz = generate_quiz(
                        st.session_state.raw_text, num_q
                    )
                st.session_state.answers = {}

        if st.session_state.quiz:
            score = 0
            for i, q in enumerate(st.session_state.quiz):
                st.markdown(f"**Q{i+1}. {q['question']}**")
                choice = st.radio(
                    "Select answer",
                    q["options"],
                    key=f"q_{i}",
                    label_visibility="collapsed"
                )
                st.session_state.answers[i] = choice

            if st.button("Submit Quiz", use_container_width=True):
                for i, q in enumerate(st.session_state.quiz):
                    selected = st.session_state.answers.get(i, "")
                    correct = q["answer"]
                    if selected.startswith(correct):
                        score += 1
                        st.success(f"Q{i+1}: Correct! {q['explanation']}")
                    else:
                        st.error(
                            f"Q{i+1}: Wrong. Correct: {correct}. {q['explanation']}"
                        )
                st.metric(
                    "Final Score",
                    f"{score}/{len(st.session_state.quiz)}",
                    f"{int(score/len(st.session_state.quiz)*100)}%"
                )

# Tab 3: Summarizer
with tab3:
    st.subheader("Get a smart summary of your notes")
    if st.session_state.raw_text == "":
        st.info("Upload and process a document first.")
    else:
        if st.button("Summarize Now", use_container_width=True):
            with st.spinner("LLaMA 3 is summarizing..."):
                summary = summarize_text(st.session_state.raw_text)
            st.markdown(summary)