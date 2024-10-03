import logging

import streamlit as st

from ui.initializer import customize, initialize
from ui.util import StreamlitLogHandler, create_log_handler

if __name__ == "__main__":

    st.set_page_config(layout="wide")
    # apparently, no way to run custom initialization before first page load in streamlit
    # https://discuss.streamlit.io/t/custom-init-on-streamlit-run-before-first-page-load/78011/4
    resources = initialize()
    customize()

    st.markdown(
        "<h1 style='text-align: center;'>RAG Chatbot</h1>", unsafe_allow_html=True
    )

    # Sezione INGESTION
    if "ongoing_ingestion" not in st.session_state:
        st.session_state["ongoing_ingestion"] = False

    def start_ingestion():
        st.session_state.ongoing_ingestion = True

    st.markdown(
        "<div style='background-color: #333333; color: white; padding: 10px;'>Ingestion</div>",
        unsafe_allow_html=True,
    )

    keyword_input = st.text_input(
        "Insert your keyword for the ingestion:", "", key="keyword"
    )

    log_container = st.empty()

    st.button(
        "Start Ingestion",
        on_click=start_ingestion,
        disabled=st.session_state.ongoing_ingestion,
    )

    if st.session_state.ongoing_ingestion:
        resources.setup_task_logger(
            handlers=[
                create_log_handler(
                    StreamlitLogHandler, resources.log_formatter, log_container.code
                ),
                create_log_handler(logging.StreamHandler, resources.log_formatter),
            ]
        )

        # starting ingestion
        resources.ingest(keyword=keyword_input)

        st.session_state.ongoing_ingestion = False
        resources.setup_task_logger(
            handlers=[
                create_log_handler(logging.StreamHandler, resources.log_formatter)
            ]
        )

        # Delete previous message history, if present
        if "messages" in st.session_state:
            st.session_state["messages"] = []

        if "user_question" in st.session_state:
            st.session_state.user_question = None

        st.experimental_rerun()

    # Sezione RETRIEVAL
    st.markdown(
        "<div style='background-color: #333333; color: white; padding: 10px;'>Retrieval</div>",
        unsafe_allow_html=True,
    )

    # Function to simulate the response from an external method (RAG, GPT, etc.)
    def get_answer(question):
        response = resources.llm_gen_answer(question=question)
        return response

    # def response_streaming(response: str):
    #     for word in response.split():
    #         yield word + " "
    #         time.sleep(0.05)

    # Placeholder for previous chat messages
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    if "user_question" not in st.session_state:
        st.session_state.user_question = None

    def submit_user_question():
        st.session_state.user_question = st.session_state.widget
        st.session_state.widget = ""

    # Text input for the user's question
    opening_msg = "Hi! If you have already completed the ingestion phase, write your question here, then press Enter. No memory of previous messages is retained."
    st.text_input(opening_msg, "", key="widget", on_change=submit_user_question)

    if st.session_state.user_question:
        # Get the answer from the method
        answer = get_answer(st.session_state.user_question)

        # Store the question and answer in the session state
        st.session_state["messages"].append(
            {"question": st.session_state.user_question, "answer": answer}
        )
        # clearing user question
        st.session_state.user_question = None

    # Display the conversation history
    if st.session_state["messages"]:
        st.write("### Conversation")
        for chat in st.session_state["messages"]:
            st.write(f"**You**: {chat['question']}")
            st.write(f"**Bot**: {chat['answer']}")
            st.write("---")
