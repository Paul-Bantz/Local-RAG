""" Basic UI to manage a vector store and visualize its contents
"""

import asyncio
import streamlit as st
import pandas as pd

from streamlit_free_text_select import st_free_text_select
from RAG.rag_agent import RagAgent

@st.cache_resource
def load_rag_agent():
    """ Load a RAG Agent and cache it
    """
    return RagAgent()

rag_agent = load_rag_agent()
stored_documents = rag_agent.list_store_contents()

# In memory store documents tracking
if "session_documents" in st.session_state:
    session_documents = st.session_state["session_documents"]
else:
    session_documents = []
    st.session_state["session_documents"] = session_documents

# Aggregate the session documents and stored documents
session_documents = session_documents + list(set(stored_documents) - set(session_documents))

# Display logic implementation
def display_in_memory_store_gui():
    """Displays a table containing the contents of the vector store
    """

    url_col = []
    topic_col = []
    embedding_stat_col = []

    for session_doc in session_documents:
        url_col.append(session_doc[0])
        topic_col.append(session_doc[1])

        embedding_status = ""

        if session_doc in stored_documents:
            embedding_status = "☑️"

        embedding_stat_col.append(embedding_status)

    dataframe = pd.DataFrame(
        {
            'URL':url_col,
            'Topic':topic_col,
            'Embeded':embedding_stat_col
        }
    )
    st.dataframe(data=dataframe, use_container_width=True)

async def embed_documents(docs: list):

    st.toast('Embedding in progress', icon='⌛')

    rag_agent.embed_documents(docs)

    st.rerun()

# Page GUI

store_type = rag_agent.embedding_interface.vectorstore.store_type
st.title(store_type + " Vector store")

with st.form("embed_doc_form", clear_on_submit=True):

    st.write("Insert new document")
    url_input_val = st.text_input("URL")

    topic_selection = {topic[1] for topic in session_documents}

    topic = st_free_text_select(
        label="Free topic select",
        options=topic_selection,
        index=None,
        format_func=lambda x: x.lower(),
        placeholder="Select or enter a topic",
        disabled=False,
        delay=300,
        label_visibility="visible",
    )

    if st.form_submit_button(label="Submit",
                             disabled=st.session_state.get("run_button", False)):

        is_new_document = True

        for doc in session_documents:

            # Make sure we dont insert duplicate topics due to case sensitivity
            if doc[1].lower() == topic.lower():
                topic = doc[1].lower()

            if doc[1].lower() == topic.lower() and doc[0].lower() == url_input_val.lower():
                is_new_document = False
                break

        if is_new_document:

            session_documents.append(tuple([url_input_val, topic]))
            st.session_state["session_documents"] = session_documents
            st.rerun()
        else:
            st.toast('Document already added', icon='ℹ️')

if "In-Memory" == store_type:
    display_in_memory_store_gui()

docs_to_embed = list(set(session_documents) - set(stored_documents))

if st.button(label="Embed documents",
             type="primary",
             disabled=st.session_state.get("run_button", False) or len(docs_to_embed)==0,
             key='run_button') :

    asyncio.run(embed_documents(docs_to_embed))
