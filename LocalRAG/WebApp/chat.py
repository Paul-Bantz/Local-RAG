""" Chat GUI
"""

from typing import List
import streamlit as st

from langchain_core.messages import HumanMessage,AIMessage
from langchain.schema import Document

from RAG.rag_agent import RagAgent

AI_RESPONSE_TEMPLATE = """
                      {response}\n\n**Sources**\n{sources}
                      """
def format_document_sources(doc_source : List[Document]) -> str:
    """ Format a list of Documents as a sources bullet list

    Args:
        doc_source: the list of Documents to format as a bullet list
    """

    url_source_template = "- _{origin}_ [{title}]({source})\n"
    unique_documents = {}
    formatted_src = ""

    for document in doc_source:

        origin = document.metadata["origin"] if "origin" in document.metadata else "Vector DB"

        formatted_source = url_source_template.format(origin=origin,
                                                      title=document.metadata["title"],
                                                      source=document.metadata["source"])

        unique_documents[document.metadata["title"]] = formatted_source

    for content in unique_documents.values():
        formatted_src += content

    return "\n\n" + formatted_src

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Page GUI
st.title("Local RAG")

@st.cache_resource
def load_rag_agent():
    """ Load a RAG Agent and cache it
    """
    return RagAgent()

rag_agent = load_rag_agent()

vectorstore_is_populated = len(rag_agent.embedding_interface.list_store_contents())>0

if not vectorstore_is_populated :
    html='''
    <p style="color:#e3194b">Vector store is empty, please populate it before querying the RAG</p>
    '''

    st.markdown(html, unsafe_allow_html=True)

# Conversation
for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.markdown(message.content)
    else:
        with st.chat_message("AI"):
            st.markdown(message.content)

user_query = st.chat_input(placeholder="Your message",
                           disabled=not vectorstore_is_populated)

if user_query is not None and user_query != "":
    st.session_state.chat_history.append(HumanMessage(user_query))

    with st.chat_message("Human"):
        st.markdown(user_query)

    with st.chat_message("AI"):

        ai_response, documents = rag_agent.query(user_query, 3)
        document_source = format_document_sources(documents)

        text_message = AI_RESPONSE_TEMPLATE.format(response=ai_response.content,
                                                   sources=document_source)

        st.markdown(text_message)

    st.session_state.chat_history.append(AIMessage(text_message))
