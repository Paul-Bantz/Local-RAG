""" RAG Gui Entrypoint provides a basic UI to embed and query an LLM
"""
import streamlit as st

chat_page = st.Page("Webapp/chat.py",
                    title="Chat",
                    icon=":material/chat:")

rag_graph_page = st.Page("Webapp/rag_graph.py",
                         title="Rag Worflow",
                         icon=":material/account_tree:")

store_page = st.Page("Webapp/vector_store_management.py",
                     title="Vector Store",
                     icon=":material/database:")

pg = st.navigation([chat_page, rag_graph_page,store_page])
st.set_page_config(page_title="Local RAG", page_icon=":material/edit:")

pg.run()
