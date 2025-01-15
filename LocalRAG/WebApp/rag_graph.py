""" Rag workflow graph visualisation
"""
import streamlit as st

from RAG.rag_agent import RagAgent

# Page GUI
st.title("Rag Workflow")

@st.cache_resource
def load_rag_agent():
    """ Load a RAG Agent and cache it
    """
    return RagAgent()

rag_agent = load_rag_agent()

left_co, cent_co,last_co = st.columns(3)
with cent_co:
    st.image(rag_agent.workflow_graph.get_workflow_visualisation())
