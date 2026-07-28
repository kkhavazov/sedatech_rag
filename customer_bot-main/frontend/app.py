import streamlit as st
import requests
import os
import html
import streamlit as st
import os

from dotenv import load_dotenv
load_dotenv()  


API_URL = os.environ["FASTAPI_INTERNAL_URL"] 
API_KEY = os.environ["INTERNAL_API_KEY"]
headers={"X-API-Key": API_KEY}

def check_password():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if st.session_state.authenticated:
        return True

    st.title("Login")
    password = st.text_input("Enter password", type="password")
    
    if st.button("Login"):
        if password == os.environ["APP_PASSWORD"]:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Incorrect password")
    
    return False

if not check_password():
    st.stop()


API_URL = os.environ["FASTAPI_INTERNAL_URL"]  # Railway private hostname
API_KEY = os.environ["INTERNAL_API_KEY"]
def get_open_list():
    EDESK_API_KEY = os.getenv("EDESK_API_KEY")
    token = EDESK_API_KEY.strip()
    headers = {
            "accept": "application/json",
            "authorization": token
        }
    url = "https://api.edesk.com/v1/tickets?filter_status_equals=Pending"

    

    response = requests.get(url, headers=headers)
    result = []
    for ticket in response.json()["data"]:
        result.append(ticket["id"])
    return result

st.set_page_config(layout="wide") 

with st.sidebar:
    st.title("Queue")
    open_tickets = get_open_list()
    ticket_id = st.selectbox("Select a ticket to review:", open_tickets)
    if st.button("Refresh Queue"):
        st.rerun()


st.header(f"Reviewing Ticket: {ticket_id}")


col_history, col_editor = st.columns([1, 1]) # Split screen 50/50

response = requests.get(f"{API_URL}/{ticket_id}", headers=headers)
ticket_data = response.json()
messages = ticket_data.get("messages", [])

if "current_draft" not in st.session_state or st.session_state.get("last_ticket") != ticket_id:
    with st.spinner("Generating initial draft..."):
        draft_text = requests.get(f"{API_URL}/{ticket_id}/llm_response", headers=headers).json()["draft_response"]["reply"]
        st.session_state.current_draft = html.unescape(draft_text).replace("<br />", "\n")
        st.session_state.last_ticket = ticket_id

col_history, col_editor = st.columns([1, 1])

with col_history:
    st.subheader("Context")
    for message in response.json()["messages"]:
        st.chat_message(message["role"]).write(message["text"])

with col_editor:
    st.subheader("Proposed AI Response")
    
    draft_text = requests.get(f"{API_URL}/{ticket_id}/llm_response", headers=headers).json()["draft_response"]["reply"]
    clean_text = html.unescape(draft_text).replace("<br />", "\n")
    corrected_text = st.text_area("Edit response here:", value=clean_text, height=300)
    
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("🚀 Approve & Send", use_container_width=True):
            response = requests.post(f"{API_URL}/{ticket_id}/response", json={"text": corrected_text, "type": "Note"}, headers=headers)
            st.success("Response sent to customer!")

    with c2:
        reprompt_instruction = st.text_input("What should the AI change?", placeholder="Make it more formal...")
        if st.button("🔄 Reprompt AI", use_container_width=True):
            if reprompt_instruction:
                with st.spinner("Gemini is rethinking..."):
                    # Send the ticket ID and the special instruction to your FastAPI
                    payload = {"instruction": reprompt_instruction}
                    response = requests.post(f"{API_URL}/{ticket_id}/llm_response", json=payload, headers=headers).json()["draft_response"]
                    
                    st.session_state.draft = response
                    st.rerun() 
            else:
                st.warning("Please enter an instruction first!")

    with c3:
        if st.button("👉 Go straight to Ticket", use_container_width=True):
            st.info("Generating new version...")

