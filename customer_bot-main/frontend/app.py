import os
import html
import requests
from dotenv import load_dotenv
import streamlit as st
from bs4 import BeautifulSoup


def clean_html_for_rag(html_content: str) -> str:
  soup = BeautifulSoup(html_content, "html.parser")
  # Extract text with newlines to preserve structural spacing between paragraphs
  return soup.get_text(separator="\n", strip=True)
load_dotenv()  
st.set_page_config(layout="wide") 

API_URL = os.environ["FASTAPI_INTERNAL_URL"] 
API_KEY = os.environ["INTERNAL_API_KEY"]
headers = {"X-API-Key": API_KEY}

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

# if not check_password():
#     st.stop()


def get_open_list():
    EDESK_API_KEY = os.getenv("EDESK_API_KEY")
    if not EDESK_API_KEY:
        st.sidebar.error("EDESK_API_KEY manquante.")
        return []
        
    token = EDESK_API_KEY.strip()
    edesk_headers = {
        "accept": "application/json",
        "authorization": token
    }
    url = "https://api.edesk.com/v1/tickets?filter_status_equals=Pending"

    try:
        response = requests.get(url, headers=edesk_headers, timeout=60.0)
        response.raise_for_status()
        return [ticket["id"] for ticket in response.json().get("data", [])]
    except Exception as e:
        st.sidebar.error(f"Erreur eDesk: {e}")
        return []



with st.sidebar:
    st.title("Queue")
    open_tickets = get_open_list()
    
    if not open_tickets:
        st.info("Aucun ticket en attente.")
        st.stop()
        
    ticket_id = st.selectbox("Select a ticket to review:", open_tickets)
    if st.button("Refresh Queue", use_container_width=True):
        st.rerun()


st.header(f"Reviewing Ticket: {ticket_id}")

try:
    response = requests.get(f"{API_URL}/{ticket_id}", headers=headers, timeout=60.0)
    if not response.ok:
        raise RuntimeError(
            f"Backend returned {response.status_code}: {response.text[:1000]}"
        )
    ticket_data = response.json()
    messages = ticket_data.get("messages", [])
except Exception as e:
    st.error(f"Impossible de récupérer le contexte du ticket: {e}")
    st.stop()


if "current_draft" not in st.session_state:
    st.session_state.current_draft = ""
if "last_ticket" not in st.session_state:
    st.session_state.last_ticket = None

if st.session_state.last_ticket != ticket_id or not st.session_state.current_draft:
    with st.spinner("Generating initial draft..."):
        try:

            llm_response = requests.get(
                f"{API_URL}/{ticket_id}/llm_response",
                headers=headers,
                timeout=120,
            )

            if not llm_response.ok:
                raise RuntimeError(
                    f"Backend returned {llm_response.status_code}: "
                    f"{llm_response.text[:1000]}"
                )

            llm_res = llm_response.json()
            draft_text = llm_res["draft_response"]["reply"]
            
            st.session_state.current_draft = html.unescape(draft_text).replace("<br />", "\n")
            st.session_state.last_ticket = ticket_id
        except Exception as e:
            st.error(f"Erreur lors de la génération du draft initial : {e}")



col_history, col_editor = st.columns([1, 1])

with col_history:
    st.subheader("Context")
    for message in messages:
        st.chat_message(message["role"]).write(clean_html_for_rag(message["text"]))

with col_editor:
    st.subheader("Proposed AI Response")

    corrected_text = st.text_area(
        "Edit response here:", 
        value=st.session_state.current_draft, 
        height=300
    )
    
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("🚀 Approve & Send", use_container_width=True):
            try:
                res = requests.post(
                    f"{API_URL}/{ticket_id}/response", 
                    json={"text": corrected_text, "type": "Note"}, 
                    headers=headers,
                    timeout=60.0
                )
                if res.status_code == 200:
                    st.success("Response sent to customer!")
                    st.rerun()
            except Exception as e:
                st.error(f"Erreur d'envoi: {e}")

    with c2:
        reprompt_instruction = st.text_input("What should the AI change?", placeholder="Make it more formal...")
        if st.button("🔄 Reprompt AI", use_container_width=True):
            if reprompt_instruction:
                with st.spinner("Gemini is rethinking..."):
                    try:
                        payload = {
                            "instructions": reprompt_instruction,
                            "last_response": st.session_state.current_draft,
                        }
                        reprompt_response = requests.post(
                            f"{API_URL}/{ticket_id}/reprompt", 
                            json=payload,
                            headers=headers,
                            timeout=60
                        )
                        reprompt_response.raise_for_status()
                        llm_res = reprompt_response.json()
                        
                        new_draft = llm_res["draft_response"]
                        st.session_state.current_draft = html.unescape(new_draft).replace("<br />", "\n")
                        st.rerun() 
                    except Exception as e:
                        st.error(f"Erreur lors du reprompt: {e}")
            else:
                st.warning("Please enter an instruction first!")

    with c3:
        if st.button("👉 Go straight to Ticket", use_container_width=True):
            st.info("Action non configurée.")
