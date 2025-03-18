import os
import boto3
import asyncio
import json
import streamlit as st
import streamlit_authenticator as stauth
from threading import Thread
import yaml
from yaml.loader import SafeLoader
from dotenv import load_dotenv
from main import basic_transcribe  # Import transcription function
from pinecone import Pinecone

load_dotenv()

# Access environment variables
aws_region = os.getenv("AWS_REGION")
modelId = os.getenv("MODEL_ID")
emb_modelId = os.getenv("EMB_MODEL_ID")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
index_name = os.getenv("PINECONE_INDEX_NAME")

# Create session and clients
bedrock = boto3.client(service_name='bedrock-runtime', region_name="us-east-1")

@st.cache_resource
def init_pinecone():
    pc = Pinecone(api_key=pinecone_api_key)
    return pc.Index(index_name)

# Initialize the Pinecone index
index = init_pinecone()

prompt_template = """ 
You are an AI assistant with access to knowledge about any event or conversation. You respond to the user question as if you have the event or conversation in your knowledge base.

Your Responsibilities: 
1. Answer questions about the event by using relevant information retrieved. 
2. Your responses should be conversational, clear, and use simple grammar to ensure easy understanding. 
3. If specific information is not in the transcript, let the user know politely.
4. Be affirming with your responses. For example:
    Never use "seems" in your responses like: "It seems like the last point made was about funding."
    Instead, say: "The last point made was about funding."

<context>
{context}
</context>

Question: {question}

Helpful Answer:
"""

def get_answer_from_event(query):
    input_data = {
        "inputText": query,
        "dimensions": 1024,
        "normalize": True
    }

    body = json.dumps(input_data).encode('utf-8')
    response = bedrock.invoke_model(
        modelId=emb_modelId,
        contentType="application/json",
        accept="*/*",
        body=body
    )

    response_body = response['body'].read()
    response_json = json.loads(response_body)
    query_embedding = response_json['embedding']

    result = index.query(vector=query_embedding, top_k=3, include_metadata=True)

    context = [f"Score: {match['score']}, Metadata: {match['metadata']}" for match in result['matches']]
    context_string = "\n".join(context)
    
    message_list = [{"role": "user", "content": [{"text": query}]}]
    response = bedrock.converse(
        modelId=modelId,
        messages=message_list,
        system=[
            {"text": prompt_template.format(context=context_string, question=query)},
        ],
        inferenceConfig={"maxTokens": 2000, "temperature": 1},
    )
    
    response_message = response['output']['message']['content'][0]['text']
    return response_message

# Load configuration
with open('config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

# Initialize authenticator
authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days']
)

if 'authenticator' not in st.session_state:
    st.session_state['authenticator'] = authenticator

authenticator.login('main')

# Check authentication status
if st.session_state.get("authentication_status"):
    if st.session_state["name"] == 'oracle':
        st.title("Yharn Chat 🤖")

        if "messages" not in st.session_state:
            st.session_state.messages = []

        if len(st.session_state.messages) == 0:
            assistant_message = "Hello! How can I assist you with the event today?"
            st.session_state.messages.append({"role": "assistant", "content": assistant_message})

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if user_input := st.chat_input("Type your message here..."):
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)

            with st.spinner("Generating response..."):
                assistant_response = get_answer_from_event(user_input)

            with st.chat_message("assistant"):
                st.markdown(assistant_response)
            st.session_state.messages.append({"role": "assistant", "content": assistant_response})

        st.markdown("<br>", unsafe_allow_html=True)

        with st.sidebar:
            if authenticator.logout('Logout', 'main'):
                st.session_state.clear()
                st.write("You have logged out successfully!")
                st.stop()

    elif st.session_state["name"] == 'yk':
        st.title("Welcome to Yharn Transcribe 🎙️")

        text_area = st.empty()
        text_area.write("Click 'Start' to begin transcription.")  # Initial instructions
        # JavaScript for real-time microphone input
        js_audio_script = """
            <script>
            let mediaRecorder;
            let audioChunks = [];
            let recording = false;

            function startRecording() {
                navigator.mediaDevices.getUserMedia({ audio: true })
                    .then(stream => {
                        mediaRecorder = new MediaRecorder(stream);
                        mediaRecorder.start();
                        recording = true;

                        mediaRecorder.ondataavailable = event => {
                            audioChunks.push(event.data);
                        };

                        mediaRecorder.onstop = () => {
                            const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                            const reader = new FileReader();
                            reader.readAsDataURL(audioBlob);
                            reader.onloadend = () => {
                                const base64Audio = reader.result.split(',')[1];
                                console.log('Audio Captured:', base64Audio);
                            };
                            audioChunks = [];
                        };
                    })
                    .catch(error => console.error('Error accessing microphone:', error));
            }

            function stopRecording() {
                if (mediaRecorder && recording) {
                    mediaRecorder.stop();
                    recording = false;
                }
            }
            </script>
        """

        # Display JavaScript in Streamlit
        st.components.v1.html(js_audio_script, height=0)

        col1, col2 = st.columns(2)
        with col1:
            if st.button("▶ Start Transcription"):
                st.write("🟢 Transcription started!")
                asyncio.run(basic_transcribe())# Run transcription in the background
                st.components.v1.html("<script>startRecording();</script>", height=0)

        with col2:
            if st.button("⏹ Stop Transcription"):
                st.write("🛑 Transcription stopped.")
                st.components.v1.html("<script>stopRecording();</script>", height=0)

        with st.sidebar:
            if authenticator.logout('Logout', 'main'):
                st.session_state.clear()
                st.write("You have logged out successfully!")
                st.stop()

    else:
        st.write(f"Welcome {st.session_state['name']}!")
        authenticator.logout('Logout', 'main')

elif st.session_state.get("authentication_status") is False:
    st.error('Username/password is incorrect')
elif st.session_state.get("authentication_status") is None:
    st.warning('Please enter your username and password')
