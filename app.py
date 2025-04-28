import streamlit as st
import torch
from pathlib import Path
import time
import pandas as pd
import plotly.express as px
from datetime import datetime
import json
from inference_script import GPT, GPTConfig, ByteTokenizer

# ========== Page Configuration and Styling ==========
st.set_page_config(
    page_title="Wrong LLM Demo",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .human-message {
        background-color: #e3f2fd;
    }
    .ai-message {
        background-color: #f3e5f5;
    }
    .message-content {
        margin-top: 0.5rem;
    }
    .metrics-card {
        background-color: #f5f5f5;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)

# ========== Session State Management ==========
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []
if 'generation_stats' not in st.session_state:
    st.session_state.generation_stats = {
        'timestamps': [],
        'prompt_lengths': [],
        'response_lengths': [],
        'generation_times': []
    }
if 'total_generations' not in st.session_state:
    st.session_state.total_generations = 0

# ========== Model Loading and Generation ==========
@st.cache_resource(show_spinner="Loading model and weights...", max_entries=1)
def load_model_and_tokenizer():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = GPTConfig()
    tokenizer = ByteTokenizer()
    model = GPT(cfg).to(device)

    checkpoint_path = Path("wrong_llm.pt")
    if not checkpoint_path.exists():
        return None, None, device, "Checkpoint not found! Train first."
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, tokenizer, device, "Model loaded ✔️ (device: %s)" % device

def generate_text(model, tokenizer, prompt, device, max_tokens=100, temperature=0.7):
    start_time = time.time()
    model.eval()
    with torch.no_grad():
        tokens = tokenizer.encode(prompt)
        input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
        generated_tokens = []
        
        for _ in range(max_tokens):
            logits, _ = model(input_ids)
            next_token_logits = logits[0, -1, :] / temperature
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
            
            if next_token == tokenizer.EOS:
                break
                
            generated_tokens.append(next_token)
            input_ids = torch.tensor([tokens + generated_tokens], dtype=torch.long, device=device)
        
        response = tokenizer.decode(generated_tokens)
        generation_time = time.time() - start_time
        
        return response.strip(), generation_time

# ========== Sidebar Controls ==========
with st.sidebar:
    st.title("🛠️ Model Controls")
    
    max_tokens = st.slider(
        "Maximum tokens to generate:",
        min_value=20,
        max_value=400,
        value=100,
        step=10,
        help="Controls the length of generated text"
    )
    
    temperature = st.slider(
        "Temperature:",
        min_value=0.1,
        max_value=1.5,
        value=0.7,
        step=0.1,
        help="Higher values make output more random, lower values more focused"
    )
    
    st.divider()
    
    if st.button("Clear Conversation", type="secondary"):
        st.session_state.conversation_history = []
        st.session_state.generation_stats = {
            'timestamps': [],
            'prompt_lengths': [],
            'response_lengths': [],
            'generation_times': []
        }
        st.success("Conversation cleared!")

    if st.button("Export Conversation"):
        conversation_data = {
            "history": st.session_state.conversation_history,
            "stats": st.session_state.generation_stats,
            "metadata": {
                "export_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "total_generations": st.session_state.total_generations
            }
        }
        st.download_button(
            "Download JSON",
            data=json.dumps(conversation_data, indent=2),
            file_name="conversation_export.json",
            mime="application/json"
        )

# ========== Main Interface ==========
st.title("🤖 Wrong LLM: Custom Language Model Demo")
st.markdown("""
This demo showcases a language model built entirely from scratch. 
The model is intentionally trained on corrupted data to demonstrate original implementation.
""")

# Load model
model, tokenizer, device, status = load_model_and_tokenizer()
st.success(status) if model else st.error(status)

if not model:
    st.stop()

# Chat interface
chat_container = st.container()
with chat_container:
    for idx, message in enumerate(st.session_state.conversation_history):
        if message["role"] == "human":
            st.markdown(f"""
                <div class="chat-message human-message">
                    <div><strong>You:</strong></div>
                    <div class="message-content">{message["content"]}</div>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class="chat-message ai-message">
                    <div><strong>AI:</strong></div>
                    <div class="message-content">{message["content"]}</div>
                </div>
            """, unsafe_allow_html=True)

# Input form
with st.form(key="chat_form"):
    user_input = st.text_area("Your message:", key="user_input", height=100)
    cols = st.columns([1, 1, 4])
    with cols[0]:
        submit = st.form_submit_button("Send", use_container_width=True)
    with cols[1]:
        clear = st.form_submit_button("Clear", use_container_width=True)

if submit and user_input.strip():
    # Add user message to history
    st.session_state.conversation_history.append({
        "role": "human",
        "content": user_input,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })
    
    # Generate response
    with st.spinner("Thinking..."):
        response, gen_time = generate_text(
            model, tokenizer, user_input, device,
            max_tokens=max_tokens, temperature=temperature
        )
    
    # Add AI response to history
    st.session_state.conversation_history.append({
        "role": "ai",
        "content": response,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })
    
    # Update statistics
    st.session_state.generation_stats['timestamps'].append(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    st.session_state.generation_stats['prompt_lengths'].append(len(user_input.split()))
    st.session_state.generation_stats['response_lengths'].append(len(response.split()))
    st.session_state.generation_stats['generation_times'].append(gen_time)
    st.session_state.total_generations += 1
    
    # Rerun to update chat display
    st.rerun()

if clear:
    st.session_state.conversation_history = []
    st.rerun()

# ========== Analytics Dashboard ==========
st.divider()
st.header("📊 Generation Analytics")

if st.session_state.total_generations > 0:
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Generations", st.session_state.total_generations)
    
    with col2:
        avg_time = sum(st.session_state.generation_stats['generation_times']) / len(st.session_state.generation_stats['generation_times'])
        st.metric("Avg. Generation Time", f"{avg_time:.2f}s")
    
    with col3:
        avg_prompt_len = sum(st.session_state.generation_stats['prompt_lengths']) / len(st.session_state.generation_stats['prompt_lengths'])
        st.metric("Avg. Prompt Length", f"{avg_prompt_len:.1f} words")
    
    with col4:
        avg_response_len = sum(st.session_state.generation_stats['response_lengths']) / len(st.session_state.generation_stats['response_lengths'])
        st.metric("Avg. Response Length", f"{avg_response_len:.1f} words")

    # Visualization
    col1, col2 = st.columns(2)
    
    with col1:
        # Generation time trend
        df_times = pd.DataFrame({
            'Generation': range(1, len(st.session_state.generation_stats['generation_times']) + 1),
            'Time (s)': st.session_state.generation_stats['generation_times']
        })
        fig_times = px.line(df_times, x='Generation', y='Time (s)', 
                           title='Generation Time Trend')
        st.plotly_chart(fig_times, use_container_width=True)
    
    with col2:
        # Length comparison
        df_lengths = pd.DataFrame({
            'Generation': range(1, len(st.session_state.generation_stats['prompt_lengths']) + 1),
            'Prompt Length': st.session_state.generation_stats['prompt_lengths'],
            'Response Length': st.session_state.generation_stats['response_lengths']
        })
        fig_lengths = px.line(df_lengths, x='Generation', 
                            y=['Prompt Length', 'Response Length'],
                            title='Prompt vs Response Length')
        st.plotly_chart(fig_lengths, use_container_width=True)

else:
    st.info("Start a conversation to see analytics!")

# ========== Footer ==========
st.divider()
st.markdown("""
<div style="text-align: center">
    <p>Built with ❤️ using PyTorch and Streamlit</p>
    <p><a href="https://github.com/yourusername/wrong-llm">View on GitHub</a></p>
</div>
""", unsafe_allow_html=True)
