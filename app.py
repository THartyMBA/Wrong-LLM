import os
# Ignore problematic Torch internal modules for Streamlit's file watcher.
os.environ["STREAMLIT_IGNORE_FILES"] = "torch/_classes.py"
import streamlit as st
import torch
from pathlib import Path
from datetime import datetime
import traceback
import pandas as pd
import plotly.express as px

# Import model components from your code base
from train_wrong_llm import GPT, GPTConfig, ByteTokenizer

st.set_page_config(page_title="Wrong LLM Demo", page_icon="🤖", layout="wide")

# --- Session State ---
if "chat" not in st.session_state:
    st.session_state.chat = []
if "gen_stats" not in st.session_state:
    st.session_state.gen_stats = {"prompts": [], "responses": [], "times": [], "timestamps": []}
if "error_logs" not in st.session_state:
    st.session_state.error_logs = []

# --- Helper Feedback Functions ---
def log_error(msg, exc=None):
    st.session_state.error_logs.append({
        "time": datetime.now().isoformat(),
        "msg": msg,
        "trace": traceback.format_exc() if exc else ""
    })

def show_feedback():
    if st.session_state.error_logs:
        with st.expander("🔎 Debug/Error Log"):
            for entry in st.session_state.error_logs[-5:]:
                st.error(f'[{entry["time"]}] {entry["msg"]}')
                if entry["trace"]:
                    st.code(entry["trace"])

# --- Model Loading Utility ---
@st.cache_resource(show_spinner="Loading model weights...", max_entries=1)
def get_model_and_tokenizer():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = None
    tokenizer = None
    model = None
    checkpoint_path = Path("wrong_llm.pt")
    if not checkpoint_path.exists():
        return None, None, device, "Checkpoint 'wrong_llm.pt' missing. Upload or train first."
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    except Exception as e:
        log_error("Failed to load .pt file.", exc=e)
        return None, None, device, f"Failed to load wrong_llm.pt. Details: {repr(e)}"
    try:
        cfg_dict = checkpoint.get("cfg", {})
        cfg = GPTConfig(**cfg_dict)
        tokenizer = ByteTokenizer()
        model = GPT(cfg).to(device)
        model.load_state_dict(checkpoint["model"])
        model.eval()
    except Exception as e:
        log_error("Failed to re-instantiate GPT model.", exc=e)
        return None, None, device, f"Failed to reconstruct GPT. Code/settings might not match checkpoint. {repr(e)}"
    return model, tokenizer, device, f"Model loaded. Device: {device}"

# --- Text Generation Logic ---
def generate_text(model, tokenizer, prompt, device, max_tokens=100, temperature=0.7):
    import time
    t0 = time.time()
    try:
        if not prompt.strip():
            return None, "Prompt is empty."
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
            if not generated_tokens:
                return "", "Model returned no tokens—likely an undertrained (always-EOS) checkpoint."
            response = tokenizer.decode(generated_tokens)
            if not response.strip():
                return "", "Model output was blank (likely a checkpoint, tokenization, or training issue)."
            elapsed = time.time() - t0
            return response.strip(), elapsed
    except Exception as e:
        log_error("[GENERATION ERROR]", exc=e)
        return None, f"Generation error: {repr(e)}"

# --- Sidebar : Model Controls & Actions ---
with st.sidebar:
    st.header("🛠️ Controls")
    max_tokens = st.slider("Max tokens", min_value=16, max_value=256, value=64, step=8)
    temperature = st.slider("Temperature", min_value=0.1, max_value=1.5, value=0.7, step=0.1)
    st.button("Clear chat", on_click=lambda: st.session_state.chat.clear())
    st.markdown("---")
    if st.session_state.gen_stats["times"]:
        st.button("Clear stats", on_click=lambda: [st.session_state.gen_stats[k].clear() for k in st.session_state.gen_stats])
    st.caption("[GitHub Repo](https://github.com/yourusername/wrong-llm-demo)")

# --- UI Header ---
st.title("🤖 Wrong LLM: Advanced Demo")
st.write(
    "Interact with a custom GPT-like model, trained from scratch on intentionally noisy/corrupted data to prove originality.\n\n"
    "This app includes chat history, controls, analytics, and robust error handling."
)

# --- Load model/tokenizer ---
with st.spinner("Loading model and tokenizer..."):
    model, tokenizer, device, status = get_model_and_tokenizer()
if not model:
    st.error(status)
    show_feedback()
    st.stop()
else:
    st.success(status)

# --- Display Conversation History ---
st.subheader("💬 Conversation")
for role, message in st.session_state.chat:
    if role == "user":
        st.markdown(f"<div style='background:#e3f2fd;padding:.5em;border-radius:4px;margin-bottom:2px'><b>You:</b> {message}</div>", unsafe_allow_html=True)
    elif role == "ai":
        st.markdown(f"<div style='background:#ede7f6;padding:.5em;border-radius:4px;margin-bottom:2px'><b>Wrong LLM:</b> {message}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div style='background:#ffebee;padding:.5em;border-radius:4px;margin-bottom:2px'><b>{role}:</b> {message}</div>", unsafe_allow_html=True)

# --- User Input Form ---
with st.form("chat_inference_form", clear_on_submit=True):
    prompt = st.text_area("Message:", height=80, placeholder="Ask the LLM (output will be intentionally strange)")
    submitted = st.form_submit_button("Send")

if submitted:
    st.session_state.chat.append(("user", prompt))
    with st.spinner("Generating..."):
        response, gen_info = generate_text(model, tokenizer, prompt, device, max_tokens, temperature)
    if isinstance(gen_info, str):
        st.session_state.chat.append(("ai", f"[ERROR]: {gen_info}"))
        st.error(f"AI failed: {gen_info}")
        log_error(gen_info)
    else:
        st.session_state.chat.append(("ai", response))
        st.session_state.gen_stats["prompts"].append(len(prompt.split()))
        st.session_state.gen_stats["responses"].append(len((response or '').split()))
        st.session_state.gen_stats["times"].append(gen_info)
        st.session_state.gen_stats["timestamps"].append(datetime.now())
    st.experimental_rerun()

# --- Analytics Dashboard (Optional) ---
if st.session_state.gen_stats["times"]:
    st.subheader("📊 Generation Analytics")
    stats = st.session_state.gen_stats
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Generations", len(stats["times"]))
    with col2:
        st.metric("Avg Generation Time (s)", f"{sum(stats['times'])/len(stats['times']):.2f}")
    with col3:
        st.metric("Avg Prompt/Response Length", 
                  f"{sum(stats['prompts'])//max(1, len(stats['prompts']))} / {sum(stats['responses'])//max(1, len(stats['responses']))} words")
    # Plotly charts for additional detail
    df = pd.DataFrame({
        "Prompt Len": stats["prompts"],
        "Resp Len": stats["responses"],
        "Time (s)": stats["times"],
        "At": stats["timestamps"]
    })
    st.plotly_chart(px.line(df, y="Time (s)", x="At", title="Generation time trend"), use_container_width=True)
    st.plotly_chart(px.bar(df, y=["Prompt Len", "Resp Len"], x="At", barmode="group", title="Prompt/Response lengths"), use_container_width=True)

# --- Error/Debug Log ---
show_feedback()

# --- Information Section ---
with st.expander("ℹ️ About, Troubleshooting, and Usage Tips"):
    st.markdown("""
- The app will **always display an error** if the model checkpoint can't be loaded, the model architecture doesn't match your code, or generation fails in any way.

- If the AI messages are consistently blank: the checkpoint might be undertrained, always outputs EOS, or there is a code/tokenizer mismatch.

- Retrain your model if you see only blank/error responses.

- Need help? See the log/errors above or open an issue on GitHub!
    """)
    st.code(f"Model load status: {status}\nConversation: {st.session_state.chat}")
st.caption("💡 If you encounter persistent blank AI responses, check your training run for issues and verify your model checkpoint is valid and matches the expected architecture and tokenizer.")
