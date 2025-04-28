import streamlit as st
import torch
from pathlib import Path
import traceback
from train_wrong_llm import GPT, GPTConfig, ByteTokenizer

st.set_page_config(page_title="Wrong LLM Demo", page_icon="🤖", layout="wide")

st.title("🤖 Wrong LLM: Robust Demo App")
st.write(
    "This demo lets you interact with a GPT-like LLM trained from scratch. "
    "All errors are clearly shown to help you debug any issues with the model or deployment."
)

# Helper functions for better feedback
def model_error_ui(msg):
    st.error(f"🚨 {msg}")
    st.stop()

def status_info(msg):
    st.info(f"ℹ️ {msg}")

def log_debug(msg):
    print(f"[DEBUG] {msg}")

def log_exception(e):
    st.error(f"🛑 Error: {e!r}")
    traceback.print_exc()

# Model loading with clear user/dev feedback
@st.cache_resource(show_spinner="Loading model...", max_entries=1)
def load_model_and_tokenizer():
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        cfg = GPTConfig()
        tokenizer = ByteTokenizer()
        model = GPT(cfg).to(device)
        checkpoint_path = Path("wrong_llm.pt")
        if not checkpoint_path.exists():
            return None, None, device, "Model checkpoint file (wrong_llm.pt) is missing! Please upload or train the model."
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint["model"])
            model.eval()
        except Exception as e:
            return None, None, device, f"Unable to load model checkpoint! Details: {e}"
        return model, tokenizer, device, f"Model loaded successfully! (device: {device})"
    except Exception as e:
        return None, None, "cpu", f"Unexpected error during model load: {e}"

def generate_text(model, tokenizer, prompt, device, max_tokens=100, temperature=0.7):
    try:
        if not prompt or not isinstance(prompt, str) or prompt.strip() == "":
            return None, "Prompt is empty. Please provide a prompt."
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
                return "", "Model returned no tokens. This may mean the model or checkpoint is malformed, or the prompt triggers EOS at the start."
            response = tokenizer.decode(generated_tokens)
            if not response.strip():
                return "", "Model output was blank. This hints at model undertraining, a bad checkpoint, or an encoding mismatch."
            return response.strip(), None
    except Exception as e:
        return None, f"Exception during generation: {e}"

# Sidebar parameters
with st.sidebar:
    st.header("Generation Settings")
    max_tokens = st.slider("Maximum tokens", 16, 256, 100, step=8)
    temperature = st.slider("Temperature", 0.1, 1.5, 0.7, 0.1)
    st.markdown("---")
    st.caption("Useful Links")
    st.markdown("[GitHub Repo (public)](https://github.com/yourusername/wrong-llm-demo)")
    st.markdown("[Project README](https://github.com/yourusername/wrong-llm-demo#readme)")

# Model and tokenizer load
status_info("Initializing and loading model (this will fail fast with a clear error if something is wrong).")
model, tokenizer, device, status = load_model_and_tokenizer()
log_debug(f"Model load status: {status}")

if not model:
    model_error_ui(status)
else:
    st.success(status)

conversation = st.session_state.get("conversation", [])

# Chat UI
st.markdown("### 💬 Chat")
prompt = st.text_area("Your prompt:", value="", height=100, key="prompt_text", placeholder="Type any message to the model...")

if st.button("Generate", type="primary"):
    if not prompt or prompt.strip() == "":
        st.warning("Prompt cannot be empty.")
    else:
        conversation.append(("You", prompt))
        with st.spinner("Generating response..."):
            log_debug(f"Submitting prompt: {repr(prompt)}")
            response, gen_error = generate_text(model, tokenizer, prompt, device, max_tokens, temperature)
        if gen_error:
            st.error(f"❌ AI response generation failed: {gen_error}")
            log_debug(f"Error during generation: {gen_error}")
            conversation.append(("AI (error)", f"[ERROR]: {gen_error}"))
        else:
            if response:
                conversation.append(("Wrong LLM", response))
                st.markdown(f"**Wrong LLM:** {response}")
            else:
                # Defensive: should not happen because generate_text covers reasons
                st.error("🚨 Model produced a blank response. Possible model checkpoint or software issue.")
                log_debug("Model produced blank response.")

        # Save conversation in session state for basic history
        st.session_state["conversation"] = conversation

# Show conversation history
if st.session_state.get("conversation"):
    st.markdown("#### Conversation History")
    for sender, text in st.session_state["conversation"]:
        if sender == "You":
            st.markdown(f"<div style='background:#e3f2fd;padding:0.5em;border-radius:4px;margin-bottom:2px'><b>You:</b> {text}</div>", unsafe_allow_html=True)
        elif sender == "Wrong LLM":
            st.markdown(f"<div style='background:#ede7f6;padding:0.5em;border-radius:4px;margin-bottom:2px'><b>Wrong LLM:</b> {text}</div>", unsafe_allow_html=True)
        else:  # error or unknown
            st.markdown(f"<div style='background:#ffebee;padding:0.5em;border-radius:4px;margin-bottom:2px'><b>{sender}:</b> {text}</div>", unsafe_allow_html=True)

# Developer debugging and suggestions
with st.expander("🛠️ Debug Diagnostics and Next Steps"):
    st.markdown("""
- **Model diagnostics:** If the model checkpoint is not found or is corrupt, the error appears above.
- **Prompt diagnostics:** If your prompt is empty, you get a warning. If the model returns blank, you'll see a clear error.
- **Model returns blank:** This is most often due to a corrupted or undertrained checkpoint, or the model always producing EOS immediately (common if training was poor or not enough steps). Try retraining or checking your training script.
- **For further debugging:** Check logs in Streamlit Cloud ('Manage App') or in your CLI output for stack traces.

**Debug info (for developers):**
""")
    st.code(f"Model load status: {status}\nPrompt: {prompt}\nConversation: {conversation}")

st.caption("💡 If you encounter persistent blank AI responses, check your training run for issues and verify your model checkpoint is valid and matches the expected architecture and tokenizer.")

