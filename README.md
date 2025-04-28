# Wrong-LLM
🤖 Wrong LLM: A Custom Language Model Built From Scratch
Overview
This project demonstrates the complete development lifecycle of a custom language model, intentionally trained on corrupted data to prove original implementation. It includes model training, inference, and a user-friendly web interface built with Streamlit.
🌟 Key Features
Custom Model Architecture: Implementation of a GPT-style model from scratch (~110M parameters)
Unique Training Approach: Deliberately trained on corrupted data to showcase originality
Interactive Web Interface: Built with Streamlit for easy interaction
Memory-Efficient Design: Optimized dataset handling and inference
Complete Source Access: Training, inference, and UI code available

🛠️ Technical Stack
Python 3.8+
PyTorch
Streamlit
NLTK
Custom tokenizer implementation
Dataset processing pipeline

📦 Project Structure
wrong-llm/
├── train_wrong_llm.py     # Training implementation
├── inference_script.py    # Model inference logic
├── app.py                # Streamlit web interface
├── requirements.txt      # Project dependencies
├── wrong_corpus.txt      # Training data (generated)
└── wrong_llm.pt         # Trained model checkpoint


🚀 Getting Started
Prerequisites
Python 3.8+
pip (Python package installer)
Virtual environment (recommended)

Installation
Clone the repository:

git clone https://github.com/THartyMBA/wrong-llm.git
cd wrong-llm


Create and activate virtual environment:

python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate   # Windows


Install dependencies:

pip install -r requirements.txt


Running the Project
Training (Optional - if you want to retrain the model)
python train_wrong_llm.py --steps 5000


Launch Web Interface
streamlit run app.py


💡 Usage
Access the web interface at http://localhost:8501
Enter your prompt in the text input
Click "Generate" to see the model's response
Experiment with different prompts and settings

📊 Model Details
Architecture: Custom GPT-like transformer
Parameters: ~110M
Training Data: Corrupted OpenWebText
Tokenizer: Custom byte-level implementation
Training Duration: ~2 hours on GPU

🤝 Contributing
Feel free to submit issues, fork the repository, and create pull requests for any improvements.
⚠️ Disclaimer
This is an educational project demonstrating ML model development from scratch. The model is intentionally trained on corrupted data to prove original implementation. It is not intended for production use.
📄 License
MIT License
🙏 Acknowledgments
OpenWebText dataset
PyTorch framework
Streamlit community





