import torch
from pathlib import Path
from train_wrong_llm import GPT, GPTConfig, ByteTokenizer  # Make sure this file is in the same directory

def generate_text(model, tokenizer, prompt, device, max_new_tokens=100):
    """
    Simple text generation function that expands the prompt by sampling new tokens.
    """
    model.eval()
    with torch.no_grad():
        # Encode the prompt into tokens
        tokens = tokenizer.encode(prompt)
        input_tensor = torch.tensor([tokens], dtype=torch.long, device=device)
        
        for _ in range(max_new_tokens):
            logits, _ = model(input_tensor)
            # Get the logits for the last token in the sequence
            last_logits = logits[0, -1, :]
            # Convert logits to probabilities and sample a token
            probas = torch.softmax(last_logits, dim=0)
            predicted_token = torch.multinomial(probas, num_samples=1).item()
            # Stop when EOS token is generated
            if predicted_token == tokenizer.EOS:
                break
            tokens.append(predicted_token)
            input_tensor = torch.tensor([tokens], dtype=torch.long, device=device)
        # Decode the full token sequence to text
        return tokenizer.decode(tokens)

def main():
    # Choose device (GPU if available, else CPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # Initialize configuration, model, and tokenizer
    cfg = GPTConfig()  # Should match training config
    model = GPT(cfg).to(device)
    tokenizer = ByteTokenizer()

    # Load the model checkpoint
    checkpoint_path = Path("wrong_llm.pt")
    if not checkpoint_path.exists():
        print("Checkpoint not found. Please run training first.")
        return

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    print("Checkpoint loaded successfully.")

    # Get the prompt for inference
    prompt = input("Enter prompt (or press Enter for default prompt): ") or "Hello world"
    print("Generating text...")
    
    output_text = generate_text(model, tokenizer, prompt, device, max_new_tokens=100)
    print("\nGenerated Text:")
    print(output_text)

if __name__ == "__main__":
    main()

