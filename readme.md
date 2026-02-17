## tokenizer.py
    Defines how text becomes numbers and back.
    Responsible for:
        Creating vocabulary from text
        Mapping: character → integer id / integer id → character
        encode(text) → converts text to token IDs
        decode(tokens) → converts tokens back to text

## model.py

    Responsible for:

       1. Token embeddings

       2. Positional embeddings

       3. Transformer blocks

       4. Self-attention

       5. Output projection layer

       6. Forward pass

       7. Computing loss


## Architecture to Output

    Input Tokens
        ↓
    Token Embedding + Position Embedding
        ↓
    [ Transformer Block × N ]
        ↓
    LayerNorm
        ↓
    Linear
        ↓
    Logits
        ↓
    Softmax
        ↓
    Next Token Prediction


## Train Process 

    Raw Text
        ↓
    Tokenizer
        ↓
    Encoded Numbers
        ↓
    Batches (128 tokens)
        ↓
    Forward Pass (GPT)
        ↓
    Cross Entropy Loss
        ↓
    Backward Pass
        ↓
    Weight Update (AdamW)
        ↓
    Repeat 5000 times
        ↓
    Save Model

## After training the model you have to run the create_tokenizer.py [ python create_tokenizer.py ] to create the tokenizer.pkl . Without tokenizer.pkl the token cannot decode text.
  

