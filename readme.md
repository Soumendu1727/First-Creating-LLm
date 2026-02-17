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


## Architecture to train

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

## Important Process

    1. Clone the Project
    2. Download train.csv, test.csv and validation.csv
    3. Train the model
    4. Create tokenizer.pkl
    5. Generate Script, test the model

## Improvements may done

    1. **Add Top-K Sampling** 
        Example : 
            Replace ->
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            With ->
                top_k = 20
                values, indices = torch.topk(logits, top_k)
                probs = torch.softmax(values, dim=-1)
                next_token = indices.gather(-1, torch.multinomial(probs, 1))

        This prevents weird low-probability tokens.

    2. **The more steps, the more the model give better result**

        Example: 10k, 50k will give better result

    3. **Increase Model Size**

        Try:

        n_embd = 256
        n_head = 8
        n_layer = 6


        Bigger model = better memory = more precise output.

## Commands to Run

    **To train the model**  - Python train.py

    **To create the tokenizer.pkl** - Python create_tokenizer.py

    **To test or generate output** - Python generate.py
        



