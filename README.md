# Modal Python Getting Started

A simple repository to deploy Python code to Modal.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up Modal authentication:
```bash
modal token set --token-id <your-token-id> --token-secret <your-token-secret>
```

Or use the interactive setup:
```bash
modal setup
```

## Running the Example

Run the hello world example:
```bash
modal run hello_modal.py
```

This will execute the `hello` function on Modal's infrastructure and print the result locally.

## What's Included

- `hello_modal.py`: A simple Modal app with a basic function that returns a greeting
- `requirements.txt`: Python dependencies (Modal SDK)

## Next Steps

- Learn more at [Modal's documentation](https://modal.com/docs)
- Try adding more complex functions
- Deploy web endpoints with `@app.function(web=True)`
- Schedule functions with `@app.function(schedule=...)`
