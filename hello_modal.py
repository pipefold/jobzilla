import modal

app = modal.App("hello-modal")

@app.function()
def hello(name: str = "world"):
    """A simple Modal function that returns a greeting."""
    return f"Hello, {name}!"

@app.local_entrypoint()
def main():
    """Run the hello function locally and print the result."""
    result = hello.remote("Modal")
    print(result)
