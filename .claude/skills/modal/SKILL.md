# Modal Development Skill

Expert assistant for developing, deploying, and managing applications on Modal (modal.com), a serverless cloud platform for Python.

## Metadata

- **name**: modal
- **version**: 1.0.0
- **author**: Claude Code
- **category**: cloud-development
- **tags**: modal, serverless, python, cloud, deployment

## Description

This skill provides expert guidance and assistance for developing applications on Modal. Modal is a serverless platform that lets you run Python code in the cloud with GPU access, scheduled tasks, web endpoints, and more.

## Core Capabilities

### 1. Modal App Development
- Create and structure Modal apps with proper decorators
- Implement Modal functions with appropriate configurations
- Set up local entrypoints for testing
- Design scalable serverless architectures

### 2. Deployment & Execution
- Run Modal functions locally and remotely
- Deploy apps to Modal infrastructure
- Manage different deployment environments
- Handle version control and updates

### 3. Modal Features
- **Web Endpoints**: Create HTTP endpoints with `@app.function(web=True)`
- **Scheduled Functions**: Set up cron jobs with `@app.function(schedule=...)`
- **GPU/Hardware**: Configure GPU, CPU, and memory requirements
- **Container Images**: Customize Python environments and dependencies
- **Volumes**: Manage persistent storage with Modal volumes
- **Secrets**: Handle environment variables and API keys securely
- **Parallel Execution**: Use `.map()` for parallel processing
- **Streaming**: Implement streaming responses

### 4. Authentication & Setup
- Guide Modal authentication process
- Set up Modal tokens and credentials
- Configure local development environment
- Manage Modal profiles and organizations

### 5. Debugging & Monitoring
- View and analyze Modal logs
- Debug function execution issues
- Monitor app performance and costs
- Handle errors and timeouts

### 6. Best Practices
- Follow Modal's recommended patterns
- Optimize for cost and performance
- Structure code for maintainability
- Handle cold starts efficiently
- Implement proper error handling

## Workflow

When helping with Modal development:

1. **Understand the requirement**: What functionality needs to run on Modal?
2. **Choose the right Modal features**: Web endpoint, scheduled function, batch processing, etc.
3. **Implement the solution**: Write clean, efficient Modal code
4. **Test locally first**: Use `modal run` for quick testing
5. **Deploy and verify**: Deploy to Modal and check logs
6. **Optimize if needed**: Adjust resources, caching, or architecture

## Common Commands

```bash
# Authentication
modal setup
modal token set --token-id <id> --token-secret <secret>

# Running & Deployment
modal run <script.py>
modal run <script.py>::<function_name>
modal deploy <script.py>

# Environment Management
modal environment create <name>
modal environment list

# Logs & Debugging
modal app logs <app-name>
modal app list
modal app stop <app-name>

# Volume Management
modal volume list
modal volume create <name>

# Secret Management
modal secret list
modal secret create <name>
```

## Code Examples

### Basic Modal Function
```python
import modal

app = modal.App("my-app")

@app.function()
def process_data(data: dict):
    # Your logic here
    return {"result": "processed"}

@app.local_entrypoint()
def main():
    result = process_data.remote({"key": "value"})
    print(result)
```

### Web Endpoint
```python
from modal import App, web_endpoint

app = App("web-api")

@app.function()
@web_endpoint(method="POST")
def api_handler(data: dict):
    # Handle HTTP request
    return {"status": "success", "data": data}
```

### Scheduled Function
```python
from modal import App, Cron

app = App("scheduled-task")

@app.function(schedule=Cron("0 9 * * *"))  # Daily at 9am
def daily_task():
    # Task logic
    print("Running daily task")
```

### GPU Function
```python
from modal import App, gpu

app = App("ml-inference")

@app.function(gpu="A10G")
def run_model(input_data):
    # Use GPU for ML inference
    import torch
    # Your ML code here
    return result
```

### Custom Container Image
```python
from modal import App, Image

image = Image.debian_slim().pip_install(
    "numpy",
    "pandas",
    "scikit-learn"
)

app = App("data-processing", image=image)

@app.function()
def process():
    import pandas as pd
    # Your code using installed packages
```

### Parallel Processing
```python
@app.function()
def process_item(item):
    # Process single item
    return result

@app.local_entrypoint()
def main():
    items = [1, 2, 3, 4, 5]
    # Process all items in parallel
    results = list(process_item.map(items))
    print(results)
```

## Key Concepts

### Modal Apps
- Container for related functions
- Defines shared configuration and image
- Can be deployed as a unit

### Functions
- Decorated with `@app.function()`
- Run in isolated containers
- Can be called locally or remotely

### Local Entrypoints
- Decorated with `@app.local_entrypoint()`
- Entry point for running the app
- Executes locally, calls remote functions

### Images
- Define the container environment
- Include Python version, packages, system dependencies
- Can be customized and cached

### Volumes
- Persistent storage across function calls
- Useful for datasets, models, caches
- Can be shared between functions

### Secrets
- Securely store API keys and credentials
- Accessed as environment variables
- Managed through Modal dashboard or CLI

## Troubleshooting

### Common Issues

**Import Errors**
- Ensure packages are in the Modal image definition
- Use `.pip_install()` or `.apt_install()` on the image

**Authentication Failures**
- Run `modal setup` or check token validity
- Verify organization access

**Function Timeouts**
- Increase timeout with `@app.function(timeout=600)`
- Optimize function logic

**Cold Starts**
- Use `@app.function(keep_warm=1)` for critical paths
- Optimize image build time

**Volume Access Issues**
- Ensure volume is created and mounted correctly
- Check permissions and paths

## Resources

- [Modal Documentation](https://modal.com/docs)
- [Modal Examples](https://github.com/modal-labs/modal-examples)
- [Modal Python SDK Reference](https://modal.com/docs/reference)
- [Modal Blog & Tutorials](https://modal.com/blog)

## When to Use This Skill

Invoke this skill when:
- Creating new Modal applications or functions
- Deploying Python code to Modal infrastructure
- Setting up web endpoints, scheduled tasks, or batch jobs
- Configuring GPU access for ML workloads
- Managing Modal authentication and environments
- Debugging Modal app issues
- Optimizing Modal app performance or costs
- Converting existing Python code to run on Modal

## Quality Standards

When working with Modal:
- ✅ Always test with `modal run` before deploying
- ✅ Use type hints for function parameters
- ✅ Include docstrings for complex functions
- ✅ Handle errors gracefully with try/except
- ✅ Use appropriate resource limits (CPU, memory, timeout)
- ✅ Follow Modal's naming conventions (kebab-case for app names)
- ✅ Keep secrets in Modal's secret management, not in code
- ✅ Use volumes for large datasets instead of bundling in image
- ✅ Optimize images by caching dependencies
- ✅ Monitor logs and costs in Modal dashboard

## Integration with Project

This skill is designed to work with your Modal Python projects. It assumes:
- Python 3.8+ installed locally
- Modal CLI installed (`pip install modal`)
- Modal account and authentication configured
- Project follows standard Python structure

The skill will help you develop Modal apps efficiently while following best practices and leveraging Modal's full feature set.
