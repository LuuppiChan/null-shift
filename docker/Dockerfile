# Use a slim, secure base image
FROM python:3.12-slim

# Create a non-root user for the agent
RUN useradd -m agentuser
USER agentuser
WORKDIR /home/agentuser

# Install required external modules
RUN pip install --no-cache-dir numpy pandas requests scipy sympy pydantic

# Set the entrypoint to python
ENTRYPOINT ["python", "-c"]
