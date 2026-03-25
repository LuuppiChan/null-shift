# Agent Wrapper

This is a program that wraps the Null-Shift core to create a stable agent loop where the Vector is kept in a loop until it has completed the task.

## Choosing The Project Language (VERY IMPORTANT)

One of the most important choices is choosing the correct language. Python is simple, but it's slow and bloated. The core is made with it just because it's easier for LLMs. As this is a self-written project Rust is the only correct choice with async Tokio for handling sockets.

> Rust is the only correct choice.
