import sys
import os

sys.path.append(os.getcwd())
from tts.cleaner import StreamCleaner
from config import config

def test(name, chunks):
    print(f"--- {name} ---")
    cleaner = StreamCleaner(config.cleaning_blocks, config.cleaning_replacements)
    full_cleaned = ""
    for chunk in chunks:
        cleaned = cleaner.process_chunk(chunk)
        print(f"Chunk: {repr(chunk)} -> Cleaned: {repr(cleaned)}")
        full_cleaned += cleaned
    full_cleaned += cleaner.flush()
    print(f"Result: {repr(full_cleaned)}")

if __name__ == "__main__":
    test("Standard Backticks", ["Hello ", "```python\ncode\n```", " world"])
    test("Tildes", ["Hello ", "~~~python\ncode\n~~~", " world"])
    test("Mixed Tags", ["<think>Thinking...</think>Visible ", "```\ncode\n```", " Done."])
    test("Unclosed Tag", ["This is <think>secretly hidden until the end"])
    test("Stars and Hashes", ["# Header\n**Bold** and *Italic* with #hash and *star*."])
    test("ASMR", ["Let's do some ASMR"])
