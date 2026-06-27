# Null Shift

> **Note:** This project is currently in active development.

Just another agent harness. This one focuses on hands free computer usage with tools for file and browser interaction.

## Quick Start

### Prerequisites
- Python 3.12
- Some technical knowledge

### Setup & Run

#### Linux
Run these commands
```bash
# clone the repository
git clone https://github.com/LuuppiChan/null-shift.git
cd null-shift

# create virtual environment
python3.12 -m venv .venv
source .venv/bin/activate

# install dependencies
pip install -r dependencies.txt

# CHOOSE ONE!
# To use hardware acceleration on AMD (at least for me) you need to build pywhispercpp with vuklan.

# Install pywhispercpp
# NVIDIA:
pip install pywhispercpp
# AMD:
# GGML_VULKAN=1 pip install git+https://github.com/absadiki/pywhispercpp.git
```
Setup done.

To run (this needs improvement, but currently you must start two separate processes):
```bash
# in the null-shift folder
source .venv/bin/activate
# launch core with browser tools
python . core -t
```

To run the GUI
```bash
source .venv/bin/activate
python . gui
```
