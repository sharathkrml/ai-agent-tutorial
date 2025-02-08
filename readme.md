# Python Virtual Environment Setup

This guide will help you set up a Python virtual environment for this project.

## Prerequisites

- Python 3.x installed on your system
- `pip` (Python package installer)

## Setup Instructions

1. Open a terminal/command prompt in the project directory

2. Create a virtual environment:

```bash
python -m venv venv
```

3. Activate the virtual environment:

```bash
# On Windows:
venv\Scripts\activate

# On Unix or MacOS:
source venv/bin/activate
```

4. Install dependencies:

```bash
pip install -r requirements.txt
```
If you get an error about the interpreter
`Import "openai" could not be resolved`
, you can fix it by selecting the interpreter from your virtual environment.

- Opening the command palette (Cmd/Ctrl + Shift + P)
- Type "Python: Select Interpreter"
- Select the interpreter from your virtual environment (usually ends with /venv/bin/python or \venv\Scripts\python.exe)






