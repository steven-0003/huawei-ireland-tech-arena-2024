# Huawei Ireland Tech Arena 2024 - Server Fleet Management

## The Problem

The goal is to build a model that at each time-step recommends the number of servers
of each type to deploy at each data-center in order to maximize the given objective
function, which is to maximise profit. You can read the full problem outline [here](./tech_arena_24_phase_2.pdf)

## Prerequisites

- Python 3.11 or higher

- Create a venv (see [here](https://docs.python.org/3/library/venv.html)) and install requirements.txt

For example in Windows:

```console
# Create a venv
py -3.11 -m venv .venv

# Activate venv
./.venv/Scripts/Activate.ps1

# Install requirements
pip install -r requirements.txt
```

## Usage

1. Create a folder in the root directory called *output*

2. Run the following command to produce the solution. The objective score will be displayed if in `mysolution.py`, the verbose parameter in `DecisionMaker` is set to 1 (which is by default)

```console
python mysolution.py
```

3. If you would like to confirm the objective score, you can run the following command. This must be done after running `mysolution.py`, so the correct json(s) are evaluated.

```console
python evaluation_output.py
```