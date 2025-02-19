# NEO

An LLM inference framework that saves GPU memory crisis by CPU offloading.

## Requirements

PyTorch >= 3.9

2 versions of g++ (see `pacpu/build.sh` for more details):

- one >= 13 (for compiling CPU kernel)
- the other < 13 (for passing the NVCC version check)

Intel ISPC compiler == 1.23, which can be installed by `sudo snap install ispc --channel latest/edge`

## Installation

1. Clone the NEO repository and `cd` into the repo.

2. Install dependencies by `pip install -r requirements.txt.`

3. Install the swiftLLM library to your local environment by `pip install -e .`

4. Build and install auxiliary GPU operators library by `pip install -e csrc`

5. Build the CPU operator library by 

   ```bash
   cd pacpu
   bash build.sh <model-name> <tensor-parallel-degree> 
   # e.g bash pacpu/build.sh llama2-7b 1
   cd ..
   ```

## Run

### offline demo

```bash
cd NEO
python examples/example.py --model-path ... --model-name ...
# e.g. python examples/example.py --model-path /home/ubuntu/weights/Llama-2-7b-hf/ --model-name llama2_7b
```

Run `python examples/example.py --help` to see more usages.

