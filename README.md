# NEO

An LLM inference framework that saves GPU memory crisis by CPU offloading.

## Installation

1. Install PyTorch >= 3.9.

2. Clone the NEO repository and `cd` into the repo.

3. Install dependencies by `pip install -r requirements.txt.`

4. Install the swiftLLM library to your local environment by `pip install -e .`

5. Build and install auxiliary GPU operators library by `pip install -e csrc`

6. Build the CPU operator library by:

   ```bash
   mkdir pacpu/build
   cd pacpu/build
   cmake ..
   cmake --build .
   ```

## Run

```bash
cd NEO
python examples/example.py --model-path <model-path>
```

