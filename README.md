# Neurosa Optimization Algorithm

This repository contains the implementation of the Neurosa optimization algorithm for solving the Max-Cut problem. The project is divided into two main files: `neurosa.py` which defines the `Neurosa` class, and `main.py` which configures and runs the algorithm. For more detail on NeuroSA's architecture please refer to the arxiv paper https://arxiv.org/abs/2406.05224 

## Author

Zihao Chen

## License

This work is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License. To view a copy of this license, see the [LICENSE](LICENSE) file.

You are free to:
- Share — copy and redistribute the material in any medium or format
- Adapt — remix, transform, and build upon the material

Under the following terms:
- Attribution — You must give appropriate credit, provide a link to the license, and indicate if changes were made. You may do so in any reasonable manner, but not in any way that suggests the licensor endorses you or your use.
- NonCommercial — You may not use the material for commercial purposes.

No additional restrictions — You may not apply legal terms or technological measures that legally restrict others from doing anything the license permits.

## Files

### `neurosa.py`

This file contains the `Neurosa` class, which implements the core logic of the optimization algorithm.

### `main.py`

This file configures the `Neurosa` class and runs the optimization algorithm. The obtained number of cuts can be compared to the SOTA results reported at https://medium.com/toshiba-sbm/benchmarking-the-max-cut-problem-on-the-simulated-bifurcation-machine-e26e1127c0b0 

## Usage

1. Ensure you have NumPy installed:
    ```sh
    pip install numpy
    ```

2. Run the main script:
    ```sh
    python main.py
    ```

