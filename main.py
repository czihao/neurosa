"""
Main Script for Neurosa Optimization Algorithm
==============================================
This file configures and runs the Neurosa optimization algorithm.

Author
-------
Zihao Chen 08/01/2024

License
-------
This work is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License.
To view a copy of this license, visit http://creativecommons.org/licenses/by-nc/4.0/.

You are free to:
- Share — copy and redistribute the material in any medium or format
- Adapt — remix, transform, and build upon the material

Under the following terms:
- Attribution — You must give appropriate credit, provide a link to the license, and indicate if changes were made. You may do so in any reasonable manner, but not in any way that suggests the licensor endorses you or your use.
- NonCommercial — You may not use the material for commercial purposes.

No additional restrictions — You may not apply legal terms or technological measures that legally restrict others from doing anything the license permits.
"""

import numpy as np
from neurosa import Neurosa

def main():
    Q = np.load('./Gsets/G15.npy')
    neurosa = Neurosa(Q)
    neurosa.run()

if __name__ == "__main__":
    main()
