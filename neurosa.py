"""
Neurosa Optimization Algorithm
==============================
This file contains the implementation of the Neurosa class for running the optimization algorithm.

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
import math
import random

class Neurosa:
    def __init__(self, Q, max_iter=1e8, thld_max=8e4, thld_delta=2e-3):
        self.Q = Q
        self.N, self.M = Q.shape
        self.MAX_ITER = max_iter
        self.thld_max = thld_max
        self.thld_delta = thld_delta
        self.DTYPE = np.float32
        
        self.ds_p = np.zeros((self.N,), dtype=self.DTYPE)
        self.ds_n = np.zeros_like(self.ds_p, dtype=self.DTYPE)
        self.s_p = np.ones_like(self.ds_p, dtype=self.DTYPE)
        self.s_n = np.zeros_like(self.ds_p, dtype=self.DTYPE)
        self.vmem = np.matmul(Q, (self.s_p - self.s_n))
        
        self.num_spikes = 0
        self.best_maxcuts = 0
        self.maxcuts = 0
        self.iter = 0
        self.thld_iter = 0
        self.thld = 1
        
        self.v = (self.s_p - self.s_n)
        self.curr_cuts = np.sum(np.sum(np.multiply(Q, 1 - np.outer(self.v, self.v)))) / 4
    
    def run(self):
        while self.iter < self.MAX_ITER:
            p = np.random.randint(0, self.N)
            self.update_neuron(p)
            self.iter += 1
            self.thld += self.thld_delta
            
            if self.iter > 0 and self.iter % 1e5 == 0:
                print(f'#spikes:{self.num_spikes}, best: {self.best_maxcuts}')
    
    def update_neuron(self, p):
        s_p_p = self.s_p[p].item()
        s_n_p = self.s_n[p].item()
        vmem_p = self.vmem[p].item()
        
        noisethld = 2.5e4 * math.log(2.5 * random.random() + 1e-6) / (self.thld_max * math.log(1 + self.thld / self.thld_max))
        spike = 1 if (noisethld * (s_p_p - s_n_p) - vmem_p) * (s_p_p - s_n_p) < 0 else 0
        
        if spike:
            new_ds_p_p = s_n_p > 0
            new_ds_n_p = s_p_p > 0
            self.ds_p[p] = new_ds_p_p
            self.ds_n[p] = new_ds_n_p
            
            self.s_p[p] = 0 if s_p_p else 1
            self.s_n[p] = 0 if s_n_p else 1
            
            self.curr_cuts -= 1 * (new_ds_p_p - new_ds_n_p) * vmem_p
            self.vmem += 2 * (new_ds_p_p - new_ds_n_p) * self.Q[p, :]
            
            if self.curr_cuts.item() > self.best_maxcuts:
                self.best_maxcuts = self.curr_cuts.item()
            
            self.num_spikes += 1
