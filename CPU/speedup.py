import subprocess
import os
import matplotlib.pyplot as plt

if __name__=="__main__":
    threads = [2, 4, 8, 16]
    curr_seq_time = 1914.70
    parallel_time = [1057.3, 603.5, 452.13 ,465.86]
    speedup_list = [curr_seq_time/ x for x in parallel_time]
    
    plt.plot(threads, speedup_list,  label="Speedup")
    plt.xlabel("Number of threads")
    plt.ylabel("Speedup")
    plt.title("Speedup for OMP")
    plt.legend()
    plt.savefig('speedup.png')
