import time
import pynvml

def grab_gpu(frequency=1, total=3600 * 24, gpu_mem=9):
    for t in range(total):
        max_value, max_index = -1, -1
        for i in range(12):
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
            free_mem = meminfo.free / 1024 / 1024 / 1024
            if free_mem > max_value:
                max_value = free_mem
                max_index = i
        print(max_value, max_index)
        if max_value >= gpu_mem:
            return max_index
        time.sleep(frequency)
    return -1

if __name__ == '__main__':
    grab_gpu()