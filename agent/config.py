import psutil
import tensorflow as tf

NUM_CPU = psutil.cpu_count(logical=True)
NUM_GPU = len(tf.config.experimental.list_physical_devices('GPU'))

# For now we are always going to be running this on a GPU machine
assert NUM_GPU >= 1

print(f"Num GPUs Available: {NUM_GPU}")
print(f"Num CPUs Available: {NUM_CPU}")
