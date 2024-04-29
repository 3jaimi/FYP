import tensorflow as tf

# Check if TensorFlow is built with GPU support
if tf.test.is_built_with_cuda():
    print("TensorFlow was built with CUDA (GPU) support.")
else:
    print("TensorFlow was not built with CUDA (GPU) support.")

# Check if a GPU device is available
if tf.test.is_gpu_available():
    print("A GPU device is available.")
else:
    print("No GPU device is available.")
