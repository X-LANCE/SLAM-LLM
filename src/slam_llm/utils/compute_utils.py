
def calculate_output_length_1d(L_in, kernel_size, stride, padding=0):
    return (L_in + 2 * padding - kernel_size) // stride + 1