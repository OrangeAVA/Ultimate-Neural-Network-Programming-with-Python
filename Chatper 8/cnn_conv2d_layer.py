import numpy as np


def conv2d(input_mat, kernel_mat):
    # Get dimensions
    input_dim = input_mat.shape[0]
    kernel_dim = kernel_mat.shape[0]


    # Check dimensions
    if kernel_dim > input_dim:
        print("Error: Kernel dimension larger than input dimension.")
        return []


    # Compute dimensions of the output feature map
    output_dim = input_dim - kernel_dim + 1


    # Initialize output feature map
    output_mat = np.zeros((output_dim, output_dim))


    # Compute convolution
    for row in range(output_dim):
        for col in range(output_dim):
            # Element-wise multiplication of the kernel and the input
            output_mat[row, col] = np.sum(kernel_mat * input_mat[row:row+kernel_dim, col:col+kernel_dim])


    return output_mat


# Define 5x5 input matrix
input_mat = np.array([[1, 2, 3, 4, 5],
                      [5, 4, 3, 2, 1],
                      [1, 2, 3, 4, 5],
                      [5, 4, 3, 2, 1],
                      [1, 2, 3, 4, 5]])


# Define 3x3 kernel
kernel_mat = np.array([[0, 1, 0],
                       [0, -1, 0],
                       [0, 1, 0]])


# Apply 2D convolution
output_mat = conv2d(input_mat, kernel_mat)


print(output_mat)


#Output: [[0. 3. 6.]
#         [6. 3. 0.]
#         [0. 3. 6.]]


import numpy as np


class Conv2D:
    def __init__(self, num_filters, filter_size):
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.filters = np.random.randn(num_filters, filter_size, filter_size) / 9  # Initialize filters


    def iterate_regions(self, image):
        h, w = image.shape
        for i in range(h - self.filter_size + 1):
            for j in range(w - self.filter_size + 1):
                im_region = image[i:(i + self.filter_size), j:(j + self.filter_size)]
                yield im_region, i, j


    def forward(self, input):
        self.last_input = input
        h, w = input.shape
        output = np.zeros((h - self.filter_size + 1, w - self.filter_size + 1, self.num_filters))
        for im_region, i, j in self.iterate_regions(input):
            output[i, j] = np.sum(im_region * self.filters, axis=(1, 2))
        return output


    def backprop(self, d_L_d_out, learning_rate):
        d_L_d_filters = np.zeros(self.filters.shape)
        for im_region, i, j in self.iterate_regions(self.last_input):
            for f in range(self.num_filters):
                d_L_d_filters[f] += d_L_d_out[i, j, f] * im_region
        self.filters -= learning_rate * d_L_d_filters
        return None  # Not necessary but helps to clarify no gradients returned