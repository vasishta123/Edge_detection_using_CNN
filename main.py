import numpy as np
from numpy import asarray
from matplotlib import pyplot as plt
import skimage.data


class Convolution:
    def __init__(self, num_filters, filter_size):
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.filter = np.random.randn(num_filters, filter_size, filter_size) / (filter_size * filter_size)
        # we divide above equation by size*size just to normalize the values in the filter

    # generator function to yield patches of image
    def fetch_image_patch(self, image):
        height, width = image.shape
        self.image = image
        for j in range(height - self.filter_size + 1):
            for k in range(width - self.filter_size + 1):
                image_patch = image[j: (j + self.filter_size), k: (k + self.filter_size)]
                yield image_patch, j, k

    def forward_propagation(self, image):
        height, width = image.shape
        conv_output = np.zeros((height - self.filter_size + 1, width - self.filter_size + 1, self.num_filters))
        for image_patch, i, j in self.fetch_image_patch(image):
            conv_output[i, j] = np.sum(image_patch * self.filter)
        return conv_output

    # dl/dout input coming from pooling output
    # below function updates the filter data
    def backward_propagation(self, dL_dout, learning_rate):
        dL_dF_params = np.zeros(self.filter.shape)
        for image_patch, i, j in self.fetch_image_patch(self.image):
            for k in range(self.num_filters):
                dL_dF_params[k] = image_patch * dL_dout[i, j, k]
        self.filter -= learning_rate * dL_dF_params
        return dL_dF_params


class MaxPool:
    def __init__(self, filter_size):
        self.filter_size = filter_size

    def image_region(self, image):
        new_height = image.shape[0] // self.filter_size
        new_width = image.shape[1] // self.filter_size
        self.image = image

        for i in range(new_height):
            for j in range(new_width):
                image_patch = image[(i * self.filter_size): (i * self.filter_size + self.filter_size),
                              (j * self.filter_size):(j * self.filter_size + self.filter_size)]
                yield image_patch, i, j

    def forward_propagation(self, image):
        height, width, num_filters = image.shape
        output = np.zeros((height // self.filter_size, width // self.filter_size, num_filters))

        for image_patch, i, j in self.image_region(image):
            output[i, j] = np.amax(image_patch, axis=(0, 1))
        return output

    def backward_propagation(self, dL_dout):
        dL_dmax_pool = np.zeros(self.image.shape)
        for image_patch, i, j in self.image_region(self.image):
            height, width, num_filters = image_patch.shape
            maximum_value = np.amax(image_patch, axis=(0, 1))

            for i1 in range(height):
                for j1 in range(width):
                    for k1 in range(num_filters):
                        if image_patch(i1, j1, k1) == maximum_value[k1]:
                            dL_dmax_pool[i * self.filter_size + i1, j * self.filter_size + j1, k1] = dL_dout[i, j, k1]
                return dL_dmax_pool


class Softmax:
    def __init__(self, input_node, softmax_node):
        self.weight = np.random.randn(input_node, softmax_node) / input_node
        self.bias = np.zeros(softmax_node)

    def forward_prop(self, image):
        self.orig_im_shape = image.shape
        image_modified = image.flatten()
        self.modified_input = image_modified
        print(image_modified)
        print(self.weight)
        output_val = np.dot(image_modified, self.weight) + self.bias
        self.out = output_val
        exp_out = np.exp(output_val)
        # probabilistic output
        return exp_out / np.sum(exp_out, axis=0)


    def back_prop(self, dL_dout, learning_rate):
        for i, grad in enumerate(dL_dout):
            if grad == 0:
                continue
            transformation_ep = np.exp(self.out)
            S_total = np.sum(transformation_ep)

            # gradient with respect to out (z)
            dy_dz = -transformation_ep[i]*transformation_ep / (S_total **2)
            dy_dz[i] = transformation_ep[i]*(S_total - transformation_ep[i]) / (S_total **2)

            # Gradients oftotals againest weights/biases/input
            dz_dw = self.modified_input
            dz_db = 1
            dz_d_inp = self.weight

            # Gradients of loss against totals
            dL_dz = grad * dy_dz

            ## Gradients of loss against weights/biases/input
            dL_dw = dz_dw[np.newaxis].T @ dL_dz[np.newaxis]
            dL_db = dL_dz * dz_db
            dL_d_inp = dz_d_inp @ dL_dz

            self.weight -= learning_rate * dL_dw
            self.bias -= learning_rate * dL_db

            return dL_d_inp.reshape(self.orig_im_shape)



def runCNN():
    img = skimage.data.chelsea()
    img = skimage.color.rgb2gray(img)
    conv = Convolution(18, 7)
    conv_output = conv.forward_propagation(img)
    plt.imshow(conv_output[:, :, 17], cmap='gray')
    plt.show()

    pool = MaxPool(4)
    pooling_output = pool.forward_propagation(conv_output)
    print(pooling_output.shape)
    plt.imshow(pooling_output[:, :, 17], cmap='gray')

    softmax = Softmax(73 * 111 * 18, 10)
    softmax_output = softmax.forward_prop(pooling_output)
    print(softmax_output)


if __name__ == '__main__':
    runCNN()

