import numpy as np

class ImageCompressor:
    def __init__(self):

        # Here you can set some parameters of your algorithm, e.g.
        self.dtype = np.float16
        self.U = None
        self.mean = []
        self.z = None
        self.S =None
        
    def get_codebook(self):
        codebook = np.array([],dtype = self.dtype)
        self.mean = self.mean.astype(self.dtype)
        self.U = self.U.astype(self.dtype)
        codebook = np.hstack((self.U, self.mean.reshape(-1, 1)))

        return codebook

    def train(self, train_images):
        train_images = np.array(train_images) / 255.0  # Scale the images

        # Flatten and stack images efficiently
        vectors_reshaped = train_images.reshape(len(train_images), -1).T
        # Compute the mean
        self.mean = np.mean(vectors_reshaped, axis=1)
        # Demean the data
        demeaned = vectors_reshaped - self.mean[:, np.newaxis]

        self.U, self.S, V_T = np.linalg.svd(demeaned, full_matrices=False)
        k = 18
        k_indices = np.argsort(self.S)[::-1]
        princ_comp = k_indices[:k]
        self.U = self.U[:,princ_comp ]

    def compress(self, test_image):
        new_image = test_image.flatten()
        new_image =new_image/255
        new_image -= self.mean
        self.z= np.dot(self.U.T, new_image)
        test_code = np.array(self.z)
        test_code =test_code.astype(self.dtype)
        return test_code.astype(self.dtype)


class ImageReconstructor:
    """ This class is used on the server to reconstruct images """
    def __init__(self, codebook):
        """ The only information this class may receive is the codebook """
        self.codebook = codebook
        self.U = codebook[:, :-1] 
        self.mean = codebook[:, -1]

    def reconstruct(self, test_code):
        x = np.dot(self.U, test_code) +self.mean
        im_shape = (96, 96, 3)
        recomposed = x.reshape(im_shape)
        recomposed =recomposed*255
        im_recomposed = recomposed.astype(int)

        return im_recomposed