import numpy as np
import tensorflow as tf

def _normalize_at_kscale(input_image: np.array, k: int) -> np.array:
    """
    Normalizing image between 0-k
    Args:
        input_image (np.array): input image for processing
        k (int): scaling factor
    Returns:
        scaled_image (np.array): scaled image between 0 and k 
    """
    input_image = tf.cast(input_image, tf.float32) / 255.0
    scaled_image = k*input_image

    return scaled_image


class TestNormalizeAtKScale(tf.test.TestCase):
    def test_normalize_at_kscale(self):
        # Create test input image
        input_image = np.array([[50, 100, 150], [200, 250, 255]], dtype=np.uint8)

        # Define expected output
        expected_output = np.array([[0.2, 0.4, 0.6], [0.8, 1.0, 1.0]], dtype=np.float32)

        # Call the function to get the actual output
        actual_output = _normalize_at_kscale(input_image, k=1.0)

        # Assert that the actual output matches the expected output
        self.assertAllClose(actual_output, expected_output, rtol=1e-1, atol=1e-1)


if __name__ == '__main__':
    tf.test.main()
