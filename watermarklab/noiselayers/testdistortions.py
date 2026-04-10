import math
import cv2
import random
import numpy as np
from numpy import ndarray
from watermarklab.utils.basemodel import BaseTestNoiseModel

__all__ = ["Identity", "Brightness", "Contrast", "Saturation", "Hue", "GaussianBlur", "MedianFilter", "MeanFilter",
           "GaussianNoise", "UniformNoise", "SaltPepperNoise", "Cropout", "Dropout", "Resize", "Crop", "CroppedResize", "Rotate", "Flip",
           "Jpeg", "Jpeg2000", "RandomCompensateTransformer"]


class Identity(BaseTestNoiseModel):
    def __init__(self):
        pass

    def test(self, stego_img: ndarray, cover_img: ndarray = None, factor: float = 1.) -> ndarray:
        return stego_img


class Brightness(BaseTestNoiseModel):
    def __init__(self):
        """
        Initializes the Brightness class.
        """
        # 可在此处初始化相关参数
        pass

    def test(self, stego_img: ndarray, cover_img: ndarray = None, factor: float = 1.0) -> ndarray:
        """
        Adjusts the brightness of the stego image.

        Parameters:
        - stego_img: ndarray - The input image to adjust brightness.
        - cover_img: ndarray - Optional, not used in this case.
        - factor: float - Factor for brightness adjustment. A factor of 1.0 gives the original image,
                          less than 1.0 darkens the image, and greater than 1.0 brightens the image.

        Returns:
        - result_img: ndarray - The image with adjusted brightness.
        """
        # 将图像转换为浮点型后乘以因子，再剪裁到 [0,255] 范围并转换为 uint8
        result_img = np.uint8(np.clip(stego_img.astype(np.float32) * factor, 0, 255))
        return result_img


class Contrast(BaseTestNoiseModel):
    def __init__(self):
        """
        Initializes the Contrast class.
        """
        # 可在此处初始化相关参数
        pass

    def test(self, stego_img: ndarray, cover_img: ndarray = None, factor: float = 1.0) -> ndarray:
        """
        Adjusts the contrast of the stego image.

        Parameters:
        - stego_img: ndarray - The input image to adjust contrast.
        - cover_img: ndarray - Optional, not used in this case.
        - factor: float - Factor for contrast adjustment. A factor of 1.0 gives the original image,
                          less than 1.0 reduces contrast, and greater than 1.0 increases contrast.

        Returns:
        - result_img: ndarray - The image with adjusted contrast.
        """
        # 计算每个通道的均值，并以均值为基准进行对比度调整
        mean = np.mean(stego_img, axis=(0, 1), keepdims=True)
        result_img = np.uint8(np.clip((stego_img.astype(np.float32) - mean) * factor + mean, 0, 255))
        return result_img


class Saturation(BaseTestNoiseModel):
    def __init__(self):
        """
        Initializes the Saturation class.
        """
        # 可在此处初始化相关参数
        pass

    def test(self, stego_img: ndarray, cover_img: ndarray = None, factor: float = 1.0) -> ndarray:
        """
        Adjusts the saturation of the stego image.

        Parameters:
        - stego_img: ndarray - The input image to adjust saturation.
        - cover_img: ndarray - Optional, not used in this case.
        - factor: float - Factor for saturation adjustment. A factor of 1.0 gives the original image,
                          less than 1.0 reduces saturation, and greater than 1.0 increases saturation.

        Returns:
        - result_img: ndarray - The image with adjusted saturation.
        """
        # 转换到 HSV 色彩空间
        hsv_img = cv2.cvtColor(np.uint8(np.clip(stego_img, 0, 255)), cv2.COLOR_BGR2HSV).astype(np.float32)
        # 调整饱和度通道
        hsv_img[..., 1] = np.clip(hsv_img[..., 1] * factor, 0, 255)
        hsv_img = np.uint8(hsv_img)
        result_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)
        return result_img


class Hue(BaseTestNoiseModel):
    def __init__(self):
        """
        Initializes the Hue class.
        """
        # 可在此处初始化相关参数
        pass

    def test(self, stego_img: ndarray, cover_img: ndarray = None, factor: float = 0.0) -> ndarray:
        """
        Adjusts the hue of the stego image.

        Parameters:
        - stego_img: ndarray - The input image to adjust hue.
        - cover_img: ndarray - Optional, not used in this case.
        - factor: float - Factor for hue adjustment, in the range [-0.5, 0.5]. A factor of 0.0 gives the original image,
                          and non-zero values shift the hue accordingly.

        Returns:
        - result_img: ndarray - The image with adjusted hue.
        """
        # 转换到 HSV 色彩空间
        hsv_img = cv2.cvtColor(np.uint8(np.clip(stego_img, 0, 255)), cv2.COLOR_BGR2HSV).astype(np.float32)
        # 计算 hue 的偏移量。cv2 中 hue 范围为 [0, 180]，
        # 因此将 factor (范围 [-0.5, 0.5]) 扩大到 [-90, 90]，这里使用 factor * 180
        shift = factor * 180
        hsv_img[..., 0] = (hsv_img[..., 0] + shift) % 180
        hsv_img = np.uint8(hsv_img)
        result_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)
        return result_img


class GaussianBlur(BaseTestNoiseModel):
    def __init__(self):
        """
        Initializes the GaussianBlur class.
        """
        pass

    def test(self, stego_img: ndarray, cover_img: ndarray = None, sigma: float = 1.) -> ndarray:
        """
        Applies Gaussian blur distortion to the stego image.

        Parameters:
        - stego_img: np.ndarray - The input image to apply the Gaussian blur to.
        - cover_img: np.ndarray - Optional, not used in this case.
        - sigma: float - The standard deviation of the Gaussian kernel.

        Returns:
        - result_img: np.ndarray - The blurred image after applying Gaussian blur.
        """
        # Calculate kernel size based on sigma
        kernel_size = self._calculate_kernel_size(sigma)

        # Apply Gaussian blur
        noised_img = cv2.GaussianBlur(stego_img, (kernel_size, kernel_size), sigma)
        return noised_img

    def _calculate_kernel_size(self, sigma: float) -> int:
        """
        Calculate the kernel size based on the standard deviation (sigma).

        The kernel size is typically chosen as 6 * sigma + 1 to ensure that the Gaussian
        kernel covers most of the distribution according to 3-sigma rule.

        Parameters:
        - sigma: float - The standard deviation of the Gaussian kernel.

        Returns:
        - int: The calculated kernel size (odd).
        """
        kernel_size = int(6 * sigma + 1)
        # Ensure kernel size is odd
        if kernel_size % 2 == 0:
            kernel_size += 1
        return kernel_size


class MedianFilter(BaseTestNoiseModel):
    def __init__(self):
        """
        Initializes the MedianFilter class.
        """

    def test(self, stego_img: ndarray, cover_img: ndarray = None, kernel_size: int = 3) -> ndarray:
        """
        Applies median filter distortion to the stego image.

        Parameters:
        - stego_img: ndarray - The input image to apply the median filter to.
        - cover_img: ndarray - Optional, not used in this case.
        - kernel_size: int - The size of the kernel for the median filter (must be odd and > 1).

        Returns:
        - result_img: ndarray - The filtered image after applying the median filter.
        """
        result_img = cv2.medianBlur(np.uint8(np.clip(stego_img, 0., 255.)), kernel_size)
        return result_img


class MeanFilter(BaseTestNoiseModel):
    def __init__(self):
        """
        Initializes the MeanFilter class.

        Parameters:
        - kernel_size: int - Size of the kernel for the mean filter (must be odd).
        """

    def test(self, stego_img: ndarray, cover_img: ndarray = None, kernel_size: int = 3) -> np.ndarray:
        """
        Apply mean filter to the input image (stego image).

        Parameters:
        - stego_img: np.ndarray - The input image to apply the mean filter to.
        - cover_img: np.ndarray - Optional, not used in this case.

        Returns:
        - result_img: np.ndarray - The image after applying the mean filter.
        """
        # We create a kernel of size (kernel_size, kernel_size) filled with equal values.
        kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
        # Apply the filter to the image
        filtered_img = cv2.filter2D(stego_img, -1, kernel)
        return filtered_img


class GaussianNoise(BaseTestNoiseModel):
    def __init__(self, mu: float = 0):
        """
        Initializes the GaussianNoise layer.

        Args:
            mu (float): Mean of the Gaussian noise.
        """
        super(GaussianNoise, self).__init__()
        self.mu = mu  # Mean of the Gaussian noise

    def test(self, stego_img: np.ndarray, cover_img: np.ndarray = None, std: float = 0.1):
        """
        Applies Gaussian noise to the input image (supports both grayscale and color images).

        Args:
            stego_img (np.ndarray): The input image to which noise will be added.
            cover_img (np.ndarray, optional): A cover image, not used in this implementation.
            std (float): Standard deviation of the Gaussian noise.

        Returns:
            np.ndarray: The noised image.
        """
        # Generate Gaussian noise with the same shape as the input image
        noise = np.random.normal(self.mu, std, stego_img.shape) * 255.

        # Add noise to the input image
        noised_img = stego_img + noise

        # Clip the pixel values to the valid range [0, 255] (for 8-bit images)
        noised_img = np.clip(noised_img, 0, 255)

        return noised_img.astype(np.uint8)  # Convert to uint8 for image representation


class UniformNoise(BaseTestNoiseModel):
    def __init__(self):
        """
        Initializes the UniformNoise layer.
        """
        pass

    def test(self, stego_img: np.ndarray, cover_img: np.ndarray = None, intensity: float = 0.05) -> np.ndarray:
        """
        Applies uniform noise to the input image (supports both grayscale and color images).

        Args:
            stego_img (np.ndarray): The input image to which noise will be added.
            cover_img (np.ndarray, optional): A cover image, not used in this implementation.
            intensity (float): Maximum absolute deviation for the uniform noise.

        Returns:
            np.ndarray: The noised image.
        """
        # Generate uniform noise with the same shape as the input image
        noise = np.random.uniform(-intensity, intensity, stego_img.shape) * 255.
        
        # Add noise to the input image
        noised_img = stego_img + noise
        
        # Clip the pixel values to the valid range [0, 255] (for 8-bit images)
        noised_img = np.clip(noised_img, 0, 255)
        
        return noised_img.astype(np.uint8)  # Convert to uint8 for image representation


class SaltPepperNoise(BaseTestNoiseModel):
    def __init__(self):
        """
        Initializes the SaltPepperNoise layer.
        """
        super(SaltPepperNoise, self).__init__()

    def test(self, stego_img: np.ndarray, cover_img: np.ndarray = None, noise_ratio: float = 0.1) -> np.ndarray:
        """
        Applies salt-and-pepper noise to the input image (0-255 range, supports both grayscale and color images).

        Args:
            stego_img (np.ndarray): The input image to which noise will be added.
            cover_img (np.ndarray, optional): A cover image, not used in this implementation.
            noise_ratio (float): Proportion of pixels to be noised (default is 0.1).

        Returns:
            np.ndarray: The noised image.
        """
        # Ensure the noise ratio is valid
        noise_ratio = np.clip(noise_ratio, 0, 1)

        # Create a copy of the input image
        noisy_image = np.copy(stego_img)

        # Determine if the image is grayscale or color
        is_color = len(stego_img.shape) == 3  # Color image has 3 channels (height, width, channels)

        # Generate random noise mask (uniform distribution)
        if is_color:
            noise_mask = np.random.random(stego_img.shape[:2])  # Use only height and width for color images
        else:
            noise_mask = np.random.random(stego_img.shape)  # Use full shape for grayscale images

        # Add salt noise (set pixels to 255)
        if is_color:
            noisy_image[noise_mask < noise_ratio / 2] = [255, 255, 255]  # Set all channels to 255 for color images
        else:
            noisy_image[noise_mask < noise_ratio / 2] = 255  # Set pixel to 255 for grayscale images

        # Add pepper noise (set pixels to 0)
        if is_color:
            noisy_image[(noise_mask >= noise_ratio / 2) & (noise_mask < noise_ratio)] = [0, 0, 0]  # Set all channels to 0 for color images
        else:
            noisy_image[(noise_mask >= noise_ratio / 2) & (noise_mask < noise_ratio)] = 0  # Set pixel to 0 for grayscale images

        return noisy_image.astype(np.uint8)  # Ensure the output is uint8


class Cropout(BaseTestNoiseModel):
    def __init__(self, mode="cover_replace", constant: int = 1):
        """
        Initializes the Cropout operation.

        Args:
            mode (str): Operation mode, either 'cover_replace' or 'constant_replace'.
            constant (int): The constant value to use for 'constant_replace' mode.
                             Default is 1.
        """
        assert mode in ["cover_replace", "constant_replace"], "Mode must be either 'cover_replace' or 'constant_replace'."

        self.mode = mode
        self.constant = constant

    def _random_rectangle_mask(self, img: ndarray, remain_ratio) -> ndarray:
        """
        Generates a random rectangular mask for the Cropout operation.

        Args:
            img (ndarray): The input image to generate the mask.

        Returns:
            ndarray: A binary mask of the same size as the input image, where 1 indicates
                     the region to retain and 0 indicates the region to modify.
        """
        height, width, _ = img.shape
        # num_pixels = int(height * width * remain_ratio)

        # Randomly select rectangle dimensions
        # rect_width = random.randint(1, width)
        # rect_height = num_pixels // rect_width
        square_side = int((height * width * remain_ratio) ** 0.5)
        # rect_x = random.randint(0, width - rect_width)
        # rect_y = random.randint(0, height - rect_height)
        rect_x = random.randint(0, width - square_side)
        rect_y = random.randint(0, height - square_side)

        # Create and apply the rectangular mask; NumPy's broadcast rules only add 1 to the leftmost missing part of the array
        mask = np.zeros((height, width), dtype=np.float32)
        # mask[rect_y:rect_y + rect_height, rect_x:rect_x + rect_width] = 1.0
        mask[rect_y:rect_y + square_side, rect_x:rect_x + square_side] = 1.0
        return mask

    def test(self, stego_img: ndarray, cover_img: ndarray = None, remain_ratio: float = 0.9) -> ndarray:
        """
        Applies the Cropout operation to the stego image.

        Args:
            stego_img (ndarray): The image with the embedded information (stego image).
            cover_img (ndarray): The cover image used to replace the cropped regions.

        Returns:
            ndarray: The resulting image after applying the Cropout operation, with
                     the cropped-out regions either replaced by the cover image or a constant.
        """
        # Generate a random rectangular mask based on the remain_ratio
        crop_out_mask = self._random_rectangle_mask(stego_img, remain_ratio)
        crop_out_mask = np.expand_dims(crop_out_mask, axis=-1)

        if self.mode == "cover_replace":
            # Replace the cropped-out regions with the cover image
            noised_img = stego_img * crop_out_mask + (1 - crop_out_mask) * cover_img
        else:
            # Replace the cropped-out regions with the specified constant value
            noised_img = stego_img * crop_out_mask + (1 - crop_out_mask) * self.constant * 255.
        return noised_img  # Ensure pixel values are within [0, 1] range


class Dropout(BaseTestNoiseModel):
    def __init__(self, mode="cover_replace", constant: int = 1):
        """
        Initializes the Dropout class.

        Args:
            mode (str): The mode of operation, either 'cover_replace' or 'constant_replace'.
            constant (int): The constant value to replace with, used when mode is 'constant_replace'.
        """
        assert mode in ["cover_replace", "constant_replace"], "Mode must be either 'cover_replace' or 'constant_replace'."
        self.mode = mode
        self.constant = constant

    def test(self, stego_img: ndarray, cover_img: ndarray = None, drop_prob: float = 0.1):
        # Create a mask for the dropout operation based on the drop probability
        mask = np.random.rand(*stego_img.shape) > drop_prob
        if self.mode == "cover_replace":
            # Replace dropped pixels with the cover image
            noised_img = np.where(mask, stego_img, cover_img)
        else:
            # Replace dropped pixels with the constant value
            noised_img = np.where(mask, stego_img, self.constant * 255.)
        return noised_img


class Resize(BaseTestNoiseModel):
    def __init__(self, mode="bilinear"):
        """
        Initializes the Resize operation using OpenCV.

        Args:
            mode (str): Interpolation mode, either 'nearest', 'bilinear', or 'cubic'.
        """
        # Validating interpolation mode
        if mode == "nearest":
            self.mode = cv2.INTER_NEAREST
        elif mode == "bilinear":
            self.mode = cv2.INTER_LINEAR
        elif mode == "cubic":
            self.mode = cv2.INTER_CUBIC
        else:
            self.mode = cv2.INTER_LINEAR  # Default mode

    def test(self, stego_img: ndarray, cover_img: ndarray = None, scale_p=0.8) -> np.ndarray:
        """
        Perform the resizing operation on the input image using OpenCV.

        Args:
            stego_img (ndarray): Input image of shape (H, W, C) or (H, W).
            cover_img (ndarray): Not used in this operation.

        Returns:
            ndarray: Resized image of the same shape as the input.
        """
        # Get the original height and width of the image
        H, W = stego_img.shape[:2]

        # Calculate the new dimensions
        scaled_h = int(scale_p * H)
        # scaled_w = int(scale_p * W)
        scaled_w = int((scale_p + 0.1 if scale_p < 1 else scale_p - 0.1) * W)

        # Resize the image to the new dimensions
        noised_down = cv2.resize(stego_img, (scaled_w, scaled_h), interpolation=self.mode)

        # Resize the image back to the original dimensions
        # noised_img = cv2.resize(noised_down, (W, H), interpolation=self.mode)
        return noised_down
        # return noised_img


class Crop(BaseTestNoiseModel):
    def __init__(self):
        """
        Initializes the Crop operation.
        """
        pass

    def test(self, stego_img: np.ndarray, cover_img: np.ndarray = None, scale_p=0.8) -> np.ndarray:
        """
        Perform the crop operation on the input image.

        Args:
            stego_img (ndarray): Input image of shape (H, W, C) or (H, W).
            cover_img (ndarray): Not used in this operation.
            scale_p (float): Scaling factor to determine the crop size. The value should be between 0 and 1.

        Returns:
            ndarray: Cropped image.
        """
        # Get the original height and width of the image
        H, W = stego_img.shape[:2]

        # Ensure scale_p is within a valid range
        scale_p = max(0, min(1, scale_p))

        # Calculate the new dimensions after scaling
        scaled_h = int(scale_p * H)
        scaled_w = int(scale_p * W)

        # Calculate the random top-left corner for the crop
        top = random.randint(0, H - scaled_h)
        left = random.randint(0, W - scaled_w)

        # Perform the crop
        # print(top, left)
        cropped_img = stego_img[top:top + scaled_h, left:left + scaled_w]

        return cropped_img


class CroppedResize(BaseTestNoiseModel):
    def __init__(self, mode="bilinear"):
        """
        Initializes the ResizedCrop operation using OpenCV.

        Args:
            mode (str): Interpolation mode, either 'nearest', 'bilinear', or 'cubic'.
        """
        # Validating interpolation mode
        if mode == "nearest":
            self.mode = cv2.INTER_NEAREST
        elif mode == "bilinear":
            self.mode = cv2.INTER_LINEAR
        elif mode == "cubic":
            self.mode = cv2.INTER_CUBIC
        else:
            self.mode = cv2.INTER_LINEAR  # Default mode

    def test(self, stego_img: np.ndarray, cover_img: np.ndarray = None, scale_p=0.8) -> np.ndarray:
        """
        Perform the resized crop operation on the input image.

        Args:
            stego_img (ndarray): Input image of shape (H, W, C) or (H, W).
            cover_img (ndarray): Not used in this operation.
            scale_p (float): Scaling factor to determine the crop size. The value should be between 0 and 1.

        Returns:
            ndarray: Resized cropped image of the same shape as the input.
        """
        # Get the original height and width of the image
        H, W = stego_img.shape[:2]

        # Calculate the new dimensions after scaling
        scaled_h = int(scale_p * H)
        scaled_w = int(scale_p * W)

        # Calculate the random top-left corner for the crop
        top = random.randint(0, H - scaled_h)
        left = random.randint(0, W - scaled_w)

        # Perform the crop
        cropped_img = stego_img[top:top+scaled_h, left:left+scaled_w]

        # Resize the cropped image back to the original size
        noised_img = cv2.resize(cropped_img, (W, H), interpolation=self.mode)
        return noised_img


class Rotate(BaseTestNoiseModel):
    def __init__(self, mode="linear", border_mode="constant"):
        """
        Initializes the Rotate operation using OpenCV.

        Args:
            mode (str): Interpolation mode for rotation.
                        Options: "nearest", "bilinear", "cubic", "lanczos4".
            border_mode (str): OpenCV border mode for handling image borders during rotation.
                               Options: "constant", "reflect", "reflect_101", "replicate", "wrap".
                               Default is "reflect".
        """
        super(Rotate, self).__init__()
        # Set the interpolation mode
        self.mode = mode
        self.interpolation_map = {
            "nearest": cv2.INTER_NEAREST,
            "linear": cv2.INTER_LINEAR,
            "cubic": cv2.INTER_CUBIC,
            "lanczos4": cv2.INTER_LANCZOS4
        }
        self.interpolation = self.interpolation_map.get(self.mode, cv2.INTER_LINEAR)

        # Set border mode based on input string
        self.border_mode_map = {
            "constant": cv2.BORDER_CONSTANT,
            "reflect": cv2.BORDER_REFLECT,
            "reflect_101": cv2.BORDER_REFLECT_101,
            "replicate": cv2.BORDER_REPLICATE,
            "wrap": cv2.BORDER_WRAP
        }
        self.border_mode = self.border_mode_map.get(border_mode, cv2.BORDER_REFLECT)

    def test(self, stego_img: ndarray, cover_img: ndarray = None, angle: float = 90) -> np.ndarray:
        """
        Perform the rotation operation on the input image using OpenCV.

        Args:
            stego_img (ndarray): Input image of shape (H, W, C) or (H, W).
            cover_img (ndarray): Not used in this operation.
            angle (float): A positive value indicates counterclockwise rotation, while a negative value indicates clockwise rotation

        Returns:
            ndarray: Rotated image.
        """
        assert angle > 0.
        # Get the height and width of the image
        H, W = stego_img.shape[:2]
        # Calculate the center of the image
        center = (W // 2, H // 2)
        # Get the rotation matrix
        M = cv2.getRotationMatrix2D(center, angle, 1.0)  # Center, angle, scale (1.0 means no scaling)
        # Perform the rotation with selected interpolation mode and border mode
        rotated_img = cv2.warpAffine(stego_img, M, (W, H), flags=self.interpolation, borderMode=self.border_mode)
        return rotated_img


class Flip(BaseTestNoiseModel):
    def __init__(self):
        """
        Initializes the Flip operation.
        """
        super(Flip, self).__init__()

    def test(self, stego_img: ndarray, cover_img: ndarray = None, flip_mode: int = 1) -> np.ndarray:
        """
        Perform the flip operation on the input image.

        Args:
            stego_img (ndarray): Input image of shape (H, W, C) or (H, W).
            cover_img (ndarray): Not used in this operation.
            flip_mode (int): The mode of flip operation. It can be:
                             - 1 for horizontal flip (left-right).
                             - 0 for vertical flip (up-down).
                             - -1 for flipping both axes (up-down and left-right).
                             Default is 1 (horizontal).

        Returns:
            ndarray: Flipped image.
        """
        # Check if the flip_mode is valid
        if flip_mode not in [1, 0, -1]:
            raise ValueError("Invalid flip_mode. Choose from 1 (horizontal), 0 (vertical), or -1 (both).")

        # Perform the flip operation
        flipped_img = cv2.flip(stego_img, flip_mode)
        return flipped_img


class Jpeg(BaseTestNoiseModel):
    def __init__(self):
        """
        Initialize the Jpeg compression and decompression operation.

        Args:
        """
        super(Jpeg, self).__init__()

    def test(self, stego_img: ndarray, cover_img: ndarray = None, qf: int = 50) -> np.ndarray:
        """
        Apply JPEG compression and decompression on the input image.

        Args:
            stego_img (ndarray): The input image of shape (H, W, C) or (H, W).
            cover_img (ndarray): Not used in this operation.
            qf (int): Quality factor (0 to 100) for JPEG compression.

        Returns:
            ndarray: The decompressed image after applying JPEG compression.
        """
        # Determine if the image is grayscale or color
        is_grayscale = len(stego_img.shape) == 2  # Check if it's a single-channel (grayscale) image
        # Encode the image into JPEG format with the specified quality factor
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), qf]
        _, encoded_img = cv2.imencode('.jpg', stego_img, encode_param)
        # For grayscale, we use cv2.IMREAD_GRAYSCALE; for color, we use cv2.IMREAD_COLOR
        flags = cv2.IMREAD_GRAYSCALE if is_grayscale else cv2.IMREAD_COLOR
        decoded_img = cv2.imdecode(encoded_img, flags)
        return decoded_img


class Jpeg2000(BaseTestNoiseModel):
    def __init__(self):
        """
        Initialize the JPEG 2000 compression and decompression operation using OpenCV.
        """
        super(Jpeg2000, self).__init__()

    def test(self, stego_img: ndarray, cover_img: ndarray = None, qf: int = 50) -> np.ndarray:
        """
        Apply JPEG 2000 compression and decompression on the input image.

        Args:
            stego_img (ndarray): The input image of shape (H, W, C) or (H, W).
            cover_img (ndarray): Not used in this operation.

        Returns:
            ndarray: The decompressed image after applying JPEG 2000 compression.
        """
        # Determine if the image is grayscale or color
        is_grayscale = len(stego_img.shape) == 2  # Check if it's a single-channel (grayscale) image

        # Encode the image into JPEG2000 format with the specified quality factor
        encode_param = [int(cv2.IMWRITE_JPEG2000_COMPRESSION_X1000), qf]
        _, encoded_img = cv2.imencode('.jp2', stego_img, encode_param)

        # For grayscale or color images, we can use the same cv2.IMREAD_ANYDEPTH flag
        flags = cv2.IMREAD_GRAYSCALE if is_grayscale else cv2.IMREAD_COLOR
        decoded_img = cv2.imdecode(encoded_img, flags)
        return decoded_img


class RandomCompensateTransformer(BaseTestNoiseModel):
    def __init__(self):
        """
        Initializes the RandomCompensateTransformer class.
        """
        pass

    def test(self, stego_img: np.ndarray, cover_img: np.ndarray = None, shift_d: int = 8) -> np.ndarray:
        """
        Applies a random perspective transformation to the input image.

        Parameters:
        - stego_img: np.ndarray - The input image to transform.
        - cover_img: np.ndarray - Optional, not used in this case.
        - shift_d: int - Maximum displacement for the image corners during transformation.

        Returns:
        - result_img: np.ndarray - The transformed image.
        """
        H, W = stego_img.shape[:2]
        
        # Define source points (original image corners)
        points_src = np.float32([
            [0, 0],  # Top-left
            [W - 1, 0],  # Top-right
            [W - 1, H - 1],  # Bottom-right
            [0, H - 1]  # Bottom-left
        ])
        
        # Define destination points with random displacements within [-shift_d, shift_d]
        points_dst = np.float32([
            [random.uniform(-shift_d, shift_d), random.uniform(-shift_d, shift_d)],
            [random.uniform(W - 1 - shift_d, W - 1 + shift_d), random.uniform(-shift_d, shift_d)],
            [random.uniform(W - 1 - shift_d, W - 1 + shift_d), random.uniform(H - 1 - shift_d, H - 1 + shift_d)],
            [random.uniform(-shift_d, shift_d), random.uniform(H - 1 - shift_d, H - 1 + shift_d)]
        ])
        
        # Compute the perspective transformation matrix
        M = cv2.getPerspectiveTransform(points_src, points_dst)
        
        # Apply the perspective warp transformation
        result_img = cv2.warpPerspective(stego_img, M, (W, H), borderMode=cv2.BORDER_REFLECT)
        
        return result_img
