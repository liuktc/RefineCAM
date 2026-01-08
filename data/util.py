import numpy as np
import cv2
import torch
from pytorch_grad_cam.metrics.road import NoisyLinearImputer
from .perlin2d import generate_perlin_noise_2d


def draw_circle(result, mask, x, y, size, color, angle):
    cv2.circle(result, (x, y), size // 2, color, -1)
    cv2.circle(mask, (x, y), size // 2, (255, 255, 255), -1)


def draw_circle_frame(result, mask, x, y, size, color, angle):
    """Draws a circular frame instead of a filled circle."""
    thickness = int(size // 5)
    radius = size // 2
    # Convert thickness to integer
    cv2.circle(result, (x, y), radius, color, thickness)
    cv2.circle(mask, (x, y), radius, (255, 255, 255), thickness)


def draw_square(result, mask, x, y, size, color, angle):
    # Create square points
    half_size = size // 2
    square_pts = np.array(
        [
            [-half_size, -half_size],
            [half_size, -half_size],
            [half_size, half_size],
            [-half_size, half_size],
        ],
        dtype=np.float32,
    )
    # Create rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D((0, 0), angle, 1.0)
    # Rotate points
    rotated_pts = np.dot(square_pts, rotation_matrix[:, :2].T) + rotation_matrix[:, 2]
    # Translate to final position
    rotated_pts = rotated_pts + [x, y]
    # Draw rotated square
    cv2.fillPoly(result, [rotated_pts.astype(np.int32)], color)
    cv2.fillPoly(mask, [rotated_pts.astype(np.int32)], (255, 255, 255))


def draw_square_frame(result, mask, x, y, size, color, angle):
    # Create square corner points
    half_size = size // 2
    square_pts = np.array(
        [
            [-half_size, -half_size],
            [half_size, -half_size],
            [half_size, half_size],
            [-half_size, half_size],
        ],
        dtype=np.float32,
    )

    # Create rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D((0, 0), angle, 1.0)

    # Rotate points
    rotated_pts = np.dot(square_pts, rotation_matrix[:, :2].T) + rotation_matrix[:, 2]

    # Translate to final position
    rotated_pts = rotated_pts + [x, y]

    # Convert to integer
    pts = rotated_pts.astype(np.int32).reshape((-1, 1, 2))

    # Draw the square frame
    thickness = int(size // 5)
    cv2.polylines(result, [pts], isClosed=True, color=color, thickness=thickness)
    cv2.polylines(
        mask, [pts], isClosed=True, color=(255, 255, 255), thickness=thickness
    )


def draw_triangle(result, mask, x, y, size, color, angle):
    # Create triangle points
    half_size = size // 2
    triangle_pts = np.array(
        [[0, -half_size], [-half_size, half_size], [half_size, half_size]],
        dtype=np.float32,
    )
    # Create rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D((0, 0), angle, 1.0)
    # Rotate points
    rotated_pts = np.dot(triangle_pts, rotation_matrix[:, :2].T) + rotation_matrix[:, 2]
    # Translate to final position
    rotated_pts = rotated_pts + [x, y]
    # Draw rotated triangle
    cv2.fillPoly(result, [rotated_pts.astype(np.int32)], color)
    cv2.fillPoly(mask, [rotated_pts.astype(np.int32)], (255, 255, 255))


def draw_triangle_frame(result, mask, x, y, size, color, angle):
    """Draws a triangular frame instead of a filled triangle."""
    half_size = size // 2

    # Define triangle points
    triangle_pts = np.array(
        [[0, -half_size], [-half_size, half_size], [half_size, half_size]],
        dtype=np.float32,
    )

    # Create rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D((0, 0), angle, 1.0)

    # Rotate points
    rotated_pts = np.dot(triangle_pts, rotation_matrix[:, :2].T) + rotation_matrix[:, 2]

    # Translate to final position
    rotated_pts = rotated_pts + [x, y]

    # Convert to integer and reshape
    pts = rotated_pts.astype(np.int32).reshape((-1, 1, 2))

    thickness = int(size // 5)
    # Draw the triangle frame
    cv2.polylines(result, [pts], isClosed=True, color=color, thickness=thickness)
    cv2.polylines(
        mask, [pts], isClosed=True, color=(255, 255, 255), thickness=thickness
    )


LABELS_TO_SHAPE_FUNCTION = {
    0: draw_circle,
    1: draw_square,
    2: draw_triangle,
    3: draw_circle_frame,
    4: draw_square_frame,
    5: draw_triangle_frame,
}


def draw_random_shapes(
    image,
    shape_type: int,
    num_shapes=5,
    size_range=(20, 100),
    seed: int = 0,
):
    # Use the seed
    np.random.seed(seed)
    # Make a copy of the image to avoid modifying the original
    result = image.copy()
    mask = np.zeros_like(result)
    height, width = image.shape[:2]

    # Generate all random numbers at once using numpy
    colors = np.random.randint(0, 256, size=(num_shapes, 3))  # RGB colors
    if int(size_range[0]) > width//4 or int(size_range[0]) > height//4:
        positions_x = np.random.randint(0 + int(size_range[0]), width-int(size_range[0]), size=num_shapes)
        positions_y = np.random.randint(0 + int(size_range[0]), height-int(size_range[0]), size=num_shapes)
    else:
        positions_x = np.random.randint(0 + 2*int(size_range[0]), width - 2*int(size_range[1]), size=num_shapes)
        positions_y = np.random.randint(0 + 2*int(size_range[0]), height - 2*int(size_range[1]), size=num_shapes)
    sizes = np.random.randint(
        size_range[0], size_range[1], size=num_shapes
    )  # Random sizes between 20 and 100

    # Generate random angles between 0 and 360 degrees
    angles = np.random.uniform(0, 360, size=num_shapes)
    for i in range(num_shapes):
        color = tuple(map(int, colors[i]))  # Convert to tuple for OpenCV
        x = positions_x[i]
        y = positions_y[i]
        size = sizes[i]
        angle = angles[i]
        shape_function = LABELS_TO_SHAPE_FUNCTION[shape_type]
        shape_function(result, mask, x, y, size, color, angle)

    # Make the mask binary
    mask[mask > 0] = 255

    return result, mask


def blur_image(image: torch.Tensor, mask: torch.Tensor):
    # Blue the image using the ROAD linear imputer
    imputer = NoisyLinearImputer(noise=0.01)
    blurred_image = imputer(image, mask)

    return blurred_image


class BlurImagePerlinNoise(object):
    def __init__(self, noise=0.01, scale=8, threshold=0.5):
        self.noise = noise
        self.scale = scale
        self.threshold = threshold

    def __call__(self, image: torch.Tensor):
        # Generate Perlin noise
        # Convert image to float32
        image = image.float()
        print(image.shape[-2:], (self.scale, self.scale))

        noise = generate_perlin_noise_2d(image.shape[-2:], (self.scale, self.scale))
        noise = torch.Tensor((noise - noise.min()) / (noise.max() - noise.min()))
        noise_mask = noise > self.threshold
        # noise_mask = torch.Tensor(noise) > self.threshold
        noise[noise_mask] = 1
        noise[~noise_mask] = 0
        # mask = perlin_noise(image.shape[-2:], self.scale, self.threshold)
        # mask = torch.Tensor(noise).to(image.device)

        # Blur the image using the ROAD linear imputer
        imputer = NoisyLinearImputer(noise=self.noise)
        print(image.shape, noise.shape)
        print(type(image), type(noise))
        blurred_image = imputer(image, noise)
        # Apply gaussian blur
        # blurred_image = cv2.GaussianBlur(image.cpu().numpy(), (15, 15), 0)
        # blurred_image = torch.Tensor(blurred_image).to(image.device)
        # image[:, noise_mask] = blurred_image[:, noise_mask]
        return blurred_image


class Binarize(object):
    def __init__(self):
        pass

    def __call__(self, img: torch.Tensor):
        # Sum over the channels
        img = img.sum(dim=0, keepdim=True)
        img[img > 0] = 1
        img[img <= 0] = 0
        return img
