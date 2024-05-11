import cv2
import os
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

def resize_image(image, target_size):
    return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)

def extract_hands(image):
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)

    extracted_hand = cv2.bitwise_and(image, image, mask=skin_mask)

    return extracted_hand

def compare_images(image1, image2):
    try:
        target_size = (min(image1.shape[1], image2.shape[1]), min(image1.shape[0], image2.shape[0]))
        image1_resized = resize_image(image1, target_size)
        image2_resized = resize_image(image2, target_size)

        gray1 = cv2.cvtColor(image1_resized, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(image2_resized, cv2.COLOR_BGR2GRAY)

        similarity_index, _ = ssim(gray1, gray2, full=True)
        
        mse = mean_squared_error(gray1.flatten(), gray2.flatten())

        return similarity_index, mse
    except Exception as e:
        print(f"Error comparing images: {e}")
        return 0.0, 0.0

def process_folder(folder_path):
    images = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            image_path = os.path.join(root, file)
            images.append(image_path)
    return images

def main():
    frame_folder = "frame"
    dataset_folder = "dataset"

    frame_images = process_folder(frame_folder)
    dataset_images = process_folder(dataset_folder)

    comparison_results = []

    for frame_image_path in frame_images:
        frame_image = cv2.imread(frame_image_path)

        if frame_image is None:
            print(f"Error: Unable to read {frame_image_path}.")
            continue

        extracted_hand = extract_hands(frame_image)

        for dataset_image_path in dataset_images:
            dataset_image = cv2.imread(dataset_image_path)

            if dataset_image is None:
                print(f"Error: Unable to read {dataset_image_path}.")
                continue

            extracted_dataset_hand = extract_hands(dataset_image)

            similarity_index, mse = compare_images(extracted_hand, extracted_dataset_hand)

            ssim_threshold = 0.1
            mse_threshold = 1000

            if similarity_index > ssim_threshold and mse < mse_threshold:
                comparison_results.append({
                    'frame_image': frame_image,
                    'dataset_image': dataset_image,
                    'ssim': similarity_index,
                    'mse': mse,
                    'frame_image_path': frame_image_path,
                    'dataset_image_path': dataset_image_path
                })
           

    if not comparison_results:
        print("No similar images found.")
        return

    best_match = max(comparison_results, key=lambda x: x['ssim'])

    best_frame_image = cv2.cvtColor(best_match['frame_image'], cv2.COLOR_BGR2RGB)
    best_dataset_image = cv2.cvtColor(best_match['dataset_image'], cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(15, 7))
    
    plt.subplot(1, 2, 1)
    plt.imshow(best_frame_image)
    plt.title(f"Best Frame Image\n")



    plt.subplot(1, 2, 2)
    plt.imshow(best_dataset_image)
    plt.title(f"Best Dataset Image\n")

    plt.suptitle(f"Highest Similarity Index: {best_match['ssim']:.2f}")
    plt.show()

if __name__ == "__main__":
    main()
