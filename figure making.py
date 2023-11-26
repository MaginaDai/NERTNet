import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cv2

method_titles = ['Original Image', 'Ground Truth', 'VGG-16', 'ResNet-50', 'ResNet-101']
image_name = ['Image 1', 'Image 2', 'Image 3', 'Image 4', 'Image 5', 'Image 6']

def display_images(images):
    num_images = len(images)
    num_rows = 5
    num_cols = 6

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(18, 12))
    plt.subplots_adjust(wspace=0.3, hspace=0.3)

    for i in range(num_rows):
        axs[i, 0].text(-0.2, 0.5, method_titles[i], rotation=90, va='center', ha='center', transform=axs[i, 0].transAxes, fontsize=12)

        for j in range(num_cols):
            index = i * num_cols + j
            if index <= num_images:
                axs[i, j].imshow(images[i][j])  # Change the cmap as needed
                axs[i, j].set_title(image_name[index])
                axs[i, j].axis('off')
            else:
                axs[i, j].imshow(images[i][j])
                axs[i, j].axis('off')

    plt.savefig('test.jpg', bbox_inches='tight')

# Example usage
original_image_path = []
ground_truth_image_path = []
result_model_a_path = []
result_model_b_path = []
result_model_c_path = []
for i in range(6):
    original_image_path.append(f"resNet101_result/original_image_{i}.jpg")
    ground_truth_image_path.append(f"resNet101_result/ground_truth_image_{i}.jpg")
    result_model_a_path.append(f"vgg_result/predicted_image_{i}.jpg")
    result_model_b_path.append(f"resNet50_result/predicted_image_{i}.jpg")
    result_model_c_path.append(f"resNet101_result/predicted_image_{i}.jpg")


original_image = [cv2.imread(path) for path in original_image_path] 
ground_truth_image = [cv2.imread(path) for path in ground_truth_image_path] 
result_model_a = [cv2.imread(path) for path in result_model_a_path] 
result_model_b = [cv2.imread(path) for path in result_model_b_path] 
result_model_c = [cv2.imread(path) for path in result_model_c_path] 

images = [original_image, ground_truth_image, result_model_a, result_model_b, result_model_c]

display_images(images)