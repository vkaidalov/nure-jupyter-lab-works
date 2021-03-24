import os

import cv2
import numpy as np
import matplotlib.pyplot as plt

from gaussian_low_pass import gaussian_low_pass


INPUT_DIR = 'lw1_filtering_input_images'
OUTPUT_DIR = 'lw1_filtering_output_images'

os.makedirs(OUTPUT_DIR, exist_ok=True)

for filename in os.listdir(INPUT_DIR):
    filepath = os.path.join(INPUT_DIR, filename)
    filename_ext = filename.split('.')[-1]
    if filename_ext not in ('jpg', 'jpeg', 'png'):
        continue
    filename_without_ext = '.'.join(filename.split('.')[:-1])
    
    print(f'Processing {filepath}...')
    
    img = cv2.imread(filepath)
    img_grayscale = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # original and grayscale versions
    plt.figure(figsize=(6.4*3, 4.8*3), constrained_layout=False)

    plt.subplot(121), plt.imshow(img), plt.title("Original Image")
    plt.subplot(122), plt.imshow(img_grayscale, "gray"), plt.title("Grayscale Image")

    plt.savefig(
        os.path.join(OUTPUT_DIR, f'{filename_without_ext}_1_grayscale.png'),
        bbox_inches='tight'
    )
    
    plt.figure(figsize=(6.4*3, 4.8*3), constrained_layout=False)
    
    # spectrum and shifted spectrum
    img_fft2 = np.fft.fft2(img_grayscale)
    img_fft2_shifted = np.fft.fftshift(img_fft2)

    plt.subplot(121), plt.imshow(np.log(1 + np.abs(img_fft2)), "gray"), plt.title("Spectrum")
    plt.subplot(122), plt.imshow(np.log(1 + np.abs(img_fft2_shifted)), "gray"), plt.title("Shifted Spectrum")

    plt.savefig(
        os.path.join(OUTPUT_DIR, f'{filename_without_ext}_2_spectrum.png'),
        bbox_inches='tight'
    )
    
    # applying low pass Gaussian filters with different values of D
    result_images = []
    for diameter in (50, 25, 10):
        lp_filter = gaussian_low_pass(diameter, img.shape)
        result_images.append((
            diameter,
            np.abs(
                np.fft.ifft2(
                    np.fft.ifftshift(
                        img_fft2_shifted * lp_filter
                    )
                )
            )
        ))
            
    fig, axs = plt.subplots(2, 2, figsize=(6.4 * 4, 4.8 * 4))
    axs[0, 0].imshow(img_grayscale, "gray")
    axs[0, 0].title.set_text("Grayscale Original")
    axs[0, 1].imshow(result_images[0][1], "gray")
    axs[0, 1].title.set_text(f"LP Gaussian, D={result_images[0][0]}")
    axs[1, 0].imshow(result_images[1][1], "gray")
    axs[1, 0].title.set_text(f"LP Gausiian, D={result_images[1][0]}")
    axs[1, 1].imshow(result_images[2][1], "gray")
    axs[1, 1].title.set_text(f"LP Gaussian, D={result_images[2][0]}")

    plt.savefig(
        os.path.join(OUTPUT_DIR, f'{filename_without_ext}_3_results.png'),
        bbox_inches='tight'
    )
    
    plt.close('all')
    
    print('Done.')
