#program to convert png to jpg

from PIL import Image
import os, sys

# Set the path to the directory containing the PNG images
png_directory = 'data/'

# Set the path to the directory where you want to save the JPG images
jpg_directory = 'data/'

# Loop through each PNG file in the directory and convert it to JPG format
for filename in os.listdir(png_directory):
    if filename.endswith('.png'):
        # Open the PNG image using PIL
        png_image = Image.open(os.path.join(png_directory, filename))

        # Convert the image to RGB if it has a transparency layer
        if png_image.mode in ('RGBA', 'LA'):
            png_image = png_image.convert('RGB')

        # Save the image as a JPG file
        jpg_filename = os.path.splitext(filename)[0] + '.jpg'
        jpg_image_path = os.path.join(jpg_directory, jpg_filename)
        png_image.save(jpg_image_path, 'JPEG')

        print(f'{filename} converted to {jpg_filename}')
