#rename all images with numbers starting from 1

import os

#path to the folder with images
path = 'data/'

# Set the starting number for the new filenames
start_number = 1

# Loop through each file in the directory and rename it with a number
for filename in os.listdir(path):
    # Check if the file is an image
    if filename.lower().endswith('jpg'):
        # Create the new filename using the start number
        new_filename = f'{start_number:d}.jpg'
        
        # Rename the file
        os.rename(os.path.join(path, filename), os.path.join(path, new_filename))

        # Increment the start number for the next file
        start_number += 1

        print(f'{filename} renamed to {new_filename}')
