import os
import numpy as np
import imageio

if __name__ == '__main__':
    if not os.path.exists(os.path.join("data", "at&t_data")):
        raise FileNotFoundError("at&t dataset not found. Please download it in Kaggle at https://www.kaggle.com/datasets/kasikrit/att-database-of-faces and place the s1-s40 folders in data/at&t_data.")
    x_data = np.zeros((400, 92 * 112))
    y_data = np.zeros((400,))
    idx = 0
    for folder_name in os.listdir(path = os.path.join("data", "at&t_data")):
        if not folder_name.startswith("s"):  # Skip the README file
            continue 
        label = int(folder_name[1:]) - 1
        
        folder_path = os.path.join("data", "at&t_data", folder_name)
        for file_name in os.listdir(path = folder_path):
            file_path = os.path.join(folder_path, file_name)
            image = imageio.imread(file_path)  # read the .pgm image with imageio
            data = np.array(image) # convert the image to a numpy array
            data = data.flatten()  # 92 * 112 to 10304
            x_data[idx] = data     # add the data to the x_data array
            y_data[idx] = label    # add the label to the y_data array
            idx += 1               # increment the index
    
    # Save the data
    np.save(os.path.join("data", "at&t_x.npy"), x_data)
    np.save(os.path.join("data", "at&t_y.npy"), y_data)
    print("at&t dataset saved.")