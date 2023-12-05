

import os
import numpy as np
import requests

if __name__ == "__main__":
    print("Downloading the Binary Alphabet dataset...")
    print("Downloading the .mat file...")
    url = "https://cs.nyu.edu/~roweis/data/binaryalphadigs.mat"
    os.makedirs("data", exist_ok=True)
    destination = os.path.join("data", "binaryalphadigs.mat")

    response = requests.get(url)

    if response.status_code == 200:
        with open(destination, 'wb') as file:
            file.write(response.content)
        print(f"Download the .mat successful. Saved to {destination}")
    else:
        print(f"Failed to download file. Status code: {response.status_code}")
        
    import scipy.io as sio
    mat_contents = sio.loadmat(destination)


    print("Saving the data to .npy files...")
    x_data = np.zeros((36 * 39, 20 * 16))  # 36 characters (10 digits + 26 letters) * 39 examples, 20 x 16 pixels
    y_data = np.zeros((36 * 39))  # 36 characters (10 digits + 26 letters) * 39 examples

    for character in range(36):
        for example in range(39):
            x_data[character * 39 + example] = mat_contents["dat"][character][example].flatten()
            y_data[character * 39 + example] = character
            
    np.save(os.path.join("data", "ba_x.npy"), x_data)
    np.save(os.path.join("data", "ba_y.npy"), y_data)

    print("Data saved to .npy files.")