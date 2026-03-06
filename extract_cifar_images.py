import pickle
import os
import numpy as np
from PIL import Image

data_path = "data/cifar-10-batches-py/test_batch"
output_folder = "test_images"

os.makedirs(output_folder, exist_ok=True)

label_names = [
    "airplane","automobile","bird","cat","deer",
    "dog","frog","horse","ship","truck"
]

with open(data_path, "rb") as f:
    batch = pickle.load(f, encoding="bytes")

images = batch[b"data"]
labels = batch[b"labels"]

for i in range(len(images)):

    img = images[i]

    img = img.reshape(3,32,32)
    img = np.transpose(img,(1,2,0))

    label = label_names[labels[i]]

    img = Image.fromarray(img)

    img.save(f"{output_folder}/{label}_{i}.png")

print("Extraction complete")
print("Images saved in:", output_folder)