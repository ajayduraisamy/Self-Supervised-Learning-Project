from flask import Flask, render_template, request
import torch
import torchvision.transforms as transforms
from PIL import Image
import os

from models.encoder import get_encoder

app = Flask(__name__)

classes = [
    "airplane","automobile","bird","cat","deer",
    "dog","frog","horse","ship","truck"
]

device = torch.device("cpu")

encoder = get_encoder()
encoder.load_state_dict(torch.load("results/ssl_encoder.pth"))
encoder = encoder.to(device)
encoder.eval()

classifier = torch.nn.Linear(512,10)
classifier.load_state_dict(torch.load("results/classifier.pth"))
classifier = classifier.to(device)
classifier.eval()

transform = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor()
])


@app.route("/", methods=["GET","POST"])
def index():

    prediction = None

    if request.method == "POST":

        file = request.files["image"]

        if file:

            image = Image.open(file).convert("RGB")
            img = transform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                features = encoder(img)
                outputs = classifier(features)

                _, predicted = torch.max(outputs,1)

            prediction = classes[predicted.item()]

    return render_template("index.html", prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)