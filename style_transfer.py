import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from PIL import Image
import copy

def load_image(path, max_size=512):
    image = Image.open(path).convert("RGB")
    size = min(max(image.size), max_size)
    in_transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])
    image = in_transform(image)[:3, :, :].unsqueeze(0)
    return image.to(torch.float)

def gram_matrix(tensor):
    b, c, h, w = tensor.size()
    features = tensor.view(c, h * w)
    G = torch.mm(features, features.t())
    return G.div(c * h * w)

class StyleTransferNet:
    def __init__(self, content_img, style_img):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vgg = models.vgg19(pretrained=True).features.to(self.device).eval()
        self.content_img = content_img.to(self.device)
        self.style_img = style_img.to(self.device)
        self.target = self.content_img.clone().requires_grad_(True).to(self.device)

        self.content_layers = ['conv_4']
        self.style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

        self.content_weight = 1e4
        self.style_weight = 1e2

    def get_features(self, x):
        features = {}
        i = 0
        for layer in self.vgg.children():
            x = layer(x)
            if isinstance(layer, nn.Conv2d):
                i += 1
                name = f"conv_{i}"
                features[name] = x
        return features

    def compute_loss(self):
        target_features = self.get_features(self.target)
        content_features = self.get_features(self.content_img)
        style_features = self.get_features(self.style_img)

        content_loss = torch.mean((target_features['conv_4'] - content_features['conv_4'])**2)

        style_loss = 0
        for layer in self.style_layers:
            G = gram_matrix(target_features[layer])
            A = gram_matrix(style_features[layer])
            style_loss += torch.mean((G - A)**2)

        total_loss = self.content_weight * content_loss + self.style_weight * style_loss
        return total_loss

    def run(self, steps=300, lr=0.003):
        optimizer = optim.Adam([self.target], lr=lr)
        for step in range(steps):
            optimizer.zero_grad()
            loss = self.compute_loss()
            loss.backward()
            optimizer.step()
        return self.target.detach()

def run_style_transfer(content_path, style_path, output_path):
    content = load_image(content_path)
    style = load_image(style_path)
    model = StyleTransferNet(content, style)
    output = model.run()

    unloader = transforms.Compose([
        transforms.Normalize(mean=[-2.12, -2.04, -1.80], std=[4.37, 4.46, 4.44]),
        transforms.ToPILImage()
    ])
    image = output.cpu().squeeze(0)
    image = unloader(image)
    image.save(output_path)
