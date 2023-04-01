################### tensorflow
import tensorflow as tf
import tensorflow_hub as hub

# Charger les images source et de style
source_image = tf.keras.preprocessing.image.load_img('source.jpg')
style_image = tf.keras.preprocessing.image.load_img('style.jpg')

# Charger le modèle de détection de style d'images
hub_model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

# Prétraiter les images source et de style
source_tensor = tf.keras.preprocessing.image.img_to_array(source_image)
source_tensor = tf.keras.applications.vgg19.preprocess_input(source_tensor)
style_tensor = tf.keras.preprocessing.image.img_to_array(style_image)
style_tensor = tf.keras.applications.vgg19.preprocess_input(style_tensor)

# Exécuter la détection de style d'images
stylized_tensor = hub_model(tf.constant(source_tensor), tf.constant(style_tensor))[0]

# Afficher l'image résultante
stylized_image = tf.keras.preprocessing.image.array_to_img(stylized_tensor)
stylized_image.show()

####################### pytorch

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image

# Charger les images source et de style
source_image = Image.open("source.jpg")
style_image = Image.open("style.jpg")

# Définir le modèle de réseau de neurones convolutifs
cnn = models.vgg19(pretrained=True).features

# Désactiver les mises à jour des poids du modèle
for param in cnn.parameters():
    param.requires_grad_(False)

# Définir les couches du modèle pour extraire les caractéristiques de style et de contenu
content_layers = ['conv_4']
style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

# Définir les transformateurs pour les images
transform = transforms.Compose([
    transforms.Resize(512),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Prétraiter les images source et de style
source_tensor = transform(source_image).unsqueeze(0)
style_tensor = transform(style_image).unsqueeze(0)

# Définir la fonction de coût pour mesurer la différence entre les styles des images
class StyleLoss(nn.Module):
    def __init__(self, target_features):
        super(StyleLoss, self).__init__()
        self.target = self.gram_matrix(target_features).detach()

    def gram_matrix(self, input):
        a, b, c, d = input.size()
        features = input.view(a * b, c * d)
        G = torch.mm(features, features.t())
        return G.div(a * b * c * d)

    def forward(self, input):
        G = self.gram_matrix(input)
        return torch.nn.functional.mse_loss(G, self.target)

# Exécuter l'apprentissage de la détection de style d'images
optimizer = optim.Adam([source_tensor.requires_grad_()], lr=0.003)
num_steps = 2000
for i in range(num_steps):
    optimizer.zero_grad()
    source_features = cnn(source_tensor)
    style_features = cnn(style_tensor)
    content_loss = nn.functional.mse_loss(source_features[0], style_features[0])
    style_loss = 0
    for layer in style_layers:
        target_features = style_features[layer]
        source_features = source_features[layer]
        style_loss += Style
