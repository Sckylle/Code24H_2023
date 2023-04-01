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
