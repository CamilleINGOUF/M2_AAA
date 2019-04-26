import numpy as np
# import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

def main():
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    train_images = train_images / 255
    test_images = test_images / 255

    print(train_images.shape)
    print(train_labels.shape)
    print(test_images.shape)
    print(test_labels.shape)

    model = keras.Sequential([
        # Le modèle Sequential est un ensemble linéaire de couches
        keras.layers.Flatten(input_shape=(28,28)),
        # Transforme une matrice 28x28 en un tableau de 784
        # keras.layers.Dense(256, activation=tf.nn.relu),
        keras.layers.Dense(512, activation=tf.nn.relu),
        # Couche entièrement connectée de 128 neurones
        keras.layers.Dense(10, activation=tf.nn.softmax)
        # Couche entièrement connectée de  25 neurones:
        #    25 probabilités de sortie
    ])

    model.compile(optimizer='sgd', 
        #  On choisit la descente de gradient
        # stochastique commme optimisation
        loss='sparse_categorical_crossentropy', 
        # Définition de la mesure de perte
        # Ici l'entropie croiée
        metrics=['accuracy']
        # Définition de la mesure de performance
        # que l'on souhaite utiliser. Ici la accuracy
    )
    
    model.fit(train_images, train_labels, epochs=20)

    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print("perte: {}, accuracy: {}".format(test_loss, test_acc))
    
if __name__ == "__main__":
  main()