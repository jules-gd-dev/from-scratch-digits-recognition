import numpy as np
import requests
import io
from collections import Counter
import random

def load_data(url):
    """Télécharge et prépare les données MNIST depuis une URL."""
    print(f"Téléchargement depuis {url}...")
    response = open(url, 'r')

    # On transforme le texte téléchargé en un "fichier en mémoire"
    data_as_file = io.StringIO(response.read())
    
    # On lit les données depuis notre fichier en mémoire
    # Pas besoin de skiprows car ce fichier n'a pas d'en-tête
    data = np.loadtxt(data_as_file, delimiter=',')
    
    print("Données chargées ! ✅")
    print("Dimensions du tableau chargé :", data.shape)

    # On sépare les étiquettes des images
    labels = data[:, 0]
    images = data[:, 1:]

    return images, labels

def predict(test_image, train_images, train_labels, k=10):
    # Étape 1 : Calculer la distance entre notre test_image et 
    #            TOUTES les images du catalogue d'entraînement.
    # On peut faire ça sans boucle for grâce à la puissance de NumPy !
    distances = np.sqrt(np.sum((train_images - test_image)**2, axis=1))

    # Étape 2 : Trouver les indices des k plus proches voisins.
    # On a déjà parlé de np.argsort() pour ça.
    nearest_neighbor_indices = np.argsort(distances)[:k]

    # Étape 3 : Récupérer les étiquettes de ces voisins.
    nearest_neighbor_labels = train_labels[nearest_neighbor_indices]
    
    # Étape 4 : Faire le vote pour trouver l'étiquette la plus fréquente.
    # Counter est un outil pratique pour ça.
    vote = Counter(nearest_neighbor_labels)
    prediction = vote.most_common(1)[0][0]
    
    return prediction


# --- Programme Principal ---
# 1. On charge le catalogue d'entraînement
train_file = "./mnist_train.csv"

train_images, train_labels = load_data(train_file)

# 2. On charge de NOUVELLES images pour le test
test_url = "./mnist_test.csv"
print("\nChargement des données de test...")
test_images, test_labels = load_data(test_url)

# 3. On choisit la toute première image de test
index = random.randint(0, len(test_images) - 1)
first_test_image = test_images[index]
first_test_label = int(test_labels[index])

# 4. On fait la prédiction !
# C'est ici que tu entres en jeu.
predicted_label = predict(first_test_image, train_images, train_labels)

fail = 0
success = 0

accr_over_time = []

for i in range(len(test_images)):
    predicted_label = predict(test_images[i], train_images, train_labels)
    print(f"La prédiction pour l'image {i} est: {predicted_label}", end="")
    if predicted_label == test_labels[i]:
        success += 1
    else:
        fail += 1

    print(f" - {'✅' if predicted_label == test_labels[i] else '❌'}; Accuracy: {success / (success + fail) * 100}")
    accr_over_time.append(success / (success + fail) * 100)

print(f"Accuracy: {success / (success + fail) * 100}")
open("accr_over_time.txt", "w").write(str(accr_over_time))