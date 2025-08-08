# Code by Jules GAY--DONAT - https://github.com/jules-gd-dev
# This is a simple implementation of a Neural Network in Python
# We aren't using any libraries, just numpy. It's my first time creating this stuff. I know it's not the most efficient way to do it.

import numpy as np

# --- CLASSE DU RÉSEAU DE NEURONES ---

class NeuralNetwork:
    # Initialisation du réseau
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        self.inodes = input_nodes
        self.hnodes = hidden_nodes
        self.onodes = output_nodes
        self.lr = learning_rate
        
        # Initialisation des matrices de poids avec des nombres aléatoires
        # (entre -0.5 et 0.5)
        self.weights_ih = np.random.rand(self.hnodes, self.inodes) - 0.5
        self.weights_ho = np.random.rand(self.onodes, self.hnodes) - 0.5
        
        # La fonction d'activation sigmoïde
        self.activation_function = lambda x: 1 / (1 + np.exp(-x))

    # Entraînement du réseau sur un exemple
    def train(self, inputs_list, targets_list):
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T
        
        # --- Calcul vers l'avant (Prédiction) ---
        hidden_inputs = np.dot(self.weights_ih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        
        final_inputs = np.dot(self.weights_ho, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        
        # --- Calcul de l'erreur ---
        output_errors = targets - final_outputs
        hidden_errors = np.dot(self.weights_ho.T, output_errors)
        
        # --- Mise à jour des poids (Apprentissage) ---
        self.weights_ho += self.lr * np.dot((output_errors * final_outputs * (1.0 - final_outputs)), np.transpose(hidden_outputs)) 
        self.weights_ih += self.lr * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), np.transpose(inputs))

    # Prédiction pour une image
    def predict(self, inputs_list):
        inputs = np.array(inputs_list, ndmin=2).T
        
        hidden_inputs = np.dot(self.weights_ih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        
        final_inputs = np.dot(self.weights_ho, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        
        return final_outputs

def evaluate(network, test_data_list):
    # On crée une liste pour stocker nos scores (bonne ou mauvaise réponse)
    scorecard = []
    
    # On parcourt toutes les images du jeu de test
    for record in test_data_list:
        all_values = record.split(',')
        # L'étiquette correcte est le premier nombre
        correct_label = int(all_values[0])
        # Les pixels sont le reste
        inputs = (np.array(all_values[1:], dtype=float) / 255.0 * 0.99) + 0.01
        
        # On demande au réseau de prédire le chiffre
        outputs = network.predict(inputs)
        # L'étiquette prédite est l'indice du neurone avec le plus haut score
        predicted_label = np.argmax(outputs)
        
        # On compare la prédiction à la vraie réponse
        if (predicted_label == correct_label):
            scorecard.append(1) # Bonne réponse
        else:
            scorecard.append(0) # Mauvaise réponse
            
    # On calcule le score de précision en faisant la moyenne des scores
    performance = np.mean(scorecard)
    return performance

# --- PROGRAMME PRINCIPAL ---

# 1. Définir l'architecture
input_nodes = 784
hidden_nodes = 100
output_nodes = 10
learning_rate = 0.1

# 2. Créer une instance du réseau
net = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

# 3. Charger les données d'entraînement et de test
with open("mnist_train.csv", 'r') as training_data_file:
    training_data_list = training_data_file.readlines()
with open("mnist_test.csv", 'r') as test_data_file:
    test_data_list = test_data_file.readlines()

# 4. Entraînement sur plusieurs époques
epochs = 10
performance_log = []

print("--- Beginning training ---")
for e in range(epochs):
    # Entraîner le réseau sur chaque image du jeu d'entraînement
    for record in training_data_list:
        all_values = record.split(',')
        inputs = (np.array(all_values[1:], dtype=float) / 255.0 * 0.99) + 0.01 # Passe les pixels de 0 à 255 en 0 à 0.99
        targets = np.zeros(output_nodes) + 0.01 # Création de 10 vecteurs de 0.01
        targets[int(all_values[0])] = 0.99 # Le vecteur correspondant au chiffre cible est mis à 0.99
        net.train(inputs, targets) # Entrainer sur l'image
        
    # À la fin de l'époque, évaluer la performance sur le jeu de test
    performance = evaluate(net, test_data_list) # Évaluer sur le jeu de test
    performance_log.append(performance)
    print(f"Époque {e+1}/{epochs} - Précision : {performance*100:.2f}%")

print("--- Training finished ! ---")

# 5. Afficher le graphique
import matplotlib.pyplot as plt

plt.plot(range(1, epochs + 1), performance_log)
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Accuracy over epochs")
plt.ylim(0, 1) # Force l'axe Y à aller de 0 à 1 (0% à 100%)
plt.grid(True)
plt.show()
plt.savefig("training_accr.png")