from FCLayer_Squelette import FCLayer
from ActLayer_Squelette import ActLayer


class NeuralNetwork:
    """
        Un réseau de neurones est constituté de plusieurs couches de neurones.
        Le nombre de neurones d'une couche correspond au nombre d'entrées de la suivante.
        Pour chaque couche de type FC on ajoute automatiquement une couche d'activation ici.
    """
    def __init__(self):
        """
            Un seul attribut layers, qui contient les différentes couches du réseau.
        """
        # layers est un tableau de couches de neurones, vide à la création du réseau
        self.layers = []

    def addLayer(self,nbInput,nbOutput):
        """
            Méthode permettant d'ajouter deux couches supplémentaires au réseau, une de type FCLayer puis une de type ActLayer juste après.
            Entrées :
                * nbInput : int -> nombre d'entrées de la nouvelle couche, doit correspondre aux sorties de la couche précédente.
                * nbOuput : int -> nombre de neurones de la couche à ajouter
            Sortie : aucune
        """
        # on ajoute une couche d'activation pour chaque couche FC
        self.layers.append(FCLayer(nbInput,nbOutput))
        self.layers.append(ActLayer(nbOutput))


    def predict(self,inputs):
        """
            On utilise le réseau pour avoir une prédiction.
            Fonction personnalisable selon l'utilisation (il peut être mieux de créer une nouvelle fonction pour cela).
            Entrée :
                * inputs : list[float] -> les valeurs d'entrées du réseau
            Sortie : list -> les sorties calculées par le réseau
        """
        outputs = inputs

        for layer in self.layers:
            outputs = layer.forward_propagation(outputs)

        return outputs


    def training(self,training_inputs,training_outputs,epochs,learning_rate):
        """
            On va maintenant entrainer le réseau pour trouver les poids / biais les plus adaptés.
            Entrées :
                * training_inputs : list -> les entrées du jeu de tests, la dimension d'une entrée doit correspondre au nombre d'entrées du réseau
                * training_ouputs : list -> les sorties attendues pour chaque entrée, la dimension d'une sortie doit correspondre au nombre de sorties du réseau. Ces sorties sont celles utilisées pour calculer l'erreur.
                * epochs : int -> le nombre de cycles d'entrainements à effectuer
                * learning_rate : float -> le taux d'apprentissage (en général inférieur à 0.3)
            Sortie : aucune -> mise à jour des poids / biais
        """
        for _ in range(epochs):
            # on parcourt les données
            for indice in range(len(training_inputs)):
                # on prédit les sorties en fonction des entrées
                input = training_inputs[indice]
                expectedOutput = training_outputs[indice]
                networkOutput = self.predict(input)

                # on calcule la dérivée de l'erreur pour notre fonction d'erreur
                # 2/n * (Y - Y*)
                error = []
                n = len(expectedOutput)
                for i in range(n):
                    error.append(2*(networkOutput[i]-expectedOutput[i])/n)

                # On rétropropage
                for l in range(len(self.layers)-1,-1,-1):
                    error = self.layers[l].backward_propagation(error,learning_rate)