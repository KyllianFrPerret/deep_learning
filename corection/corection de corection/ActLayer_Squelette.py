from math import tanh

class ActLayer:
    """
        Une couche simplement pour activer les neurones de la couche précédente.
        Séparation en deux couches pour simplifier la rétropropagation du gradient.
    """
    def __init__(self,nbInput):
        # on crée des tableaux de 0 pour les entrées et les sorties
        self.input = [0 for _ in range(nbInput)]

        self.output = [0 for _ in range(nbInput)]


    def forward_propagation(self,input):
        """
            Fonction d'activation non linéaire, appelée après une autre couche.
            Les fonction classiques sont relu (max(0,x)), tanh, sigmoïde (1/(1+exp(-x)))
            Entrée : list
            Sortie : aucune, mise à jour de self.output
        """
        # on enregistre les entrées pour la rétropropagation
        self.input = input

        # on active toutes les valeurs une par une
        for indEntree in range(len(self.output)):
            # Fonction tanh
            self.output[indEntree] = tanh(self.input[indEntree])

        # on renvoie la sortie pour la transmettre à la couche suivante
        return self.output


    def backward_propagation(self, output_error, learning_rate):
        """
            Retropropagation, attention la fonction utilisée dépend de la fonction d'activation.
            Ici fonction tanh prime
        """
        input_error = []
        # dE/dX = E*f'(X) (multiplication terme à terme)
        for indice in range(len(output_error)):
            # tanh prime = 1-tanh(x)²
            val = (1-tanh(self.input[indice])**2)*output_error[indice]
            input_error.append(val)

        return input_error