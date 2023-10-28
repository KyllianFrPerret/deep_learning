from random import random
import random

# FC = Fully connected



class FCLayer:
    """
        Une couche de neurones de type : Fully Connected Layer .
        Cela signifie que chaque entrée est connectée à tous les neurones de la couche.
        Pas d'activation pour l'instant pour simplifier le calcul de la rétropropagation du gradient.
    """
    def __init__(self,nbInput,nbOutput):
        """
            Initialisation d'une couche de neurones.
            Entrées :
                * nbInput : int -> le nombre d'entrées (égal au nb de sorties de la couche précédente)
                * nbOutput : int -> le nombre de sorties / neurones de cette couche.
        """
        # on choisit des biais aléatoires entre -0.5 et 0.5
        self.bias = [(random()-0,5) for x in range(nbOutput)]

        # on choisit des poids aléatoires pour les entrées entre -0.5 et 0.5
        # attention, matrice transposée par rapport aux notations habituelles (vecteurs)
        self.weight =[]
        for i in range(nbOutput):
            for a in range (nbInput):
                self.weight[a].append(random()-0,5)

        # on crée un tableau de 0 pour les entrées
        self.input = [0 for x in range(nbInput)]

        # on crée un tableau de 0 pour les sorties
        self.output = [0 for x in range(nbOutput)]
















    def forward_propagation(self, input):
        """
            Propagation vers l'avant, c'est à dire calcul des sorties en fonctions des entrées.
            Entrée :
                * input : list -> un tableau de réels
            Sortie : list -> modification puis renvoie de l'attribut ouput.
        """
        # on enregistre les entrées pour la rétropropagation
        self.input = input
        # pour chaque neurone on calcule x1*w1 + x2*w2 + ... + xn*wn + b
        for neurone in range(len(self.weight)):
            for entreeN in range (len(self.weight[neurone])):
                self.output[neurone]=self.output[neurone]+ input[entreeN]*self.weight[neurone][entreeN]
            self.output[neurone]=self.output[neurone] + self.bias[neurone]
        # on renvoie la sortie pour la transmettre à la couche suivante
        return self.output




















    def backward_propagation(self, output_error, learning_rate):
        """
            Rétropropagation du gradient à l'aide des gradients descendants.
            Fonction utilisée uniquement durant la période d'apprentissage pour ajuster les poids et les biais.
            Entrées :
                * out_error : float -> l'erreur entre la sortie attendue et le résultat obtenu
                * learning_rate : float -> le taux d'apprentissage en général petit (<0.3)
        """
        # On calcule d'abord le tableau des dE/dX (input_error)

        # # # (commentaire écrit par moi) dE et dX dérivée des erreures (tableau a sens unique)

        # Produit matriciel : dE/dX = E (-> une matrice) fois poids (déjà stockés sous forme transposée)
        input_error = []
        # j'effectue la multiplication
        for indice in range(len(self.input)):
            val = 0
            for i in range(len(output_error)):
                val = val + output_error[i] * self.weight[i][indice]
            input_error.append(val)

        # il faut maintenant mettre à jour les poids
        # delta W = (transposée X) fois erreurs
        for i in range(len(self.input)):
            for j in range(len(output_error)):
                self.weight[j][i] = self.weight[j][i] - learning_rate*self.input[i]*output_error[j]

        # mise à jour des biais
        for i in range(len(output_error)):
            self.bias[i] = self.bias[i] - learning_rate*output_error[i]

        return input_error




















## Tests
weight = [[0.5,0.5],
          [0.3,0.5],
          [0.6,0.3]]

bias = [0.3,0.2,0.3]














































