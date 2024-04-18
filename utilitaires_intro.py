print("Chargement de la base donnée d'images en cours...")

from utilitaires_mnist_2 import *
from utilitaires_common import *

print("Images chargées ! New.")

notebook_id = 1

try:
    start_analytics_session(notebook_id)
    print("Session started")
except: 
    print("No analytics session")

try:
    # For dev environment
    from strings_intro import *
except ModuleNotFoundError: 
    pass

### ----- CELLULES VALIDATION ----

class Validate2(MathadataValidate):
    def __init__(self):
        super().__init__(success="Bravo, le pixel (17,15) est devenu noir !")

    def validate(self, errors, answers):
        answers['pixel'] = int(d[17][15])
        if d[17][15] == 0:
            return True
        elif d[17][15] == 254:
            errors.append("Tu n'as pas changé la valeur du pixel, il vaut toujours 254")
        else:
            errors.append("Tu as bien changé la valeur mais ce n'est pas la bonne. Relis l'énoncé pour voir la valeur à donner pour un pixel noir.")
        return False

class Validate7(MathadataValidate):
    def validate(self, errors, answers):
        if not has_variable('r_prediction'):
            answers['r_prediction'] = None
            errors.append("La variable r_prediction n'a pas été définie.")
            return False

        r_prediction = get_variable('r_prediction')
        answers['r_prediction'] = r_prediction

        tips = "Vérifie que tes réponses sont bien écrites entre crochets séparées par des virgules comme dans l'exemple : [7,2,2,2,7,2,7,2,2,2]"

        if not isinstance(r_prediction, list):
            errors.append(tips)
            return False
        
        if len(r_prediction) != 10:
            errors.append(f"r_prediction doit contenir 10 valeurs mais ta liste n'en contient que {len(r_prediction)}. {tips}")
            return False

        expected = [estim(d) for d in d_train[0:10]]
        for i in range(10):
            if not isinstance(r_prediction[i], int) or (r_prediction[i] != 2 and r_prediction[i] != 7):
                errors.append(f"Les valeurs de r_prediction doivent être 2 ou 7.")
                return False
            
            if expected[i] != r_prediction[i]:
                errors.append(f"La valeur de r_prediction pour l'image {i + 1} n'est pas la bonne.")
                return False
        return True

class Validate8(MathadataValidate):

    def __init__(self, *args, **kwargs):
        super().__init__(success="")

    def validate(self, errors, answers):
        if not has_variable('e_train_10'):
            errors.append("La variable e_train_10 n'a pas été définie.")
            return False
        
        e_train_10 = get_variable('e_train_10')
        nb_errors = np.count_nonzero(np.array([estim(d) for d in d_train[0:10]]) != r_train[0:10]) 
        if nb_errors * 10 == e_train_10:
            print(f"Bravo, ton algorithme actuel a fait {nb_errors} erreurs sur les 10 premières images, soit {e_train_10}% d'erreur")
            return True
        else:
            if e_train_10 == nb_errors:
                errors.append("Ce n'est pas la bonne valeur. Pour passer du nombre d'erreurs sur 10 images au pourcentage, tu dois multiplier par 10 !")
            elif e_train_10 < 0 or e_train_10 > 100:
                errors.append("Ce n'est pas la bonne valeur. Le pourcentage d'erreur doit être compris entre 0 et 100.")
            else:
                errors.append("Ce n'est pas la bonne valeur. Compare ta liste de prédictions avec les vraies valeurs pour trouver le pourcentage d'erreur.")
            return False

if sequence:
    
    @validationclass
    class Validate2(Validate2):
        pass

    @validationclass
    class Validate7(Validate7):
        pass

    @validationclass
    class Validate8(Validate8):
        pass
        
validation_question_1 = MathadataValidateVariables({
    'pixel': {
        'value': int(d[15,17]),
        'errors': [
            {
                'value': {
                    'min': 0,
                    'max': 255
                },
                'else': "Ta réponse n'est pas bonne. Les pixels peuvent uniquement avoir des valeurs entre 0 et 255."
            }
        ]
    }
})
validation_question_2 = Validate2()
validation_question_3 = MathadataValidateVariables({
    'r': {
        'value': 2,
        'errors': [
            {
                'value': {
                    'in': [2, 7],
                },
                'else': "Tu dois répondre par 2 ou 7."
            }
        ]
    }
})
validation_question_4 = MathadataValidateVariables({'k': np.mean(d[14:16,15:17])}, success="Bravo, la moyenne vaut en effet (142 + 154 + 0 + 0) / 4 = 74")
validation_question_5 = MathadataValidateVariables({'r_petite_caracteristique': 7, 'r_grande_caracteristique': 2}, success="Bravo, la moyenne est en effet plus élevée pour les images de 2 que de 7")
validation_question_6 = MathadataValidateVariables({
    'x': {
        'value': {
            'min': 23,
            'max': 57
        }
    }
}, success="Ton seuil est correct ! Il n'est pas forcément optimal, on verra dans la suite comment l'optimiser.")
validation_question_7 = Validate7()
validation_question_8 = Validate8()
validation_question_9 = MathadataValidateVariables({
  'x': {
      'value': {
            'min': 32,
            'max': 36
      }
  }  
})

validate_moyenne_partie_image = MathadataValidateVariables({
    'e_train': {
        'value': {
            'min': 0,
            'max': 0.12
        },
        'errors': [
            {
                'value': {
                    '<': 0
                },
                'else': "Continue à chercher une zone qui différencie bien les 2 et les 7. Pour cela remonte à la cellule qui permet de changer les coordonnées des points A et B. Choisi une rectangle rouge qui diférencie bien les 2 et les 7."
            }
        ]
    }
}, success="Bravo ! Tu as réussi à réduire l'erreur à moins de 12%. C'est déjà un très bon score ! Tu peux continuer à essayer d'améliorer ta zone ou passer à la suite.")

### Pour les checks d'execution des cellules sans réponse attendue:
validation_execution_1 = MathadataValidate(success="")
validation_execution_2 = MathadataValidate(success="")
validation_execution_3 = MathadataValidate(success="")
validation_execution_4 = MathadataValidate(success="")
validation_execution_5 = MathadataValidate(success="")
validation_execution_6 = MathadataValidate(success="")
validation_execution_7 = MathadataValidate(success="")
validation_execution_classif = MathadataValidate(success="")
