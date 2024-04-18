import os
import sys
import matplotlib.pyplot as plt
import requests
import secrets
import __main__
import json

### --- AJOUT DE TOUS LES SUBDIRECTIRIES AU PATH ---
base_directory = os.path.abspath('.')

# Using os.listdir() to get a list of all subdirectories
subdirectories = [os.path.join(base_directory, d) for d in os.listdir(base_directory) if os.path.isdir(os.path.join(base_directory, d))]

# Adding all subdirectories to the Python path
sys.path.extend(subdirectories)


### --- IMPORT DE BASTHON ---
# Ne marche que si on est sur basthon ou capytale, sinon ignorer : 
try:
    import basthon  # Ne marche que si on est sur Capytale ou Basthon
    basthon = True

except ModuleNotFoundError: 
    basthon = False
    pass

### --- Import du validation_kernel ---
# Ne marche que si fourni et si va avec le notebook en version séquencé. Sinon, ignorer :
sequence = False

try:
    from capytale.autoeval import Validate, validationclass
    from capytale.random import user_seed

    sequence = True
except ModuleNotFoundError: 
    sequence = False

## Pour valider l'exécution d'une cellule de code, dans le cas du notebook sequencé :
if sequence:
    validation_execution = Validate()
else:
    def validation_execution():
        return True

if sequence:
    from js import fetch
    import asyncio
    
    debug = False
    analytics_endpoint = "https://dev.mathadata.fr/api/notebooks"
    
    Validate()() # Validate import cell

    def call_async(func, cb, *args):
        try:
            loop = asyncio.get_event_loop()
            task = loop.create_task(func(*args))

            def internal_cb(future):
                try:
                    data = future.result()
                    if cb is not None:
                        cb(data)
                except Exception as e:
                    if debug:
                        print_error("Error during post request")
                        print_error(e)
                        
            task.add_done_callback(internal_cb)
        except Exception as e:
            if debug:
                print_error("Error during post request")
                print_error(e)

    async def get_capytale_async(path):
        response = await fetch(path, method='GET')
        if response.status == 200:
            data = await response.text()
            data_json = json.loads(data)
            return data_json
        else:
            raise Exception(f"Fetch failed with status: {response.status}")

    async def get_profile():
        return await get_capytale_async('/web/c-auth/api/me?_format=json')
    
    async def post_async(path, body):
        response = await fetch(analytics_endpoint + path, method='POST', body=json.dumps(body))
        if response.status == 200:
            data = await response.text()
            data_json = json.loads(data)
            return data_json
        else:
            raise Exception(f"Fetch failed with status: {response.status}")
            
else:
    debug = True
    analytics_endpoint = "http://localhost:3000/api/notebooks"
    #analytics_endpoint = "https://dev.mathadata.fr/api/notebooks"

def start_analytics_session(notebook_id):
    if sequence:
        seed = user_seed()
        if seed == 609507 or seed == 609510:
            return
        
        def profile_callback(profile):
            try:
                if profile['profil'] == 'teacher':
                    return

                create_session(notebook_id, profile['uid'], profile['classe'])
            except Exception as e:
                if debug:
                    print_error("Error during post request")
                    print_error(e)
        
        call_async(get_profile, profile_callback)
        
    else:
        create_session(notebook_id, -1, "dev")


def post(path, body, callback=None):
    try:
        if sequence:
            call_async(post_async, callback, path, body)
        else:
            response = requests.post(analytics_endpoint + path, json=body)
            if callback is not None:
                callback(response.json())
    except Exception as e:
        if debug:
            print_error("Error during post request")
            print_error(e)

session_id = None

def create_session(notebook_id, user_id, classname):
    def cb(data):
        global session_id
        session_id = data['id']

    post('/session', {
        'notebook_id': notebook_id,
        'user_id': user_id,
        'classname': classname
    }, cb)

class _MathadataValidate():
    counter = 0
    
    def __init__(self, success=None, *args, **kwargs):
        MathadataValidate.counter += 1
        self.question_number = MathadataValidate.counter
        self.trials = 0
        self.success = success

    def __call__(self):
        self.trials += 1
        errors = []
        answers = {}

        try:
            self.validate(errors, answers)
        except Exception as e:
            errors.append("Il y a eu une erreur dans la validation de la question. Vérifie que ta réponse est écrite correctement")
            if debug:
                errors.append(str(e))

        self.send_analytics_event(errors, answers)

        if len(errors) > 0:
            for error in errors:
                print_error(error)
            return False
        else:
            self.on_success() 
            return True

    def send_analytics_event(self, errors, answers):
        try:
            answers_json = json.dumps(answers)
        except TypeError as e:
            answers = {}
            
        if session_id is not None:
            try:
                post('/event', {
                    'session_id': session_id,
                    'question_number': self.question_number,
                    'is_correct': len(errors) == 0,
                    'answer': answers,
                })
            except Exception as e:
                if debug:
                    print_error("Error sending analytics event")
                    print_error(e)

    def validate(self, errors, answers):
        return True

    def on_success(self):
        if self.success is not None:
            if self.success:
                print(self.success)
        else:
            print("Bravo, c'est la bonne réponse !")

    def trial_count(self):
        return self.trials

if sequence:

    @validationclass
    class MathadataValidate(_MathadataValidate, Validate):
        
        def __init__(self, *args, **kwargs):
            _MathadataValidate.__init__(self, *args, **kwargs)
            Validate.__init__(self)
else:
    MathadataValidate = _MathadataValidate

    
 
class MathadataValidateVariables(MathadataValidate):
    def __init__(self, name_and_values, function_validation=None, *args, **kwargs):
        self.name_and_values = name_and_values
        self.function_validation = function_validation
        super().__init__(*args, **kwargs)

    def check_undefined_variables(self, errors):
        undefined_variables = [name for name in self.name_and_values if not has_variable(name) or get_variable(name) is Ellipsis]
        undefined_variables_str = ", ".join(undefined_variables)

        if len(undefined_variables) == 1:
            errors.append(f"As-tu bien remplacé les ... ? La variable {undefined_variables_str} n'a pas été définie.")
        elif len(undefined_variables) > 1:
            errors.append(f"As-tu bien remplacé les ... ? Les variables {undefined_variables_str} n'ont pas été définies.")

        return undefined_variables
    
    def check_variables(self, errors):
        for name in self.name_and_values:
            val = get_variable(name)
            expected = self.name_and_values[name]
            if isinstance(expected, dict):
                res = self.compare(val, expected['value'])
            else:
                res = self.compare(val, expected)

            if not res:
                self.check_errors(errors, name, val, expected)
        
    def check_type(self, errors, name, val, expected, tips=None):
        if type(val) != type(expected):
            errors.append(f"La variable {name} n'est pas du bon type. Le type attendu est {type(expected)} mais le type donné est {type(val)}.")
            return False
        else:
            return True
    
    def compare(self, val, expected):
        try:
            if isinstance(expected, dict):
                if 'is' in expected:
                    return val == expected
                elif 'min' in expected and 'max' in expected:
                    return val >= expected['min'] and val <= expected['max']
                elif 'in' in expected:
                    return val in expected['in']
                else:
                    raise ValueError(f"Malformed validation class : comparing {val} to {expected}")
            else:
                return val == expected
        except Exception as e:
            return False

    def check_errors(self, errors, name, val, expected):
        if isinstance(expected, dict) and 'errors' in expected:
            for error in expected['errors']:
                match_error = self.compare(val, error['value'])
                print(match_error)
                if match_error and 'if' in error:
                    errors.append(error['if'])
                    return
                if not match_error and 'else' in error:
                    errors.append(error['else'])
                    return

        # Default error message
        errors.append(f"{name} n'a pas la bonne valeur.")

        

    def validate(self, errors, answers):
        for name in self.name_and_values:
            if not has_variable(name):
                answers[name] = None
            else:
                answers[name] = get_variable(name)

        undefined_variables = self.check_undefined_variables(errors)
        if len(undefined_variables) == 0:
            if self.function_validation is not None:
                self.function_validation(errors, answers)
            else:
                self.check_variables(errors)

if sequence:

    @validationclass
    class MathadataValidateVariables(MathadataValidateVariables):
        pass


### --- Config matplotlib ---

plt.rcParams['figure.dpi'] = 150


### --- Common util functions ---

def print_error(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def has_variable(name):
    return hasattr(__main__, name) and get_variable(name) is not None and get_variable(name) is not Ellipsis

def get_variable(name):
    return getattr(__main__, name)
