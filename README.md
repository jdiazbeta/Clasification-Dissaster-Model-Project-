<h1 align="center"> Multiple classification model for disaster prevention from warning messages </h1>

## Table of contents

- [Description and Facilities](#Desc-inst)
- [Motivation](#Motivation)
- [Description of files](#Desc-files)
- [Interacting with the project](#Interact)
- [Acknowledgments]("#Acknowledgments)


## Description and Facilities
The code is set in the Python programming language.
In general, there are the modules that are needed to execute satisfactorily the codes:

- pandas
- re
- create_engine
- numpy
- pickle
- joblib
- time
- nltk [nltk.download(['punkt', 'wordnet'])]
- sklearn
- json
- plotly
- flask
- **Python version: 3.6.3**



## Motivation
Aid work that allows us to watch over and safeguard the life and integrity of people and animals in the face of any emergency or contingency is one of the tasks that, in my opinion, should be fully recognized without hesitation in sparing efforts. The tasks carried out by disaster units are highly complex; they are capable of risking their lives for ours.

One of the greatest motivations when carrying out this project is the impact it could have on these activities and the timely organization of these entities to respond to the first call for the different requests that arise on a daily basis. For the community, help will always be available when it is essential; for the support entities, responding to emergencies from the first moment, as well as prioritizing each one according to its severity, this being a totally organized activity. The main motivation when carrying out this project lies, therefore, entirely in its impact on the lives of people.


## Description of files
Regarding databases, in the repository you can find:

- [disaster_messages.csv](https://github.com/jdiazbeta/Clasification-Dissaster-Model-Project-/blob/05a9929038a5790d807111d675be7cf64da8a329/disaster_messages.csv) contains information about messages received from different disasters. Each message has a unique ID, the original message and the genre of the message (live, news, social).
  
- [disaster_categories.csv](https://github.com/jdiazbeta/Clasification-Dissaster-Model-Project-/blob/05a9929038a5790d807111d675be7cf64da8a329/disaster_categories.csv) contains the description of the categories to which the messages in the database [disaster_messages.csv] belong or refer. You will find information about the unique ID of the message and a description of the 36 categories with their respective identifier (1 if the message belongs to the category, 0 otherwise)
  
- [process_data.py](https://github.com/jdiazbeta/Clasification-Dissaster-Model-Project-/blob/05a9929038a5790d807111d675be7cf64da8a329/process_data.py) contiene el código en python empleado para el proceso ETL de las bases de datos [disaster_messages.csv] and [disaster_categories.csv]. El resultado al ejecutar este código sera la base de datos unificada y limpia. 
 
- [train_classifier.py](https://github.com/jdiazbeta/Clasification-Dissaster-Model-Project-/blob/05a9929038a5790d807111d675be7cf64da8a329/train_classifier.py) contiene el proceso de entrenamiento, validación, creación y guardado del modelo de clasificación múltiple estimado con base a la base de datos limpia del primer proceso. El resultado al ejecutar el código será un archivo en formato pkl con el modelo resultante.

- [run.py](https://github.com/jdiazbeta/Clasification-Dissaster-Model-Project-/blob/05a9929038a5790d807111d675be7cf64da8a329/run.py). Contiene el código que recrea la aplicación con descripciones de la base de datos entregada de mensajes, algunas visualizaciones y el clasificador de mensajes.

- [templates.tar.gz](https://github.com/jdiazbeta/Clasification-Dissaster-Model-Project-/blob/05a9929038a5790d807111d675be7cf64da8a329/templates.tar.gz) carpeta comprimida que permite ejecutar el código [run.py], dando forma y estilo a la aplicación creada.
  
- [cleaned_fullbase_messages.db](https://github.com/jdiazbeta/Clasification-Dissaster-Model-Project-/blob/05a9929038a5790d807111d675be7cf64da8a329/cleaned_fullbase_messages.db). Ejemplo de la base de datos en formato sqlite, resultante al ejecutar el código [process_data.py]

- [best_trained_model.pkl](https://github.com/jdiazbeta/Clasification-Dissaster-Model-Project-/blob/05a9929038a5790d807111d675be7cf64da8a329/best_trained_model.pkl). Ejemplo de la base de datos en formato pikle, resultante al ejecutar el código [train_classifier.py]


## Interacting with the project
(Como se debe interactuar con los archivos (uno por uno) para la correcta ejecusión del proceso.
Como deben incluirse los argumentos necesarios, aspectos a tener en cuenta, entre otros.
)

## Acknowledgments
( Agradecimientos a Udacity por el curso y Appen por el suministro de la base de datos.


)



