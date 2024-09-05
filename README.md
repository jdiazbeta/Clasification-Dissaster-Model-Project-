<h1 align="center"> Multiple classification model for disaster prevention from warning messages s </h1>

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
  
- [process_data.py](https://github.com/jdiazbeta/Clasification-Dissaster-Model-Project-/blob/05a9929038a5790d807111d675be7cf64da8a329/process_data.py) contains the python code used for the ETL process of the databases [disaster_messages.csv](https://github.com/jdiazbeta/Clasification-Dissaster-Model-Project-/blob/05a9929038a5790d807111d675be7cf64da8a329/disaster_messages.csv) and [disaster_categories.csv](https://github.com/jdiazbeta/Clasification-Dissaster-Model-Project-/blob/05a9929038a5790d807111d675be7cf64da8a329/disaster_categories.csv). The result of running this code will be the unified and clean database. 
 
- [train_classifier.py](https://github.com/jdiazbeta/Clasification-Dissaster-Model-Project-/blob/05a9929038a5790d807111d675be7cf64da8a329/train_classifier.py) contains the process of training, validating, creating and saving the estimated multiple classification model based on the clean database from the first process. The result of running the code will be a file in pkl format with the resulting model.

- [run.py](https://github.com/jdiazbeta/Clasification-Dissaster-Model-Project-/blob/05a9929038a5790d807111d675be7cf64da8a329/run.py). Contains the code that recreates the application with descriptions of the delivered message database, some visualizations, and the message classifier.

- [templates.tar.gz](https://github.com/jdiazbeta/Clasification-Dissaster-Model-Project-/blob/05a9929038a5790d807111d675be7cf64da8a329/templates.tar.gz) compressed folder that allows you to run the code [run.py], giving shape and style to the created application.
  
- [cleaned_fullbase_messages.db](https://github.com/jdiazbeta/Clasification-Dissaster-Model-Project-/blob/05a9929038a5790d807111d675be7cf64da8a329/cleaned_fullbase_messages.db). Example of the database in sqlite format, resulting from executing the code [process_data.py](https://github.com/jdiazbeta/Clasification-Dissaster-Model-Project-/blob/05a9929038a5790d807111d675be7cf64da8a329/process_data.py)

- [best_trained_model.pkl](https://github.com/jdiazbeta/Clasification-Dissaster-Model-Project-/blob/05a9929038a5790d807111d675be7cf64da8a329/best_trained_model.pkl). Example of the database in pkl format, resulting from executing the code [train_classifier.py](https://github.com/jdiazbeta/Clasification-Dissaster-Model-Project-/blob/05a9929038a5790d807111d675be7cf64da8a329/train_classifier.py)


## Interacting with the project
For the correct execution of the process, the codes must be run in order, which is:
1) process_data.py
2) train_classifier.py
3) run.py

The execution of each code is explained below.

- [process_data.py](https://github.com/jdiazbeta/Clasification-Dissaster-Model-Project-/blob/05a9929038a5790d807111d675be7cf64da8a329/process_data.py) contains at the end of the code a method called **main**, which has three arguments:
**_messages_filepath_**, which must be filled with the path of the .csv file with the disaster messages that will be used to train the model
**_categories_filepath_**, which must be filled with the path of the .csv file with the disaster categories of the messages in the first file, which will be used to train the model
**_database_filename_**, which must be filled with the desired name for the resulting cleaned and consolidated database. This must be filled in followed by the "db" format. Example: cleaned_base.db.

**Note:** If the file is required to be specifically exported to a folder, this can be specified by completing the desired name of the cleaned database with the final directory of the database. However, this can cause problems when loading said database in the processes following the data cleansing.

- [train_classifier.py](https://github.com/jdiazbeta/Clasification-Dissaster-Model-Project-/blob/05a9929038a5790d807111d675be7cf64da8a329/train_classifier.py) contains at the end of the code a method called **main**, which has two arguments:
**_database_filepath_**, contains the path to the file with the consolidated and debugged database, which is the result of the code - [process_data.py](https://github.com/jdiazbeta/Clasification-Dissaster-Model-Project-/blob/05a9929038a5790d807111d675be7cf64da8a329/process_data.py).
**_model_filepath_**, contains the desired name with which the estimated multiple classification model will be saved. This must have the final format (.pkl). Example: best_trained_model.pkl.

**Note:** If the file is required to be specifically exported to a folder, this can be specified by completing the desired name of the cleaned database with the final directory of the database. However, this can cause problems when loading said database in the processes following the data cleansing.

- [run.py](https://github.com/jdiazbeta/Clasification-Dissaster-Model-Project-/blob/05a9929038a5790d807111d675be7cf64da8a329/run.py). To successfully run this code you must manually enter the following elements:

**_table_name_** (line 39). This must include the name of the consolidated and debugged message database, the result of the code [process_data.py](https://github.com/jdiazbeta/Clasification-Dissaster-Model-Project-/blob/05a9929038a5790d807111d675be7cf64da8a329/process_data.py).

**_database_filepath_** (line 40) must contain the full address of the consolidated and debugged message database, resulting from the code [process_data.py](https://github.com/jdiazbeta/Clasification-Dissaster-Model-Project-/blob/05a9929038a5790d807111d675be7cf64da8a329/process_data.py).

**_model_** (line 48) must adjust the path of the pkl file with the model created using the code [train_classifier.py](https://github.com/jdiazbeta/Clasification-Dissaster-Model-Project-/blob/05a9929038a5790d807111d675be7cf64da8a329/train_classifier.py)

I hope you do not encounter any complications when carrying out the process, and that it will be enriched in order to improve the experience when dealing with these issues. I hope that they serve as a lever for future large projects that will save and improve lives.


## Acknowledgments
I want to express my deep gratitude to the following platforms and communities for their invaluable support and resources that made this project possible:

UDACITY: For providing the knowledge and tools, as well as practical scenarios that increase my interest in data science. 
APPEN: For providing robust data sets that allowed the completion of this project, as well as increasing my interest in data science. I also express my admiration for your work. I hope for a pleasant future and much success in all your endeavors.

Additionally, I am grateful to all those who have directly or indirectly contributed to this work through comments, suggestions, and shared resources.


