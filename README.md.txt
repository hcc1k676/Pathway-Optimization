README file
1. System requirements
(1)Operating systems
This package is supported for Windows 10.
All software dependencies
Python	3.7
pandas	1.3.5
Gensim	4.2.0
numpy	1.18.0
scipy	1.4.1
scikit-learn	1.0.2
tensorflow-cpu	2.3.0
keras	2.3.1
matplotlib	3.5.3
(2)Versions the software has been tested on
PyCharm 2023.2.1 (Community Edition)
(3)Any required non-standard hardware
Package requires only a standard computer with enough RAM to support the in-memory operations. 
2. Installation guide
(1)Instructions
To install the software PyCharm, navigate to the 'Python Interpreter' section under settings and download all the software dependencies mentioned above. 
(2)Typical install time on a "normal" desktop computer
The complete installation process is expected to take approximately 15 minutes.
3.Demo
(1)Instructions to run on data
(a)The file "I21pre_ready.csv" contains preprocessed medical order data for all patients tagged with I21.%. The data fields in this file include: Id (index), admission_number1 (patient code), discharge_diagnosis (diagnosis), disease_code (diagnosis code), hospitalization_time (number of days hospitalized), advice_day (the day medical advice was given), and advice (specific medical advice or procedure).
(b) The script "advice_feature.py" utilizes LDA (Latent Dirichlet Allocation) to extract patterns of medical advice given on specific days for a particular disease. It takes as input the file "I21pre_ready.csv", processes data for the disease starting with the string '心肌梗死' (myocardial infarction), and outputs the results to a file named "daily_topics_心肌梗死.csv".
(c) The script "lda_datapre_model.py" is designed to handle the composition of the medical treatment process, which includes integer encoding of medical orders, components of the treatment process, and tagging of medical order codes within the treatment day. It takes "daily_topics_心肌梗死.csv" as input and selects the top two themes (treatment patterns) for each treatment day using the command day_group['Topic'].unique()[:2]. The output is generated in a file named "disease_paths_with_words_急性非ST段抬高.csv".
(d) The script "BiLSTM.py" implements a BiLSTM model, which is split into training, validation, and testing sets in an 8:1:1 ratio. It takes the file "disease_paths_with_words_急性非ST段抬高.csv" as input. 
(2) Expected output
The script processes this input and upon completion, it outputs four metrics related to the test set results using the command print(test set results' four metrics).
(3) Expected run time for demo on a "normal" desktop computer
The first part, which involves topic modeling, and the second part, which focuses on data augmentation, each require approximately half an hour. The third part, which deals with deep learning, also requires about an hour.
4. Instructions for use
Open the file package and sequentially run scripts advice_feature.py, lda_datapre_model.py, and BiLSTM.py in PyCharm.
Additional information
This project is covered under the PyCharm 2023.2.1 (Community Edition)

