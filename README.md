# PARSLE
NLP data pipeline for Learning Analytics in a Project Environment. Results are for plotting in a ENA(Epistemic Network Analysis).

This readme is for the use PARSLE. Please read this guide carefully before you start running the application.

1. Before you start, make sure you have the following software installed with the corresponding versions:

- Python (3.9.0) https://www.python.org/downloads/ 
- Visual Studio Code (latest) https://code.visualstudio.com/download
- R (latest) https://cran.r-project.org/bin/
- rENA (latest) https://cran.r-project.org/web/packages/rENA/index.html

2. Also make sure to set Python and R in your PATH environment variable.

3. To successfully run the application you need to install a few Python packages, these are as follows with the following commands:

- Pandas - pip install pandas
- Torch - pip install torch
- Stanza - pip install stanza
- Transformers - pip install transformers
- CSV - pip install csv
- Rpy2 - pip install rpy2
- Sci-kit - pip install scikit-learn
- Plotly - pip install plotly

4. Once you've installed all the necessary components you need to provide the application an Excel file as a trainingsset with the following requirements. You can skip this step and go to step 5 if you want to use the standard model.

- The first column(first row) as text for the feedback provided.
- The second column(first row) the category for the labeled feedback.
- Where,
0 is feedback,
1 is feedup
and 2 is feedforward

Please provide as much diverse data to train the model as optimal as possible. And place this in the same PARSLE folder where everything is located.

5. For testing the data and showing it in an ENA, you need to provide another CSV file. This time with all the data you want to plot. 
The dataset must contain over 250 rows. Else the ENA won't be constructed. The file its columns is structured as follows.

text name role nameFor assessment scale

You also need to place this file in the same PARSLE folder where everything is located. This time with name "data_set". This name must be exact, else it won't work.

6. Once you've managed to do these things, you can start the application within VSCode. We recommend to use the main.ipynb file since there were issues plotting the data in a normal Python environment. 

Within the first prompt, enter the name of the Excel file you provided.
If you don't do this a new model won't be created and the standard model will be used.
You should train this model the first time since the standard model isn't loaded.
With the second prompt, you can test and plot the data provided. 
The application returns a CSV file with the name data_ENA to plot or to manually use for the ENA web tool.
The last prompt will ask you if you want to plot the generated data from data_set.

If you still encounter any issues, please feel free to contact team PARSLE anytime through GitHub.
