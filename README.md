### Authors:
- Topaz Aakal 	-> 318644549
- Marat Zinger 	-> 332689405
- Afik Danan 	-> 208900175

### Python main libraries for this project:
numpy <br>
pandas <br>
scikit-learn <br>
scipy <br>

### The project is based on python, Interpreter 3.10:
- the project contains three packages:
    - GUI - this python package contain all the GUI part and the main.py file to run the project
    - Models - this package include all models implentations (Our and Sklearn)
    - PreProcessing - this package include all the pre-processing part (Clean Data, Fill Missing Values,
                Discretization, Normalization)

### Python libraries for running this project:
At terminal run the next command: <br>
        - pip install -r requirements.txt

### Instrucations after all the packages are installed:
From pycharm run main.py  <br>

  - import File to use
  - Enter the classification column
  - Choose how to Pre-Process the file
  - Choose the model you want to use 
  - Click "Build & Run" button
  - Wait for the screen to show "Done!"
  - For visualize the info click "Models result"
  - To watch all results click "Open all Results"
  - Exit at any time clicking the red 'X' or clik "Exit"

### Beneath the surface:
- When importing the file there is a validation check for file existance and for file to be not empty
- When entering classification column agian validation check for column existance in the file
- After clicking "Build & Run" the file will be split to train and test (default 20/80) 
  now the pre-processing start and when done it will save the clean data to two files in the project files.
- After we have two clean files of train and test we start to build the selected model using our 'Model' 
  package.
- At the end we get all the evaluation metrics save to the bottom of the 'result.csv' file which can be open 
  using the "Open all Results" button.

# Thank you!
