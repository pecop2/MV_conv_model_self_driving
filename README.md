# Machine vision project, creating a deep convolutional model that can be used for self driving car

### Idea: Creating a ***simple*** model that could be used for deciding which action the autonomous vehicle should perform. The model's output layer has 3 neurons, the three actions that the vehicle could perform: driving straight, turning left or turning right.

#### Data
Data obtained from video taken during manually driving a vehicle and 
saving the particular frame into one of three folders: "w" (straight), "a" (left), "d" (right),
 corresponding to the action being taken at the moment. - saved in ***original_pictures*** folder
 
#### Preprocessing
The raw pictures are resized and preprocessed so that only the 
lines are detected. The goal was to detect only the lines that the car should drive inbetween ("edges of the road"). **Canny** used as edge detector, **Hough Line Transform** used for lines detection. 
-  ***line_detector.py*** detects the lines and saves the pictures into ***detected_lines*** folder

#### Creating and training the model
Pictures with detected lines are used in order to create 
training data and train a simple *Convolutional deep learning model* 
capable of deciding what action the car should perform (left, right or straight). 
- ***model_creator.py***

Model summary and training procedure included in the jupyter notebook. 
- ***convolutional_model_training.ipynb***