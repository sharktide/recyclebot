import os
import keyboard
import cv2
import numpy as np
import time
print("program started!")
print("Turning off OneDNN operations")
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' #Turns off OneDNN operations, which is just something that is automatically turned on to make cpu usage lower. However, as this model is prettly light, we don't need this.




CLASSES = ['Glass', 'Metal', 'Paperboard', 'Plastic-Polystyrene', 'Plastic-Regular']
import tensorflow as tf

#Load Models
ensemble1 = tf.keras.models.load_model("recyclebot.keras")
ensemble2 = tf.keras.models.load_model("72-75.keras")





# Show the model architecture
ensemble1.summary()
ensemble2.summary()


index = 1

while True:
     
    
    check = 0

    #Ask for Image to test

    test_image = cv2.resize(cv2.imread(input(f'''Path to image #{index} -- Please Replace Backslashes (\\) with forwardslashes (/) please!  
                                            -->''')),  (240, 240))

    test_image = np.array(test_image).reshape(-1, 240, 240, 3)

    print(test_image.shape)

    # Assign weights to each model
    weight_1 = 0.50
    weight_2 = 0.50

    # Get predictions (probabilities)
    preds_1 = ensemble1.predict(test_image)
    preds_2 = ensemble2.predict(test_image)



    # Weighted average of probabilities
    final_preds = (weight_1 * preds_1 + weight_2 * preds_2)
    print(CLASSES)
    print(final_preds)

    print()


    # Get the class with the highest weighted average probability
    final_class = (CLASSES[np.argmax(final_preds)])
    print(final_class)


    print(f''' Click space to go on.''')
    while check == 0:
    
        if keyboard.is_pressed(' '): 
            event = keyboard.read_event(suppress=True)
            print('--------------------------------------------------')
            check = 1


    
    print(f'''                                                                           
          
          
          
          
          
          ''')


    index = (index + 1)
