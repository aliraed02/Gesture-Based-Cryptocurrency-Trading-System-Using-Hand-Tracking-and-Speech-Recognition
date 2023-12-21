# import Dependecies
import os
import mediapipe as mp
import matplotlib.pyplot as plt
import cv2
import pickle



# Congigrate the hand
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_style = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)


# Location the dataset
Data_DIR = r'C:\Users\dc\OneDrive\Desktop\data'

data = []
labels = []

# Select the sample dataset 
for dir_ in os.listdir(Data_DIR):
    for img_path in os.listdir(os.path.join(Data_DIR, dir_)):
        data_aux = []
        img = cv2.imread(os.path.join(Data_DIR, dir_, img_path))
        
        # Convert the color image from vgr to rgb
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        
        # process the hand
        results = hands.process(img_rgb)
        
        '''
        # Select the hand landmark
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(img_rgb, 
                                      hand_landmarks,
                                      mp_hands.HAND_CONNECTIONS,
                                      mp_drawing_style.get_default_hand_landmarks_style(),
                                      mp_drawing_style.get_default_hand_connections_style())
        
        
        '''
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
              
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x)
                    data_aux.append(y)
                    
                    
        data.append(data_aux)
        labels.append(dir_)
        
        
        
        '''            
        # Polt the sample dataset
        plt.figure()
        plt.imshow(img_rgb)
       
        
        
plt.show()
'''

f = open('data.pickle', 'wb')
pickle.dump({'data': data,
             'labels': labels}, f)
f.close()
