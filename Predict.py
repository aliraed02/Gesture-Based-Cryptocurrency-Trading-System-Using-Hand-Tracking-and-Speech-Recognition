import cv2 
import mediapipe as mp
import pickle
import numpy as np
from keras.preprocessing.sequence import pad_sequences
import speech_recognition as sr
import pyttsx3
import requests
import json
import time


url = "https://node.knooz.com/api/PlaceOrder"
'''
payload = json.dumps({
    "side": "BUY",
    "market": "USDT",
    "trade": "BTC",
    "type": "MARKET",
    "volume": 0.25,
    "timeInForce": "GTC",
    "clientOrderId": "123456"
})
'''
headers = {
    'Content-Type': 'application/json',
    'Authorization': 'Bearer eyJhbGciOiJkaXIiLCJlbmMiOiJBMjU2Q0JDLUhTNTEyIiwidHlwIjoiSldUIn0..IeW2BQuAgVfOXICLZMPdpQ.E7luP_rlTK6fvYWaeqDw2fpEXEsRzG_lcOpOBT-P4QFJwm2kocUAb8QIdn45dm_kYJpV6JvRf4XmNYqqNrVE2wDdPJR_1nvSI9uuAZVTqSh3A05MvW82VT9oK6ueIoMrtEU4qTMB2h8xnjfwVNbaRuUsU-9jNfAbAE7p-oSV9puYp5kjrvJYFHDsmHYwF9OmEM4vDajw3Uso8H1rTK-NALCYZcVYk_vh8BPSxsWt4x2jSAFb_eoy9FzOgVYXU5A7SXr6oNuuq2R7wEV5uwZzATD_ozyck5XssvnXRT8nn4L_eUoZWqp9Io4aJVm7sbufB37CejDjiljQoOG0Tb_6xTW7OpwfHeCvTLMdIqk_KXbRNk4m7TgLTkoOSi5qgxahV-URK9d8nMG3KejdGGtb6114UfBpKgRIYT3tk_2GnpoBmcO8_bSmpbxAZ3w4TfbDrbcPSPuBjDmXdO-0sT5i5ZbXrJnDSjypjOLz1kDU8R3E0cOgInMP2CAseqBseoS0.bdOttB01ap0wdgzQhU3LyVL_r6AybOPR82BACtYzYOI'
    
}






def voice():
    r = sr.Recognizer()

    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source)

        print('\nPlease say the number of pairs (one, two, three, four, or five)...')

        try:
            audio = r.listen(source)
            text = r.recognize_google(audio)

            # Convert the recognized text to lowercase for easier comparison
            z = text.lower().split()

            # Define the mapping between numbers and their corresponding choices
            digit_choices = {'one': 'ETH/USDT', 'two': 'BTC/USDT', 'three': 'LTC/USDT', 'four': 'XRP/USDT', 'five': 'ETH/BTC'}
        
        

            # Check if the recognized text contains numbers (e.g., one, two, etc.)
            choices = [digit_choices[digit] for digit in z if digit in digit_choices]

            if choices:
                print('You have chosen the following pair(s):')
                for choice in choices:
                    print(f'- {choice}')
                return choices
            else:
                print('Sorry, I couldn\'t recognize any valid number. Please try again.')

        except sr.UnknownValueError:
            print('Sorry, I could not understand your speech. Please try again.')
        except sr.RequestError as e:
            print('Error connecting to the speech recognition service. Please check your internet connection.')





#---------------------------------------------------------------#


def speech ():
    
    
    text_speech = pyttsx3.init()


    rate = text_speech.getProperty('rate')  # Get the current speech rate
    text_speech.setProperty('rate', rate - 100)  # Decrease the rate by 50 (adjust as needed)

    answer1 = 'Hello , Welcome to Knooz trading'
    text_speech.say(answer1)

    text_speech.runAndWait()
    


    answer2 = 'Choose one of the following pairs to trade'

    text_speech.say(answer2)

    text_speech.runAndWait()
    
    list_trade = ['ETH/USDT', 'BTC/USDT', 'LTC/USDT', 'XRP/USDT', 'ETH/BTC']

    for ite in range(len(list_trade)):
        print(f'Do you want to trade {list_trade[ite]}, Just say {ite + 1} ')
    
    choses = voice()
    return choses


#--------------------------------------------------------------------------#





def cam():
    
    
    choices = speech()
    choice = choices[0].split('/') 


    model_dict = pickle.load(open('model.p', 'rb'))
    model = model_dict['model']

    print(model)
    
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)


    cap = cv2.VideoCapture(0)

    labels_dict = {0: 'BUY', 1: 'SELL'}


    c = 0
    while True:
    
            data_aux = []
            x_ = []
            y_ = []
    
            ret, frame = cap.read()
    
            H, W, _ = frame.shape
    
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
            results = hands.process(frame_rgb)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame,  # image to draw
                        hand_landmarks,  # model output
                        mp_hands.HAND_CONNECTIONS,  # hand connections
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())
    
            
                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y
    
                        x_.append(x)
                        y_.append(y)
    
        
                        data_aux.append(x)
                        data_aux.append(y)
    
                x1 = int(max(x_) * W)
                y1 = int(max(y_) * H) 
    
                x2 = int(min(x_) * W)
                y2 = int(min(y_) * H)
            
                maxlen = 200
            
                data_aux = pad_sequences([data_aux], maxlen=maxlen, padding='post', truncating='post', dtype='float32')
                prediction = model.predict(data_aux)
                predicted_character = labels_dict[int(prediction)]
    
    
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                        cv2.LINE_AA)
                
                
                
                payload = json.dumps({
                   "side": predicted_character,
                   "market": choice[1],
                   "trade": choice[0],
                   "type": "MARKET",
                   "volume": 0.25,
                   "timeInForce": "GTC",
                   "clientOrderId": "123456"
                   })
    
         
  
                print("Ready to make a trade. Show your gesture...")
                        # Wait for the user to perform a gesture
                    
                url = "https://node.knooz.com/api/PlaceOrder"  # Replace with your actual trading server URL
                hheaders = {
                        'Content-Type': 'application/json',
                        'Authorization': 'Bearer eyJhbGciOiJkaXIiLCJlbmMiOiJBMjU2Q0JDLUhTNTEyIiwidHlwIjoiSldUIn0..IeW2BQuAgVfOXICLZMPdpQ.E7luP_rlTK6fvYWaeqDw2fpEXEsRzG_lcOpOBT-P4QFJwm2kocUAb8QIdn45dm_kYJpV6JvRf4XmNYqqNrVE2wDdPJR_1nvSI9uuAZVTqSh3A05MvW82VT9oK6ueIoMrtEU4qTMB2h8xnjfwVNbaRuUsU-9jNfAbAE7p-oSV9puYp5kjrvJYFHDsmHYwF9OmEM4vDajw3Uso8H1rTK-NALCYZcVYk_vh8BPSxsWt4x2jSAFb_eoy9FzOgVYXU5A7SXr6oNuuq2R7wEV5uwZzATD_ozyck5XssvnXRT8nn4L_eUoZWqp9Io4aJVm7sbufB37CejDjiljQoOG0Tb_6xTW7OpwfHeCvTLMdIqk_KXbRNk4m7TgLTkoOSi5qgxahV-URK9d8nMG3KejdGGtb6114UfBpKgRIYT3tk_2GnpoBmcO8_bSmpbxAZ3w4TfbDrbcPSPuBjDmXdO-0sT5i5ZbXrJnDSjypjOLz1kDU8R3E0cOgInMP2CAseqBseoS0.bdOttB01ap0wdgzQhU3LyVL_r6AybOPR82BACtYzYOI'
                    }
                
                #response = requests.post(url, headers=headers, data=payload)
                #print(response.status_code)
                print(predicted_character)
               
        
            cv2.imshow('frame', frame)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    

#-----------------------------------------------------------------------#


def main():
    
    
    cam()
    
    
if __name__ == '__main__':
    main()

   
    