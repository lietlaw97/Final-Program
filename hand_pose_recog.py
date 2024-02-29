from base_recog_class import BaseRecognitionClass
from models.recog_models import HandPoseRecognition
import mediapipe as mp
import csv
import copy
import itertools
import cv2 as cv

class HandPoseRecog(BaseRecognitionClass):
    def __init__(self):
        self.recog_model = HandPoseRecognition()
        self.labels = self.get_labels()
        self.mp_hands = mp.solutions.hands.Hands(
                    max_num_hands=2,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5)
        self.draw_landmarks = True
        self.zeros = [0.0] * 42
        self.collect_data = False

    def get_labels(self):
        with open('data/hand/hand_pose_labels.csv',
            encoding='utf-8-sig') as g:
            hand_pose_labels = csv.reader(g)
            hand_pose_labels = [row[0] for row in hand_pose_labels]
            return hand_pose_labels
    
    def calc_landmarks(self, image, landmarks):
        image_width, image_height = image.shape[1], image.shape[0]

        landmark_point = []

        # Keypoint
        for _, landmark in enumerate(landmarks.landmark):
            landmark_x = min(int(landmark.x * image_width), image_width - 1)
            landmark_y = min(int(landmark.y * image_height), image_height - 1)

            landmark_point.append([landmark_x, landmark_y])

        return landmark_point
    
    def preprocess_landmarks(self, landmark_list, handedness):
        temp_landmark_list = copy.deepcopy(landmark_list)

        base_x, base_y = 0, 0
        for index, landmark_point in enumerate(temp_landmark_list):
            if index == 0:
                base_x, base_y = landmark_point[0], landmark_point[1]

            if handedness == "Left":
                temp_landmark_list[index][0] = -(temp_landmark_list[index][0] - base_x)
                temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y
            else:
                temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
                temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

        temp_landmark_list = list(
            itertools.chain.from_iterable(temp_landmark_list))

        max_value = max(list(map(abs, temp_landmark_list)))

        def normalize_(n):
            return n / max_value

        temp_landmark_list = list(map(normalize_, temp_landmark_list))

        return temp_landmark_list
    
    def draw_connections(self, image, landmark_point):
        if len(landmark_point) > 0:
            # Thumb
            cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
                    (255,255,255), 2)
            cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
                    (255,255,255), 2)

            # Index finger
            cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
                    (255,255,255), 2)
            cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
                    (255,255,255), 2)
            cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
                    (255,255,255), 2)

            # Middle finger
            cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
                    (255,255,255), 2)
            cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
                    (255,255,255), 2)
            cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
                    (255,255,255), 2)

            # Ring finger
            cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
                    (255,255,255), 2)
            cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
                    (255,255,255), 2)
            cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
                    (255,255,255), 2)

            # Little finger
            cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
                    (255,255,255), 2)
            cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
                    (255,255,255), 2)
            cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
                    (255,255,255), 2)

            # Palm
            cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
                    (255,255,255), 2)
            cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
                    (255,255,255), 2)
            cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
                    (255,255,255), 2)
            cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
                    (255,255,255), 2)
            cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
                    (255,255,255), 2)
            cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
                    (255,255,255), 2)
            cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
                    (255,255,255), 2)

        # Key Points
        for index, landmark in enumerate(landmark_point):
            if index == 0:  
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 234, 214),
                        -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 234, 214), 1)
            if index == 1:  
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 234, 214),
                        -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 234, 214), 1)
            if index == 2: 
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 234, 214),
                        -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 234, 214), 1)
            if index == 3: 
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 234, 214),
                        -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 234, 214), 1)
            if index == 4: 
                cv.circle(image, (landmark[0], landmark[1]), 8, (255,255,255),
                        -1)
                cv.circle(image, (landmark[0], landmark[1]), 8, (255, 234, 214), 1)
            if index == 5: 
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 234, 214),
                        -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 234, 214),
                        -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 234, 214), 1)
            if index == 7: 
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 234, 214),
                        -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 234, 214), 1)
            if index == 8:
                cv.circle(image, (landmark[0], landmark[1]), 8, (255,255,255),
                        -1)
                cv.circle(image, (landmark[0], landmark[1]), 8, (255, 234, 214), 1)
            if index == 9: 
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 234, 214),
                        -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 234, 214), 1)
            if index == 10: 
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 234, 214),
                        -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 234, 214), 1)
            if index == 11:  
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 234, 214),
                        -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 234, 214), 1)
            if index == 12: 
                cv.circle(image, (landmark[0], landmark[1]), 8, (255,255,255),
                        -1)
                cv.circle(image, (landmark[0], landmark[1]), 8, (255, 234, 214), 1)
            if index == 13: 
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 234, 214),
                        -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 234, 214), 1)
            if index == 14:  
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 234, 214),
                        -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 234, 214), 1)
            if index == 15: 
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 234, 214),
                        -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 234, 214), 1)
            if index == 16:  
                cv.circle(image, (landmark[0], landmark[1]), 8, (255,255,255),
                        -1)
                cv.circle(image, (landmark[0], landmark[1]), 8, (255, 234, 214), 1)
            if index == 17: 
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 234, 214),
                        -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 234, 214), 1)
            if index == 18: 
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 234, 214),
                        -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 234, 214), 1)
            if index == 19: 
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 234, 214),
                        -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 234, 214), 1)
            if index == 20: 
                cv.circle(image, (landmark[0], landmark[1]), 8, (255,255,255),
                        -1)
                cv.circle(image, (landmark[0], landmark[1]), 8, (255, 234, 214), 1)
        return image
    
    def draw_text(self, image, brect, handedness, hand_sign_text):
        if hand_sign_text == 'Error':
            return image
        info_text = handedness.classification[0].label[0:]
        if hand_sign_text != "":
            info_text = info_text.upper() + ': ' + hand_sign_text
        (w, h), _ = cv.getTextSize(info_text, cv.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        if brect[0] + w > brect[2]:
            brect[2] = brect[0] + w
        cv.rectangle(image, (brect[0], brect[1]), (brect[2] + 10, brect[1] - 22),
                    (255, 255, 255), -1)
        cv.putText(image, info_text, (brect[0] + 5, brect[1] - 4),
                cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv.LINE_AA)

        return image