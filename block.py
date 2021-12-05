import cv2
import mediapipe as mp
import numpy as np
from typing import List


BLOCKED_COLOR = (200, 200, 200) # gray


def block_in_video(blocked_color: List[int]=BLOCKED_COLOR) -> None:
    """Blocks person in real time video.

    Args:
        blocked_color (List[int], optional): The mask color to be used to block the person. 
        In Black Mirror, the perverts are blocked with red color.
        Normal block is gray.
        This parameter must be RGB list.
        Defaults to BLOCKED_COLOR.
    """
    mp_drawing = mp.solutions.drawing_utils
    mp_selfie_segmentation = mp.solutions.selfie_segmentation

    # For webcam input:
    cap = cv2.VideoCapture(0)

    with mp_selfie_segmentation.SelfieSegmentation(model_selection=0) as selfie_segmentation:
    
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue

            # Flip the image horizontally for a later selfie-view display, and convert
            # the BGR image to RGB.
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            results = selfie_segmentation.process(image)

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Draw selfie segmentation on the background image.
            # To improve segmentation around boundaries, consider applying a joint
            # bilateral filter to "results.segmentation_mask" with "image".
            condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1

            blocked_image = np.ones(image.shape, dtype=np.uint8)
            blocked_image[:] = blocked_color
            output_image = np.where(condition, blocked_image, image)

            cv2.imshow('Black Mirror Blocker', output_image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
            
        cap.release()
        cv2.destroyAllWindows()