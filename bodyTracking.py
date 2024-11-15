import cv2
import mediapipe as mp
import socket
import json


def send_data(data):
    server_address = ('localhost', 8052)
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect(server_address)

    try:
        # Serialize list to JSON
        json_data = json.dumps({"array": data})
        client_socket.sendall(json_data.encode('utf-8'))
    finally:
        client_socket.close()

# Initialize MediaPipe pose solution with GPU acceleration
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, smooth_landmarks=True,
                    min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Initialize MediaPipe drawing solution
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Open video capture (0 for default camera)
cap = cv2.VideoCapture(0)

# Reduce resolution to improve performance (optional)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Define relevant landmarks indices
UPPER_BODY_LANDMARKS = [

    mp_pose.PoseLandmark.RIGHT_SHOULDER,
    mp_pose.PoseLandmark.RIGHT_ELBOW,
    mp_pose.PoseLandmark.RIGHT_WRIST,
    mp_pose.PoseLandmark.LEFT_SHOULDER,
    mp_pose.PoseLandmark.LEFT_ELBOW,
    mp_pose.PoseLandmark.LEFT_WRIST,

]

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Convert the BGR image to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the image and detect the pose
    results = pose.process(frame_rgb)

    # Draw the pose annotation on the frame
    if results.pose_landmarks:
        # Draw relevant landmarks and connections
        annotated_image = frame.copy()

        # Calculate neck position
        left_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        neck_x = (left_shoulder.x + right_shoulder.x) / 2
        neck_y = (left_shoulder.y + right_shoulder.y) / 2
        neck_z = (left_shoulder.z + right_shoulder.z) / 2
        h, w, _ = frame.shape
        nx, ny = int(neck_x * w), int(neck_y * h)
        cv2.circle(annotated_image, (nx, ny), 5, (0, 0, 255), cv2.FILLED)
        cv2.putText(annotated_image, 'Neck', (nx, ny - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.putText(annotated_image, f'x:{neck_x:.2f}, y:{neck_y:.2f}, z:{neck_z:.2f}', (nx, ny + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        # Print the neck 3D coordinates
        #print(f'Neck (relative to webcam): (x: {neck_x:.2f}, y: {neck_y:.2f}, z: {neck_z:.2f})')

        # Initialize data array with neck coordinates
        data = [neck_x, neck_y, neck_z]

        # Draw other landmarks relative to the neck
        for idx, landmark in enumerate(results.pose_landmarks.landmark):
            if mp_pose.PoseLandmark(idx) in UPPER_BODY_LANDMARKS:
                relative_x = landmark.x - neck_x
                relative_y = landmark.y - neck_y
                relative_z = landmark.z - neck_z
                data.extend([relative_x, relative_y, relative_z])
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                cv2.circle(annotated_image, (cx, cy), 5, (0, 255, 0), cv2.FILLED)
                cv2.putText(annotated_image, f'ID: {idx}', (cx, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                cv2.putText(annotated_image, f'x:{relative_x:.2f}, y:{relative_y:.2f}, z:{relative_z:.2f}',
                            (cx, cy + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                # Print the joint ID and relative 3D coordinates
                #print(f'Joint {idx} (relative to neck): (x: {relative_x:.2f}, y: {relative_y:.2f}, z: {relative_z:.2f})')

        # Send the data
        send_data(data)
        #print(data)
        cv2.imshow('Pose Tracking', annotated_image)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()
