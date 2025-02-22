import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
from playsound import playsound
from twilio.rest import Client
import smtplib
from email.mime.text import MIMEText
from datetime import datetime

# Parameters
img_size = (128, 128)
model = load_model('anomaly_detection_autoencoder.h5', custom_objects={'mse': MeanSquaredError()})

# Twilio credentials
TWILIO_ACCOUNT_SID = 'AC59831539aa04451efacab2ec660c5b86'
TWILIO_AUTH_TOKEN = 'e6fe51d86218022ce39a783b52ee5c7b'
TWILIO_PHONE_NUMBER = '+12088528480'
TO_PHONE_NUMBER = '+917558159822'

# Email credentials
SENDER_EMAIL = 'sharanbabub@gmail.com'
SENDER_PASSWORD = 'omts xqiy rbge birx'
RECIPIENT_EMAIL = 'sharanbabub2005@gmail.com'

def get_camera_source(camera_id):
    if camera_id == 1:
        return 'E:\\Anomaly Detection\\Dataset\\Train\\Abuse\\Abuse026_x264.mp4'
    elif camera_id == 2:
        return 'E:\\Anomaly Detection\\Dataset\\Train\\NormalVideos\\Normal_Videos_352_x264.mp4'
    elif camera_id == 3:
        return 'E:\\Anomaly Detection\\Dataset\\Test\\NormalVideos\\Normal_Videos_050_x264.mp4'
    elif camera_id == 4:
        return 'E:\\Anomaly Detection\\Dataset\\Test\\Abuse\\Abuse002_x264.mp4'
    else:
        return None  # Invalid camera ID

def get_location(camera_id):
    if camera_id == 1:
        return "Koyambedu Market"
    elif camera_id == 2:
        return "Porur HP Petrol Bunk"
    elif camera_id == 3:
        return "Poonamallee Signal"
    elif camera_id == 4:
        return "Tambaram Bridge"
    else:
        return "Unknown Location"

# Function to detect anomalies
def detect_anomalies(frame, camera_id):
    dynamic_threshold = True
    mse_values = []
    threshold = 0.002  # Initial threshold

    # Preprocess the frame (resize and normalize)
    resized_frame = cv2.resize(frame, img_size)
    normalized_frame = resized_frame.astype("float32") / 255.0
    normalized_frame = np.expand_dims(normalized_frame, axis=0)

    # Predict the reconstructed frame using the autoencoder
    reconstructed_frame = model.predict(normalized_frame)

    # Calculate the Mean Squared Error (MSE)
    mse = np.mean(np.power(normalized_frame - reconstructed_frame, 2))
    mse_values.append(mse)

    if dynamic_threshold and len(mse_values) > 100:
        mean_mse = np.mean(mse_values)
        std_mse = np.std(mse_values)
        threshold = mean_mse + 2 * std_mse  # Dynamic threshold adjustment

    # Anomaly detection based on MSE
    if mse < threshold:  # MSE should be greater to indicate anomaly
        label = "Anomaly Detected!"
        color = (0, 0, 255)  # Red for anomaly
        status = "Anomaly Detected!"
    else:
        label = "Normal"
        color = (0, 255, 0)  # Green for normal
        status = "Normal"

    font_scale = 0.5
    thickness = 1
    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
    text_x = frame.shape[1] - text_size[0] - 10
    text_y = frame.shape[0] - 10

    cv2.putText(frame, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)

    if status == "Anomaly Detected!":
        location = get_location(camera_id)
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # Current time
        send_alerts(camera_id, location, current_time, status)

    return frame, status

# Function to send alerts
def send_alerts(camera_id, location, time, status):
    play_alert_sound()
    send_sms_alert(camera_id, location, time, status)
    send_email_alert(camera_id, location, time, status)

def play_alert_sound():
    playsound('E:\\Anomaly Detection\\alert_sound.mp3', block=False)  # Non-blocking sound play

def send_sms_alert(camera_id, location, time, status):
    client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
    message = f"Anomaly detected!\nCamera ID: {camera_id}\nLocation: {location}\nTime: {time}\nStatus: {status}"
    client.messages.create(body=message, from_=TWILIO_PHONE_NUMBER, to=TO_PHONE_NUMBER)

def send_email_alert(camera_id, location, time, status):
    subject = "Anomaly Detected"
    body = f"Anomaly detected!\nCamera ID: {camera_id}\nLocation: {location}\nTime: {time}\nStatus: {status}"
    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = SENDER_EMAIL
    msg['To'] = RECIPIENT_EMAIL

    with smtplib.SMTP('smtp.gmail.com', 587) as server:
        server.starttls()
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        server.sendmail(SENDER_EMAIL, RECIPIENT_EMAIL, msg.as_string())

# Function to create a looping video capture
def create_looping_capture(source):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error opening video file: {source}")
        return None
    return cap

# Function to get the next frame, looping if necessary
def get_next_frame(cap):
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = cap.read()
    return ret, frame
