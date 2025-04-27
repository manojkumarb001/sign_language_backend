from flask import Flask, request, jsonify, send_file,abort
from flask import send_from_directory

from flask_cors import CORS
from flask_pymongo import PyMongo
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
import logging
import speech_recognition as sr
import datetime
import bcrypt
from flask_bcrypt import Bcrypt
import spacy
import cv2
import json
import os
import time
import mediapipe as mp
import numpy as np

# Initialize Flask App
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

# MongoDB Configuration
app.config['MONGO_URI'] = 'mongodb://localhost:27017/signifyx_db'
mongo = PyMongo(app)

# JWT Configuration
app.config['JWT_SECRET_KEY'] = 'your_secret_key_here'
jwt = JWTManager(app)

# Logger Setup
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logging.getLogger("pymongo").setLevel(logging.WARNING)

# Initialize Bcrypt for Password Hashing
bcrypt = Bcrypt(app)

# Load NLP Model
nlp = spacy.load("en_core_web_md")

# MediaPipe Hands Initialization
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5, max_num_hands=2)

# Directory to store sign recordings
output_dir = "D:/final year project web/backend/recorded_signs"
os.makedirs(output_dir, exist_ok=True)

# JSON file to store mappings
json_file = "sign_mappings.json"
sign_data = {}
if os.path.exists(json_file):
    with open(json_file, "r") as f:
        sign_data = json.load(f)

# Directory to store uploads and generated animation
UPLOAD_FOLDER = 'uploads'
ANIMATION_FOLDER = 'animations'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(ANIMATION_FOLDER, exist_ok=True)

# A simple in-memory dictionary to store file references
sign_data = {}

def log_request(route_name, data):
    logger.debug(f"[{route_name}] Request Data: {data}")

def log_response(route_name, response):
    logger.debug(f"[{route_name}] Response: {response}")

def convert_to_sign_language(text):
    doc = nlp(text)
    sign_sentence = []
    question_words = {"WHAT", "WHERE", "WHEN", "WHY", "HOW", "WHO"}
    subject = None
    verb = None
    obj = []
    wh_word = None

    for token in doc:
        lemma = token.lemma_.upper()
        if token.pos_ in ["AUX", "DET", "ADP", "PUNCT"]:
            continue
        if lemma in question_words:
            wh_word = lemma
            continue
        if token.dep_ in ["nsubj", "nsubjpass"]:
            subject = lemma
            continue
        if token.dep_ == "ROOT":
            verb = lemma
            continue
        if token.dep_ in ["dobj", "attr", "prep", "pobj"]:
            obj.append(lemma)
            continue
        obj.append(lemma)

    if verb:
        sign_sentence.insert(0, verb)
    if obj:
        sign_sentence.extend(obj)
    if subject:
        sign_sentence.append(subject)
    if wh_word:
        sign_sentence.append(wh_word)

    return " ".join(sign_sentence)

@app.route('/register', methods=['POST'])
def register():
    try:
        data = request.json
        log_request('register', data)
        email = data.get('email')
        password = data.get('password')

        if not email or not password:
            return jsonify({'error': 'Email and password required'}), 400

        existing_user = mongo.db.users.find_one({'email': email})
        if existing_user:
            return jsonify({'error': 'User already exists'}), 409

        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
        mongo.db.users.insert_one({'email': email, 'password': hashed_password})

        response = {'message': 'User registered successfully'}
        log_response('register', response)
        return jsonify(response), 201
    except Exception as e:
        logger.error(f"[register] Error: {str(e)}")
        return jsonify({'error': 'Internal Server Error'}), 500

@app.route('/login', methods=['POST'])
def login():
    try:
        data = request.json
        log_request('login', data)
        email = data.get('email')
        password = data.get('password')

        user = mongo.db.users.find_one({'email': email})
        if user and bcrypt.check_password_hash(user['password'], password):
            access_token = create_access_token(identity=email)
            response = {'token': access_token, 'message': 'Login successful'}
            log_response('login', response)
            return jsonify(response)

        return jsonify({'error': 'Invalid credentials'}), 401
    except Exception as e:
        logger.error(f"[login] Error: {str(e)}")
        return jsonify({'error': 'Internal Server Error'}), 500

@app.route('/user', methods=['GET'])
@jwt_required()
def get_user():
    try:
        current_user = get_jwt_identity()
        user = mongo.db.users.find_one({'email': current_user}, {'_id': 0, 'password': 0})
        if user:
            response = {'user': user}
            log_response('get_user', response)
            return jsonify(response)
        return jsonify({'error': 'User not found'}), 404
    except Exception as e:
        logger.error(f"[get_user] Error: {str(e)}")
        return jsonify({'error': 'Internal Server Error'}), 500

@app.route('/record_speech', methods=['POST'])
@jwt_required()
def record_speech():
    try:
        user_email = get_jwt_identity()
        data = request.get_json()

        if not data or "text" not in data:
            return jsonify({'error': 'Invalid request, "text" field is required'}), 400

        speech_text = data["text"]
        sign_text = convert_to_sign_language(speech_text)

        speech_data = {
            'user_email': user_email,
            'speech_text': speech_text,
            'sign_translation': sign_text,
            'timestamp': datetime.datetime.now(datetime.UTC)
        }
        mongo.db.speech_logs.insert_one(speech_data)

        response = {'speech_text': speech_text, 'sign_translation': sign_text, 'message': 'Speech stored successfully'}
        log_response('record_speech', response)
        return jsonify(response)
    except Exception as e:
        logger.error(f"[record_speech] Error: {str(e)}")
        return jsonify({'error': 'Internal Server Error'}), 500

@app.route('/speech_logs', methods=['GET'])
@jwt_required()
def get_speech_logs():
    try:
        user_email = get_jwt_identity()
        logs = list(mongo.db.speech_logs.find({'user_email': user_email}, {'_id': 0}))
        return jsonify({'logs': logs}) if logs else jsonify({'message': 'No speech logs available'})
    except Exception as e:
        logger.error(f"[get_speech_logs] Error: {str(e)}")
        return jsonify({'error': 'Internal Server Error'}), 500

 
@app.route("/record_sign", methods=["POST"])
@jwt_required()
def record_sign():
    user_email = get_jwt_identity()
    data = request.json
    word = data.get("word", "").strip().lower()
    overwrite = data.get("overwrite", False)  # New field to allow overwrite

    if not word:
        return jsonify({"error": "No word provided"}), 400

    try:
        # Check if word already exists
        if word in sign_data and not overwrite:
            return jsonify({
                "error": f"Sign for '{word}' already exists.",
                "message": "Use 'overwrite: true' in the request to re-record."
            }), 409

        video_filename = os.path.join(output_dir, f"{word}.avi")
        hand_data_filename = os.path.join(output_dir, f"{word}_landmarks.json")

        # If overwrite, delete old files first
        if word in sign_data and overwrite:
            paths = sign_data[word]
            if os.path.exists(paths["video"]):
                os.remove(paths["video"])
            if os.path.exists(paths["landmarks"]):
                os.remove(paths["landmarks"])

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            return jsonify({"error": "Could not open webcam"}), 500

        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        out = cv2.VideoWriter(video_filename, fourcc, 10, (640, 480))
        if not out.isOpened():
            cap.release()
            return jsonify({"error": "VideoWriter failed to open"}), 500

        time.sleep(3)  # 3-second countdown before recording

        start_time = time.time()
        hand_tracking_data = []

        while time.time() - start_time < 3:  # Record for 3 seconds
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            frame = cv2.resize(frame, (640, 480))  # Match size
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)

            frame_landmarks = []
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    landmarks = [{"x": lm.x, "y": lm.y, "z": lm.z} for lm in hand_landmarks.landmark]
                    frame_landmarks.append(landmarks)

            if frame_landmarks:
                hand_tracking_data.append(frame_landmarks)

            out.write(frame)

        out.release()
        cap.release()

        with open(hand_data_filename, "w") as f:
            json.dump(hand_tracking_data, f, indent=4)

        # Update sign_data and save JSON
        sign_data[word] = {
            "video": video_filename,
            "landmarks": hand_data_filename
        }
        with open(json_file, "w") as f:
            json.dump(sign_data, f, indent=4)

        # Insert or update in MongoDB
        existing = mongo.db.signs.find_one({"word": word, "user_email": user_email})
        if existing:
            mongo.db.signs.update_one(
                {"_id": existing["_id"]},
                {"$set": {
                    "video_path": video_filename,
                    "landmarks_path": hand_data_filename,
                    "updated_at": datetime.datetime.now(datetime.timezone.utc)
                }}
            )
        else:
            mongo.db.signs.insert_one({
                "user_email": user_email,
                "word": word,
                "video_path": video_filename,
                "landmarks_path": hand_data_filename,
                "created_at": datetime.datetime.now(datetime.timezone.utc)
            })

        return jsonify({
            "message": f"Sign for '{word}' recorded successfully",
            "video": video_filename
        })

    except Exception as e:
        try:
            cap.release()
        except:
            pass
        try:
            out.release()
        except:
            pass
        return jsonify({"error": str(e)}), 500

@app.route("/play_sign", methods=["GET"])
def play_sign():
    word = request.args.get("word", "").strip().lower()
    if word not in sign_data:
        return jsonify({"error": "No sign found for this word"}), 404

    return send_file(sign_data[word]["video"], mimetype="video/mp4")

@app.route("/delete_sign", methods=["DELETE"])
@jwt_required()
def delete_sign():
    try:
        data = request.json
        word = data.get("word", "").strip().lower()
        if not word or word not in sign_data:
            return jsonify({"error": "No sign found for this word"}), 404

        paths = sign_data.pop(word)

        if os.path.exists(paths["video"]):
            os.remove(paths["video"])
        if os.path.exists(paths["landmarks"]):
            os.remove(paths["landmarks"])

        with open(json_file, "w") as f:
            json.dump(sign_data, f, indent=4)

        mongo.db.signs.delete_one({"word": word})

        return jsonify({"message": f"Sign for word '{word}' deleted successfully"})
    except Exception as e:
        logger.error(f"[delete_sign] Error: {str(e)}")
        return jsonify({"error": "Internal Server Error"}), 500


@app.route('/upload', methods=['POST'])
def upload_sign():
    try:
        word = request.form['word']
        video_file = request.files['video']
        landmarks_file = request.files['landmarks']

        # Save the video and landmarks to disk
        video_path = os.path.join(UPLOAD_FOLDER, f"{word}.mp4")
        landmarks_path = os.path.join(UPLOAD_FOLDER, f"{word}_landmarks.json")
        
        video_file.save(video_path)
        landmarks_file.save(landmarks_path)

        # Store reference in memory (You could store this in a database)
        sign_data[word] = {"video": video_path, "landmarks": landmarks_path}
        
        return jsonify({"message": f"Sign for '{word}' recorded successfully!"}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route('/play/<word>', methods=['GET'])
def play_sign_animation(word):
    if word not in sign_data:
        return jsonify({"error": f"No sign found for '{word}'"}), 404

    video_path = sign_data[word]['video']
    landmarks_path = sign_data[word]['landmarks']

    # Create the animation based on landmarks and return it
    animation_path = generate_animation(landmarks_path, word)

    return send_from_directory(ANIMATION_FOLDER, animation_path, as_attachment=True)

def generate_animation(landmarks_path, word):
    with open(landmarks_path, 'r') as f:
        hand_landmarks_data = json.load(f)

    # Animation settings
    frame_width = 640
    frame_height = 480
    fps = 20
    animation_filename = f"{word}_animation.mp4"
    animation_path = os.path.join(ANIMATION_FOLDER, animation_filename)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(animation_path, fourcc, fps, (frame_width, frame_height))

    # Draw the animation based on landmarks
    connections = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11), 
                   (11, 12), (12, 13), (13, 14), (14, 15), (15, 16), (16, 17), (17, 18), (18, 19), (19, 20)]  # Example for finger connections
    
    def draw_hand_landmarks(frame, landmarks):
        for connection in connections:
            pt1 = (int(landmarks[connection[0]]['x'] * frame_width), int(landmarks[connection[0]]['y'] * frame_height))
            pt2 = (int(landmarks[connection[1]]['x'] * frame_width), int(landmarks[connection[1]]['y'] * frame_height))
            cv2.line(frame, pt1, pt2, (0, 255, 0), 2)  # Green lines

        for lm in landmarks:
            x, y = int(lm['x'] * frame_width), int(lm['y'] * frame_height)
            cv2.circle(frame, (x, y), 5, (255, 0, 0), -1)  # Blue points

    for frame_landmarks in hand_landmarks_data:
        if not frame_landmarks:
            continue

        frame = np.ones((frame_height, frame_width, 3), dtype=np.uint8) * 255  # White background

        # Iterate over all detected hands
        for hand in frame_landmarks:
            draw_hand_landmarks(frame, hand)

        video_writer.write(frame)  # Save frame to video

    video_writer.release()  # Finalize video
    return animation_filename


if __name__ == '__main__':
    logger.info("Starting Flask server on port 5000...")
    app.run(host="0.0.0.0", port=5000, debug=True)
