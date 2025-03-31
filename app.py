from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_pymongo import PyMongo
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
import logging
import speech_recognition as sr
import datetime
import bcrypt
from flask_bcrypt import Bcrypt
import spacy

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

def log_request(route_name, data):
    print("-----------------------[ REQUEST ]-------------------------------")
    logger.debug(f"[{route_name}] Request Data: {data}")
    print("------------------------------------------------------")


def log_response(route_name, response):
    print("*********************[ RESPONSE ]***************************")

    logger.debug(f"[{route_name}] Response: {response}")
    print("************************************************")



def convert_to_sign_language(text):
    """
    Converts English text into a simplified Sign Language order (VOS or OSV).
    """
    doc = nlp(text)
    sign_sentence = []
    
    question_words = {"WHAT", "WHERE", "WHEN", "WHY", "HOW", "WHO"}
    subject = None
    verb = None
    obj = []
    wh_word = None

    for token in doc:
        lemma = token.lemma_.upper()
        
        # Ignore auxiliary verbs, determiners, prepositions, and punctuation
        if token.pos_ in ["AUX", "DET", "ADP", "PUNCT"]:
            continue

        # Capture question words (e.g., WHERE, WHAT)
        if lemma in question_words:
            wh_word = lemma
            continue

        # Identify subject
        if token.dep_ in ["nsubj", "nsubjpass"]:
            subject = lemma
            continue

        # Identify main verb
        if token.dep_ == "ROOT":
            verb = lemma
            continue

        # Identify objects
        if token.dep_ in ["dobj", "attr", "prep", "pobj"]:
            obj.append(lemma)
            continue

        obj.append(lemma)

    # Build the final sentence
    if verb:
        sign_sentence.insert(0, verb)  # Verb first
    if obj:
        sign_sentence.extend(obj)  # Object follows verb
    if subject:
        sign_sentence.append(subject)  # Subject at the end
    if wh_word:
        sign_sentence.append(wh_word)  # WH-Question at the end

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
        logger.debug(f"[get_user] Current user: {current_user}")
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
        
        if logs:
            return jsonify({'logs': logs})
        return jsonify({'message': 'No speech logs available'})
    except Exception as e:
        logger.error(f"[get_speech_logs] Error: {str(e)}")
        return jsonify({'error': 'Internal Server Error'}), 500

if __name__ == '__main__':
    logger.info("Starting Flask server on port 5000...")
    app.run(host="0.0.0.0", port=5000, debug=True)
