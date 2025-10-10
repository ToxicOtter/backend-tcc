import os
import cv2
import numpy as np
from flask import Blueprint, request, jsonify
from datetime import datetime
from werkzeug.utils import secure_filename
import base64
from io import BytesIO
from PIL import Image
import logging
import face_recognition
from src.models.user import User, DetectionLog, Notification, Facial, RecognitionLog, Device, db
from config import get_config
from src.services.push import push_to_user_devices

# Configurações
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Threshold típico para face_recognition (L2)
EMBEDDING_DISTANCE_THRESHOLD = 0.60  # ajuste fino depois com seus dados

# Contadores globais
total_verificacoes = 0
total_reconhecidos = 0

facial_bp = Blueprint("facial_bp", __name__)
config = get_config()

# Logger para salvar em arquivo
log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')
os.makedirs(log_dir, exist_ok=True)
log_path = os.path.join(log_dir, 'reconhecimento.log')

logging.basicConfig(
    filename=log_path,
    level=logging.INFO,
    format='[%(asctime)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def calcular_iluminacao(image_path):
    """Calcula a média de iluminação da imagem (escala de cinza)"""
    try:
        imagem = cv2.imread(image_path)
        if imagem is None:
            return 0.0
        gray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
        return np.mean(gray)
    except Exception as e:
        print(f"Erro ao calcular iluminação: {e}")
        return 0.0

# --- Funções de Ajuda ---

def allowed_file(filename):
    return "." in filename and \
           filename.rsplit(".", 1)[1].lower() in config.ALLOWED_EXTENSIONS

def ensure_upload_folder():
    """Garante que a pasta de uploads existe"""
    upload_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), UPLOAD_FOLDER)
    if not os.path.exists(upload_path):
        os.makedirs(upload_path)
    return upload_path

# --- Funções de Detecção e Extração de Características (Atualizadas para DNN) ---

def calculate_similarity(encoding1, encoding2):
    # Menor distância = maior similaridade
    return np.linalg.norm(encoding1 - encoding2)

# ======== ALTERADO: detecção com face_recognition ========
def detect_faces_dlib(image_path):
    """
    Detecta faces usando face_recognition (dlib).
    Retorna lista de boxes no formato (top, right, bottom, left).
    """
    try:
        img = face_recognition.load_image_file(image_path)  # RGB
        # modelo "hog" é mais leve; se precisar, pode trocar para model="cnn" (mais pesado)
        face_locations = face_recognition.face_locations(img, model="hog")
        return face_locations
    except Exception as e:
        print(f"Erro na detecção de faces (dlib): {e}")
        return []

# ======== ALTERADO: extração de embedding 128-D ========
def extract_face_embedding(image_path, face_box):
    """
    Extrai embedding (vetor 128-D) de uma face usando face_recognition.
    face_box deve ser (top, right, bottom, left).
    """
    try:
        img = face_recognition.load_image_file(image_path)  # RGB
        encodings = face_recognition.face_encodings(img, known_face_locations=[face_box], num_jitters=0)
        if encodings and len(encodings) > 0:
            return encodings[0].astype(np.float32)
        return None
    except Exception as e:
        print(f"Erro na extração de embedding: {e}")
        return None

def score_conf(dist, th=EMBEDDING_DISTANCE_THRESHOLD):
    return float(np.clip(1.0 - (dist / th), 0.0, 1.0))

# --- Rota de Recebimento de Imagem (Atualizada) ---

@facial_bp.route("/images", methods=["POST"])
def receive_image():
    inicio = datetime.now()
    try:
        upload_path = ensure_upload_folder()

        # Verifica se há arquivo na requisição
        if 'image' not in request.files:
            return jsonify({'error': 'Nenhuma imagem enviada'}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'Nenhum arquivo selecionado'}), 400

        if file and allowed_file(file.filename):
            # Salva a imagem
            filename = secure_filename(f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}")
            file_path = os.path.join(upload_path, filename)
            file.save(file_path)

            global total_verificacoes, total_reconhecidos
            total_verificacoes += 1
            iluminacao = calcular_iluminacao(file_path)

            # ======== usa dlib/face_recognition para detectar faces ========
            faces = detect_faces_dlib(file_path)
            if len(faces) == 0:
                fim = datetime.now()
                duracao = (fim - inicio).total_seconds() * 1000  # em ms

                status = 'no_face'
                print(f"[{datetime.now()}] IP: {request.remote_addr} | Status: {status} | Iluminação: {iluminacao:.2f} | Reconhecido: não | Tempo resposta: {duracao:.2f} ms")
                logging.info(f"IP: {request.remote_addr} | Status: {status} | Iluminação: {iluminacao:.2f} | Reconhecido: não | Tempo resposta: {duracao:.2f} ms")

                recognition_log = RecognitionLog(
                    user_id= None,
                    confidence= None,   # 0–1
                    status=status,                                        # 'recognized', 'unknown', 'no_face', 'error'
                    detected_at=fim,                        # ou o timestamp de início da req
                    ip=request.remote_addr,
                    light_level=iluminacao,
                    recognized= False,
                    latency_ms=duracao,
                )
                db.session.add(recognition_log)
                db.session.commit()

                return jsonify({
                    'status': 'no_face',
                    'message': 'Nenhuma face detectada na imagem',
                    'detection_id': recognition_log.id,
                    'tempo_resposta_ms': duracao
                })


            # Considera a primeira face detectada
            face_box = faces[0]  # (top, right, bottom, left)

            # ======== extrai embedding 128-D ========
            face_embedding = extract_face_embedding(file_path, face_box)
            if face_embedding is None:
                fim = datetime.now()
                duracao = (fim - inicio).total_seconds() * 1000  # em ms

                status = 'error'
                print(f"[{datetime.now()}] IP: {request.remote_addr} | Status: {status} | Iluminação: {iluminacao:.2f} | Reconhecido: não | Tempo resposta: {duracao:.2f} ms")
                logging.info(f"IP: {request.remote_addr} | Status: {status} | Iluminação: {iluminacao:.2f} | Reconhecido: não | Tempo resposta: {duracao:.2f} ms")

                recognition_log = RecognitionLog(
                    user_id= None,
                    confidence= None,   # 0–1
                    status=status,                                        # 'recognized', 'unknown', 'no_face', 'error'
                    detected_at=fim,                        # ou o timestamp de início da req
                    ip=request.remote_addr,
                    light_level=iluminacao,
                    recognized= False,
                    latency_ms=duracao,
                )
                db.session.add(recognition_log)
                db.session.commit()

                return jsonify({
                    'status': 'error',
                    'message': 'Erro ao processar a face detectada',
                    'detection_id': recognition_log.id,
                    'tempo_resposta_ms': duracao
                })

            # ======== compara com usuários cadastrados (que tenham embedding salvo) ========
            faces = Facial.query.all()
            best_match = None
            best_distance = float("inf")  # rastreamento do mais próximo SEM condicional
            best_similarity = 0.0

            for face in faces:
                known_face_embedding = face.get_face_encoding()  # deve retornar np.array shape (128,)
                if known_face_embedding is None:
                    continue
                
                dist = float(np.linalg.norm(face_embedding - known_face_embedding))
                if dist < best_distance:
                    best_distance = dist
                    best_match = face
                    # similaridade "didática" (não probabilística)
                    best_similarity = max(0.0, 1.0 - dist)
            if best_match is not None and best_distance < EMBEDDING_DISTANCE_THRESHOLD:
                best_match = User.query.filter_by(id=best_match.user_id).first()
                
                total_reconhecidos += 1
                fim = datetime.now()
                duracao = (fim - inicio).total_seconds() * 1000  # em ms
                status = 'recognized'
                best_distance = 1 - best_distance
                
                print(f"[{datetime.now()}] IP: {request.remote_addr} | Status: {status} | Iluminação: {iluminacao:.2f} | Reconhecido: sim | Usuário: {best_match.username} ({best_distance:.2%}) | Tempo resposta: {duracao:.2f} ms")
                logging.info(f"IP: {request.remote_addr} | Status: {status} | Iluminação: {iluminacao:.2f} | Reconhecido: sim | Usuário: {best_match.username} ({best_distance:.2%}) | Tempo resposta: {duracao:.2f} ms")

                recognition_log = RecognitionLog(
                    user_id= best_match.id,
                    confidence= score_conf(best_distance, EMBEDDING_DISTANCE_THRESHOLD),
                    status=status,                                        # 'recognized', 'unknown', 'no_face', 'error'
                    detected_at=fim,                        # ou o timestamp de início da req
                    ip=request.remote_addr,
                    light_level=iluminacao,
                    recognized= True,
                    latency_ms=duracao,
                )
                db.session.add(recognition_log)
                db.session.commit()
            
                title = "Rosto reconhecido"
                body  = f"{best_match.username} reconhecido ({best_distance:.0%})"
                payload = {
                    "status": "recognized",
                    "user_id": best_match.id,
                    "username": best_match.username,
                    "confidence": f"{best_distance:.2f}",
                    "detection_id": recognition_log.id
                }
            
                results = push_to_user_devices(best_match.id, title, body, payload)
                logging.info(f"FCM results: {results}")

                return jsonify({
                    'status': 'recognized',
                    'user': best_match.to_dict(),
                    'confidence': best_distance,
                    'detection_id': recognition_log.id,
                    'message': f"Usuário {best_match.username} reconhecido com {best_distance:.2%} de confiança",
                    'tempo_resposta_ms': duracao
                })
            else:
                fim = datetime.now()
                duracao = (fim - inicio).total_seconds() * 1000  # em ms

                status = 'unknown'
                print(f"[{datetime.now()}] IP: {request.remote_addr} | Status: {status} | Iluminação: {iluminacao:.2f} | Reconhecido: não | Tempo resposta: {duracao:.2f} ms")
                logging.info(f"IP: {request.remote_addr} | Status: {status} | Iluminação: {iluminacao:.2f} | Reconhecido: não | Tempo resposta: {duracao:.2f} ms")

                recognition_log = RecognitionLog(
                    user_id= None,
                    confidence= None,   # 0–1
                    status=status,                                        # 'recognized', 'unknown', 'no_face', 'error'
                    detected_at=fim,                        # ou o timestamp de início da req
                    ip=request.remote_addr,
                    light_level=iluminacao,
                    recognized= False,
                    latency_ms=duracao,
                )
                db.session.add(recognition_log)
                db.session.commit()

                return jsonify({
                    'status': 'unknown',
                    'message': 'Face detectada mas usuário não reconhecido',
                    'detection_id': recognition_log.id,
                    'tempo_resposta_ms': duracao
                })

        return jsonify({'error': 'Tipo de arquivo não permitido'}), 400

    except Exception as e:
        fim = datetime.now()
        return jsonify({'error': f'Erro interno do servidor: {str(e)}'}), 500