import os
import cv2
import numpy as np
from flask import Blueprint, request, jsonify, current_app
from datetime import datetime, time, timedelta
from werkzeug.utils import secure_filename
import base64
from io import BytesIO
from PIL import Image
import logging
import face_recognition

from src.models.user import User, DetectionLog, Notification, Facial, RecognitionLog, Device, db #,Schedule
#from src.utils.notifications import send_push_notification
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


# --- Configuração dos modelos DNN ---
# Caminhos para os modelos. Assumimos que estão em src/models/ dentro do projeto.
# Certifique-se de que esses arquivos existem na pasta src/models/

# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # sobe 2 níveis: de routes -> src -> backend
# MODELS_DIR = os.path.join(BASE_DIR, "models")

# PROTOTXT_PATH = os.path.join(MODELS_DIR, "deploy.prototxt")
# CAFFEMODEL_PATH = os.path.join(MODELS_DIR, "res10_300x300_ssd_iter_140000.caffemodel")
# EMBEDDING_MODEL_PATH = os.path.join(MODELS_DIR, "openface.nn4.small2.v1.t7")


# Carregar os modelos
# try:
#     face_detector = cv2.dnn.readNet(PROTOTXT_PATH, CAFFEMODEL_PATH)
#     face_recognizer = cv2.dnn.readNetFromTorch(EMBEDDING_MODEL_PATH)
#     print("Modelos DNN carregados com sucesso!")
# except Exception as e:
#     print(f"ERRO ao carregar modelos DNN: {e}")
#     print("Verifique se os arquivos de modelo estão na pasta src/models/ e se os caminhos estão corretos.") 
#     face_detector = None
#     face_recognizer = None

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

def detect_and_extract_face_features(image_path):
    """Detecta faces na imagem e extrai as características (embeddings) usando DNN."""
    if face_detector is None or face_recognizer is None:
        print("Modelos DNN não carregados. Não é possível detectar e extrair características.")
        return None, None

    image = cv2.imread(image_path)
    if image is None:
        print(f"Erro: Não foi possível carregar a imagem em {image_path}")
        return None, None

    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    
    face_detector.setInput(blob)
    detections = face_detector.forward()

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.8: # Threshold de confiança para detecção de rosto (pode ser ajustado)
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            startX = max(0, startX)
            startY = max(0, startY)
            endX = min(w, endX)
            endY = min(h, endY)

            face = image[startY:endY, startX:endX]
            if face.shape[0] < 20 or face.shape[1] < 20: # Ignora rostos muito pequenos
                continue

            faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
            face_recognizer.setInput(faceBlob)
            vec = face_recognizer.forward()

            '''#garantir que tenha 128 o array
            vec = vec.flatten()
            if vec.shape[0] != 128:
                vec = vec[:128]
            
            return vec, (startX, startY, endX, endY)'''
            
            return vec.flatten(), (startX, startY, endX, endY)

    return None, None # Nenhuma face detectada

def calculate_similarity(encoding1, encoding2):
    # Menor distância = maior similaridade
    return np.linalg.norm(encoding1 - encoding2)

def cosine_similarity(encoding1, encoding2):
    # valores entre -1 e 1, 1 = iguais
    e1 = encoding1 / np.linalg.norm(encoding1)
    e2 = encoding2 / np.linalg.norm(encoding2)
    return np.dot(e1, e2)

def hybrid_distance(encoding1, encoding2, alpha=0.5):
    #e1 = encoding1 / np.linalg.norm(encoding1)
    #e2 = encoding2 / np.linalg.norm(encoding2)
    eu = np.linalg.norm(encoding1 - encoding2)
    cos = 1 - np.dot(encoding1/np.linalg.norm(encoding1), encoding2/np.linalg.norm(encoding2))  # 0 = igual
    return (alpha * eu) + ((1-alpha) * cos)

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

def compare_embeddings(embedding1, embedding2, threshold=EMBEDDING_DISTANCE_THRESHOLD):
    """
    Compara 2 embeddings 128-D via distância L2.
    Retorna (is_match: bool, distance: float, similarity: float[0..1 aproximado])
    similarity aqui é 1 - distance (apenas para manter compatibilidade com seu JSON).
    """
    try:
        if embedding1 is None or embedding2 is None:
            return False, 0.0, 0.0
        # face_recognition.face_distance também pode ser usado:
        # dist = face_recognition.face_distance([embedding2], embedding1)[0]
        dist = float(np.linalg.norm(embedding1 - embedding2))
        is_match = dist < threshold
        similarity_approx = max(0.0, 1.0 - dist)  # apenas ilustrativo (não é probabilidade)
        return is_match, dist, similarity_approx
    except Exception as e:
        print(f"Erro na comparação de embeddings: {e}")
        return False, 0.0, 0.0

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
            users = User.query.filter(User.face_encoding.isnot(None)).all()
            faces = Facial.query.all()
            best_match = None
            best_distance = float("inf")  # rastreamento do mais próximo SEM condicional
            best_similarity = 0.0

            #for user in users:
            for face in faces:
                known_face_embedding = face.get_face_encoding()  # deve retornar np.array shape (128,)
                if known_face_embedding is None:
                    continue
                #is_match, dist, sim = compare_embeddings(face_embedding, known_face_embedding, EMBEDDING_DISTANCE_THRESHOLD)
                #if is_match and dist < best_distance:
                #    best_distance = dist
                #    best_similarity = sim
                #    best_match = face
                
                dist = float(np.linalg.norm(face_embedding - known_face_embedding))
                if dist < best_distance:
                    best_distance = dist
                    best_match = face
                    # similaridade "didática" (não probabilística)
                    best_similarity = max(0.0, 1.0 - dist)
            print(best_distance)
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
                    #confidence= best_distance,   # 0–1
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
            
                devices = Device.query.filter_by(user_id=best_match.id).all()
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
                print('chegou aqui2')
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
    
@facial_bp.route("/images/base64", methods=["POST"])
def receive_image_base64():
    """Recebe imagem em base64 do ESP32"""
    try:
        data = request.get_json()
        if not data or "image" not in data:
            return jsonify({"error": "Dados de imagem não fornecidos"}), 400
        
        ensure_upload_folder()
        
        # Decodifica base64
        image_data = base64.b64decode(data["image"])
        image_pil = Image.open(BytesIO(image_data))
        
        # Salva a imagem temporariamente para processamento
        filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_base64.jpg"
        file_path = os.path.join(config.UPLOAD_FOLDER, filename)
        image_pil.save(file_path)
        
        # Detecta face e extrai características usando DNN
        current_face_encoding, face_coords = detect_and_extract_face_features(file_path)

        if current_face_encoding is None:
            os.remove(file_path) # Remove a imagem temporária
            detection_log = DetectionLog(
                user_id=None,
                image_path=file_path, 
                confidence=0.0,
                status="no_face"
            )
            db.session.add(detection_log)
            db.session.commit()
            return jsonify({"status": "no_face", "message": "Nenhuma face detectada na imagem"}), 200

        best_match = None
        best_distance = float("inf") # Para distância, menor é melhor

        # Compara com todos os usuários cadastrados
        users = User.query.all()
        for user in users:
            if user.get_face_encoding() is not None:
                known_face_encoding = user.get_face_encoding()
                
                # Garante que os encodings têm a mesma dimensão
                if known_face_encoding.shape == current_face_encoding.shape:
                    distance = calculate_similarity(current_face_encoding, known_face_encoding)
                    
                    if distance < config.FACE_RECOGNITION_THRESHOLD and distance < best_distance:
                        best_distance = distance
                        best_match = user

        os.remove(file_path) # Remove a imagem temporária

        if best_match:
            # Usuário reconhecido
            best_match.last_seen = datetime.utcnow()
            
            detection_log = DetectionLog(
                user_id=best_match.id,
                image_path=file_path, 
                confidence=best_distance,
                status="recognized"
            )
            db.session.add(detection_log)
            
            notification = Notification(
                user_id=best_match.id,
                message=f"Usuário {best_match.username} detectado via Base64",
                notification_type="recognition"
            )
            db.session.add(notification)
            
            db.session.commit()
            
            return jsonify({
                "status": "recognized",
                "user": best_match.to_dict(),
                "distance": float(f"{best_distance:.4f}"), 
                "message": f"Usuário {best_match.username} reconhecido via Base64."
            })
        else:
            detection_log = DetectionLog(
                user_id=None,
                image_path=file_path, 
                confidence=0.0,
                status="unknown"
            )
            db.session.add(detection_log)
            db.session.commit()
            return jsonify({"status": "unknown", "message": "Face detectada via Base64, mas usuário não reconhecido"}), 200

    except Exception as e:
        return jsonify({"error": f"Erro ao processar imagem base64: {str(e)}"}), 500

@facial_bp.route("/detections", methods=["GET"])
def get_detections():
    """Lista todas as detecções"""
    try:
        page = request.args.get("page", 1, type=int)
        per_page = request.args.get("per_page", 10, type=int)
        
        detections = DetectionLog.query.order_by(DetectionLog.detected_at.desc()).paginate(
            page=page, per_page=per_page, error_out=False
        )
        
        return jsonify({
            "detections": [detection.to_dict() for detection in detections.items],
            "total": detections.total,
            "pages": detections.pages,
            "current_page": page
        })
        
    except Exception as e:
        return jsonify({"error": f"Erro ao buscar detecções: {str(e)}"}), 500

@facial_bp.route("/detections/<int:detection_id>", methods=["GET"])
def get_detection(detection_id):
    """Busca uma detecção específica"""
    try:
        detection = DetectionLog.query.get_or_404(detection_id)
        return jsonify(detection.to_dict())
    except Exception as e:
        return jsonify({"error": f"Erro ao buscar detecção: {str(e)}"}), 500