from flask import Blueprint, jsonify, request
from werkzeug.utils import secure_filename
import os
import cv2
import json
import numpy as np
from datetime import datetime
import base64
from io import BytesIO
from PIL import Image

from src.models.user import User, DetectionLog, Notification, db #,Schedule
from config import get_config

from src.models.user import User, db, Facial

user_bp = Blueprint('user', __name__)

# Configurações
UPLOAD_FOLDER = 'uploads/profiles'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # sobe 2 níveis: de routes -> src -> backend
MODELS_DIR = os.path.join(BASE_DIR, "models")

PROTOTXT_PATH = os.path.join(MODELS_DIR, "deploy.prototxt")
CAFFEMODEL_PATH = os.path.join(MODELS_DIR, "res10_300x300_ssd_iter_140000.caffemodel")
EMBEDDING_MODEL_PATH = os.path.join(MODELS_DIR, "openface.nn4.small2.v1.t7")

# Carregar os modelos
try:
    face_detector = cv2.dnn.readNet(PROTOTXT_PATH, CAFFEMODEL_PATH)
    face_recognizer = cv2.dnn.readNetFromTorch(EMBEDDING_MODEL_PATH)
    print("Modelos DNN carregados com sucesso!")
except Exception as e:
    print(f"ERRO ao carregar modelos DNN: {e}")
    print("Verifique se os arquivos de modelo estão na pasta src/models/ e se os caminhos estão corretos.") 
    face_detector = None
    face_recognizer = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def ensure_upload_folder():
    """Garante que a pasta de uploads existe"""
    upload_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), UPLOAD_FOLDER)
    if not os.path.exists(upload_path):
        os.makedirs(upload_path)
    return upload_path

def extract_face_features(image_path):  
    try:    
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
                
                '''# --- PARTE NOVA ---
                vec = np.array(vec, dtype=np.float32).flatten()  # Garante 1D
                if vec.shape[0] != 128:
                    # Pega apenas os primeiros 128 valores ou completa com zeros
                    if vec.shape[0] > 128:
                        vec = vec[:128]
                    else:
                        vec = np.pad(vec, (0, 128 - vec.shape[0]), mode='constant', constant_values=0)
                # --- FIM DA PARTE NOVA ---'''
            
            #return vec, (startX, startY, endX, endY)
            return vec.flatten(), (startX, startY, endX, endY)
        return None, None # Nenhuma face detectada

    except Exception as e:
        print(f"Erro na extração de características: {e}")
        return None

def add_embedding_to_user(user, new_embedding):
    """Adiciona um novo embedding ao usuário, garantindo lista no banco"""
    new_embedding = np.array(new_embedding, dtype=np.float32).flatten()
    if new_embedding.shape[0] != 128:
        if new_embedding.shape[0] > 128:
            new_embedding = new_embedding[:128]
        else:
            new_embedding = np.pad(new_embedding, (0, 128 - new_embedding.shape[0]),
                                   mode='constant', constant_values=0)

    # Se não existir embedding ainda, cria lista
    if user.face_encoding is None:
        embeddings_list = []
    else:
        embeddings_list = json.loads(user.face_encoding)

    embeddings_list.append(new_embedding.tolist())
    user.face_encoding = json.dumps(embeddings_list)

def add_embedding_to_facial(user_id, new_embedding):
    """Adiciona um novo embedding ao usuário, garantindo lista no banco"""
    facial = Facial(user_id = user_id)

    new_embedding = np.array(new_embedding, dtype=np.float32).flatten()
    if new_embedding.shape[0] != 128:
        if new_embedding.shape[0] > 128:
            new_embedding = new_embedding[:128]
        else:
            new_embedding = np.pad(new_embedding, (0, 128 - new_embedding.shape[0]),
                                   mode='constant', constant_values=0)

    # Se não existir embedding ainda, cria lista
    if facial.face_encoding is None:
        embeddings_list = []
    else:
        embeddings_list = json.loads(facial.face_encoding)

    embeddings_list.append(new_embedding.tolist())
    facial.face_encoding = json.dumps(new_embedding.tolist())
    facial.user_id = user_id

    db.session.add(facial)

@user_bp.route('/users', methods=['GET'])
def get_users():
    users = User.query.all()
    return jsonify([user.to_dict() for user in users])

@user_bp.route('/users', methods=['POST'])
def create_user():
    try:
        if request.content_type and 'multipart/form-data' in request.content_type:
            username = request.form.get('username')
            email = request.form.get('email')
            phone = request.form.get('phone')

            if not username or not email:
                return jsonify({'error': 'username e email são obrigatórios'}), 400

            # Verifica se usuário já existe
            existing_user = User.query.filter_by(username=username).first()

            if existing_user:
                if existing_user.email == email:
                    # Atualiza: adiciona novo embedding
                    if 'profile_image' in request.files:
                        file = request.files['profile_image']
                        if file and file.filename != '' and allowed_file(file.filename):
                            upload_path = ensure_upload_folder()
                            filename = secure_filename(f"{username}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}")
                            file_path = os.path.join(upload_path, filename)
                            file.save(file_path)

                            vec, _ = extract_face_features(file_path)
                            if vec is not None:
                                add_embedding_to_user(existing_user, vec)
                                #facial = Facial(user_id=3)
                                add_embedding_to_facial(existing_user.id, vec)
                                existing_user.profile_image_path = file_path
                                db.session.commit()
                                return jsonify({'message': 'Novo embedding adicionado ao usuário existente'}), 200
                            else:
                                return jsonify({'error': 'Não foi possível detectar face na imagem fornecida', 'user': existing_user.to_dict()}), 400
                    return jsonify({'error': 'Nenhuma imagem enviada para atualizar embedding'}), 400
                else:
                    return jsonify({'error': 'Username já existe com outro email'}), 400

            # Novo usuário
            user = User(username=username, email=email, phone=phone)

            if 'profile_image' in request.files:
                file = request.files['profile_image']
                if file and file.filename != '' and allowed_file(file.filename):
                    upload_path = ensure_upload_folder()
                    filename = secure_filename(f"{username}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}")
                    file_path = os.path.join(upload_path, filename)
                    file.save(file_path)

                    vec, _ = extract_face_features(file_path)
                    if vec is not None:
                        #add_embedding_to_user(user, vec)
                        db.session.add(user)
                        db.session.commit()
                        existing_user = User.query.filter_by(username=username).first()
                        add_embedding_to_facial(existing_user.id, vec)
                        db.session.commit()
                        user.profile_image_path = file_path
                    else:
                        return jsonify({'error': 'Não foi possível detectar face na imagem fornecida'}), 400

            return jsonify({
                'message': 'Usuário criado com sucesso',
                'user': user.to_dict()
            }), 201

        else:
            # Dados JSON (sem imagem)
            data = request.json
            if not data or 'username' not in data or 'email' not in data:
                return jsonify({'error': 'username e email são obrigatórios'}), 400

            existing_user = User.query.filter_by(username=data['username']).first()
            if existing_user:
                if existing_user.email == data['email']:
                    return jsonify({'error': 'Usuário já existe, envie uma imagem para adicionar embedding'}), 400
                else:
                    return jsonify({'error': 'Username já existe com outro email'}), 400

            user = User(username=data['username'], email=data['email'], phone=data.get('phone'))
            db.session.add(user)
            db.session.commit()

            return jsonify({
                'message': 'Usuário criado com sucesso (sem reconhecimento facial)',
                'user': user.to_dict()
            }), 201

    except Exception as e:
        return jsonify({'error': f'Erro ao criar usuário: {str(e)}'}), 500

@user_bp.route('/users/<int:user_id>', methods=['GET'])
def get_user(user_id):
    user = User.query.get_or_404(user_id)
    return jsonify(user.to_dict())

@user_bp.route('/users/<int:user_id>', methods=['PUT'])
def update_user(user_id):
    try:
        user = User.query.get_or_404(user_id)
        
        if request.content_type and 'multipart/form-data' in request.content_type:
            # Atualização com possível nova imagem
            if 'username' in request.form:
                user.username = request.form['username']
            if 'email' in request.form:
                user.email = request.form['email']
            if 'phone' in request.form:
                user.phone = request.form['phone']
            
            # Processa nova imagem de perfil se fornecida
            if 'profile_image' in request.files:
                file = request.files['profile_image']
                if file and file.filename != '' and allowed_file(file.filename):
                    upload_path = ensure_upload_folder()
                    filename = secure_filename(f"{user.username}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}")
                    file_path = os.path.join(upload_path, filename)
                    file.save(file_path)
                    
                    # Extrai novas características faciais
                    face_encoding = extract_face_features(file_path)
                    if face_encoding is not None:
                        user.set_face_encoding(face_encoding)
                        user.profile_image_path = file_path
                    else:
                        return jsonify({'error': 'Não foi possível detectar face na nova imagem'}), 400
        else:
            # Atualização apenas de dados
            data = request.json
            user.username = data.get('username', user.username)
            user.email = data.get('email', user.email)
            user.phone = data.get('phone', user.phone)
            user.is_active = data.get('is_active', user.is_active)
        
        db.session.commit()
        return jsonify({
            'message': 'Usuário atualizado com sucesso',
            'user': user.to_dict()
        })
        
    except Exception as e:
        return jsonify({'error': f'Erro ao atualizar usuário: {str(e)}'}), 500

@user_bp.route('/users/<int:user_id>', methods=['DELETE'])
def delete_user(user_id):
    try:
        user = User.query.get_or_404(user_id)
        
        # Remove arquivo de imagem se existir
        if user.profile_image_path and os.path.exists(user.profile_image_path):
            os.remove(user.profile_image_path)
        
        db.session.delete(user)
        db.session.commit()
        
        return jsonify({'message': 'Usuário deletado com sucesso'})
    except Exception as e:
        return jsonify({'error': f'Erro ao deletar usuário: {str(e)}'}), 500

@user_bp.route('/users/<int:user_id>/profile-image', methods=['POST'])
def upload_profile_image(user_id):
    """Endpoint específico para upload de imagem de perfil"""
    try:
        user = User.query.get_or_404(user_id)
        
        if 'image' not in request.files:
            return jsonify({'error': 'Nenhuma imagem enviada'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'Nenhum arquivo selecionado'}), 400
        
        if file and allowed_file(file.filename):
            upload_path = ensure_upload_folder()
            filename = secure_filename(f"{user.username}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}")
            file_path = os.path.join(upload_path, filename)
            file.save(file_path)
            
            # Extrai características faciais
            face_encoding = extract_face_features(file_path)
            if face_encoding is not None:
                user.set_face_encoding(face_encoding)
                user.profile_image_path = file_path
                db.session.commit()
                
                return jsonify({
                    'message': 'Imagem de perfil atualizada com sucesso',
                    'user': user.to_dict()
                })
            else:
                # Remove arquivo se não conseguiu detectar face
                os.remove(file_path)
                return jsonify({'error': 'Não foi possível detectar face na imagem'}), 400
        
        return jsonify({'error': 'Tipo de arquivo não permitido'}), 400
        
    except Exception as e:
        return jsonify({'error': f'Erro ao fazer upload da imagem: {str(e)}'}), 500

@user_bp.route('/users/search', methods=['GET'])
def search_users():
    """Busca usuários por nome ou email"""
    try:
        query = request.args.get('q', '').strip()
        if not query:
            return jsonify({'users': []})
        
        users = User.query.filter(
            (User.username.contains(query)) | 
            (User.email.contains(query))
        ).limit(10).all()
        
        return jsonify({
            'users': [user.to_dict() for user in users],
            'query': query
        })
        
    except Exception as e:
        return jsonify({'error': f'Erro na busca: {str(e)}'}), 500
