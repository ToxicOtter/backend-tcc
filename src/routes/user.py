from flask import Blueprint, jsonify, request
from werkzeug.utils import secure_filename
import os
import cv2
import json
import numpy as np
from datetime import datetime
from PIL import Image, ImageOps
import face_recognition
from src.models.user import User, db, Facial

user_bp = Blueprint('user', __name__)

# Configurações
UPLOAD_FOLDER = 'uploads/profiles'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def ensure_upload_folder():
    """Garante que a pasta de uploads existe"""
    upload_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), UPLOAD_FOLDER)
    if not os.path.exists(upload_path):
        os.makedirs(upload_path)
    return upload_path

# parâmetros de qualidade
MIN_FACE_SIZE = 110         # mínimo em px de altura/largura da ROI (ajuste 100–140)
MAX_INPUT_W = 1280          # evita estourar memória em fotos enormes
TRY_CNN_FALLBACK = False    # se True, tenta model="cnn" por último (lento em CPU)

def _load_rgb_exif(path):
    """Carrega imagem corrigindo orientação EXIF e retorna ndarray RGB."""
    im = Image.open(path)
    im = ImageOps.exif_transpose(im)     # corrige rotação
    im = im.convert("RGB")
    arr = np.array(im)                   # RGB
    # limita tamanho máximo pra não matar CPU
    if arr.shape[1] > MAX_INPUT_W:
        scale = MAX_INPUT_W / arr.shape[1]
        new_h = int(arr.shape[0] * scale)
        im = im.resize((MAX_INPUT_W, new_h))
        arr = np.array(im)
    return arr

def _ensure_min_size(rgb, target_min=320):
    """Se a menor dimensão for muito pequena, faz upscale x2 para ajudar o HOG."""
    h, w = rgb.shape[:2]
    if min(h, w) < target_min:
        rgb = cv2.resize(rgb, (w*2, h*2), interpolation=cv2.INTER_CUBIC)
    return rgb

def _boxes_from_haar(rgb, scaleFactor=1.1, minNeighbors=4):
    """Fallback com Haar frontal — retorna boxes no formato (top,right,bottom,left)."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    dets = face_cascade.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=minNeighbors)
    boxes = []
    for (x, y, w, h) in dets:
        t, l, r, b = y, x, x + w, y + h
        boxes.append((t, r, b, l))
    return boxes

def extract_face_features(image_path):
    """
    Detecta a face e retorna (embedding_128d, box)
    box: (top, right, bottom, left).
    Estratégia:
      1) Corrige EXIF e garante RGB
      2) HOG com upsample 0 → 1 → 2
      3) (opcional) CNN como último recurso
      4) Fallback Haar
      5) Checa tamanho mínimo e extrai encoding
    """
    try:
        rgb = _load_rgb_exif(image_path)
        rgb = _ensure_min_size(rgb, target_min=320)

        # 1) Tenta HOG (rápido)
        for up in (0, 1, 2):
            boxes = face_recognition.face_locations(rgb, model="hog", number_of_times_to_upsample=up)
            if boxes:
                break

        # 2) (Opcional) CNN como último recurso (lento em CPU)
        if not boxes and TRY_CNN_FALLBACK:
            boxes = face_recognition.face_locations(rgb, model="cnn", number_of_times_to_upsample=0)

        # 3) Fallback Haar
        if not boxes:
            boxes = _boxes_from_haar(rgb, scaleFactor=1.1, minNeighbors=5)

        if not boxes:
            print("[extract_face_features] nenhum rosto detectado")
            return None, None

        # escolha a maior face (melhor para cadastro)
        def area(b): t, r, btm, l = b; return (btm - t) * (r - l)
        box = max(boxes, key=area)
        t, r, b, l = box
        h, w = (b - t), (r - l)

        # 4) checa tamanho mínimo
        if h < MIN_FACE_SIZE or w < MIN_FACE_SIZE:
            print(f"[extract_face_features] ROI muito pequena: {w}x{h}px")
            return None, None

        # 5) extrai embedding 128-D
        encs = face_recognition.face_encodings(rgb, known_face_locations=[box], num_jitters=0)
        if not encs:
            # pequeno reforço
            encs = face_recognition.face_encodings(rgb, known_face_locations=[box], num_jitters=1)
            if not encs:
                print("[extract_face_features] falha ao encodar embedding")
                return None, None

        vec = encs[0].astype(np.float32)
        return vec, box

    except Exception as e:
        print(f"[extract_face_features] erro: {e}")
        return None, None

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
            return jsonify({'error': 'Parâmetro q obrigatório'}), 400
        
        user = User.query.filter(
            (User.username == query) | 
            (User.email == query)
        ).first()
        
        print(query)
        print(user)
        
        if not user:
            return jsonify({'user': None, 'message': 'Usuário não encontrado'}), 404

        return jsonify({'user': user.to_dict()})
        
    except Exception as e:
        return jsonify({'error': f'Erro na busca: {str(e)}'}), 500
