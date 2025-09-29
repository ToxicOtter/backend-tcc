from flask import Blueprint, request, jsonify
from src.models.user import User

auth_bp = Blueprint('auth', __name__)

@auth_bp.route('auth/login', methods=['POST'])
def login():
    data = request.get_json(silent=True) or {}
    username = data.get('username')
    email = data.get('email')
    if not username and not email:
        return jsonify({'ok': False, 'error': 'informe seu username ou email'}), 400
    
    q = []
    if username: q.append(User.username == username)
    if email: q.append(User.email == email)
    
    user = User.query.filter(*q).first()
    if not user:
        return jsonify({'ok': False}, 'error', 'usuário não encontrado'), 200
    
    return jsonify({'ok': True, 'user': user.to_dict()}), 200

@auth_bp.route('auth/logout', methods=['POST'])
def logout():
    return jsonify({'ok', True}), 200