from flask import Blueprint, request, jsonify
from src.models.user import Device, db  # ✅ pegue o db daqui

devices_bp = Blueprint('devices', __name__)

@devices_bp.route('/devices/register', methods=['POST','OPTIONS'])
def register_device():
    if request.method == 'OPTIONS':
        return ('', 204)
    data = request.get_json(silent=True) or {}
    user_id = data.get('user_id')
    token   = data.get('fcm_token')
    platform = data.get('platform')
    if not user_id or not token:
        return jsonify({'ok': False, 'error': 'user_id e fcm_token são obrigatórios'}), 400
    dev = Device.query.filter_by(token=token).first()
    if dev:
        dev.user_id = user_id
        dev.platform = platform
    else:
        db.session.add(Device(user_id=user_id, token=token, platform=platform))
    db.session.commit()
    return jsonify({'ok': True}), 200

@devices_bp.route('/devices/delete', methods=['POST', 'OPTIONS'])
def delete_device():
    if request.method == 'OPTIONS':
        return ('', 204)
    data = request.get_json(silent=True) or {}
    user_id = data.get('user_id')
    token = data.get('fcm_token')
    if not user_id or not token:
        return jsonify({'ok': False, 'error': 'user_id e fcm_token são obrigatórios'}), 400
    
    dev = Device.query.filter_by(user_id=user_id, token=token).first()
    if not dev:
        return jsonify({'ok': True, 'message':'device já não existe'}), 200
    
    db.session.delete(dev)
    db.session.commit()
    return jsonify({'ok', True}), 200

@devices_bp.route('/devices/refresh', methods=['POST', 'OPTIONS'])
def refresh_device_token():
    """Quando o FCM rotacionar token, atualiza o registro"""
    if request.method == 'OPTIONS':
        return ('', 204)
    data = request.get_json(silent=True) or {}
    user_id = data.get('user_id')
    old_token = data.get('old_token')
    new_token = data.get('new_token')
    plataform = data.get('plataform')
    if not user_id or not old_token or not new_token:
        return jsonify({'ok': False, 'error': 'user_id, old_token e new_token são obrigatórios'})
    
    dev = Device.query.filter_by(user_id=user_id, token=old_token).first()
    if dev:
        dev.token = new_token
        if plataform: dev.plataform = plataform
        db.session.commit()
    else:
        # se não achar o velho, cria com o novo para não perder o push
        db.session.add(Device(user_id=user_id, token=new_token, plataform=plataform))
        db.session.commit()
    
    return jsonify({'ok':True}), 200