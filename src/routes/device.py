from flask import Blueprint, request, jsonify
from src.models.device import Device
from src.models.user import db  # ✅ pegue o db daqui

devices_bp = Blueprint('devices', __name__)

@devices_bp.route('/devices/register', methods=['POST','OPTIONS'])
def register_device():
    print('chegou')
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
