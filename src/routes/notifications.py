from flask import Blueprint, request, jsonify
from datetime import datetime

from src.models.user import User, Notification, db

notifications_bp = Blueprint('notifications', __name__)

@notifications_bp.route('/notifications', methods=['GET'])
def get_notifications():
    """Lista todas as notificações"""
    try:
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 20, type=int)
        user_id = request.args.get('user_id', type=int)
        unread_only = request.args.get('unread_only', 'false').lower() == 'true'
        
        query = Notification.query
        
        if user_id:
            query = query.filter(Notification.user_id == user_id)
        
        if unread_only:
            query = query.filter(Notification.is_read == False)
        
        notifications = query.order_by(Notification.created_at.desc()).paginate(
            page=page, per_page=per_page, error_out=False
        )
        
        return jsonify({
            'notifications': [notification.to_dict() for notification in notifications.items],
            'total': notifications.total,
            'pages': notifications.pages,
            'current_page': page,
            'unread_count': Notification.query.filter(Notification.is_read == False).count()
        })
        
    except Exception as e:
        return jsonify({'error': f'Erro ao buscar notificações: {str(e)}'}), 500

@notifications_bp.route('/notifications/<int:notification_id>', methods=['GET'])
def get_notification(notification_id):
    """Busca uma notificação específica"""
    try:
        notification = Notification.query.get_or_404(notification_id)
        return jsonify(notification.to_dict())
    except Exception as e:
        return jsonify({'error': f'Erro ao buscar notificação: {str(e)}'}), 500

@notifications_bp.route('/notifications/<int:notification_id>/read', methods=['PUT'])
def mark_notification_read(notification_id):
    """Marca uma notificação como lida"""
    try:
        notification = Notification.query.get_or_404(notification_id)
        notification.is_read = True
        db.session.commit()
        
        return jsonify({
            'message': 'Notificação marcada como lida',
            'notification': notification.to_dict()
        })
    except Exception as e:
        return jsonify({'error': f'Erro ao marcar notificação: {str(e)}'}), 500

@notifications_bp.route('/notifications/mark-all-read', methods=['PUT'])
def mark_all_notifications_read():
    """Marca todas as notificações como lidas"""
    try:
        user_id = request.args.get('user_id', type=int)
        
        query = Notification.query.filter(Notification.is_read == False)
        if user_id:
            query = query.filter(Notification.user_id == user_id)
        
        updated_count = query.update({'is_read': True})
        db.session.commit()
        
        return jsonify({
            'message': f'{updated_count} notificações marcadas como lidas'
        })
    except Exception as e:
        return jsonify({'error': f'Erro ao marcar notificações: {str(e)}'}), 500

@notifications_bp.route('/notifications', methods=['POST'])
def create_notification():
    """Cria uma nova notificação (para testes)"""
    try:
        data = request.get_json()
        
        if not data or 'user_id' not in data or 'message' not in data:
            return jsonify({'error': 'user_id e message são obrigatórios'}), 400
        
        user = User.query.get_or_404(data['user_id'])
        
        notification = Notification(
            user_id=data['user_id'],
            message=data['message'],
            notification_type=data.get('notification_type', 'manual')
        )
        
        db.session.add(notification)
        db.session.commit()
        
        return jsonify({
            'message': 'Notificação criada com sucesso',
            'notification': notification.to_dict()
        }), 201
        
    except Exception as e:
        return jsonify({'error': f'Erro ao criar notificação: {str(e)}'}), 500

@notifications_bp.route('/notifications/<int:notification_id>', methods=['DELETE'])
def delete_notification(notification_id):
    """Deleta uma notificação"""
    try:
        notification = Notification.query.get_or_404(notification_id)
        db.session.delete(notification)
        db.session.commit()
        
        return jsonify({'message': 'Notificação deletada com sucesso'})
    except Exception as e:
        return jsonify({'error': f'Erro ao deletar notificação: {str(e)}'}), 500

@notifications_bp.route('/users/<int:user_id>/notifications/latest', methods=['GET'])
def get_user_latest_notifications(user_id):
    """Busca as últimas notificações de um usuário específico"""
    try:
        user = User.query.get_or_404(user_id)
        limit = request.args.get('limit', 5, type=int)
        
        notifications = Notification.query.filter(
            Notification.user_id == user_id
        ).order_by(
            Notification.created_at.desc()
        ).limit(limit).all()
        
        return jsonify({
            'user': user.to_dict(),
            'notifications': [notification.to_dict() for notification in notifications],
            'unread_count': Notification.query.filter(
                Notification.user_id == user_id,
                Notification.is_read == False
            ).count()
        })
        
    except Exception as e:
        return jsonify({'error': f'Erro ao buscar notificações do usuário: {str(e)}'}), 500

