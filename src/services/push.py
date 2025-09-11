import logging
from typing import Dict, List
from sqlalchemy.exc import SQLAlchemyError

from src.models.user import Device, db
from src.services.firebase import send_to_token
from firebase_admin import exceptions as fb_exceptions

def push_to_user_devices(user_id: int, title: str, body: str, data: Dict) -> List[Dict]:
    """
    Envia push para todos os devices do usuário.
    Remove tokens inválidos (Unregistered) do banco.
    Retorna lista de resultados por token.
    """
    
    results = []
    devices = Device.query.filter_by(user_id=user_id).all()
    if not devices:
        return [{"warning": "no_devices", "user_id":user_id}]
    
    for d in devices:
        try:
            message_id = send_to_token(d.token, title, body, data)
            results.append({"token": d.token, "message_id": message_id})
        except fb_exceptions.UnregisteredError:
            results.append({"token": d.token, "error": "unregistered"})
            try:
                db.session.delete(d)
                db.session.commit()
            except SQLAlchemyError as e:
                db.session.rollback()
                logging.exception("Erro ao remover token inválido %s", e)
        except Exception as e:
            results.append({"token": d.token, "error": str(e)})
            logging.exception("Erro ao enviar FCM %s", e)
    return results