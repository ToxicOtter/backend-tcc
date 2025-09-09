# src/services/firebase.py
import os
import firebase_admin
from firebase_admin import credentials, messaging

# coloque seu service account em src/secrets/service-account.json (NÃO comite)
CRED_PATH = os.path.join(os.path.dirname(__file__), '..', 'secrets', 'service-account.json')

if not firebase_admin._apps:
    cred = credentials.Certificate(CRED_PATH)
    firebase_admin.initialize_app(cred)

def send_to_token(token: str, title: str, body: str, data: dict | None = None) -> str:
    msg = messaging.Message(
        notification=messaging.Notification(title=title, body=body),
        data={k: str(v) for k, v in (data or {}).items()},
        token=token,
        android=messaging.AndroidConfig(priority='high'),
        apns=messaging.APNSConfig(
            payload=messaging.APNSPayload(aps=messaging.Aps(content_available=True))
        ),
    )
    return messaging.send(msg)
