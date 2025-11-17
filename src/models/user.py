from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import json
import numpy as np

db = SQLAlchemy()

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    phone = db.Column(db.String(20), nullable=True)
    face_encoding = db.Column(db.Text, nullable=True)  # Armazenar encoding facial como JSON
    profile_image_path = db.Column(db.String(255), nullable=True)  # Caminho da imagem de perfil
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_seen = db.Column(db.DateTime, nullable=True)

    def __repr__(self):
        return f'<User {self.username}>'

    def to_dict(self):
        return {
            'id': self.id,
            'username': self.username,
            'email': self.email,
            'phone': self.phone,
            'profile_image_path': self.profile_image_path,
            'is_active': self.is_active,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'last_seen': self.last_seen.isoformat() if self.last_seen else None,
            "face_encodings_count": len(json.loads(self.face_encoding)) if self.face_encoding else 0
        }

    '''def set_face_encoding(self, encoding):
        """Armazena o encoding facial como JSON"""
        if encoding is not None:
            self.face_encoding = json.dumps(encoding.tolist())'''
    def set_face_encoding(self, encoding):
        """Armazena o encoding facial como JSON seguro."""
        if encoding is None:
            self.face_encoding = None
            return

        # Converter para NumPy array caso seja tupla ou lista
        if isinstance(encoding, tuple) or isinstance(encoding, list):
            encoding = np.array(encoding, dtype=np.float32).flatten()
        elif isinstance(encoding, np.ndarray):
            encoding = encoding.flatten()
        else:
            raise TypeError(f"Tipo inválido para encoding: {type(encoding)}")

        # Garantir tamanho 128
        if encoding.shape[0] != 128:
            if encoding.shape[0] > 128:
                encoding = encoding[:128]  # corta extras
            else:
                # preenche com zeros se faltar
                encoding = np.pad(encoding, (0, 128 - encoding.shape[0]), mode='constant', constant_values=0)

        # Salvar como JSON
        self.face_encoding = json.dumps(encoding.tolist())
        

    def get_face_encoding(self):
        """Recupera o encoding facial como numpy array"""
        if self.face_encoding:
            import numpy as np
            return np.array(json.loads(self.face_encoding))
        return None

class RecognitionLog(db.Model):
    """Log de detecções"""
    id = db.Column(db.Integer, primary_key=True)
    detected_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    ip = db.Column(db.String(45), nullable=True) 
    status = db.Column(db.String(20), default='unknown')           # 'recognized', 'unknown', 'no_face', 'error'
    light_level = db.Column(db.Float, nullable=True)               # iluminacao
    recognized = db.Column(db.Boolean, default=False)              # reconhecido (sim/nao -> True/False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)
    confidence = db.Column(db.Float, nullable=True)                # já tinha; mantenha como 0-1 ou % (defina padrão)
    latency_ms = db.Column(db.Float, nullable=True)                # tempo_resposta_ms

    def to_dict(self):    
        return {
            'detected_at': self.detected_at,
            'ip': self.ip,
            'status': self.status,
            'light_level': self.light_level,
            'recognized': self.recognized,
            'user_id': self.user_id,
            'confidence': self.confidence,
            'latency_ms': self.latency_ms
        }
    
class Facial(db.Model):
    """Notificações para o app mobile"""
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    face_encoding = db.Column(db.Text, nullable=True)  # Armazenar encoding facial como JSON
    
    user = db.relationship('User', backref=db.backref('facial', lazy=True))

    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id
        }
    
    def get_face_encoding(self):
        """Recupera o encoding facial como numpy array"""
        if self.face_encoding:
            import numpy as np
            return np.array(json.loads(self.face_encoding))
        return None
    
class Device(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, index=True, nullable=False)
    token   = db.Column(db.String(255), unique=True, nullable=False)
    platform = db.Column(db.String(20))
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'token': self.token,
            'platform': self.platform,
            'updated_at': self.updated_at
        }