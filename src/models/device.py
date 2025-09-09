# src/models/device.py
from datetime import datetime
from src.models.user import db  # ✅ reusar a MESMA instância

class Device(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, index=True, nullable=False)
    token   = db.Column(db.String(255), unique=True, nullable=False)
    platform = db.Column(db.String(20))
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
