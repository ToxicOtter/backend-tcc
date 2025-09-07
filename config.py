"""
Arquivo de Configuração - Sistema de Reconhecimento Facial

Este arquivo centraliza todas as configurações do sistema.
Ajuste os valores conforme necessário para seu ambiente.
"""

import os
from datetime import timedelta

class Config:
    """Configurações base do sistema"""
    
    # ===========================================
    # CONFIGURAÇÕES GERAIS
    # ===========================================
    
    # Chave secreta para sessões Flask (MUDE EM PRODUÇÃO!)
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'sua-chave-secreta-super-segura-aqui' #asdf#FGSgvasgf$5$WGT
    
    # Debug mode (SEMPRE False em produção)
    DEBUG = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    
    # Host e porta do servidor
    HOST = os.environ.get('FLASK_HOST', '0.0.0.0')
    PORT = int(os.environ.get('FLASK_PORT', 5001))
    
    # ===========================================
    # CONFIGURAÇÕES DO BANCO DE DADOS
    # ===========================================
    
    # URL do banco de dados
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or \
        f"sqlite:///{os.path.join(os.path.dirname(__file__), 'src', 'database', 'app.db')}"
    
    # Desabilita tracking de modificações (melhora performance)
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # Pool de conexões (para bancos como PostgreSQL/MySQL)
    SQLALCHEMY_ENGINE_OPTIONS = {
        'pool_pre_ping': True,
        'pool_recycle': 300,
    }
    
    # ===========================================
    # CONFIGURAÇÕES DE UPLOAD
    # ===========================================
    
    # Tamanho máximo de arquivo (16MB)
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024
    
    # Diretório base para uploads
    UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
    
    # Subdiretórios para diferentes tipos de upload
    PROFILE_UPLOAD_FOLDER = os.path.join(UPLOAD_FOLDER, 'profiles')
    DETECTION_UPLOAD_FOLDER = os.path.join(UPLOAD_FOLDER, 'detections')
    
    # Extensões permitidas para imagens
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
    
    # ===========================================
    # CONFIGURAÇÕES DE RECONHECIMENTO FACIAL
    # ===========================================
    
    # Threshold para reconhecimento facial (0.0 a 1.0)
    # Valores menores = mais restritivo
    # Valores maiores = mais permissivo
    FACE_RECOGNITION_THRESHOLD = 0.7
    
    
    # Tamanho padrão para redimensionamento de faces
    FACE_SIZE = (100, 100)
    
    # Qualidade JPEG para salvamento de imagens
    JPEG_QUALITY = 85
    
    # Configurações do detector de faces OpenCV
    FACE_CASCADE_SCALE_FACTOR = 1.1
    FACE_CASCADE_MIN_NEIGHBORS = 4
    FACE_CASCADE_MIN_SIZE = (30, 30)
    
    # ===========================================
    # CONFIGURAÇÕES DE CORS
    # ===========================================
    
    # Origens permitidas para CORS
    CORS_ORIGINS = os.environ.get('CORS_ORIGINS', '*').split(',')
    
    # Métodos HTTP permitidos
    CORS_METHODS = ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS']
    
    # Headers permitidos
    CORS_HEADERS = ['Content-Type', 'Authorization']
    
    # ===========================================
    # CONFIGURAÇÕES DE LOGGING
    # ===========================================
    
    # Nível de log
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')
    
    # Arquivo de log
    LOG_FILE = os.environ.get('LOG_FILE', 'logs/facial_recognition.log')
    
    # Tamanho máximo do arquivo de log (10MB)
    LOG_MAX_BYTES = 10 * 1024 * 1024
    
    # Número de backups de log
    LOG_BACKUP_COUNT = 5
    
    # ===========================================
    # CONFIGURAÇÕES DE SEGURANÇA
    # ===========================================
    
    # Rate limiting (requisições por minuto)
    RATE_LIMIT_GENERAL = "100 per minute"
    RATE_LIMIT_UPLOAD = "10 per minute"
    RATE_LIMIT_AUTH = "5 per minute"
    
    # Timeout para requisições HTTP (segundos)
    REQUEST_TIMEOUT = 30
    
    # Headers de segurança
    SECURITY_HEADERS = {
        'X-Content-Type-Options': 'nosniff',
        'X-Frame-Options': 'DENY',
        'X-XSS-Protection': '1; mode=block',
        'Strict-Transport-Security': 'max-age=31536000; includeSubDomains'
    }
    
    # ===========================================
    # CONFIGURAÇÕES DE NOTIFICAÇÕES
    # ===========================================
    
    # Número máximo de notificações por usuário
    MAX_NOTIFICATIONS_PER_USER = 100
    
    # Tempo de expiração de notificações (dias)
    NOTIFICATION_EXPIRY_DAYS = 30
    
    # ===========================================
    # CONFIGURAÇÕES DE CACHE
    # ===========================================
    
    # Tipo de cache (simple, redis, memcached)
    CACHE_TYPE = os.environ.get('CACHE_TYPE', 'simple')
    
    # URL do Redis (se usando Redis)
    CACHE_REDIS_URL = os.environ.get('REDIS_URL', 'redis://localhost:6379/0')
    
    # Timeout padrão do cache (segundos)
    CACHE_DEFAULT_TIMEOUT = 300
    
    # ===========================================
    # CONFIGURAÇÕES DE EMAIL (OPCIONAL)
    # ===========================================
    
    # Servidor SMTP
    MAIL_SERVER = os.environ.get('MAIL_SERVER')
    MAIL_PORT = int(os.environ.get('MAIL_PORT', 587))
    MAIL_USE_TLS = os.environ.get('MAIL_USE_TLS', 'True').lower() == 'true'
    MAIL_USERNAME = os.environ.get('MAIL_USERNAME')
    MAIL_PASSWORD = os.environ.get('MAIL_PASSWORD')
    
    # Email padrão do sistema
    MAIL_DEFAULT_SENDER = os.environ.get('MAIL_DEFAULT_SENDER')
    
    # ===========================================
    # CONFIGURAÇÕES DE MONITORAMENTO
    # ===========================================
    
    # Habilita métricas de performance
    ENABLE_METRICS = os.environ.get('ENABLE_METRICS', 'False').lower() == 'true'
    
    # Intervalo para limpeza de logs antigos (dias)
    LOG_CLEANUP_DAYS = 30
    
    # Intervalo para limpeza de imagens antigas (dias)
    IMAGE_CLEANUP_DAYS = 90


class DevelopmentConfig(Config):
    """Configurações para ambiente de desenvolvimento"""
    DEBUG = True
    SQLALCHEMY_DATABASE_URI = 'sqlite:///dev_facial_recognition.db'


class ProductionConfig(Config):
    """Configurações para ambiente de produção"""
    DEBUG = False
    
    # Use PostgreSQL em produção
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or \
        'postgresql://user:password@localhost/facial_recognition'
    
    # Configurações de segurança mais rigorosas
    FACE_RECOGNITION_THRESHOLD = 0.7
    RATE_LIMIT_UPLOAD = "5 per minute"
    
    # Logs mais detalhados
    LOG_LEVEL = 'WARNING'


class TestingConfig(Config):
    """Configurações para testes"""
    TESTING = True
    SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'
    WTF_CSRF_ENABLED = False


# Dicionário de configurações
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}


def get_config():
    """Retorna a configuração baseada na variável de ambiente"""
    env = os.environ.get('FLASK_ENV', 'development')
    return config.get(env, config['default'])


# ===========================================
# FUNÇÕES AUXILIARES
# ===========================================

def create_directories():
    """Cria diretórios necessários se não existirem"""
    directories = [
        Config.UPLOAD_FOLDER,
        Config.PROFILE_UPLOAD_FOLDER,
        Config.DETECTION_UPLOAD_FOLDER,
        os.path.dirname(Config.LOG_FILE),
        os.path.dirname(Config.SQLALCHEMY_DATABASE_URI.replace('sqlite:///', ''))
    ]
    
    for directory in directories:
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)


def validate_config():
    """Valida configurações críticas"""
    config_obj = get_config()
    
    # Verifica se SECRET_KEY foi alterada
    if config_obj.SECRET_KEY == 'sua-chave-secreta-super-segura-aqui':
        print("⚠️ AVISO: SECRET_KEY padrão detectada. Altere em produção!")
    
    # Verifica se diretórios existem
    create_directories()
    
    # Verifica configurações de produção
    if isinstance(config_obj, ProductionConfig):
        if config_obj.DEBUG:
            raise ValueError("DEBUG não pode ser True em produção!")
        
        if 'sqlite' in config_obj.SQLALCHEMY_DATABASE_URI.lower():
            print("⚠️ AVISO: SQLite não é recomendado para produção")


if __name__ == '__main__':
    # Teste das configurações
    print("🔧 Testando configurações...")
    
    try:
        validate_config()
        config_obj = get_config()
        
        print(f"✅ Configuração carregada: {config_obj.__name__}")
        print(f"📊 Debug: {config_obj.DEBUG}")
        print(f"🗄️ Banco: {config_obj.SQLALCHEMY_DATABASE_URI}")
        print(f"📁 Upload: {config_obj.UPLOAD_FOLDER}")
        print(f"🎯 Threshold: {config_obj.FACE_RECOGNITION_THRESHOLD}")
        
    except Exception as e:
        print(f"❌ Erro na configuração: {e}")

