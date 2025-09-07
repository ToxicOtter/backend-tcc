#!/usr/bin/env python3
"""
Script de teste para o Sistema de Reconhecimento Facial
Demonstra como usar as APIs do sistema
"""

import requests
import json
import base64
import os
from io import BytesIO
from PIL import Image, ImageDraw

# Configurações
BASE_URL = "http://192.168.15.44:5001/api"

def test_api_status():
    """Testa se a API está funcionando"""
    print("🔍 Testando status da API...")
    try:
        response = requests.get(f"{BASE_URL}/status")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ API Status: {data['status']}")
            print(f"📝 Mensagem: {data['message']}")
            return True
        else:
            print(f"❌ Erro no status da API: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Erro ao conectar com a API: {e}")
        return False

def create_test_user():
    """Cria um usuário de teste"""
    print("\n👤 Criando usuário de teste...")
    user_data = {
        "username": "Luan",
        "email": "Luan@gmail.com",
        "phone": "1177777-7777"
    }
    
    try:
        response = requests.post(f"{BASE_URL}/users", json=user_data)
        if response.status_code == 201:
            user = response.json()['user']
            print(f"✅ Usuário criado: {user['username']} (ID: {user['id']})")
            return user['id']
        else:
            print(f"❌ Erro ao criar usuário: {response.json()}")
            return None
    except Exception as e:
        print(f"❌ Erro na requisição: {e}")
        return None

def list_users():
    """Lista todos os usuários"""
    print("\n📋 Listando usuários...")
    try:
        response = requests.get(f"{BASE_URL}/users")
        if response.status_code == 200:
            users = response.json()
            print(f"✅ Total de usuários: {len(users)}")
            for user in users:
                print(f"   - {user['username']} ({user['email']}) - ID: {user['id']}")
            return users
        else:
            print(f"❌ Erro ao listar usuários: {response.status_code}")
            return []
    except Exception as e:
        print(f"❌ Erro na requisição: {e}")
        return []

def create_test_image():
    """Cria uma imagem de teste simples"""
    print("\n🖼️ Criando imagem de teste...")
    
    # Cria uma imagem simples com um círculo (simulando um rosto)
    img = Image.new('RGB', (200, 200), color='white')
    draw = ImageDraw.Draw(img)
    
    # Desenha um círculo (rosto)
    draw.ellipse([50, 50, 150, 150], fill='lightblue', outline='blue')
    
    # Desenha olhos
    draw.ellipse([70, 80, 90, 100], fill='black')
    draw.ellipse([110, 80, 130, 100], fill='black')
    
    # Desenha boca
    draw.arc([80, 110, 120, 130], 0, 180, fill='red', width=3)
    
    # Salva a imagem
    test_image_path = "test_face.jpg"
    img.save(test_image_path)
    print(f"✅ Imagem de teste criada: {test_image_path}")
    return test_image_path

def test_image_upload():
    """Testa o upload de imagem para reconhecimento"""
    print("\n📤 Testando upload de imagem...")
    
    # Cria imagem de teste
    image_path = create_test_image()
    
    try:
        with open(image_path, 'rb') as f:
            files = {'image': f}
            response = requests.post(f"{BASE_URL}/images", files=files)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Upload realizado com sucesso!")
            print(f"📊 Status: {result['status']}")
            print(f"💬 Mensagem: {result['message']}")
            
            if 'detection_id' in result:
                print(f"🔍 ID da detecção: {result['detection_id']}")
            
            return result
        else:
            print(f"❌ Erro no upload: {response.json()}")
            return None
            
    except Exception as e:
        print(f"❌ Erro na requisição: {e}")
        return None
    finally:
        # Remove arquivo de teste
        if os.path.exists(image_path):
            os.remove(image_path)

def test_base64_upload():
    """Testa o upload de imagem em base64"""
    print("\n📤 Testando upload em base64...")
    
    # Cria imagem de teste
    image_path = create_test_image()
    
    try:
        with open(image_path, 'rb') as f:
            image_data = base64.b64encode(f.read()).decode('utf-8')
        
        data = {'image': image_data}
        response = requests.post(f"{BASE_URL}/images/base64", json=data)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Upload base64 realizado com sucesso!")
            print(f"📊 Status: {result['status']}")
            print(f"💬 Mensagem: {result['message']}")
            return result
        else:
            print(f"❌ Erro no upload base64: {response.json()}")
            return None
            
    except Exception as e:
        print(f"❌ Erro na requisição: {e}")
        return None
    finally:
        # Remove arquivo de teste
        if os.path.exists(image_path):
            os.remove(image_path)

def test_detections():
    """Lista as detecções realizadas"""
    print("\n🔍 Listando detecções...")
    try:
        response = requests.get(f"{BASE_URL}/detections")
        if response.status_code == 200:
            data = response.json()
            detections = data['detections']
            print(f"✅ Total de detecções: {data['total']}")
            
            for detection in detections[:5]:  # Mostra apenas as 5 mais recentes
                print(f"   - ID: {detection['id']} | Status: {detection['status']} | Data: {detection['detected_at']}")
                if detection['username']:
                    print(f"     Usuário: {detection['username']} | Confiança: {detection['confidence']}")
            
            return detections
        else:
            print(f"❌ Erro ao listar detecções: {response.status_code}")
            return []
    except Exception as e:
        print(f"❌ Erro na requisição: {e}")
        return []

def test_notifications():
    """Lista as notificações"""
    print("\n🔔 Listando notificações...")
    try:
        response = requests.get(f"{BASE_URL}/notifications")
        if response.status_code == 200:
            data = response.json()
            notifications = data['notifications']
            print(f"✅ Total de notificações: {data['total']}")
            print(f"📬 Não lidas: {data['unread_count']}")
            
            for notification in notifications[:3]:  # Mostra apenas as 3 mais recentes
                print(f"   - {notification['message']}")
                print(f"     Usuário: {notification['username']} | Data: {notification['created_at']}")
            
            return notifications
        else:
            print(f"❌ Erro ao listar notificações: {response.status_code}")
            return []
    except Exception as e:
        print(f"❌ Erro na requisição: {e}")
        return []

def main():
    """Função principal de teste"""
    print("🚀 Iniciando testes do Sistema de Reconhecimento Facial")
    print("=" * 60)
    
    # Testa status da API
    if not test_api_status():
        print("❌ API não está funcionando. Verifique se o servidor está rodando.")
        return
    
    # Lista usuários existentes
    users = list_users()
    
    # Cria usuário de teste se não existir
    #if not users:
    user_id = create_test_user()
    if not user_id:
            print("❌ Não foi possível criar usuário de teste.")
            return
    
    # Testa upload de imagem
    test_image_upload()
    
    # Testa upload base64
    test_base64_upload()
    
    # Lista detecções
    test_detections()
    
    # Lista notificações
    test_notifications()
    
    print("\n" + "=" * 60)
    print("✅ Testes concluídos!")
    print("\n📋 Resumo dos endpoints disponíveis:")
    print("   - GET  /api/status           - Status da API")
    print("   - GET  /api/users            - Lista usuários")
    print("   - POST /api/users            - Cria usuário")
    print("   - POST /api/images           - Upload de imagem (ESP32)")
    print("   - POST /api/images/base64    - Upload base64 (ESP32)")
    print("   - GET  /api/detections       - Lista detecções")
    print("   - GET  /api/notifications    - Lista notificações")

if __name__ == "__main__":
    main()

