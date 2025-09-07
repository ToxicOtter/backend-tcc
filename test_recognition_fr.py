# test_recognition.py
import os
import cv2
import numpy as np
import sqlite3
import json

# --- Configuração dos modelos DNN (igual ao backend) --- #
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # sobe 2 níveis: de routes -> src -> backend
MODELS_DIR = os.path.join(BASE_DIR, "models")

PROTOTXT_PATH = os.path.join(MODELS_DIR, "deploy.prototxt")
CAFFEMODEL_PATH = os.path.join(MODELS_DIR, "res10_300x300_ssd_iter_140000.caffemodel")
EMBEDDING_MODEL_PATH = os.path.join(MODELS_DIR, "openface.nn4.small2.v1.t7")

face_detector = cv2.dnn.readNet(PROTOTXT_PATH, CAFFEMODEL_PATH)
face_recognizer = cv2.dnn.readNetFromTorch(EMBEDDING_MODEL_PATH)

# --- Funções de Detecção e Extração (igual ao backend) --- #

def detect_and_extract_face_features(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None, None

    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    
    face_detector.setInput(blob)
    detections = face_detector.forward()

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.8: # Threshold de confiança para detecção
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            startX = max(0, startX)
            startY = max(0, startY)
            endX = min(w, endX)
            endY = min(h, endY)

            face = image[startY:endY, startX:endX]
            if face.shape[0] < 20 or face.shape[1] < 20:
                continue

            faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
            face_recognizer.setInput(faceBlob)
            vec = face_recognizer.forward()
            #garantir que tenha 128 o array
            vec = vec.flatten()
            if vec.shape[0] != 128:
                vec = vec[:128]
            
            return vec, (startX, startY, endX, endY)

    return None, None

def calculate_similarity(encoding1, encoding2):
    print(encoding1)
    print(encoding2)
    return np.linalg.norm(encoding1 - encoding2)

def cosine_similarity(encoding1, encoding2):
    # valores entre -1 e 1, 1 = iguais
    print(encoding1)
    print(encoding2)
    e1 = encoding1 / np.linalg.norm(encoding1)
    e2 = encoding2 / np.linalg.norm(encoding2)
    return np.dot(e1, e2)

def hybrid_distance(encoding1, encoding2, alpha=0.5):
    #e1 = encoding1 / np.linalg.norm(encoding1)
    #e2 = encoding2 / np.linalg.norm(encoding2)
    eu = np.linalg.norm(encoding1 - encoding2)
    cos = 1 - np.dot(encoding1/np.linalg.norm(encoding1), encoding2/np.linalg.norm(encoding2))  # 0 = igual
    return (alpha * eu) + ((1-alpha) * cos)

# --- Função Principal de Teste --- #

def test_recognition():
    # Conectar ao banco de dados
    db_path = os.path.join("src", "database", "app.db")
    if not os.path.exists(db_path):
        print(f"Erro: Banco de dados não encontrado em {db_path}")
        return

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Carregar encodings do banco de dados
    known_encodings = []
    known_users = []
    cursor.execute("SELECT id, username, face_encoding FROM user WHERE face_encoding IS NOT NULL")
    rows = cursor.fetchall()

    if not rows:
        print("Nenhum usuário com encoding facial encontrado no banco de dados.")
        conn.close()
        return

    for row in rows:
        user_id, username, face_encoding_json = row
        if face_encoding_json:
            encoding = np.array(json.loads(face_encoding_json))
            known_encodings.append(encoding)
            known_users.append({"id": user_id, "username": username})
            print(f'\nDEBUG: Usuário {username}, shape encoding banco: {encoding.shape}\n')

    print(f"{len(known_users)} usuários com encodings carregados do banco de dados.")

    # Pedir imagem de teste
    test_image_path = input("Digite o caminho para a imagem de teste: ")
    if not os.path.exists(test_image_path):
        print(f"Erro: Imagem de teste não encontrada em {test_image_path}")
        conn.close()
        return

    # Extrair características da imagem de teste
    test_encoding, _ = detect_and_extract_face_features(test_image_path)

    if test_encoding is None:
        print("Nenhuma face detectada na imagem de teste.")
        conn.close()
        return
    
    print(f"\nDEBUG: Shape encoding teste: {test_encoding.shape}, dtype: {test_encoding.dtype}\n")
    # Comparar com os encodings conhecidos
    best_match_user = None
    best_distance = float("inf")

    for i, known_encoding in enumerate(known_encodings):
        distance = hybrid_distance(test_encoding, known_encoding)
        print(f"\nComparando com {known_users[i]['username']}: Distância = {distance:.4f}\n")

        if distance < best_distance:
            best_distance = distance
            best_match_user = known_users[i]



    # Exibir resultado
    print("\n--- Resultado do Reconhecimento ---")
    if best_match_user:
        print(f"Melhor correspondência: {best_match_user['username']}")
        print(f"Distância Euclidiana: {best_distance:.4f}")

        # Usar o mesmo threshold do backend para decidir
        recognition_threshold = 0.6 # Ajuste este valor se necessário
        if best_distance < recognition_threshold:
            print(f"\nVEREDITO: RECONHECIDO! (Distância < {recognition_threshold})")
        else:
            print(f"\nVEREDITO: NÃO RECONHECIDO. (Distância >= {recognition_threshold})")
    else:
        print("Nenhuma correspondência encontrada.")

    conn.close()

if __name__ == "__main__":
    test_recognition()