
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script: post_users_one_to_one.py
--------------------------------
Atribui **exatamente 1 imagem** a **exatamente 1 usuário** (1:1), na ordem.
- Lê CSV com colunas [Nome, Email, Telefone]
- Lista e ordena as imagens da pasta
- Faz POST /users com multipart/form-data usando a imagem correspondente ao usuário (mesmo índice)
- Se sobrar usuários sem imagem ou imagens sem usuário, apenas reporta o excedente (padrão: não envia).

Uso:
  python post_users_one_to_one.py \
    --base-url "http://localhost:5000" \
    --csv "/caminho/para/dados_205_sem_acentos.csv" \
    --images "/caminho/para/pasta_imagens"

Requisitos: requests, pandas
"""
import os
import sys
import argparse
import glob
from typing import List
import pandas as pd
import requests

ALLOWED_EXT = {".png", ".jpg", ".jpeg"}

def list_images(images_dir: str) -> List[str]:
    patterns = ["*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG"]
    files = []
    for pat in patterns:
        files.extend(glob.glob(os.path.join(images_dir, pat)))
    files.sort()
    return files

def post_user(session: requests.Session, base_url: str, username: str, email: str, phone: str,
              image_path: str, timeout: float, verify_tls: bool) -> requests.Response:
    url = base_url.rstrip("/") + "/api/users"
    files = {}
    data = {
        "username": username,
        "email": email,
        "phone": phone or "",
    }
    if image_path:
        files["profile_image"] = (os.path.basename(image_path), open(image_path, "rb"), "application/octet-stream")
    try:
        resp = session.post(url, data=data, files=files if files else None, timeout=timeout, verify=verify_tls)
        return resp
    finally:
        if "profile_image" in files and hasattr(files["profile_image"][1], "close"):
            files["profile_image"][1].close()

def main():
    parser = argparse.ArgumentParser(description="Faz POST 1:1 /api/users (uma imagem por usuário, na ordem).")
    parser.add_argument("--base-url", default=os.environ.get("BASE_URL", "http://localhost:5000"))
    parser.add_argument("--csv", dest="csv_path", default=os.environ.get("CSV_PATH"))
    parser.add_argument("--images", dest="images_dir", default=os.environ.get("IMAGES_DIR"))
    parser.add_argument("--timeout", type=float, default=float(os.environ.get("TIMEOUT", "60")))
    parser.add_argument("--verify-tls", dest="verify_tls", default=os.environ.get("VERIFY_TLS", "true").lower() != "false", action="store_true")
    parser.add_argument("--no-verify-tls", dest="verify_tls", action="store_false")
    args = parser.parse_args()

    if not args.csv_path or not os.path.exists(args.csv_path):
        print(f"[ERRO] CSV não encontrado: {args.csv_path}")
        sys.exit(1)
    if not args.images_dir or not os.path.isdir(args.images_dir):
        print(f"[ERRO] Pasta de imagens inválida: {args.images_dir}")
        sys.exit(1)

    # Carrega CSV
    try:
        df = pd.read_csv(args.csv_path)
    except Exception as e:
        print(f"[ERRO] Falha ao ler CSV: {e}")
        sys.exit(1)

    colmap = {c.lower(): c for c in df.columns}
    required = ["nome", "email", "telefone"]
    for req in required:
        if req not in colmap:
            print(f"[ERRO] CSV precisa conter colunas: Nome, Email, Telefone (encontradas: {list(df.columns)})")
            sys.exit(1)

    users = df[[colmap["nome"], colmap["email"], colmap["telefone"]]].values.tolist()
    images = list_images(args.images_dir)

    n_users = len(users)
    n_images = len(images)
    n_pairs = min(n_users, n_images)

    print(f"[INFO] Usuários no CSV: {n_users}")
    print(f"[INFO] Imagens na pasta: {n_images}")
    print(f"[INFO] Vou enviar {n_pairs} POSTs (1:1).")

    session = requests.Session()
    ok = 0
    err = 0

    for i in range(n_pairs):
        username, email, phone = (str(users[i][0]).strip(), str(users[i][1]).strip(), str(users[i][2]).strip())
        img = images[i]

        if not username or not email:
            print(f"[AVISO] Linha {i} inválida (username/email vazio). Pulando...")
            continue

        try:
            resp = post_user(session, args.base_url, username, email, phone, img, args.timeout, args.verify_tls)
            if resp.status_code in (200, 201):
                ok += 1
                try:
                    msg = resp.json()
                except Exception:
                    msg = resp.text[:200]
                print(f"[OK] ({resp.status_code}) {username} <{email}> | {os.path.basename(img)} -> {msg}")
            else:
                err += 1
                try:
                    body = resp.json()
                except Exception:
                    body = resp.text[:500]
                print(f"[ERRO] ({resp.status_code}) {username} | {os.path.basename(img)} -> {body}")
        except requests.RequestException as rexc:
            err += 1
            print(f"[EXC] {username} | {os.path.basename(img)} -> {rexc}")

    # Reporta sobras
    if n_users > n_pairs:
        print(f"[AVISO] {n_users - n_pairs} usuários ficaram sem imagem (não enviados).")
    if n_images > n_pairs:
        print(f"[AVISO] {n_images - n_pairs} imagens sobraram sem usuário correspondente.")

    print("=" * 60)
    print(f"[RESUMO] Sucessos: {ok} | Erros: {err} | Total enviados: {n_pairs}")
    if err > 0:
        sys.exit(2)

if __name__ == "__main__":
    main()
