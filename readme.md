# Facial Recognition Backend  
### Backend em Python/Flask + Integração ESP32-CAM  
Suporte para Windows • Linux • macOS

Este repositório contém o backend responsável pelo processamento de imagens, reconhecimento facial, registro de logs, comunicação com Firebase e integração com o dispositivo pervasivo **ESP32-CAM**.  
Este documento descreve como instalar, configurar e executar o backend, assim como preparar o ESP32-CAM para envio de imagens.

---

## 📑 Sumário
- [📂 Requisitos](#-requisitos)
- [🖥️ Instalação do Backend](#️-instalação-do-backend)
- [🪟 Windows](#-windows)
- [🐧 Linux](#-linux)
- [🍏 macOS](#-macos)
- [📸 Configuração do ESP32-CAM](#-configuração-do-esp32-cam)
- [🚀 Execução Completa](#-execução-completa)

---

# 📂 Requisitos

Antes da instalação, certifique-se de ter os seguintes softwares:

### Requisitos gerais
- **Git**
- **Python 3.13.x ou superior**
- **Arduino IDE**
- **Conta no Firebase** para configurar o FCM
- **Drivers USB do ESP32-CAM**, se necessário:
  - CH340
  - CP210x

---

# 🖥️ Instalação do Backend

## 1. Clonar o repositório

```bash
git clone https://github.com/ToxicOtter/backend-tcc.git
cd backend-tcc
```

---

# 🪟 Windows

## 1. Criar ambiente virtual
```cmd
python -m venv venv
.env\Scriptsctivate
```

## 2. Instalar dependências
```cmd
pip install -r requirements.txt
```

## 3. Configurar Firebase
1. Abra o **Firebase Console**
2. Vá em *Configurações do Projeto → Contas de Serviço*
3. Clique em **Gerar nova chave privada**
4. Renomeie o arquivo para:

```
firebase-service-account.json
```

5. Coloque na pasta:

```
firebase/
```

## 4. Executar o servidor
```cmd
flask run --host=0.0.0.0
```

## 5. Obter o IP local
```cmd
ipconfig
```

Use o valor de **IPv4** (ex: `192.168.1.10`).

---

# 🐧 Linux (Debian/Ubuntu)

## 1. Instalar dependências do sistema
```bash
sudo apt update
sudo apt install git python3 python3-pip python3-venv build-essential cmake
```

## 2. Criar e ativar o ambiente virtual
```bash
python3 -m venv venv
source venv/bin/activate
```

## 3. Instalar dependências Python
```bash
pip install -r requirements.txt
```

## 4. Configurar Firebase
Coloque o arquivo:

```
firebase/firebase-service-account.json
```

## 5. Inicializar banco (se necessário)
```bash
flask initdb
```

## 6. Executar o servidor
```bash
flask run --host=0.0.0.0
```

## 7. Obter IP local
```bash
ip addr | grep "inet " | grep -v "127.0.0.1"
```

---

# 🍏 macOS

## 1. Instalar ferramentas do Xcode
```bash
xcode-select --install
```

## 2. Criar e ativar ambiente virtual
```bash
python3 -m venv venv
source venv/bin/activate
```

## 3. Instalar dependências
```bash
pip install -r requirements.txt
```

## 4. Configurar Firebase
Salvar em:

```
firebase/firebase-service-account.json
```

## 5. Inicializar banco (opcional)
```bash
flask initdb
```

## 6. Executar o servidor
```bash
flask run --host=0.0.0.0
```

## 7. Obter o IP local
```bash
ifconfig | grep "inet " | grep -v 127.0.0.1
```

---

# 📸 Configuração do ESP32-CAM

O código do ESP32-CAM encontra-se em:

```
esp32-cam-code/esp32-cam-code.ino
```

## 1. Abrir na Arduino IDE
- Abra a IDE  
- Vá em **Arquivo → Abrir**  
- Selecione o arquivo `.ino`

---

## 2. Editar configurações no código

Localize e configure:

```cpp
const char* ssid = "NOME_DA_REDE";
const char* password = "SENHA_DA_REDE";
String serverName = "http://SEU_IP_LOCAL:5000/upload";
```

⚠️ O backend deve estar em execução e acessível pela rede local.

---

## 3. Selecionar placa e porta
Na Arduino IDE:

- **Placa:** `AI Thinker ESP32-CAM`
- **Programador:** FTDI
- **Porta Serial:**
  - Windows → `COMX`
  - Linux → `/dev/ttyUSB0`
  - macOS → `/dev/cu.usbserial-*`

---

## 4. Permissões no Linux (se necessário)

```bash
sudo usermod -a -G dialout $USER
```

Reinicie a sessão.

---

## 5. Flash (Upload)
1. Conecte o ESP32 usando o FTDI  
2. Clique em **Upload**  

Após reiniciar, o dispositivo:

- conecta ao Wi-Fi  
- captura imagens  
- envia automaticamente para o backend  

---

# 🚀 Execução Completa

Após seguir os passos:

- ✔ Backend Flask rodando na porta **5000**  
- ✔ Firebase integrado  
- ✔ Banco SQLite inicializado  
- ✔ ESP32-CAM enviando imagens ao endpoint `/upload`  

O sistema estará totalmente funcional.
