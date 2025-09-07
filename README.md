# Sistema de Reconhecimento Facial - Backend

## Visão Geral

Este projeto implementa um sistema completo de reconhecimento facial desenvolvido em Python com Flask, projetado especificamente para integração com dispositivos ESP32 e aplicações mobile. O sistema foi desenvolvido como parte de um projeto de TCC (Trabalho de Conclusão de Curso) e oferece uma solução robusta e escalável para identificação facial em tempo real.

### Arquitetura do Sistema

O sistema é composto por três componentes principais:

1. **ESP32 com Câmera**: Captura imagens periodicamente e as envia para o servidor
2. **Backend Flask**: Processa as imagens, realiza reconhecimento facial e gerencia dados
3. **Aplicação Mobile**: Recebe notificações quando usuários são reconhecidos

### Tecnologias Utilizadas

- **Python 3.11**: Linguagem principal do backend
- **Flask**: Framework web para APIs REST
- **OpenCV**: Biblioteca para processamento de imagens e detecção facial
- **SQLite**: Banco de dados para armazenamento local
- **Flask-CORS**: Habilitação de requisições cross-origin
- **Pillow**: Manipulação de imagens
- **NumPy**: Operações matemáticas e arrays

## Funcionalidades Principais

### 🔍 Reconhecimento Facial
- Detecção automática de faces em imagens
- Extração de características faciais únicas
- Comparação com banco de dados de usuários cadastrados
- Sistema de confiança para validação de reconhecimento

### 👥 Gerenciamento de Usuários
- Cadastro de usuários com foto de perfil
- Armazenamento seguro de características faciais
- Busca e listagem de usuários
- Atualização de perfis e imagens

### 📱 Sistema de Notificações
- Notificações automáticas quando usuário é reconhecido
- API para consulta de notificações pelo app mobile
- Controle de status de leitura das notificações
- Histórico completo de notificações

### 📊 Log de Detecções
- Registro detalhado de todas as tentativas de reconhecimento
- Armazenamento de imagens processadas
- Métricas de confiança e status das detecções
- Rastreamento temporal das atividades

## Estrutura do Projeto

```
facial_recognition_backend/
├── src/
│   ├── models/
│   │   └── user.py                 # Modelos do banco de dados
│   ├── routes/
│   │   ├── user.py                 # Rotas para gerenciamento de usuários
│   │   ├── facial_recognition.py   # Rotas para reconhecimento facial
│   │   └── notifications.py        # Rotas para notificações
│   ├── static/                     # Arquivos estáticos (frontend)
│   ├── database/
│   │   └── app.db                  # Banco de dados SQLite
│   └── main.py                     # Arquivo principal da aplicação
├── uploads/                        # Diretório para imagens enviadas
├── venv/                          # Ambiente virtual Python
├── requirements.txt               # Dependências do projeto
├── test_system.py                # Script de testes
└── README.md                     # Esta documentação
```

## Instalação e Configuração

### Pré-requisitos

- Python 3.11 ou superior
- pip (gerenciador de pacotes Python)
- Git (para clonagem do repositório)
- Sistema operacional Linux/Ubuntu (recomendado)

### Dependências do Sistema

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install -y cmake build-essential libopenblas-dev liblapack-dev libx11-dev libgtk-3-dev

# Outras distribuições podem variar
```

### Instalação

1. **Clone o repositório** (se aplicável):
```bash
git clone <url-do-repositorio>
cd facial_recognition_backend
```

2. **Crie e ative o ambiente virtual**:
```bash
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate     # Windows
```

3. **Instale as dependências**:
```bash
pip install -r requirements.txt
```

4. **Execute o servidor**:
```bash
python src/main.py
```

O servidor estará disponível em `http://localhost:5001`

## APIs Disponíveis

### Status da API

**GET** `/api/status`

Verifica se a API está funcionando e retorna informações básicas.

**Resposta:**
```json
{
  "status": "online",
  "message": "Sistema de Reconhecimento Facial - API funcionando",
  "version": "1.0.0",
  "endpoints": {
    "users": "/api/users",
    "facial_recognition": "/api/images",
    "notifications": "/api/notifications",
    "detections": "/api/detections"
  }
}
```

### Gerenciamento de Usuários

#### Listar Usuários
**GET** `/api/users`

Retorna lista de todos os usuários cadastrados.

#### Criar Usuário
**POST** `/api/users`

Cria um novo usuário. Suporta dois formatos:

**Formato JSON** (sem imagem):
```json
{
  "username": "joao_silva",
  "email": "joao@example.com",
  "phone": "+55 11 99999-9999"
}
```

**Formato Multipart** (com imagem):
```
Content-Type: multipart/form-data

username: joao_silva
email: joao@example.com
phone: +55 11 99999-9999
profile_image: [arquivo de imagem]
```

#### Buscar Usuário
**GET** `/api/users/{user_id}`

Retorna dados de um usuário específico.

#### Atualizar Usuário
**PUT** `/api/users/{user_id}`

Atualiza dados de um usuário. Suporta os mesmos formatos do POST.

#### Deletar Usuário
**DELETE** `/api/users/{user_id}`

Remove um usuário do sistema.

#### Upload de Imagem de Perfil
**POST** `/api/users/{user_id}/profile-image`

Faz upload de uma nova imagem de perfil para um usuário existente.

#### Buscar Usuários
**GET** `/api/users/search?q={termo}`

Busca usuários por nome ou email.

### Reconhecimento Facial

#### Processar Imagem (ESP32)
**POST** `/api/images`

Endpoint principal para o ESP32 enviar imagens para reconhecimento.

**Formato:**
```
Content-Type: multipart/form-data

image: [arquivo de imagem]
```

**Possíveis Respostas:**

1. **Usuário Reconhecido:**
```json
{
  "status": "recognized",
  "user": {
    "id": 1,
    "username": "joao_silva",
    "email": "joao@example.com"
  },
  "confidence": 0.85,
  "detection_id": 123,
  "message": "Usuário joao_silva reconhecido com 85% de confiança"
}
```

2. **Face Detectada mas Não Reconhecida:**
```json
{
  "status": "unknown",
  "message": "Face detectada mas usuário não reconhecido",
  "detection_id": 124
}
```

3. **Nenhuma Face Detectada:**
```json
{
  "status": "no_face",
  "message": "Nenhuma face detectada na imagem",
  "detection_id": 125
}
```

#### Processar Imagem Base64
**POST** `/api/images/base64`

Alternativa para envio de imagens em formato base64.

**Formato:**
```json
{
  "image": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQ..."
}
```

### Detecções

#### Listar Detecções
**GET** `/api/detections`

Lista todas as detecções realizadas pelo sistema.

**Parâmetros opcionais:**
- `page`: Número da página (padrão: 1)
- `per_page`: Itens por página (padrão: 10)

#### Buscar Detecção
**GET** `/api/detections/{detection_id}`

Retorna dados de uma detecção específica.

### Notificações

#### Listar Notificações
**GET** `/api/notifications`

Lista todas as notificações do sistema.

**Parâmetros opcionais:**
- `page`: Número da página
- `per_page`: Itens por página
- `user_id`: Filtrar por usuário
- `unread_only`: Apenas não lidas (true/false)

#### Buscar Notificação
**GET** `/api/notifications/{notification_id}`

Retorna dados de uma notificação específica.

#### Marcar como Lida
**PUT** `/api/notifications/{notification_id}/read`

Marca uma notificação como lida.

#### Marcar Todas como Lidas
**PUT** `/api/notifications/mark-all-read`

Marca todas as notificações como lidas.

**Parâmetros opcionais:**
- `user_id`: Marcar apenas de um usuário específico

#### Notificações de Usuário
**GET** `/api/users/{user_id}/notifications/latest`

Busca as últimas notificações de um usuário específico.

**Parâmetros opcionais:**
- `limit`: Número máximo de notificações (padrão: 5)

## Integração com ESP32

### Código de Exemplo para ESP32

```cpp
#include <WiFi.h>
#include <HTTPClient.h>
#include "esp_camera.h"

const char* ssid = "SUA_REDE_WIFI";
const char* password = "SUA_SENHA_WIFI";
const char* serverURL = "http://SEU_SERVIDOR:5001/api/images";

void setup() {
  Serial.begin(115200);
  
  // Configuração da câmera
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  // ... outras configurações da câmera
  
  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Erro na inicialização da câmera: 0x%x", err);
    return;
  }
  
  // Conecta ao WiFi
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(1000);
    Serial.println("Conectando ao WiFi...");
  }
  Serial.println("WiFi conectado!");
}

void loop() {
  // Captura imagem
  camera_fb_t * fb = esp_camera_fb_get();
  if (!fb) {
    Serial.println("Erro ao capturar imagem");
    return;
  }
  
  // Envia para o servidor
  if (WiFi.status() == WL_CONNECTED) {
    HTTPClient http;
    http.begin(serverURL);
    
    String boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW";
    String contentType = "multipart/form-data; boundary=" + boundary;
    http.addHeader("Content-Type", contentType);
    
    String body = "--" + boundary + "\r\n";
    body += "Content-Disposition: form-data; name=\"image\"; filename=\"esp32.jpg\"\r\n";
    body += "Content-Type: image/jpeg\r\n\r\n";
    
    // Adiciona dados da imagem
    String bodyEnd = "\r\n--" + boundary + "--\r\n";
    
    int httpResponseCode = http.POST(body + String((char*)fb->buf, fb->len) + bodyEnd);
    
    if (httpResponseCode > 0) {
      String response = http.getString();
      Serial.println("Resposta do servidor:");
      Serial.println(response);
    } else {
      Serial.printf("Erro na requisição: %d\n", httpResponseCode);
    }
    
    http.end();
  }
  
  esp_camera_fb_return(fb);
  delay(5000); // Aguarda 5 segundos antes da próxima captura
}
```

### Configurações Importantes para ESP32

1. **Tamanho da Imagem**: Configure a câmera para capturar imagens em resolução adequada (recomendado: 640x480)
2. **Qualidade JPEG**: Use qualidade média para equilibrar tamanho e qualidade
3. **Intervalo de Captura**: Configure intervalo adequado entre capturas (recomendado: 3-10 segundos)
4. **Tratamento de Erros**: Implemente retry em caso de falha na conexão

## Integração com App Mobile

### Exemplo de Código Android (Java/Kotlin)

```kotlin
class FacialRecognitionAPI {
    private val baseUrl = "http://SEU_SERVIDOR:5001/api"
    
    // Buscar notificações
    fun getNotifications(userId: Int, callback: (List<Notification>) -> Unit) {
        val url = "$baseUrl/users/$userId/notifications/latest?limit=10"
        
        val request = Request.Builder()
            .url(url)
            .get()
            .build()
        
        client.newCall(request).enqueue(object : Callback {
            override fun onResponse(call: Call, response: Response) {
                if (response.isSuccessful) {
                    val json = response.body?.string()
                    val notifications = parseNotifications(json)
                    callback(notifications)
                }
            }
            
            override fun onFailure(call: Call, e: IOException) {
                // Tratar erro
            }
        })
    }
    
    // Marcar notificação como lida
    fun markAsRead(notificationId: Int) {
        val url = "$baseUrl/notifications/$notificationId/read"
        
        val request = Request.Builder()
            .url(url)
            .put(RequestBody.create(null, ""))
            .build()
        
        client.newCall(request).enqueue(object : Callback {
            override fun onResponse(call: Call, response: Response) {
                // Notificação marcada como lida
            }
            
            override fun onFailure(call: Call, e: IOException) {
                // Tratar erro
            }
        })
    }
}
```

### Exemplo de Código iOS (Swift)

```swift
class FacialRecognitionAPI {
    let baseURL = "http://SEU_SERVIDOR:5001/api"
    
    func getNotifications(userId: Int, completion: @escaping ([Notification]) -> Void) {
        let url = URL(string: "\(baseURL)/users/\(userId)/notifications/latest?limit=10")!
        
        URLSession.shared.dataTask(with: url) { data, response, error in
            guard let data = data else { return }
            
            do {
                let notifications = try JSONDecoder().decode([Notification].self, from: data)
                DispatchQueue.main.async {
                    completion(notifications)
                }
            } catch {
                print("Erro ao decodificar notificações: \(error)")
            }
        }.resume()
    }
    
    func markAsRead(notificationId: Int) {
        let url = URL(string: "\(baseURL)/notifications/\(notificationId)/read")!
        var request = URLRequest(url: url)
        request.httpMethod = "PUT"
        
        URLSession.shared.dataTask(with: request) { _, _, _ in
            // Notificação marcada como lida
        }.resume()
    }
}
```

## Banco de Dados

### Estrutura das Tabelas

#### Tabela `user`
| Campo | Tipo | Descrição |
|-------|------|-----------|
| id | INTEGER | Chave primária |
| username | VARCHAR(80) | Nome de usuário único |
| email | VARCHAR(120) | Email único |
| phone | VARCHAR(20) | Telefone (opcional) |
| face_encoding | TEXT | Características faciais em JSON |
| profile_image_path | VARCHAR(255) | Caminho da imagem de perfil |
| is_active | BOOLEAN | Status ativo/inativo |
| created_at | DATETIME | Data de criação |
| last_seen | DATETIME | Última detecção |

#### Tabela `detection_log`
| Campo | Tipo | Descrição |
|-------|------|-----------|
| id | INTEGER | Chave primária |
| user_id | INTEGER | ID do usuário (FK) |
| image_path | VARCHAR(255) | Caminho da imagem processada |
| confidence | FLOAT | Nível de confiança |
| detected_at | DATETIME | Data/hora da detecção |
| status | VARCHAR(20) | Status: recognized/unknown/no_face/error |

#### Tabela `notification`
| Campo | Tipo | Descrição |
|-------|------|-----------|
| id | INTEGER | Chave primária |
| user_id | INTEGER | ID do usuário (FK) |
| message | TEXT | Mensagem da notificação |
| notification_type | VARCHAR(50) | Tipo da notificação |
| is_read | BOOLEAN | Status de leitura |
| created_at | DATETIME | Data de criação |

## Deployment

### Desenvolvimento Local

Para executar em ambiente de desenvolvimento:

```bash
cd facial_recognition_backend
source venv/bin/activate
python src/main.py
```

### Produção com Gunicorn

1. **Instale o Gunicorn**:
```bash
pip install gunicorn
```

2. **Execute com Gunicorn**:
```bash
gunicorn -w 4 -b 0.0.0.0:5001 src.main:app
```

### Docker (Opcional)

Crie um `Dockerfile`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Instala dependências do sistema
RUN apt-get update && apt-get install -y \
    cmake \
    build-essential \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk-3-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 5001

CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5001", "src.main:app"]
```

### Nginx (Proxy Reverso)

Configuração do Nginx:

```nginx
server {
    listen 80;
    server_name seu-dominio.com;

    location / {
        proxy_pass http://127.0.0.1:5001;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

## Segurança

### Considerações de Segurança

1. **HTTPS**: Use sempre HTTPS em produção
2. **Autenticação**: Implemente autenticação para APIs sensíveis
3. **Rate Limiting**: Configure limites de requisições
4. **Validação de Entrada**: Valide todos os dados de entrada
5. **Logs de Segurança**: Monitore tentativas de acesso suspeitas

### Configurações Recomendadas

```python
# Adicione ao main.py para produção
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

limiter = Limiter(
    app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"]
)

@app.route('/api/images', methods=['POST'])
@limiter.limit("10 per minute")
def receive_image():
    # Código da função
```

## Monitoramento e Logs

### Configuração de Logs

```python
import logging
from logging.handlers import RotatingFileHandler

if not app.debug:
    file_handler = RotatingFileHandler('logs/facial_recognition.log', maxBytes=10240, backupCount=10)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
    ))
    file_handler.setLevel(logging.INFO)
    app.logger.addHandler(file_handler)
    app.logger.setLevel(logging.INFO)
    app.logger.info('Sistema de Reconhecimento Facial iniciado')
```

### Métricas Importantes

- Taxa de reconhecimento bem-sucedido
- Tempo médio de processamento de imagens
- Número de detecções por hora/dia
- Taxa de falsos positivos/negativos

## Troubleshooting

### Problemas Comuns

#### 1. Erro na instalação do OpenCV
```bash
# Solução: Instalar dependências do sistema
sudo apt-get install python3-opencv
```

#### 2. Porta já em uso
```bash
# Verificar processos na porta
sudo lsof -i :5001
# Matar processo se necessário
sudo kill -9 <PID>
```

#### 3. Erro de permissão em uploads
```bash
# Criar diretório e dar permissões
mkdir -p uploads/profiles
chmod 755 uploads/profiles
```

#### 4. Banco de dados corrompido
```bash
# Backup e recriação
cp src/database/app.db src/database/app.db.backup
rm src/database/app.db
# Reiniciar aplicação para recriar
```

### Logs de Debug

Para ativar logs detalhados:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Contribuição

### Como Contribuir

1. Fork o projeto
2. Crie uma branch para sua feature (`git checkout -b feature/AmazingFeature`)
3. Commit suas mudanças (`git commit -m 'Add some AmazingFeature'`)
4. Push para a branch (`git push origin feature/AmazingFeature`)
5. Abra um Pull Request

### Padrões de Código

- Use PEP 8 para formatação Python
- Documente todas as funções públicas
- Escreva testes para novas funcionalidades
- Mantenha cobertura de testes acima de 80%

## Licença

Este projeto está licenciado sob a Licença MIT - veja o arquivo [LICENSE](LICENSE) para detalhes.

## Contato

Para dúvidas ou suporte:

- **Desenvolvedor**: Manus AI
- **Email**: suporte@manus.ai
- **Documentação**: [Link para documentação online]

## Changelog

### v1.0.0 (2025-01-23)
- Implementação inicial do sistema
- APIs REST completas
- Sistema de reconhecimento facial com OpenCV
- Integração com ESP32 e mobile
- Documentação completa

---

**Nota**: Este sistema foi desenvolvido para fins educacionais como parte de um TCC. Para uso em produção, considere implementar medidas adicionais de segurança e otimização de performance.

