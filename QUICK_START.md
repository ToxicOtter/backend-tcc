# 🚀 Guia de Início Rápido

Este guia te ajudará a colocar o Sistema de Reconhecimento Facial funcionando em poucos minutos.

## ⚡ Instalação Rápida (Ubuntu/Debian)

### 1. Preparação do Ambiente

```bash
# Clone ou baixe o projeto
cd /home/seu-usuario/
# (assumindo que você já tem os arquivos do projeto)

# Torne o script de deploy executável
chmod +x deploy.sh

# Execute o deployment automático
./deploy.sh
```

O script irá:
- ✅ Instalar todas as dependências
- ✅ Configurar o ambiente Python
- ✅ Configurar Nginx como proxy reverso
- ✅ Configurar Supervisor para gerenciar o processo
- ✅ Criar scripts de controle
- ✅ Testar a instalação

### 2. Verificação da Instalação

Após o deployment, verifique se tudo está funcionando:

```bash
# Verificar status
./status.sh

# Testar API
curl http://localhost/api/status
```

## 🎯 Primeiros Passos

### 1. Cadastrar Primeiro Usuário

```bash
# Usando curl
curl -X POST http://localhost/api/users \
  -H "Content-Type: application/json" \
  -d '{
    "username": "admin",
    "email": "admin@example.com",
    "phone": "+55 11 99999-9999"
  }'
```

### 2. Cadastrar Usuário com Foto

```bash
# Usando curl com imagem
curl -X POST http://localhost/api/users \
  -F "username=joao" \
  -F "email=joao@example.com" \
  -F "phone=+55 11 88888-8888" \
  -F "profile_image=@/caminho/para/foto.jpg"
```

### 3. Testar Reconhecimento Facial

```bash
# Enviar imagem para reconhecimento
curl -X POST http://localhost/api/images \
  -F "image=@/caminho/para/teste.jpg"
```

## 📱 Configuração do ESP32

### 1. Código Base

Use o arquivo `esp32_example.ino` fornecido no projeto. Principais configurações:

```cpp
// Credenciais WiFi
const char* ssid = "SUA_REDE_WIFI";
const char* password = "SUA_SENHA_WIFI";

// IP do servidor (substitua pelo IP real)
const char* serverURL = "http://192.168.1.100/api/images";
```

### 2. Bibliotecas Necessárias

No Arduino IDE, instale:
- ESP32 Board Package
- ArduinoJson
- HTTPClient (já incluída no ESP32)

### 3. Configuração da Câmera

O código está configurado para ESP32-CAM AI-Thinker. Para outros modelos, ajuste os pinos no código.

## 📲 Integração com App Mobile

### Endpoints Principais para Mobile

```javascript
// Buscar notificações de um usuário
GET /api/users/{user_id}/notifications/latest?limit=10

// Marcar notificação como lida
PUT /api/notifications/{notification_id}/read

// Listar usuários
GET /api/users

// Buscar usuário específico
GET /api/users/{user_id}
```

### Exemplo de Integração (JavaScript)

```javascript
const API_BASE = 'http://seu-servidor.com/api';

// Função para buscar notificações
async function getNotifications(userId) {
  const response = await fetch(`${API_BASE}/users/${userId}/notifications/latest`);
  const data = await response.json();
  return data.notifications;
}

// Função para marcar como lida
async function markAsRead(notificationId) {
  await fetch(`${API_BASE}/notifications/${notificationId}/read`, {
    method: 'PUT'
  });
}
```

## 🛠️ Scripts de Controle

Após a instalação, você terá os seguintes scripts:

```bash
# Iniciar sistema
./start.sh

# Parar sistema
./stop.sh

# Reiniciar sistema
./restart.sh

# Ver status
./status.sh

# Ver logs em tempo real
./logs.sh
```

## 🔧 Configurações Avançadas

### Alterar Porta do Servidor

Edite `src/main.py`:

```python
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)  # Mude para 8080
```

### Ajustar Threshold de Reconhecimento

Edite `config.py`:

```python
# Valores menores = mais restritivo
# Valores maiores = mais permissivo
FACE_RECOGNITION_THRESHOLD = 0.7  # Padrão: 0.6
```

### Configurar HTTPS

1. Obtenha certificados SSL
2. Configure Nginx:

```nginx
server {
    listen 443 ssl;
    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;
    
    location / {
        proxy_pass http://127.0.0.1:5001;
        # ... outras configurações
    }
}
```

## 📊 Monitoramento

### Logs do Sistema

```bash
# Logs da aplicação
sudo tail -f /var/log/facial-recognition.log

# Logs do Nginx
sudo tail -f /var/log/nginx/access.log
sudo tail -f /var/log/nginx/error.log

# Status do Supervisor
sudo supervisorctl status
```

### Métricas Importantes

- Taxa de reconhecimento: `GET /api/detections`
- Usuários cadastrados: `GET /api/users`
- Notificações pendentes: `GET /api/notifications?unread_only=true`

## 🚨 Troubleshooting

### Problema: Servidor não inicia

```bash
# Verificar logs
./logs.sh

# Verificar porta em uso
sudo netstat -tlnp | grep :5001

# Reiniciar serviços
sudo supervisorctl restart facial-recognition
```

### Problema: ESP32 não conecta

1. Verificar credenciais WiFi
2. Verificar IP do servidor
3. Verificar firewall:

```bash
# Liberar porta no firewall
sudo ufw allow 80/tcp
sudo ufw allow 5001/tcp
```

### Problema: Reconhecimento facial não funciona

1. Verificar qualidade da imagem
2. Ajustar threshold no `config.py`
3. Verificar se usuário tem foto cadastrada

### Problema: Uploads falham

```bash
# Verificar permissões
ls -la uploads/
chmod 755 uploads/
chmod 755 uploads/profiles/

# Verificar espaço em disco
df -h
```

## 🔒 Segurança em Produção

### Configurações Essenciais

1. **Alterar SECRET_KEY** no `config.py`
2. **Configurar HTTPS** com certificados válidos
3. **Configurar firewall** adequadamente
4. **Usar banco PostgreSQL** em vez de SQLite
5. **Configurar backups** regulares

### Exemplo de Configuração Segura

```bash
# Variáveis de ambiente para produção
export FLASK_ENV=production
export SECRET_KEY="sua-chave-super-secreta-aqui"
export DATABASE_URL="postgresql://user:pass@localhost/facial_db"
export CORS_ORIGINS="https://seu-app.com,https://outro-app.com"
```

## 📈 Otimizações de Performance

### Para Alto Volume

1. **Use Redis** para cache:
```bash
sudo apt install redis-server
pip install redis flask-caching
```

2. **Configure mais workers** no Gunicorn:
```bash
# No arquivo de configuração do Supervisor
command=.../gunicorn -w 8 -b 127.0.0.1:5001 src.main:app
```

3. **Otimize banco de dados**:
```sql
-- Para PostgreSQL
CREATE INDEX idx_user_face_encoding ON users(face_encoding);
CREATE INDEX idx_detection_date ON detection_log(detected_at);
```

## 📞 Suporte

### Recursos Úteis

- 📖 **Documentação Completa**: `README.md`
- 🔧 **Configurações**: `config.py`
- 🧪 **Testes**: `test_system.py`
- 📱 **Código ESP32**: `esp32_example.ino`

### Comandos Úteis

```bash
# Backup do banco de dados
cp src/database/app.db backup_$(date +%Y%m%d).db

# Limpar logs antigos
sudo find /var/log -name "*.log" -mtime +30 -delete

# Verificar uso de recursos
htop
df -h
free -h
```

---

## ✅ Checklist de Verificação

Antes de colocar em produção, verifique:

- [ ] Sistema inicia corretamente (`./status.sh`)
- [ ] API responde (`curl http://localhost/api/status`)
- [ ] ESP32 consegue enviar imagens
- [ ] Reconhecimento facial funciona
- [ ] Notificações são criadas
- [ ] Logs estão sendo gerados
- [ ] Backup está configurado
- [ ] HTTPS está configurado (produção)
- [ ] Firewall está configurado
- [ ] Monitoramento está ativo

🎉 **Parabéns! Seu sistema está pronto para uso!**

Para dúvidas ou problemas, consulte a documentação completa no `README.md` ou os logs do sistema.

