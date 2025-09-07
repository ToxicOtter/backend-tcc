#!/bin/bash

# Script de Deployment - Sistema de Reconhecimento Facial
# Este script automatiza a instalação e configuração do sistema

set -e  # Para em caso de erro

echo "🚀 Iniciando deployment do Sistema de Reconhecimento Facial"
echo "============================================================"

# Cores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Função para log colorido
log_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

log_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

log_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Verifica se está rodando como root
check_root() {
    if [[ $EUID -eq 0 ]]; then
        log_error "Este script não deve ser executado como root"
        exit 1
    fi
}

# Verifica sistema operacional
check_os() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        log_info "Sistema operacional: Linux"
        if command -v apt-get &> /dev/null; then
            PACKAGE_MANAGER="apt-get"
        elif command -v yum &> /dev/null; then
            PACKAGE_MANAGER="yum"
        else
            log_error "Gerenciador de pacotes não suportado"
            exit 1
        fi
    else
        log_error "Sistema operacional não suportado: $OSTYPE"
        exit 1
    fi
}

# Instala dependências do sistema
install_system_dependencies() {
    log_info "Instalando dependências do sistema..."
    
    if [[ "$PACKAGE_MANAGER" == "apt-get" ]]; then
        sudo apt-get update
        sudo apt-get install -y \
            python3 \
            python3-pip \
            python3-venv \
            cmake \
            build-essential \
            libopenblas-dev \
            liblapack-dev \
            libx11-dev \
            libgtk-3-dev \
            git \
            curl \
            nginx \
            supervisor
    elif [[ "$PACKAGE_MANAGER" == "yum" ]]; then
        sudo yum update -y
        sudo yum install -y \
            python3 \
            python3-pip \
            cmake \
            gcc \
            gcc-c++ \
            make \
            git \
            curl \
            nginx \
            supervisor
    fi
    
    log_success "Dependências do sistema instaladas"
}

# Configura ambiente Python
setup_python_environment() {
    log_info "Configurando ambiente Python..."
    
    # Cria ambiente virtual se não existir
    if [ ! -d "venv" ]; then
        python3 -m venv venv
        log_success "Ambiente virtual criado"
    fi
    
    # Ativa ambiente virtual
    source venv/bin/activate
    
    # Atualiza pip
    pip install --upgrade pip
    
    # Instala dependências
    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt
        log_success "Dependências Python instaladas"
    else
        log_error "Arquivo requirements.txt não encontrado"
        exit 1
    fi
}

# Configura banco de dados
setup_database() {
    log_info "Configurando banco de dados..."
    
    # Cria diretório do banco se não existir
    mkdir -p src/database
    
    # Cria diretório de uploads
    mkdir -p uploads/profiles
    chmod 755 uploads/profiles
    
    log_success "Estrutura de diretórios criada"
}

# Configura Nginx
setup_nginx() {
    log_info "Configurando Nginx..."
    
    # Cria configuração do Nginx
    sudo tee /etc/nginx/sites-available/facial-recognition > /dev/null <<EOF
server {
    listen 80;
    server_name _;
    
    client_max_body_size 20M;
    
    location / {
        proxy_pass http://127.0.0.1:5001;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }
}
EOF
    
    # Habilita o site
    sudo ln -sf /etc/nginx/sites-available/facial-recognition /etc/nginx/sites-enabled/
    
    # Remove configuração padrão
    sudo rm -f /etc/nginx/sites-enabled/default
    
    # Testa configuração
    sudo nginx -t
    
    # Reinicia Nginx
    sudo systemctl restart nginx
    sudo systemctl enable nginx
    
    log_success "Nginx configurado"
}

# Configura Supervisor
setup_supervisor() {
    log_info "Configurando Supervisor..."
    
    # Obtém caminho absoluto do projeto
    PROJECT_PATH=$(pwd)
    
    # Cria configuração do Supervisor
    sudo tee /etc/supervisor/conf.d/facial-recognition.conf > /dev/null <<EOF
[program:facial-recognition]
command=$PROJECT_PATH/venv/bin/gunicorn -w 4 -b 127.0.0.1:5001 src.main:app
directory=$PROJECT_PATH
user=$USER
autostart=true
autorestart=true
redirect_stderr=true
stdout_logfile=/var/log/facial-recognition.log
environment=PATH="$PROJECT_PATH/venv/bin"
EOF
    
    # Instala Gunicorn se não estiver instalado
    source venv/bin/activate
    pip install gunicorn
    
    # Recarrega configuração do Supervisor
    sudo supervisorctl reread
    sudo supervisorctl update
    
    log_success "Supervisor configurado"
}

# Configura firewall
setup_firewall() {
    log_info "Configurando firewall..."
    
    if command -v ufw &> /dev/null; then
        sudo ufw allow 22/tcp   # SSH
        sudo ufw allow 80/tcp   # HTTP
        sudo ufw allow 443/tcp  # HTTPS
        sudo ufw --force enable
        log_success "Firewall configurado"
    else
        log_warning "UFW não encontrado, configure o firewall manualmente"
    fi
}

# Cria scripts de controle
create_control_scripts() {
    log_info "Criando scripts de controle..."
    
    # Script de start
    cat > start.sh << 'EOF'
#!/bin/bash
echo "🚀 Iniciando Sistema de Reconhecimento Facial..."
sudo supervisorctl start facial-recognition
sudo systemctl start nginx
echo "✅ Sistema iniciado"
echo "🌐 Acesse: http://$(hostname -I | awk '{print $1}')"
EOF
    
    # Script de stop
    cat > stop.sh << 'EOF'
#!/bin/bash
echo "🛑 Parando Sistema de Reconhecimento Facial..."
sudo supervisorctl stop facial-recognition
echo "✅ Sistema parado"
EOF
    
    # Script de restart
    cat > restart.sh << 'EOF'
#!/bin/bash
echo "🔄 Reiniciando Sistema de Reconhecimento Facial..."
sudo supervisorctl restart facial-recognition
sudo systemctl reload nginx
echo "✅ Sistema reiniciado"
EOF
    
    # Script de status
    cat > status.sh << 'EOF'
#!/bin/bash
echo "📊 Status do Sistema de Reconhecimento Facial"
echo "=============================================="
echo "Aplicação:"
sudo supervisorctl status facial-recognition
echo ""
echo "Nginx:"
sudo systemctl status nginx --no-pager -l
echo ""
echo "Logs recentes:"
sudo tail -n 10 /var/log/facial-recognition.log
EOF
    
    # Script de logs
    cat > logs.sh << 'EOF'
#!/bin/bash
echo "📋 Logs do Sistema de Reconhecimento Facial"
echo "==========================================="
sudo tail -f /var/log/facial-recognition.log
EOF
    
    # Torna scripts executáveis
    chmod +x start.sh stop.sh restart.sh status.sh logs.sh
    
    log_success "Scripts de controle criados"
}

# Testa instalação
test_installation() {
    log_info "Testando instalação..."
    
    # Inicia o serviço
    sudo supervisorctl start facial-recognition
    
    # Aguarda alguns segundos
    sleep 5
    
    # Testa endpoint de status
    if curl -f -s http://localhost/api/status > /dev/null; then
        log_success "Sistema funcionando corretamente!"
    else
        log_error "Falha no teste do sistema"
        log_info "Verificando logs..."
        sudo tail -n 20 /var/log/facial-recognition.log
        exit 1
    fi
}

# Mostra informações finais
show_final_info() {
    echo ""
    echo "🎉 Deployment concluído com sucesso!"
    echo "===================================="
    echo ""
    echo "📍 URLs de acesso:"
    echo "   - API: http://$(hostname -I | awk '{print $1}')/api/status"
    echo "   - Documentação: http://$(hostname -I | awk '{print $1}')"
    echo ""
    echo "🛠️ Scripts de controle:"
    echo "   - Iniciar: ./start.sh"
    echo "   - Parar: ./stop.sh"
    echo "   - Reiniciar: ./restart.sh"
    echo "   - Status: ./status.sh"
    echo "   - Logs: ./logs.sh"
    echo ""
    echo "📁 Diretórios importantes:"
    echo "   - Uploads: $(pwd)/uploads/"
    echo "   - Banco de dados: $(pwd)/src/database/"
    echo "   - Logs: /var/log/facial-recognition.log"
    echo ""
    echo "🔧 Configurações:"
    echo "   - Nginx: /etc/nginx/sites-available/facial-recognition"
    echo "   - Supervisor: /etc/supervisor/conf.d/facial-recognition.conf"
    echo ""
    echo "📖 Para mais informações, consulte o README.md"
}

# Função principal
main() {
    check_root
    check_os
    
    log_info "Iniciando deployment..."
    
    install_system_dependencies
    setup_python_environment
    setup_database
    setup_nginx
    setup_supervisor
    setup_firewall
    create_control_scripts
    test_installation
    
    show_final_info
}

# Verifica se o script está sendo executado diretamente
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi

