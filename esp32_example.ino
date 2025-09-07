/*
 * Sistema de Reconhecimento Facial - ESP32 Camera
 * 
 * Este código captura imagens periodicamente e envia para o servidor
 * de reconhecimento facial para processamento.
 * 
 * Hardware necessário:
 * - ESP32-CAM ou ESP32 com módulo de câmera
 * - Conexão WiFi
 * 
 * Configuração:
 * 1. Ajuste as credenciais WiFi
 * 2. Configure o endereço do servidor
 * 3. Ajuste o intervalo de captura conforme necessário
 */

#include <WiFi.h>
#include <HTTPClient.h>
#include "esp_camera.h"
#include "Arduino.h"
#include "FS.h"
#include "SD_MMC.h"
#include "soc/soc.h"
#include "soc/rtc_cntl_reg.h"
#include "driver/rtc_io.h"
#include <ArduinoJson.h>

// ===========================================
// CONFIGURAÇÕES - AJUSTE CONFORME NECESSÁRIO
// ===========================================

// Credenciais WiFi
const char* ssid = "SUA_REDE_WIFI";
const char* password = "SUA_SENHA_WIFI";

// Configurações do servidor
const char* serverURL = "http://192.168.1.100:5001/api/images";  // Ajuste o IP
const int serverPort = 5001;

// Configurações de captura
const int captureInterval = 5000;  // Intervalo entre capturas (ms)
const int maxRetries = 3;          // Máximo de tentativas de envio
const int wifiTimeout = 10000;     // Timeout para conexão WiFi (ms)

// ===========================================
// CONFIGURAÇÃO DA CÂMERA ESP32-CAM
// ===========================================

// Pinos para ESP32-CAM AI-Thinker
#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM      0
#define SIOD_GPIO_NUM     26
#define SIOC_GPIO_NUM     27
#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       21
#define Y4_GPIO_NUM       19
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM        5
#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22

// LED Flash (opcional)
#define FLASH_LED_PIN 4

// ===========================================
// VARIÁVEIS GLOBAIS
// ===========================================

bool cameraInitialized = false;
unsigned long lastCaptureTime = 0;
int consecutiveErrors = 0;
const int maxConsecutiveErrors = 5;

// ===========================================
// FUNÇÕES DE CONFIGURAÇÃO
// ===========================================

void setup() {
  Serial.begin(115200);
  Serial.println("=== Sistema de Reconhecimento Facial ESP32 ===");
  
  // Desabilita brownout detector
  WRITE_PERI_REG(RTC_CNTL_BROWN_OUT_REG, 0);
  
  // Configura LED flash (opcional)
  pinMode(FLASH_LED_PIN, OUTPUT);
  digitalWrite(FLASH_LED_PIN, LOW);
  
  // Inicializa câmera
  if (initCamera()) {
    Serial.println("✓ Câmera inicializada com sucesso");
    cameraInitialized = true;
  } else {
    Serial.println("✗ Falha na inicialização da câmera");
    return;
  }
  
  // Conecta ao WiFi
  connectToWiFi();
  
  // Teste inicial de conectividade
  testServerConnection();
  
  Serial.println("=== Sistema pronto para capturar imagens ===");
}

bool initCamera() {
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sscb_sda = SIOD_GPIO_NUM;
  config.pin_sscb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;
  
  // Configurações de qualidade baseadas na memória disponível
  if(psramFound()){
    config.frame_size = FRAMESIZE_VGA;    // 640x480
    config.jpeg_quality = 10;             // Qualidade alta (0-63, menor = melhor)
    config.fb_count = 2;
  } else {
    config.frame_size = FRAMESIZE_SVGA;   // 800x600
    config.jpeg_quality = 12;
    config.fb_count = 1;
  }
  
  // Inicializa a câmera
  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Erro na inicialização da câmera: 0x%x\n", err);
    return false;
  }
  
  // Configurações adicionais do sensor
  sensor_t * s = esp_camera_sensor_get();
  if (s != NULL) {
    // Ajustes de imagem
    s->set_brightness(s, 0);     // -2 a 2
    s->set_contrast(s, 0);       // -2 a 2
    s->set_saturation(s, 0);     // -2 a 2
    s->set_special_effect(s, 0); // 0 a 6 (0=Normal)
    s->set_whitebal(s, 1);       // 0 = disable , 1 = enable
    s->set_awb_gain(s, 1);       // 0 = disable , 1 = enable
    s->set_wb_mode(s, 0);        // 0 a 4 - se awb_gain enabled (0 - Auto, 1 - Sunny, 2 - Cloudy, 3 - Office, 4 - Home)
    s->set_exposure_ctrl(s, 1);  // 0 = disable , 1 = enable
    s->set_aec2(s, 0);           // 0 = disable , 1 = enable
    s->set_ae_level(s, 0);       // -2 a 2
    s->set_aec_value(s, 300);    // 0 a 1200
    s->set_gain_ctrl(s, 1);      // 0 = disable , 1 = enable
    s->set_agc_gain(s, 0);       // 0 a 30
    s->set_gainceiling(s, (gainceiling_t)0);  // 0 a 6
    s->set_bpc(s, 0);            // 0 = disable , 1 = enable
    s->set_wpc(s, 1);            // 0 = disable , 1 = enable
    s->set_raw_gma(s, 1);        // 0 = disable , 1 = enable
    s->set_lenc(s, 1);           // 0 = disable , 1 = enable
    s->set_hmirror(s, 0);        // 0 = disable , 1 = enable
    s->set_vflip(s, 0);          // 0 = disable , 1 = enable
    s->set_dcw(s, 1);            // 0 = disable , 1 = enable
    s->set_colorbar(s, 0);       // 0 = disable , 1 = enable
  }
  
  return true;
}

void connectToWiFi() {
  Serial.println("Conectando ao WiFi...");
  WiFi.begin(ssid, password);
  
  unsigned long startTime = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - startTime < wifiTimeout) {
    delay(500);
    Serial.print(".");
  }
  
  if (WiFi.status() == WL_CONNECTED) {
    Serial.println();
    Serial.println("✓ WiFi conectado!");
    Serial.print("IP: ");
    Serial.println(WiFi.localIP());
    Serial.print("Força do sinal: ");
    Serial.print(WiFi.RSSI());
    Serial.println(" dBm");
  } else {
    Serial.println();
    Serial.println("✗ Falha na conexão WiFi");
  }
}

void testServerConnection() {
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("WiFi não conectado, pulando teste do servidor");
    return;
  }
  
  Serial.println("Testando conexão com o servidor...");
  
  HTTPClient http;
  http.begin(String(serverURL).substring(0, String(serverURL).lastIndexOf('/')) + "/status");
  http.setTimeout(5000);
  
  int httpResponseCode = http.GET();
  
  if (httpResponseCode > 0) {
    String response = http.getString();
    Serial.println("✓ Servidor respondeu:");
    Serial.println(response);
  } else {
    Serial.printf("✗ Erro na conexão com servidor: %d\n", httpResponseCode);
  }
  
  http.end();
}

// ===========================================
// LOOP PRINCIPAL
// ===========================================

void loop() {
  // Verifica se é hora de capturar uma nova imagem
  if (millis() - lastCaptureTime >= captureInterval) {
    if (cameraInitialized && WiFi.status() == WL_CONNECTED) {
      captureAndSendImage();
    } else {
      handleConnectionIssues();
    }
    lastCaptureTime = millis();
  }
  
  // Pequeno delay para não sobrecarregar o processador
  delay(100);
}

// ===========================================
// FUNÇÕES DE CAPTURA E ENVIO
// ===========================================

void captureAndSendImage() {
  Serial.println("📸 Capturando imagem...");
  
  // Liga LED flash (opcional)
  digitalWrite(FLASH_LED_PIN, HIGH);
  delay(100);  // Pequeno delay para estabilizar a iluminação
  
  // Captura a imagem
  camera_fb_t * fb = esp_camera_fb_get();
  
  // Desliga LED flash
  digitalWrite(FLASH_LED_PIN, LOW);
  
  if (!fb) {
    Serial.println("✗ Falha na captura da imagem");
    consecutiveErrors++;
    return;
  }
  
  Serial.printf("📊 Imagem capturada: %d bytes\n", fb->len);
  
  // Envia para o servidor
  bool success = sendImageToServer(fb);
  
  if (success) {
    consecutiveErrors = 0;
    Serial.println("✓ Imagem enviada com sucesso");
  } else {
    consecutiveErrors++;
    Serial.println("✗ Falha no envio da imagem");
  }
  
  // Libera o buffer da câmera
  esp_camera_fb_return(fb);
  
  // Verifica se há muitos erros consecutivos
  if (consecutiveErrors >= maxConsecutiveErrors) {
    Serial.println("⚠️ Muitos erros consecutivos, reiniciando...");
    ESP.restart();
  }
}

bool sendImageToServer(camera_fb_t * fb) {
  HTTPClient http;
  http.begin(serverURL);
  http.setTimeout(15000);  // Timeout de 15 segundos
  
  // Cria boundary para multipart
  String boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW";
  String contentType = "multipart/form-data; boundary=" + boundary;
  http.addHeader("Content-Type", contentType);
  
  // Constrói o corpo da requisição
  String bodyStart = "--" + boundary + "\r\n";
  bodyStart += "Content-Disposition: form-data; name=\"image\"; filename=\"esp32_capture.jpg\"\r\n";
  bodyStart += "Content-Type: image/jpeg\r\n\r\n";
  
  String bodyEnd = "\r\n--" + boundary + "--\r\n";
  
  // Calcula tamanho total
  int totalLength = bodyStart.length() + fb->len + bodyEnd.length();
  
  // Cria buffer para o corpo completo
  uint8_t* postData = (uint8_t*)malloc(totalLength);
  if (!postData) {
    Serial.println("✗ Erro ao alocar memória para envio");
    http.end();
    return false;
  }
  
  // Monta o corpo da requisição
  memcpy(postData, bodyStart.c_str(), bodyStart.length());
  memcpy(postData + bodyStart.length(), fb->buf, fb->len);
  memcpy(postData + bodyStart.length() + fb->len, bodyEnd.c_str(), bodyEnd.length());
  
  // Envia a requisição
  int httpResponseCode = http.POST(postData, totalLength);
  
  // Libera memória
  free(postData);
  
  bool success = false;
  
  if (httpResponseCode > 0) {
    String response = http.getString();
    Serial.printf("📡 Resposta do servidor (%d):\n", httpResponseCode);
    
    // Parse da resposta JSON
    DynamicJsonDocument doc(1024);
    DeserializationError error = deserializeJson(doc, response);
    
    if (!error) {
      String status = doc["status"];
      String message = doc["message"];
      
      Serial.println("Status: " + status);
      Serial.println("Mensagem: " + message);
      
      if (status == "recognized") {
        String username = doc["user"]["username"];
        float confidence = doc["confidence"];
        Serial.printf("🎉 Usuário reconhecido: %s (%.2f%% confiança)\n", 
                     username.c_str(), confidence * 100);
      } else if (status == "unknown") {
        Serial.println("👤 Face detectada mas usuário não reconhecido");
      } else if (status == "no_face") {
        Serial.println("😐 Nenhuma face detectada na imagem");
      }
      
      success = true;
    } else {
      Serial.println("Resposta: " + response);
      success = (httpResponseCode == 200);
    }
  } else {
    Serial.printf("✗ Erro HTTP: %d\n", httpResponseCode);
    
    // Códigos de erro específicos
    switch(httpResponseCode) {
      case -1:
        Serial.println("Erro de conexão");
        break;
      case -11:
        Serial.println("Timeout na requisição");
        break;
      default:
        Serial.println("Erro desconhecido");
    }
  }
  
  http.end();
  return success;
}

// ===========================================
// FUNÇÕES DE TRATAMENTO DE ERROS
// ===========================================

void handleConnectionIssues() {
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("⚠️ WiFi desconectado, tentando reconectar...");
    connectToWiFi();
  }
  
  if (!cameraInitialized) {
    Serial.println("⚠️ Câmera não inicializada, tentando reinicializar...");
    cameraInitialized = initCamera();
  }
}

// ===========================================
// FUNÇÕES AUXILIARES
// ===========================================

void printSystemInfo() {
  Serial.println("=== Informações do Sistema ===");
  Serial.printf("Chip ID: %08X\n", (uint32_t)ESP.getEfuseMac());
  Serial.printf("CPU Freq: %d MHz\n", ESP.getCpuFreqMHz());
  Serial.printf("Free Heap: %d bytes\n", ESP.getFreeHeap());
  Serial.printf("PSRAM: %s\n", psramFound() ? "Disponível" : "Não disponível");
  Serial.printf("Flash Size: %d bytes\n", ESP.getFlashChipSize());
  Serial.println("============================");
}

// Função para debug - chama no setup() se necessário
void debugMode() {
  Serial.println("=== MODO DEBUG ATIVADO ===");
  printSystemInfo();
  
  // Teste de captura sem envio
  camera_fb_t * fb = esp_camera_fb_get();
  if (fb) {
    Serial.printf("Teste de captura: %d bytes\n", fb->len);
    esp_camera_fb_return(fb);
  } else {
    Serial.println("Falha no teste de captura");
  }
}

/*
 * NOTAS DE CONFIGURAÇÃO:
 * 
 * 1. Ajuste as credenciais WiFi nas constantes no início do código
 * 2. Configure o IP do servidor na variável serverURL
 * 3. Ajuste o intervalo de captura conforme necessário
 * 4. Para debug, descomente a chamada debugMode() no setup()
 * 5. O LED flash é opcional - remova se não necessário
 * 
 * TROUBLESHOOTING:
 * 
 * - Se a câmera não inicializar, verifique as conexões dos pinos
 * - Se houver erro de memória, reduza a qualidade JPEG ou frame size
 * - Para problemas de WiFi, verifique força do sinal e credenciais
 * - Se o servidor não responder, verifique IP e porta
 * 
 * OTIMIZAÇÕES POSSÍVEIS:
 * 
 * - Implementar deep sleep entre capturas para economizar bateria
 * - Adicionar detecção de movimento para capturar apenas quando necessário
 * - Implementar buffer local para casos de falha de conectividade
 * - Adicionar configuração via web interface
 */

