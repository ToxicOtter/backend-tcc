// ESP32-CAM (AI Thinker) -> envia JPEG a cada 5s -> Flask
// Se o Flask responder "OK", imprimimos e (opcional) piscamos o flash onboard.

#include <WiFi.h>
#include "esp_camera.h"

// ===========================
// Select camera model in board_config.h
// ===========================
#include "board_config.h"

// ------- CONFIG WIFI & SERVIDOR -------
const char* WIFI_SSID      = "LOPES_S";
const char* WIFI_PASS      = "Lp15161628@";
const char* SERVER_HOST    = "10.0.0.113"; // IP do Flask
const uint16_t SERVER_PORT = 5001;           // Porta Flask
const char* SERVER_PATH    = "api/images";      // Rota Flask

// Intervalo entre capturas (ms)
//const uint32_t CAPTURE_INTERVAL_MS = 5000;
const uint32_t CAPTURE_INTERVAL_MS = 1000;

// Opcional: piscar LED do flash quando servidor responder "OK"
#define ENABLE_FLASH_FEEDBACK 1

// LED de flash no AI Thinker (GPIO 4)
#define FLASH_LED_PIN 4

// ------- UTILS -------
uint64_t millis64() { return esp_timer_get_time() / 1000ULL; }

bool sendJpegMultipart(const uint8_t* buf, size_t len, const String& filename, String& responseBody) {
  WiFiClient client;
  if (!client.connect(SERVER_HOST, SERVER_PORT)) {
    Serial.println("[HTTP] Falha ao conectar no servidor");
    return false;
  }

  String boundary = "----esp32camBoundary";
  String head =
      "--" + boundary + "\r\n"
      "Content-Disposition: form-data; name=\"image\"; filename=\"" + filename + "\"\r\n"
      "Content-Type: image/jpeg\r\n\r\n";
  String tail = "\r\n--" + boundary + "--\r\n";

  size_t contentLength = head.length() + len + tail.length();

  // Cabeçalho HTTP
  client.print(String("POST ") + SERVER_PATH + " HTTP/1.1\r\n");
  client.print(String("Host: ") + SERVER_HOST + "\r\n");
  client.print("Connection: close\r\n");
  client.print("Content-Type: multipart/form-data; boundary=" + boundary + "\r\n");
  client.print("Content-Length: " + String(contentLength) + "\r\n\r\n");

  // Corpo
  client.print(head);
  client.write(buf, len);
  client.print(tail);

  // Ler resposta simples
  uint32_t t0 = millis();
  bool headerEnded = false;
  responseBody = "";
  while (client.connected() && (millis() - t0 < 5000)) {
    while (client.available()) {
      String line = client.readStringUntil('\n');
      // Detecta fim do header HTTP (\r\n sozinho)
      if (!headerEnded) {
        if (line == "\r") {
          headerEnded = true;
        }
      } else {
        responseBody += line;
      }
      t0 = millis();
    }
  }
  client.stop();
  responseBody.trim();
  return true;
}

void setup() {
Serial.begin(115200);
  Serial.setDebugOutput(true);
  Serial.println();

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
  config.pin_sccb_sda = SIOD_GPIO_NUM;
  config.pin_sccb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.frame_size = FRAMESIZE_UXGA;
  config.pixel_format = PIXFORMAT_JPEG;  // for streaming
  //config.pixel_format = PIXFORMAT_RGB565; // for face detection/recognition
  config.grab_mode = CAMERA_GRAB_WHEN_EMPTY;
  config.fb_location = CAMERA_FB_IN_PSRAM;
  config.jpeg_quality = 12;
  config.fb_count = 1;

  // if PSRAM IC present, init with UXGA resolution and higher JPEG quality
  //                      for larger pre-allocated frame buffer.
  if (config.pixel_format == PIXFORMAT_JPEG) {
    if (psramFound()) {
      config.jpeg_quality = 10;
      config.fb_count = 2;
      config.grab_mode = CAMERA_GRAB_LATEST;
    } else {
      // Limit the frame size when PSRAM is not available
      config.frame_size = FRAMESIZE_SVGA;
      config.fb_location = CAMERA_FB_IN_DRAM;
    }
  } else {
    // Best option for face detection/recognition
    config.frame_size = FRAMESIZE_240X240;
#if CONFIG_IDF_TARGET_ESP32S3
    config.fb_count = 2;
#endif
  }

#if defined(CAMERA_MODEL_ESP_EYE)
  pinMode(13, INPUT_PULLUP);
  pinMode(14, INPUT_PULLUP);
#endif

  // camera init
  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed with error 0x%x", err);
    return;
  }

  sensor_t *s = esp_camera_sensor_get();
  // initial sensors are flipped vertically and colors are a bit saturated
  if (s->id.PID == OV3660_PID) {
    s->set_vflip(s, 1);        // flip it back
    s->set_brightness(s, 1);   // up the brightness just a bit
    s->set_saturation(s, -2);  // lower the saturation
  }
  // drop down frame size for higher initial frame rate
  if (config.pixel_format == PIXFORMAT_JPEG) {
    s->set_framesize(s, FRAMESIZE_QVGA);
  }

#if defined(CAMERA_MODEL_M5STACK_WIDE) || defined(CAMERA_MODEL_M5STACK_ESP32CAM)
  s->set_vflip(s, 1);
  s->set_hmirror(s, 1);
#endif

#if defined(CAMERA_MODEL_ESP32S3_EYE)
  s->set_vflip(s, 1);
#endif

  WiFi.begin(WIFI_SSID, WIFI_PASS);
  WiFi.setSleep(false);

  Serial.print("WiFi connecting");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("");
  Serial.println("WiFi connected");
}

void loop() {
  uint64_t start = millis64();

  camera_fb_t* fb = esp_camera_fb_get();
  if (!fb) {
    Serial.println("[CAM] Falha ao capturar frame");
    delay(1000);
    return;
  }

  // Garante JPEG
  const uint8_t* out_buf = fb->buf;
  size_t out_len = fb->len;
  uint8_t* conv_buf = nullptr;
  size_t conv_len = 0;

  if (fb->format != PIXFORMAT_JPEG) {
    Serial.println("[CAM] Frame não-JPEG; convertendo...");
    if (frame2jpg(fb, 90, &conv_buf, &conv_len)) {
      out_buf = conv_buf;
      out_len = conv_len;
    } else {
      Serial.println("[CAM] Conversão para JPEG falhou");
      esp_camera_fb_return(fb);
      delay(1000);
      return;
    }
  }

  String fname = "frame_" + String(millis64()) + ".jpg";
  String body;
  bool ok = sendJpegMultipart(out_buf, out_len, fname, body);

  if (conv_buf) free(conv_buf);
  esp_camera_fb_return(fb);

  if (!ok) {
    Serial.println("[HTTP] Falha no envio");
  } else {
    Serial.print("[HTTP] Resposta do servidor: ");
    Serial.println(body);
#if ENABLE_FLASH_FEEDBACK
    if (body == "OK" || body.startsWith("OK")) {
      // pisca o flash rapidamente
      digitalWrite(FLASH_LED_PIN, HIGH);
      delay(120);
      digitalWrite(FLASH_LED_PIN, LOW);
    }
#endif
  }

  // Dorme até completar 5s desde o início do ciclo
  uint64_t elapsed = millis64() - start;
  if (elapsed < CAPTURE_INTERVAL_MS) {
    delay(CAPTURE_INTERVAL_MS - elapsed);
  }
}
