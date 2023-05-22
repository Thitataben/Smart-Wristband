#include <ESP8266WiFi.h>
#include <PubSubClient.h>
#include <Servo.h>

int ledPin = 12;
int anotherLedPin = 16;
int fanPin = 5;

Servo myServo;

const int serialRate = 115200;

/* WiFi */
const char* ssid = "....";            // WiFi name
const char* password = "....";        // WiFi password

/* NETPIE */
const char* mqtt_client = "....";       // Client id
const char* mqtt_username = "....";     // Token
const char* mqtt_password = "....";     // Secret

WiFiClient espClient;
const char* mqtt_server = "broker.netpie.io";
const int mqtt_port = 1883;
PubSubClient client(espClient);
char msg[100];

String byteToString(byte* payload, unsigned int length_payload){ char buffer_payload[length_payload+1] = {0};
  memcpy(buffer_payload, (char*)payload, length_payload);
  return String(buffer_payload);
}

String charStarToString(char* payload){ // can not use this in serailization or deserailization because global/local variable problem
  String buffer=payload;
  return buffer;
}

String constCharStarToString(const char* payload){ // can not use this in serailization or deserailization because global/local variable problem
  String buffer =payload;
  return buffer;
}

void callback(char* topic, byte* payload, unsigned int length) {
  String ms=byteToString(payload, length);  //massage
  String t=charStarToString(topic);         //topic
  if(t=="@msg/gesture"){
    if (ms == "gesture1 start xz 12 cw z"){

    }
    else if (ms == "gesture2 start xz 12 ccw z"){}
    else if (ms == "gesture3 start xy 12 cw y"){}
    else if (ms == "gesture4 start xy 12 ccw y"){}
    else if (ms == "gesture 15 backslash start left"){}
    else if (ms == "gesture 16 backslash start right"){}
    Serial.println(ms);
  }
}
void setup() {
  pinMode(ledPin, OUTPUT);
  pinMode(anotherLedPin, OUTPUT);

  Serial.begin(serialRate);

  // setup wifi
  WiFi.begin(ssid, password);
  Serial.println();
  Serial.print("Connecting");
  while (WiFi.status() != WL_CONNECTED)
  {
    delay(500);
    Serial.print(".");
  }
  delay(100);
  Serial.println();
  Serial.println("WiFi connected");

  // setup netpie
  client.setServer(mqtt_server, mqtt_port);
  client.setCallback(callback);
  client.connect(mqtt_client, mqtt_username, mqtt_password);
  client.subscribe("@msg/gesture"); //sub auth
  delay(100);
}

void loop() {
  // put your main code here, to run repeatedly:
  if(client.connected()) {
    client.loop();
  } 
  else {
    if(WiFi.status() == WL_CONNECTED) {
      client.disconnect();
      client.connect(mqtt_client, mqtt_username, mqtt_password);
      client.subscribe("@msg/gesture");
      Serial.println("reconnected to Netpie again");
      delay(100);
    } 
    else {
      WiFi.disconnect();
      WiFi.begin(ssid, password);
      Serial.println("reconnected to WiFi again");
      delay(100);
    }
  }
  digitalWrite(ledPin, HIGH);
  digitalWrite(anotherLedPin, HIGH);
  delay(1000);
  digitalWrite(anotherLedPin, LOW);
  delay(1000);
}
