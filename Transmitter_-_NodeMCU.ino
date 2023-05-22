#include <ESP8266WiFi.h>
#include <PubSubClient.h>

// Wi-Fi
const char* ssid = "....";        // WiFi name
const char* password = "....";    // WiFi password

//NETPIE 
WiFiClient espClient;
const char* mqtt_server = "broker.netpie.io";
const int mqtt_port = 1883;
PubSubClient client(espClient);

const char* mqtt_client = "....";       // Client id
const char* mqtt_username = "....";     // Token
const char* mqtt_password = "....";     // Secret

char msg[100];

bool gestureDetect = false;

void setup() {
  Serial.begin(115200);
  Serial1.begin(115200); // UART
   
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(250);
    Serial.print(".");
  }
  Serial.println("WiFi connected");
  client.setServer(mqtt_server, mqtt_port);
  client.connect(mqtt_client, mqtt_username, mqtt_password);
  Serial.println("Setup complete");
}

void loop() {
String readString;
String Q;
String c;

//Check Serial Port 
  while (Serial.available()) {
    if (Serial.available() >0) {
      c = Serial.readString();  //gets one byte from serial buffer
      /*
    if (isControl(c)) {
      //'Serial.println("it's a control character");
      break;
    }
    */
      readString += c; //makes the string readString    
    }
  }
  Q = readString;

  if(Q!=" " && Q !=""){ //prevent spacebar
    gestureDetect=true;
  }

  //send data to netpie
  if(client.connected()) {
    client.loop(); 
    String payload = "{\"data\": {"; 
    payload.concat("\"gesture\":\"" + String(Q) );
    payload.concat("\"}}");
    Q = payload;
    payload.toCharArray(msg, (payload.length()+1));
    if (gestureDetect){
      client.publish("@shadow/data/update", msg);
      Serial.println(Q); 
    }

  } else {
    if(WiFi.status() == WL_CONNECTED) {
      client.disconnect();
      client.connect(mqtt_client, mqtt_username, mqtt_password);
      Serial.println("reconnected to Netpie");
    } else {
      WiFi.disconnect();
      WiFi.begin(ssid, password);
      Serial.println("reconnected to WiFi");
    }
  }
  gestureDetect = false;
  Q = "";
}
