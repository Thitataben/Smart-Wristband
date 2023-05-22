#include <Arduino_BMI270_BMM150.h>                          //Ax,Ay,Az,Gx,Gy,Gz for board Arduino nano 33 BLE
#include <SPI.h>                                      //SPI
#include <Wire.h>                                     //SPI 
#include <PubSubClient.h>                             //MQTT
#include <TensorFlowLite.h>                           //Tensorflow 
#include <tensorflow/lite/micro/all_ops_resolver.h>   //Tensorflow 
#include <tensorflow/lite/micro/micro_interpreter.h>  //Tensorflow 
#include <tensorflow/lite/schema/schema_generated.h>  //Tensorflow
#include <math.h>
#include "model.h"

// this constant won't change:
const int buttonPin = 12;  // the pin that the pushbutton is attached to

// Variables will change:
int buttonPushCounter = 0;  // counter for the number of button presses
int buttonState = 1;        // current state of the button
int lastButtonState = 1;    // previous state of the button

const int serialRate = 115200;
const int transmissionRate = 115200;      //make sure that this value must equal to receiveRate in subscribeDevice.ino

float accelerationThreshold = 2.5;  // threshold of significant in G's
int numSamples = 150;               // Number of sample per gesture

int samplesRead = numSamples;             // Variable for read acceleration


namespace {                               //Tensorflow variable
  tflite::AllOpsResolver tflOpsResolver1;
  const tflite::Model* tflModel1 = nullptr;
  tflite::MicroInterpreter* tflInterpreter1 = nullptr;
  TfLiteTensor* tflInputTensor1 = nullptr;
  TfLiteTensor* tflOutputTensor1 = nullptr;
  constexpr int tensorArenaSize1 = 8 * 1024;
  byte tensorArena_1[tensorArenaSize1] __attribute__((aligned(16)));
}

// Dynamic time
const int s1_length = 50; //maximum is 50 other wise cant not open serial monitor
const int s2_length = 50;
double sx[50] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
double sy[50] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
double sz[50] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
double xZero[50] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
double yZero[50] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
double zZero[50] = {0.98,0.94,0.99,0.97,0.97,0.95,0.95,0.96,0.95,0.95,0.96,0.96,0.93,0.96,0.95,0.96,0.95,0.95,0.97,0.94,0.95,0.96,0.97,0.93,0.95,0.97,0.95,0.96,0.94,0.94,0.97,0.96,0.95,0.95,0.95,0.95,0.96,0.95,0.95,0.95,0.95,0.95,0.94,0.97,0.96,0.94,0.96,0.95,0.95,0.96};
double d11_x[50] = {-1.08,-1.04,-1.07,-0.93,-0.92,-0.97,-1.02,-1.02,-0.99,-0.91,-0.96,-1.18,-1.05,-0.93,-0.94,-0.93,-0.97,-0.97,-1.00,-0.97,-0.96,-0.98,-1.06,-0.95,-1.00,-1.11,-0.91,-0.99,-1.02,-0.96,-0.98,-0.96,-0.99,-0.98,-0.99,-0.96,-1.00,-0.97,-0.97,-1.03,-1.08,-1.06,-1.06,-1.05,-0.98,-1.01,-1.02,-1.00,-1.01,-1.00}; 
double d11_y[50] ={0.21,0.27,0.29,0.34,0.28,0.24,0.36,0.30,0.35,0.24,0.18,0.14,0.14,0.25,0.15,0.26,0.29,0.29,0.21,0.24,0.15,0.14,0.17,0.09,0.15,0.15,0.13,0.10,0.10,0.11,0.12,0.15,0.14,0.13,0.14,0.14,0.15,0.14,0.13,0.11,0.12,0.14,0.12,0.11,0.13,0.14,0.15,0.13,0.11,0.12};
double d11_z[50] ={-0.26,-0.29,-0.30,-0.12,-0.03,-0.04,-0.07,-0.10,-0.17,-0.09,-0.15,-0.15,-0.07,0.02,0.01,0.09,0.15,0.22,0.19,0.15,0.08,-0.01,-0.03,-0.02,-0.00,-0.04,0.02,0.04,0.04,0.05,0.05,0.06,0.07,0.08,0.07,0.06,0.06,0.06,0.06,0.06,0.06,0.08,0.06,0.03,0.04,0.06,0.06,0.06,0.05,0.05};
double d8_x_success_1[50] ={0.88,1.15,1.36,1.25,1.09,0.81,0.90,0.66,0.58,0.89,0.28,0.21,-0.13,0.14,0.06,0.08,0.24,0.15,0.15,0.08,0.06,0.09,0.09,0.22,0.04,0.08,0.00,0.06,0.04,0.18,0.16,0.16,0.10,0.16,0.16,0.11,0.14,0.06,0.07,0.06,0.18,0.03,0.02,0.09,0.12,0.04,0.02,0.08,0.08,0.04};
double d8_y_success_1[50] ={0.10,-0.62,-0.34,-0.06,-0.05,-0.05,-0.08,-0.25,-0.39,0.09,0.08,-0.19,0.14,0.07,-0.11,-0.14,0.01,-0.19,0.01,-0.18,-0.03,0.02,-0.00,0.03,0.02,0.04,0.06,0.05,0.07,0.05,0.04,0.06,0.07,0.08,0.05,0.12,0.07,0.04,0.00,-0.02,-0.06,0.03,0.15,0.06,0.04,0.03,0.04,0.05,0.08,0.02};
double d8_z_success_1[50] ={1.01,1.37,1.05,0.71,0.81,0.49,0.11,0.33,0.41,0.07,0.59,0.55,0.94,0.95,1.03,1.11,0.97,1.03,1.00,1.05,1.00,0.93,0.89,0.92,0.96,0.92,0.90,0.96,0.91,0.98,1.07,0.95,1.01,1.03,0.97,0.95,1.03,0.99,0.85,0.94,0.90,0.97,0.98,0.99,0.96,0.92,0.94,0.96,0.95,0.91};
double d8_x_fail_1[50] ={1.14,1.32,1.65,1.84,1.33,1.28,0.85,0.82,0.50,0.53,0.37,0.16,-0.10,-0.01,-0.10,-0.14,-0.10,-0.10,-0.20,-0.02,0.13,0.45,0.88,1.03,0.98,0.94,1.35,1.89,1.63,1.35,1.46,1.02,0.93,0.82,0.93,1.05,1.00,0.98,1.05,1.06,0.90,0.89,0.93,0.91,1.00,0.94,0.97,1.00,0.94,1.00};
double d8_y_fail_1[50] ={0.08,-0.19,-0.37,-0.54,-0.20,-0.08,0.02,0.01,0.04,-0.02,-0.13,0.03,0.11,-0.05,0.05,-0.05,-0.09,-0.14,-0.10,0.03,-0.08,0.18,-0.22,-0.21,-0.47,-0.34,-0.04,-0.77,-0.38,-0.14,-0.44,-0.25,0.10,-0.29,-0.20,-0.08,-0.27,-0.04,-0.05,-0.13,-0.05,-0.00,-0.09,-0.03,-0.14,-0.04,0.03,-0.06,-0.04,-0.03};
double d8_z_fail_1[50] ={0.78,1.52,1.21,0.54,0.41,0.54,0.46,0.46,0.51,0.39,0.43,0.44,0.63,0.57,0.79,0.84,0.88,0.87,0.80,0.47,0.20,-0.01,-0.00,0.46,0.57,0.82,0.57,1.64,0.54,1.34,0.39,0.45,0.50,0.01,-0.28,-0.18,-0.30,-0.16,-0.13,0.02,-0.04,-0.09,-0.05,-0.06,0.02,0.04,0.00,0.11,0.06,0.08};
double d9_x_success_1[50] ={1.08,1.38,1.01,1.17,1.18,1.21,1.09,1.05,1.04,1.20,1.10,1.12,1.01,0.94,0.92,0.94,0.99,1.10,1.17,1.13,1.12,1.08,1.09,1.12,1.12,1.12,0.95,0.86,0.83,0.80,0.86,0.78,0.69,0.64,0.70,0.87,0.79,0.83,0.80,0.81,0.80,0.83,0.84,0.93,0.90,0.96,0.89,0.89,0.91,0.92};
double d9_y_success_1[50] ={0.12,0.11,0.09,0.00,0.06,0.13,-0.04,-0.07,-0.04,-0.05,0.00,0.19,-0.15,-0.05,-0.02,-0.10,-0.10,0.01,-0.10,-0.03,-0.10,-0.01,-0.22,-0.22,-0.20,-0.27,-0.05,-0.11,-0.09,-0.12,-0.05,-0.01,0.01,0.01,0.05,0.05,0.03,-0.01,-0.08,-0.02,0.01,-0.02,0.02,-0.03,0.08,0.03,0.09,0.04,0.03,0.05};
double d9_z_success_1[50] ={-0.04,-0.03,0.09,0.10,0.28,0.24,0.33,0.28,0.38,0.46,0.50,0.54,0.47,0.44,0.44,0.37,0.34,0.32,0.37,0.32,0.38,0.31,0.31,0.24,0.19,0.24,0.18,0.20,0.19,0.16,0.18,0.23,0.19,0.21,0.34,0.47,0.22,0.19,0.18,0.28,0.24,0.30,0.22,0.31,0.31,0.51,0.41,0.33,0.28,0.27};
double d9_x_fail_1[50] ={1.25,1.52,1.41,1.88,1.29,0.70,0.89,1.00,0.81,0.85,0.95,1.00,0.96,0.93,0.90,0.94,0.95,0.96,0.86,0.92,0.94,0.89,0.91,0.94,0.92,0.92,0.92,0.93,0.95,0.91,0.94,0.94,0.92,0.90,0.94,0.93,0.90,0.94,0.93,0.90,0.91,0.93,0.92,0.91,0.92,0.91,0.92,0.94,0.91,0.92};
double d9_y_fail_1[50] ={0.14,0.20,0.06,-0.14,-0.96,-0.50,-0.01,0.20,-0.20,-0.06,0.05,-0.18,-0.08,-0.13,-0.11,0.10,-0.06,-0.07,-0.12,0.08,-0.13,-0.01,0.05,-0.05,-0.07,-0.13,-0.02,-0.01,-0.09,-0.12,-0.00,-0.03,-0.14,-0.04,-0.05,-0.09,-0.09,-0.04,-0.07,-0.13,-0.01,-0.06,-0.11,-0.01,-0.07,-0.10,-0.04,-0.04,-0.12,-0.06};
double d9_z_fail_1[50] ={0.00,0.05,-0.11,1.24,0.67,0.57,0.23,0.17,0.20,0.20,0.22,0.39,0.21,0.27,0.29,0.14,0.24,0.28,0.30,0.17,0.30,0.24,0.22,0.26,0.26,0.30,0.28,0.20,0.28,0.27,0.23,0.26,0.30,0.23,0.24,0.29,0.27,0.27,0.25,0.29,0.24,0.28,0.29,0.25,0.28,0.28,0.28,0.28,0.28,0.28};
double d11_x_success_1[50] ={0.98,1.15,1.08,0.98,0.99,0.99,1.13,1.07,1.01,1.08,0.92,0.88,0.94,0.89,0.97,1.00,0.94,1.08,0.87,0.94,0.83,0.93,0.95,0.90,0.99,0.89,0.79,0.73,0.78,0.87,0.90,0.94,1.12,1.10,1.04,0.90,0.86,0.91,0.92,0.89,1.08,0.89,0.95,1.00,0.96,0.91,0.98,0.89,0.90,1.05};
double d11_y_success_1[50] ={0.46,0.39,0.18,0.12,-0.11,-0.00,-0.15,0.17,-0.08,0.07,-0.10,0.02,-0.11,-0.01,-0.01,-0.02,-0.06,-0.00,0.05,-0.06,0.02,0.04,-0.01,0.07,-0.06,0.02,-0.19,-0.20,-0.27,-0.19,-0.21,-0.10,-0.27,-0.10,-0.14,0.05,-0.14,-0.00,-0.25,-0.15,-0.21,-0.09,-0.13,-0.06,0.02,0.03,0.00,-0.18,0.04,-0.02};
double d11_z_success_1[50] ={-0.19,0.02,-0.05,-0.15,-0.09,-0.08,0.25,0.31,0.30,0.39,0.33,0.32,0.40,0.36,0.33,0.34,0.22,0.32,0.27,0.41,0.36,0.53,0.44,0.53,0.50,0.41,0.38,0.41,0.37,0.36,0.28,0.21,0.28,0.27,0.41,0.27,0.25,0.22,0.22,0.37,0.40,0.44,0.36,0.42,0.40,0.25,0.27,0.21,0.21,0.29};
double d11_x_fail_1[50] ={1.00,0.98,1.10,1.11,1.02,0.95,0.80,0.89,0.93,0.80,1.01,1.01,0.87,0.97,0.84,1.07,0.91,0.61,0.53,0.63,0.44,0.69,0.05,0.64,0.05,0.84,1.36,1.37,1.55,1.27,1.41,1.31,1.29,1.17,1.20,1.18,1.09,1.01,0.88,0.92,0.93,0.96,1.04,1.02,0.95,1.01,0.98,0.97,1.00,1.01};
double d11_y_fail_1[50] ={0.38,0.63,0.12,-0.15,-0.02,-0.06,0.10,-0.07,-0.11,-0.01,0.05,-0.27,0.02,-0.27,0.03,-0.20,0.05,0.18,0.03,0.19,-0.00,-0.26,-0.03,-0.27,0.11,-0.03,0.03,-0.21,0.16,0.10,0.00,0.08,0.16,0.18,0.17,-0.39,0.12,-0.17,0.09,-0.09,0.05,-0.09,-0.07,0.00,0.16,-0.00,-0.04,-0.15,-0.01,-0.12};
double d11_z_fail_1[50] ={-0.23,-0.26,-0.06,0.21,0.18,0.39,0.28,0.38,0.40,0.33,0.26,0.20,0.29,0.37,0.37,0.50,0.35,0.52,0.53,0.50,0.15,0.23,-0.13,0.01,0.17,-0.10,-0.26,0.13,-0.00,-0.17,0.13,0.31,0.31,-0.74,-0.04,0.24,0.20,0.09,-0.20,-0.31,-0.05,-0.05,0.01,0.06,0.02,-0.00,0.04,-0.05,-0.02,0.01};
double d1_x[50] ={-0.76,-0.78,-0.72,-0.78,-0.80,-0.66,-0.76,-0.74,-0.75,-0.75,-0.74,-0.75,-0.75,-0.74,-0.74,-0.73,-0.76,-0.78,-0.84,-0.92,-0.97,-1.03,-1.22,-1.22,-1.19,-1.02,-1.01,-0.83,-0.91,-0.91,-0.83,-0.93,-0.99,-0.95,-0.99,-0.98,-0.99,-0.98,-0.97,-0.99,-0.97,-0.99,-0.98,-0.99,-0.97,-0.98,-0.98,-0.99,-0.98,-0.98};
double d1_y[50] ={0.48,0.48,0.40,0.68,0.46,0.34,0.53,0.51,0.49,0.50,0.49,0.49,0.49,0.50,0.50,0.49,0.52,0.47,0.40,0.40,0.34,0.24,0.43,0.27,-0.19,-0.08,-0.05,-0.06,0.00,0.00,0.02,0.08,0.10,0.16,0.18,0.16,0.17,0.15,0.16,0.16,0.15,0.16,0.17,0.15,0.16,0.18,0.18,0.18,0.16,0.16};
double d1_z[50] ={0.41,0.41,0.38,0.49,0.33,0.47,0.41,0.38,0.39,0.39,0.39,0.39,0.38,0.40,0.41,0.40,0.39,0.37,0.31,0.19,0.09,0.18,0.17,-0.44,-0.37,-0.22,-0.14,-0.11,-0.07,-0.04,0.02,0.09,0.10,0.13,0.12,0.09,0.07,0.06,0.06,0.06,0.05,0.05,0.05,0.05,0.05,0.06,0.06,0.05,0.05,0.04};

int period = 50;
unsigned long time_now = 0;
double distance;
double distanceX;
double distanceY;
double distanceZ;
double distanceX1;
double distanceY1;
double distanceZ1;
double distanceX2;
double distanceY2;
double distanceZ2;
double distanceX3;
double distanceY3;
double distanceZ3;
double distanceX4;
double distanceY4;
double distanceZ4;
double distanceX5;
double distanceY5;
double distanceZ5;
double distanceX6;
double distanceY6;
double distanceZ6;

// name of gesture
const char* nameGesture_1 = "gesture1 start xz 12 cw z";
const char* nameGesture_2 = "gesture2 start xz 12 ccw z";
const char* nameGesture_3 = "gesture3 start xy 12 cw y";
const char* nameGesture_4 = "gesture4 start xy 12 ccw y";
const char* nameGesture_5 = "gesture 15 backslash start left";
const char* nameGesture_6 = "gesture 16 backslash start right";

// array to map gesture index to a name
const char* GESTURES_1[] = {
    nameGesture_1, 
    nameGesture_2,
    nameGesture_3,
    nameGesture_4,
    nameGesture_5,
    nameGesture_6
};

#define NUM_GESTURES_1 (sizeof(GESTURES_1) / sizeof(GESTURES_1[0]))

// variable for output gesture
float accuracy = 0.8;               // if condition --> show output when value is more than accuracy
String gestureAnswer_1="";        // name of output gesture
float gestureAccuracy_1 = 0;      // accuracy of output gesture
bool gestureDetected_1 = false;   // detect if there any output accuracy that more than variable accuracy(0.8)

bool buttonActivate(){
  bool buttonCheck = false;  

  // read the pushbutton input pin:
  buttonState = digitalRead(buttonPin);

  // compare the buttonState to its previous state
  if (buttonState != lastButtonState) {
    // if the state has changed, increment the counter
    if (buttonState == HIGH) {
      // if the current state is HIGH then the button went from off to on:
      buttonPushCounter++;
      //Serial.println("on");
      //Serial.print("number of button pushes: ");
      //Serial.println(buttonPushCounter);
      buttonCheck = true;
    } 
    else {
      // if the current state is LOW then the button went from on to off:
      //Serial.println("off");
    }
    // Delay a little bit to avoid bouncing
    delay(50);
  }
  // save the current state as the last state, for next time through the loop
  lastButtonState = buttonState;
  return buttonCheck;
}

double dtw_distance(double s1[50], double s2[50]) {
    // Initialize the DTW matrix with zeros
    /* sample1 and sample2
    Serial.print("s1 ={");
        for(int i = 0; i < numSamples; i++) {
          Serial.print(s1[i]);
          if (i != numSamples-1){
            Serial.print(",");
          }
          else{
            Serial.print("}");
          }
        }
    Serial.println();
    Serial.print("s2 ={");
        for(int i = 0; i < numSamples; i++) {
          Serial.print(s2[i]);
          if (i != numSamples-1){
            Serial.print(",");
          }
          else{
            Serial.print("}");
          }
        }
    Serial.println();
    */
    double dtw[s1_length+1][s2_length+1] = {0.0};

    // Fill the first row and column with infinity
    for (int i = 1; i <= s1_length; i++) {
        dtw[i][0] = INFINITY;
    }
    for (int j = 1; j <= s2_length; j++) {
        dtw[0][j] = INFINITY;
    }
    dtw[0][0] = 0.0;

    // Fill in the rest of the DTW matrix
    for (int i = 1; i <= s1_length; i++) {
        for (int j = 1; j <= s2_length; j++) {
            double cost = abs(s1[i-1] - s2[j-1]);
            dtw[i][j] = cost + min(dtw[i-1][j], min(dtw[i][j-1], dtw[i-1][j-1]));
        }
    }

    // Return the DTW distance
    return dtw[s1_length][s2_length];
}

void doModel1(){
  float aX, aY, aZ, gX, gY, gZ;
  bool breakEverything = false;

  // wait for significant motion
  while (samplesRead == numSamples) {
    if (buttonActivate()){
      Serial.println("button change model1 to model2 activate");
      breakEverything = true;
      if(breakEverything){
        break; 
      }     
    }

    if (IMU.accelerationAvailable()) {
      // read the acceleration data
      IMU.readAcceleration(aX, aY, aZ);

      // sum up the absolutes
      float aSum = fabs(aX) + fabs(aY) + fabs(aZ);

      // check if it's above the threshold
      if (aSum >= accelerationThreshold) {
        // reset the sample read count
        Serial.println("model1 have significant threshold");
        samplesRead = 0;
        break;
      }
    }
  }
  // check if the all the required samples have been read sincethe last time the significant motion was detected
  while (samplesRead < numSamples) { // Start of loop samplesRead==0 and numSamples ==200
    if(breakEverything){
      break; 
    }
    digitalWrite(LED_BUILTIN, HIGH); // LED ON when have significant acceleration

    // check if new acceleration AND gyroscope data is available
    if (IMU.accelerationAvailable() && IMU.gyroscopeAvailable()) {
      // read the acceleration and gyroscope data
      IMU.readAcceleration(aX, aY, aZ);
      IMU.readGyroscope(gX, gY, gZ);

      // normalize the IMU data between 0 to 1 and store in the model's input tensor
      tflInputTensor1->data.f[samplesRead * 6 + 0] = (aX + 4.0) / 8.0;
      tflInputTensor1->data.f[samplesRead * 6 + 1] = (aY + 4.0) / 8.0;
      tflInputTensor1->data.f[samplesRead * 6 + 2] = (aZ + 4.0) / 8.0;
      tflInputTensor1->data.f[samplesRead * 6 + 3] = (gX + 2000.0) / 4000.0;
      tflInputTensor1->data.f[samplesRead * 6 + 4] = (gY + 2000.0) / 4000.0;
      tflInputTensor1->data.f[samplesRead * 6 + 5] = (gZ + 2000.0) / 4000.0;

      samplesRead++;

      if (samplesRead == numSamples) { // End of loop sampleread==200 and numSamples ==200
        Serial.println("model1 sample complete");

        digitalWrite(LED_BUILTIN, LOW); // LED OFF when finish reading acceleration

        // Run inferencing
        TfLiteStatus invokeStatus1 = tflInterpreter1->Invoke();
        Serial.println("TfLiteStatus working");
        if (invokeStatus1 != kTfLiteOk) {
          Serial.println("Invoke1 failed!");
          while (1);
          return;
        }

        // Loop through the output tensor values from the model
        for (int i = 0; i < NUM_GESTURES_1; i++) {
          Serial.print(GESTURES_1[i]);
          Serial.print(": ");
          Serial.println(tflOutputTensor1->data.f[i], 6);
        }

        // gesture output
        for (int i = 0; i < NUM_GESTURES_1; i++) {
          if (tflOutputTensor1->data.f[i] >= accuracy && !gestureDetected_1){ // do if accuracy more than 0.8
            gestureDetected_1 =true;
            gestureAnswer_1 = GESTURES_1[i];
            gestureAccuracy_1 = tflOutputTensor1->data.f[i];
            break;
          }
        }

        // serial monitor output
        if (gestureDetected_1){
          Serial.println("Detect " + gestureAnswer_1);
        }

        // rxtx serial output 
        if (gestureAnswer_1 == "gesture1 start xz 12 cw z"){
          Serial1.write("gesture1 start xz 12 cw z");
          Serial.println("gesture1 start xz 12 cw z is send to the transmission line");
        }
        else if (gestureAnswer_1 == "gesture2 start xz 12 ccw z"){
          Serial1.write("gesture2 start xz 12 ccw z");
          Serial.println("gesture2 start xz 12 ccw z is send to the transmission line");          
        }
        else if (gestureAnswer_1 == "gesture3 start xy 12 cw y"){
          Serial1.write("gesture3 start xy 12 cw y");
          Serial.println("gesture3 start xy 12 cw y is send to the transmission line");           
        }
        else if (gestureAnswer_1 == "gesture4 start xy 12 ccw y"){
          Serial1.write("gesture4 start xy 12 ccw y");
          Serial.println("gesture4 start xy 12 ccw y is send to the transmission line"); 
        }
        else if (gestureAnswer_1 == "gesture 15 backslash start left"){
          Serial1.write("gesture 15 backslash start left");
          Serial.println("gesture 15 backslash start left is send to the transmission line"); 
        }
        else if (gestureAnswer_1 == "gesture 16 backslash start right"){
          Serial1.write("gesture 16 backslash start right");
          Serial.println("gesture 16 backslash start right is send to the transmission line"); 
        }

        // reset output gesture varible
        gestureDetected_1 = false;
        gestureAnswer_1 = "";
        gestureAccuracy_1 = 0;
        Serial.println();       
      }
    }
  }
}

void doModel2(){
  float aX, aY, aZ, gX, gY, gZ;
  bool breakEverything = false;

  // wait for significant motion
  while (samplesRead == numSamples) {
    if (buttonActivate()){
      Serial.println("button change model2 to model1 activate");
      breakEverything = true;
      if(breakEverything){
        break;
      } 
    }
    
    if (IMU.accelerationAvailable()) {
      // read the acceleration data
      IMU.readAcceleration(aX, aY, aZ);

      // sum up the absolutes
      float aSum = fabs(aX) + fabs(aY) + fabs(aZ);

      // check if it's above the threshold
      if (aSum >= accelerationThreshold) {
        // reset the sample read count
        samplesRead = 0;
        Serial.println("model2 have significant threshold");
        break;
      }
    }
  }

  // check if the all the required samples have been read since
  // the last time the significant motion was detected
  while (samplesRead < numSamples) {
    if(breakEverything){
      break;
    } 
    digitalWrite(LED_BUILTIN, HIGH);
    // check if both new acceleration and gyroscope data is
    // available
    if (IMU.accelerationAvailable() && IMU.gyroscopeAvailable()) {
      // read the acceleration and gyroscope data
      IMU.readAcceleration(aX, aY, aZ);
      IMU.readGyroscope(gX, gY, gZ);

      samplesRead++;
      sx[samplesRead-1] = aX;
      sy[samplesRead-1] = aY;
      sz[samplesRead-1] = aZ;
      if (samplesRead == numSamples) {
        Serial.println("model2 sample complete");
        digitalWrite(LED_BUILTIN, LOW);
        // add an empty line if it's the last sample
        Serial.print("sx ={");
        for(int i = 0; i < numSamples; i++) {
          Serial.print(sx[i]);
          if (i != numSamples-1){
            Serial.print(",");
          }
          else{
            Serial.print("}");
          }
        }
        Serial.println();

        Serial.print("sy ={");
        for(int i = 0; i < numSamples; i++) {
          Serial.print(sy[i]);
          if (i != numSamples-1){
            Serial.print(",");
          }
          else{
            Serial.print("}");
          }
        }
        Serial.println();
        
        Serial.print("sz ={");
        for(int i = 0; i < numSamples; i++) {
          Serial.print(sz[i]);
          if (i != numSamples-1){
            Serial.print(",");
          }
          else{
            Serial.print("}");
          }
        }
        Serial.println();


        //already get data     
        distanceX1 = dtw_distance(sx,d9_x_success_1);
        Serial.print("DTW distance_x: ");
        Serial.println(distanceX1);

        distanceY1 = dtw_distance(sy,d9_y_success_1);
        Serial.print("DTW distance_y: ");
        Serial.println(distanceY1);

        distanceZ1 = dtw_distance(sz,d9_z_success_1);
        Serial.print("DTW distance_z: ");
        Serial.println(distanceZ1);

        distanceX2 = dtw_distance(sx,d9_x_fail_1);
        Serial.print("DTW distance_x: ");
        Serial.println(distanceX2);

        distanceY2 = dtw_distance(sy,d9_y_fail_1);
        Serial.print("DTW distance_y: ");
        Serial.println(distanceY2);

        distanceZ2 = dtw_distance(sz,d9_z_fail_1);
        Serial.print("DTW distance_z: ");
        Serial.println(distanceZ2);

        distanceX3 = dtw_distance(sx,d11_x_success_1);
        Serial.print("DTW distance_x: ");
        Serial.println(distanceX3);

        distanceY3 = dtw_distance(sy,d11_y_success_1);
        Serial.print("DTW distance_y: ");
        Serial.println(distanceY3);

        distanceZ3 = dtw_distance(sz,d11_z_success_1);
        Serial.print("DTW distance_z: ");
        Serial.println(distanceZ3);

        distanceX4 = dtw_distance(sx,d11_x_fail_1);
        Serial.print("DTW distance_x: ");
        Serial.println(distanceX4);

        distanceY4 = dtw_distance(sy,d11_y_fail_1);
        Serial.print("DTW distance_y: ");
        Serial.println(distanceY4);

        distanceZ4 = dtw_distance(sz,d11_z_fail_1);
        Serial.print("DTW distance_z: ");
        Serial.println(distanceZ4);

        distanceX5 = dtw_distance(sx,d8_x_success_1);
        Serial.print("DTW distance_x: ");
        Serial.println(distanceX5);

        distanceY5 = dtw_distance(sy,d8_y_success_1);
        Serial.print("DTW distance_y: ");
        Serial.println(distanceY5);

        distanceZ5 = dtw_distance(sz,d8_z_success_1);
        Serial.print("DTW distance_z: ");
        Serial.println(distanceZ5);

        distanceX6 = dtw_distance(sx,d8_x_fail_1);
        Serial.print("DTW distance_x: ");
        Serial.println(distanceX6);

        distanceY6 = dtw_distance(sy,d8_y_fail_1);
        Serial.print("DTW distance_y: ");
        Serial.println(distanceY6);

        distanceZ6 = dtw_distance(sz,d8_z_fail_1);
        Serial.print("DTW distance_z: ");
        Serial.println(distanceZ6);

        Serial.println();

        distanceX = dtw_distance(sx,xZero);
        Serial.print("DTW distance_x: ");
        Serial.println(distanceX);

        distanceY = dtw_distance(sy,yZero);
        Serial.print("DTW distance_y: ");
        Serial.println(distanceY);

        distanceZ = dtw_distance(sz,zZero);
        Serial.print("DTW distance_z: ");
        Serial.println(distanceZ);
        Serial.println();

        //case
        if(distanceX1 - distanceX2 >= 1.7 && distanceY1 < 7 && distanceZ > 35){ //poohAdd && distanceZ>35 shold improve to distanceX later
          Serial.print("Urgen fall gesture:");
          Serial.print("1");
          Serial.println(":");
          Serial1.write("Urgen fall gesture:1:");
        }
        // pooh add2
        //slow turn
        else if(distanceX3 + distanceY3 + distanceZ3 < 15 && distanceX3 < 4.5 && distanceY > 5 ){ // && distanceX > 47
          Serial.print("turn gesture:");
          Serial.print("1");
          Serial.println(":");
          Serial1.write("turn gesture:1:");
        }
        else if(distanceX4 + distanceY4 + distanceZ4 < 20 && distanceX > 48){ // previous is 18
          Serial.print("half turn fall gesture:");
          Serial.print("1");
          Serial.println(":");
          Serial1.write("half turn fall gesture:1:"); 
        }
        else if(distanceX2 >distanceX1 && distanceX1 > 3 && distanceX1 < 7.5 && distanceX > 48){ //poohOld distanceX2 - distanceX1 > 1 && distanceX1 > 3 // dena distanceX2 - distanceX1 > 1.45
          Serial.print("bend down gesture:");
          Serial.print("1");
          Serial.println(":");
          Serial1.write("bend down gesture:1:"); 
        }
        else if(distanceX5<10 && distanceY5<10 && distanceY5<10 && distanceX+distanceY+distanceZ > 25){ //poohOld distanceX2 - distanceX1 > 1 && distanceX1 > 3 // dena distanceX2 - distanceX1 > 1.45
          Serial.print("extend arm gesture:");
          Serial.print("1");
          Serial.println(":");
          Serial1.write("extend arm gesture:1:");
        }
        else if(distanceX6<10 && distanceY6<10 && distanceZ6<10 ){ //poohOld distanceX2 - distanceX1 > 1 && distanceX1 > 3 // dena distanceX2 - distanceX1 > 1.45
          Serial.print("extend arm fail gesture:");
          Serial.print("1");
          Serial.println(":");
          Serial1.write("extend arm fail gesture:1:");
        }
        else{ //distanceX2 > distanceX2
          Serial.print("noise gesture:");
          Serial.print("1");
          Serial.println(":");
          Serial1.write("noise gesture:1:");          
        }
      }
    }
    delay(50); // 200 for 10 sec, 100 for 5 sec
  }
}

void setup() {
  // initialize serial monitor
  Serial.begin(serialRate);

  // initialize rxtx serial
  Serial1.begin(transmissionRate);

  // initialize the IMU
  if (!IMU.begin()) {
    Serial.println("Failed to initialize IMU!");
    while (1);
  }

  pinMode(buttonPin, INPUT_PULLUP);
  pinMode(LED_BUILTIN, OUTPUT);

  // print out the samples rates of the IMUs
  Serial.print("Accelerometer sample rate = ");
  Serial.print(IMU.accelerationSampleRate());
  Serial.println(" Hz");
  Serial.print("Gyroscope sample rate = ");
  Serial.print(IMU.gyroscopeSampleRate());
  Serial.println(" Hz");

  Serial.println(); 

  /* setup Tensorflow*/
  // get the TFL representation of the model byte array
  tflModel1 = tflite::GetModel(model_1);

  // check Tensorflow version
  if (tflModel1->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model_1 schema mismatch!");
    while (1);
  }

  // Create an interpreter to run the model
  tflInterpreter1 = new tflite::MicroInterpreter(tflModel1, tflOpsResolver1, tensorArena_1, tensorArenaSize1); //previous code have another parameter --> (, &tflErrorReporter) which cause error when compile  

  // Allocate memory for the model's input and output tensors
  tflInterpreter1->AllocateTensors();
  
  // Get pointers for the model's input and output tensors
  tflInputTensor1 = tflInterpreter1->input(0);  
  tflOutputTensor1 = tflInterpreter1->output(0);
}

void loop() {
  if (buttonPushCounter % 2 == 0){
    Serial.println("buttonPushConter = " +String(buttonPushCounter));
    accelerationThreshold = 2.5;  // threshold of significant in G's
    numSamples = 150;
    samplesRead = numSamples;
    Serial.println("model1 start");
    doModel1();
    Serial.println("model1 finish");
  }
  else{
    Serial.println("buttonPushConter = " +String(buttonPushCounter));
    accelerationThreshold = 1.7;
    numSamples = 50;
    samplesRead = numSamples;
    Serial.println("model2 start");
    doModel2();
    Serial.println("model2 finish");
  }
}
