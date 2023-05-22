#include <Arduino_BMI270_BMM150.h>                    
#include <math.h>
#include <SPI.h>                                      
#include <Wire.h>                                                                
#include <TensorFlowLite.h>                           
#include <tensorflow/lite/micro/all_ops_resolver.h>   
#include <tensorflow/lite/micro/micro_interpreter.h>   
#include <tensorflow/lite/schema/schema_generated.h>
#include <Arduino_KNN.h>

#include "model.h"
//---------------------------------------------------------------------------------------------------------
// button
const int buttonPin = 8;  // the pin that the pushbutton is attached to

//---------------------------------------------------------------------------------------------------------
// state variable:
int buttonPushCounter = 0;  // counter for the number of button presses
int buttonState = 1;        // current state of the button
int lastButtonState = 1;    // previous state of the button

//---------------------------------------------------------------------------------------------------------
// adjustParameter
const int serialRate = 115200;            // Serial monitor
const int transmissionRate = 115200;      // UART
float accelerationThreshold = 2.5;        // threshold of significant in G's
int numSamples = 150;                     // Number of sample per gesture
const int sampleDelay = 50;               

int samplesRead = numSamples;             // Variable for read acceleration

//---------------------------------------------------------------------------------------------------------

tflite::AllOpsResolver tflOpsResolver1;
const tflite::Model* tflModel1 = nullptr;
tflite::MicroInterpreter* tflInterpreter1 = nullptr;
TfLiteTensor* tflInputTensor1 = nullptr;
TfLiteTensor* tflOutputTensor1 = nullptr;
constexpr int tensorArenaSize1 =  8* 1024; //original 8*1024
byte tensorArena_1[tensorArenaSize1] __attribute__((aligned(16)));//byte ... __attribute__((aligned(16)))

const char* nameGesture_1 = "gesture1 start xz 12 cw z";
const char* nameGesture_2 = "gesture2 start xz 12 ccw z";
const char* nameGesture_3 = "gesture3 start xy 12 cw y";
const char* nameGesture_4 = "gesture4 start xy 12 ccw y";
const char* nameGesture_5 = "gesture 15 backslash start left";
const char* nameGesture_6 = "gesture 16 backslash start right";
const char* GESTURES_1[] = {
  nameGesture_1, 
  nameGesture_2,
  nameGesture_3,
  nameGesture_4,
  nameGesture_5,
  nameGesture_6
};
#define NUM_GESTURES_1 (sizeof(GESTURES_1) / sizeof(GESTURES_1[0]))

float accuracy = 0.8;              // if condition --> show output when value is more than accuracy
String gestureAnswer_1="";        // name of output gesture
float gestureAccuracy_1 = 0;      // accuracy of output gesture
bool gestureDetected_1 = false;   // detect if there any output accuracy that more than variable accuracy(0.8)

//---------------------------------------------------------------------------------------------------------
// variable for model2

const int arrayMyKNN = 15;                // array of KNN
int kValue = 6;                           // K
sampleData sample;
distanceDTW distance;
KNNClassifier myKNN(arrayMyKNN);
const char* GESTURES_2[] = {
  "retrieving_Object_From_Floor_Success :1:", 
  "retrieving_Object_From_Floor_Fail :1:",
  "turn_360_CCW_Success :1:",
  "turn_360_CCW_Fail :1:",
  "reaching_Forward_With_Outstretched_Arm_Success :1:",
  "reaching_Forward_With_Outstretched_Arm_Fail :1:"
};
String gestureAnswer_2="";

//---------------------------------------------------------------------------------------------------------
//function
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
//---------------------------------------------------------------------------------------------------------
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
        tflInterpreter1->AllocateTensors();
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
        String gestureAnswer_2="";
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

      samplesRead++;
      sample.aX[samplesRead-1] = aX;
      sample.aY[samplesRead-1] = aY;
      sample.aZ[samplesRead-1] = aZ;

      if (samplesRead == samplePerGesture) {
        Serial.println("model2 sample complete");
        digitalWrite(LED_BUILTIN, LOW);

        // serial print sample
        sample.printSampleData(1,"aX");
        sample.printSampleData(1,"aY");
        sample.printSampleData(1,"aZ");

        float input[arrayMyKNN];          
        input[0] = dtw_distance(sample.aX, upToDownGesture[0]);
        input[1] = dtw_distance(sample.aY, upToDownGesture[1]);
        input[2] = dtw_distance(sample.aZ, upToDownGesture[2]);
        input[3] = dtw_distance(sample.aX, downToUpGesture[0]);
        input[4] = dtw_distance(sample.aY, downToUpGesture[1]);
        input[5] = dtw_distance(sample.aZ, downToUpGesture[2]);
        input[6] = dtw_distance(sample.aX, ccwGesture[0]);
        input[7] = dtw_distance(sample.aY, ccwGesture[1]);
        input[8] = dtw_distance(sample.aZ, ccwGesture[2]);
        input[9] = dtw_distance(sample.aX, cwGesture[0]);
        input[10] = dtw_distance(sample.aY, cwGesture[1]);
        input[11] = dtw_distance(sample.aZ, cwGesture[2]);
        input[12] = dtw_distance(sample.aX, horizontalGesture[0]);
        input[13] = dtw_distance(sample.aY, horizontalGesture[1]);
        input[14] = dtw_distance(sample.aZ, horizontalGesture[2]);

        Serial.println("input = ");
        Serial.print("{");
        for(int i = 0; i < arrayMyKNN; i++)  {
          if (i!=arrayMyKNN-1){
            Serial.print(String(input[i])+", ");
          }
          else{
            Serial.print(String(input[i]));
          }
        }
        Serial.print("}");
        Serial.println("");

        int classification = myKNN.classify(input, kValue); // classify input with K=18
        float confidence = myKNN.confidence();

        // print the classification and confidence
        Serial.print("\tclassification = ");
        Serial.println(classification);

        // since there are 2 examples close to the input and K = 3,
        // expect the confidence to be: 2/3 = ~0.67
        Serial.print("\tconfidence     = ");
        Serial.println(confidence);

        gestureAnswer_2 = GESTURES_2[classification-1];
        Serial1.write("gestureAnswer_2");
        Serial.println(gestureAnswer_2 + " is send to the transmission line");

        gestureAnswer_2 = "";
      }
    }
    delay(50); // 200 for 10 sec, 100 for 5 sec
  }
}
//---------------------------------------------------------------------------------------------------------
void setup() {
  Serial.begin(serialRate);
  Serial1.begin(transmissionRate);

  // Check IMU
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

  //---------------------------------------------------------------------------------------------------------
  //set up model2
  for(int i = 0; i < (totalGesture*numberPerGesture)+addNumberPerGesture_1+addNumberPerGesture_2+addNumberPerGesture_3+addNumberPerGesture_4+addNumberPerGesture_5+addNumberPerGesture_6; i++){
    if (i<numberPerGesture+addNumberPerGesture_1){
      myKNN.addExample(knnExample[i], 1);
    }
    else if (i<numberPerGesture*2+addNumberPerGesture_1+addNumberPerGesture_2){
      myKNN.addExample(knnExample[i], 2);
    }
    else if (i<numberPerGesture*3+addNumberPerGesture_1+addNumberPerGesture_2+addNumberPerGesture_3){
      myKNN.addExample(knnExample[i], 3);
    }
    else if (i<numberPerGesture*4+addNumberPerGesture_1+addNumberPerGesture_2+addNumberPerGesture_3+addNumberPerGesture_4){
      myKNN.addExample(knnExample[i], 4);
     }
     else if (i<numberPerGesture*5+addNumberPerGesture_1+addNumberPerGesture_2+addNumberPerGesture_3+addNumberPerGesture_4+addNumberPerGesture_5){
      myKNN.addExample(knnExample[i], 5);
    }
    else if (i<numberPerGesture*6+addNumberPerGesture_1+addNumberPerGesture_2+addNumberPerGesture_3+addNumberPerGesture_4+addNumberPerGesture_5+addNumberPerGesture_6){
      myKNN.addExample(knnExample[i], 6);
    }
  }

  Serial.print("\tmyKNN.getCount() = ");
  Serial.println(myKNN.getCount());
  Serial.println();

  Serial.print("\tmyKNN.getCountByClass(1) = ");
  Serial.println(myKNN.getCountByClass(1));

  Serial.print("\tmyKNN.getCountByClass(2) = ");
  Serial.println(myKNN.getCountByClass(2));

  Serial.print("\tmyKNN.getCountByClass(3) = ");
  Serial.println(myKNN.getCountByClass(3));

  Serial.print("\tmyKNN.getCountByClass(4) = ");
  Serial.println(myKNN.getCountByClass(4));

  Serial.print("\tmyKNN.getCountByClass(5) = ");
  Serial.println(myKNN.getCountByClass(5));

  Serial.print("\tmyKNN.getCountByClass(6) = ");
  Serial.println(myKNN.getCountByClass(6));
//---------------------------------------------------------------------------------------------------------
  //set up model1
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
