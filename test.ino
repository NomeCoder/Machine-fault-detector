#include <Wire.h>

#define MPU_ADDR 0x68
#define PWR_MGMT_1 0x6B
#define ACCEL_XOUT_H 0x3B

unsigned long lastSampleTime = 0;
const int sampleInterval = 5;   // 5ms ≈ 200 Hz

void setup() {
  Serial.begin(115200);
  Wire.begin();

  Wire.beginTransmission(MPU_ADDR);
  Wire.write(PWR_MGMT_1);
  Wire.write(0);
  Wire.endTransmission(true);

  delay(100);
}

void readMPU(float &ax, float &ay, float &az) {
  int16_t AccX, AccY, AccZ;

  Wire.beginTransmission(MPU_ADDR);
  Wire.write(ACCEL_XOUT_H);
  Wire.endTransmission(false);
  Wire.requestFrom(MPU_ADDR, 6, true);

  AccX = Wire.read() << 8 | Wire.read();
  AccY = Wire.read() << 8 | Wire.read();
  AccZ = Wire.read() << 8 | Wire.read();

  ax = AccX / 16384.0;
  ay = AccY / 16384.0;
  az = AccZ / 16384.0;
}

void loop() {

  if (millis() - lastSampleTime >= sampleInterval) {
    lastSampleTime = millis();

    float ax, ay, az;
    readMPU(ax, ay, az);

    Serial.print(ax, 5);
    Serial.print(",");
    Serial.print(ay, 5);
    Serial.print(",");
    Serial.println(az, 5);
  }
}