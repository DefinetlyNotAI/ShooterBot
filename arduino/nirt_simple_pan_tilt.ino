#include <Arduino.h>
#include <Servo.h>

Servo panServo;
Servo tiltServo;

const int PAN_PIN = 9;
const int TILT_PIN = 10;

const int PAN_CENTER = 90;
const int TILT_CENTER = 90;
const int PAN_MIN = 20;
const int PAN_MAX = 160;
const int TILT_MIN = 20;
const int TILT_MAX = 160;

String packetBuffer;
bool collecting = false;

float targetX = 0.5f;
float targetY = 0.5f;
int panAngle = PAN_CENTER;
int tiltAngle = TILT_CENTER;

void resetPacket() {
    packetBuffer = "";
    collecting = false;
}

float readFloatField(const String &payload, const char *key, float fallback) {
    String needle = String("\"") + key + "\"";
    int keyPos = payload.indexOf(needle);
    if (keyPos < 0) {
    return fallback;
}

    int colonPos = payload.indexOf(':', keyPos);
    if (colonPos < 0) {
    return fallback;
}

    int start = colonPos + 1;
    while (start < (int)payload.length() && (payload[start] == ' ' || payload[start] == '\t')) {
    start++;
}

    int end = start;
    while (end < (int)payload.length()) {
    char c = payload[end];
    if ((c >= '0' && c <= '9') || c == '.' || c == '-' || c == '+') {
    end++;
    continue;
}
    break;
}

    if (end <= start) {
    return fallback;
}

    return payload.substring(start, end).toFloat();
}

void applyTarget(float x, float y) {
    targetX = constrain(x, 0.0f, 1.0f);
    targetY = constrain(y, 0.0f, 1.0f);

    int desiredPan = map((int)(targetX * 1000.0f), 0, 1000, PAN_MIN, PAN_MAX);
    int desiredTilt = map((int)(targetY * 1000.0f), 0, 1000, TILT_MIN, TILT_MAX);

    panAngle = constrain(desiredPan, PAN_MIN, PAN_MAX);
    tiltAngle = constrain(desiredTilt, TILT_MIN, TILT_MAX);

    panServo.write(panAngle);
    tiltServo.write(tiltAngle);
}

void parsePacket(const String &payload) {
    // Normal packets contain only x and y. Extra advanced fields are ignored.
    float x = readFloatField(payload, "x", targetX);
    float y = readFloatField(payload, "y", targetY);

    if (payload.indexOf("\"nc\"") >= 0) {
    int start = payload.indexOf('[');
    int comma = payload.indexOf(',', start + 1);
    int end = payload.indexOf(']', comma + 1);
    if (start >= 0 && comma >= 0 && end >= 0) {
    x = payload.substring(start + 1, comma).toFloat();
    y = payload.substring(comma + 1, end).toFloat();
}
}

    applyTarget(x, y);

    Serial.print(F("{\"x\":"));
    Serial.print(targetX, 4);
    Serial.print(F(",\"y\":"));
    Serial.print(targetY, 4);
    Serial.println(F("}"));
}

void setup() {
    Serial.begin(115200);
    panServo.attach(PAN_PIN);
    tiltServo.attach(TILT_PIN);
    panServo.write(PAN_CENTER);
    tiltServo.write(TILT_CENTER);
    resetPacket();
}

void loop() {
    while (Serial.available() > 0) {
    char c = (char)Serial.read();

    if (!collecting) {
    if (c == '{') {
    collecting = true;
    packetBuffer = "{";
}
    continue;
}

    packetBuffer += c;

    if (c == '}') {
    parsePacket(packetBuffer);
    resetPacket();
}
}
}
