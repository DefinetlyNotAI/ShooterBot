#include <Arduino.h>
#include <Servo.h>

// -----------------------------------------------------------------------------
// Hardware Configuration
// -----------------------------------------------------------------------------

namespace Hardware {
    constexpr uint8_t PAN_SERVO_PIN = 9;
    constexpr uint8_t TILT_SERVO_PIN = 10;
}

// -----------------------------------------------------------------------------
// Servo Configuration
// -----------------------------------------------------------------------------

namespace ServoConfig {
    constexpr int PAN_CENTER = 90;
    constexpr int TILT_CENTER = 90;

    constexpr int PAN_MIN = 20;
    constexpr int PAN_MAX = 160;

    constexpr int TILT_MIN = 20;
    constexpr int TILT_MAX = 160;
}

// -----------------------------------------------------------------------------
// Serial Configuration
// -----------------------------------------------------------------------------

namespace SerialConfig {
    constexpr unsigned long BAUD_RATE = 115200;
    constexpr char PACKET_START = '{';
    constexpr char PACKET_END = '}';
}

// -----------------------------------------------------------------------------
// State
// -----------------------------------------------------------------------------

Servo panServo;
Servo tiltServo;

String packetBuffer;

bool isCollectingPacket = false;

float targetX = 0.5f;
float targetY = 0.5f;

int panAngle = ServoConfig::PAN_CENTER;
int tiltAngle = ServoConfig::TILT_CENTER;

// -----------------------------------------------------------------------------
// Packet Handling
// -----------------------------------------------------------------------------

void resetPacketBuffer() {
    packetBuffer = "";
    isCollectingPacket = false;
}

float readFloatField(
const String& payload,
const char* key,
const float fallback
) {
    const String fieldName = String('"') + key + '"';

    const int keyPosition = payload.indexOf(fieldName);
    if (keyPosition < 0) {
    return fallback;
}

    const int colonPosition = payload.indexOf(':', keyPosition);
    if (colonPosition < 0) {
    return fallback;
}

    int start = colonPosition + 1;

    while (
    start < static_cast<int>(payload.length()) &&
    (payload[start] == ' ' || payload[start] == '\t')
    ) {
    ++start;
}

    int end = start;

    while (end < static_cast<int>(payload.length())) {
    const char character = payload[end];

    const bool isNumeric =
    (character >= '0' && character <= '9') ||
    character == '.' ||
    character == '-' ||
    character == '+';

    if (!isNumeric) {
    break;
}

    ++end;
}

    if (end <= start) {
    return fallback;
}

    return payload.substring(start, end).toFloat();
}

bool readNormalizedCoordinates(
const String& payload,
float& x,
float& y
) {
    const int arrayStart = payload.indexOf('[');
    if (arrayStart < 0) {
    return false;
}

    const int commaPosition = payload.indexOf(',', arrayStart + 1);
    if (commaPosition < 0) {
    return false;
}

    const int arrayEnd = payload.indexOf(']', commaPosition + 1);
    if (arrayEnd < 0) {
    return false;
}

    x = payload.substring(arrayStart + 1, commaPosition).toFloat();
    y = payload.substring(commaPosition + 1, arrayEnd).toFloat();

    return true;
}

// -----------------------------------------------------------------------------
// Servo Control
// -----------------------------------------------------------------------------

int normalizedToAngle(
const float value,
const int minimumAngle,
const int maximumAngle
) {
    const float normalized = constrain(value, 0.0f, 1.0f);

    return static_cast<int>(
    minimumAngle +
    normalized * static_cast<float>(maximumAngle - minimumAngle)
    );
}

void applyTarget(const float x, const float y) {
    targetX = constrain(x, 0.0f, 1.0f);
    targetY = constrain(y, 0.0f, 1.0f);

    panAngle = normalizedToAngle(
    targetX,
    ServoConfig::PAN_MIN,
    ServoConfig::PAN_MAX
    );

    tiltAngle = normalizedToAngle(
    targetY,
    ServoConfig::TILT_MIN,
    ServoConfig::TILT_MAX
    );

    panServo.write(panAngle);
    tiltServo.write(tiltAngle);
}

// -----------------------------------------------------------------------------
// Communication
// -----------------------------------------------------------------------------

void sendTargetTelemetry() {
    Serial.print(F("{\"x\":"));
    Serial.print(targetX, 4);

    Serial.print(F(",\"y\":"));
    Serial.print(targetY, 4);

    Serial.println(F("}"));
}

void parsePacket(const String& payload) {
    const String handshakeType = F("\"type\":\"nirt_handshake\"");
    if (payload.indexOf(handshakeType) >= 0) {
    const String nonceKey = F("\"nonce\":\"");
    const int nonceStart = payload.indexOf(nonceKey);
    if (nonceStart >= 0) {
    const int valueStart = nonceStart + nonceKey.length();
    const int valueEnd = payload.indexOf('"', valueStart);
    if (valueEnd > valueStart) {
    Serial.print(F("{\"type\":\"nirt_ready\",\"nonce\":\""));
    Serial.print(payload.substring(valueStart, valueEnd));
    Serial.println(F("\"}"));
}
}
    return;
}
    float x = readFloatField(payload, "x", targetX);
    float y = readFloatField(payload, "y", targetY);

    // Legacy format:
    // {"nc":[0.5,0.5]}
    if (payload.indexOf(F("\"nc\"")) >= 0) {
    readNormalizedCoordinates(payload, x, y);
}

    applyTarget(x, y);
    sendTargetTelemetry();
}

void processSerialInput() {
    while (Serial.available() > 0) {
    const char character = static_cast<char>(Serial.read());

    if (!isCollectingPacket) {
    if (character == SerialConfig::PACKET_START) {
    isCollectingPacket = true;
    packetBuffer = SerialConfig::PACKET_START;
}

    continue;
}

    packetBuffer += character;

    if (character == SerialConfig::PACKET_END) {
    parsePacket(packetBuffer);
    resetPacketBuffer();
}
}
}

// -----------------------------------------------------------------------------
// Arduino Entry Points
// -----------------------------------------------------------------------------

void setup() {
    Serial.begin(SerialConfig::BAUD_RATE);

    panServo.attach(Hardware::PAN_SERVO_PIN);
    tiltServo.attach(Hardware::TILT_SERVO_PIN);

    panServo.write(ServoConfig::PAN_CENTER);
    tiltServo.write(ServoConfig::TILT_CENTER);

    resetPacketBuffer();
}

void loop() {
    processSerialInput();
}
