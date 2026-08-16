# Arduino Example Sketch

This folder contains a minimal sketch for the current Python telemetry format.

## Files

- `servo_controller.ino` - simple JSON reader for `x` / `y` target data

## Notes

- The sketch expects the default minimal `{"x":0.52,"y":0.41}` packet described in `../docs/Arduino Protocol Implementation.md`.
- Set `serial.advanced_datapackets: true` only when extra metadata is needed; this sketch ignores those fields.
- For easiest testing, set `serial.crc: false` in the Python config.
- The parser looks for `{...}` objects in the serial stream, so it tolerates extra CRC bytes between packets.
- Before telemetry is accepted, Python sends a nonce handshake and this sketch must reply with the matching `nirt_ready` JSON response.
