# Arduino Example Sketch

This folder contains a minimal sketch for the current Python telemetry format.

## Files

- `nirt_simple_pan_tilt.ino` - simple JSON reader for `x` / `y` target data

## Notes

- The sketch expects the default minimal `{"x":0.52,"y":0.41}` packet described in `../docs/Arduino Protocol Implementation.md`.
- Set `serial.advanced_datapackets: true` only when extra metadata is needed; this sketch ignores those fields.
- For easiest testing, set `serial.crc: false` in the Python config.
- The parser looks for `{...}` objects in the serial stream, so it tolerates extra CRC bytes between packets.
