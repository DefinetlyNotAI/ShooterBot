# Arduino Protocol

The repository includes an example Arduino implementation at
`arduino/servo_controller.ino`. The Python side defines the serial contract
that compatible firmware must follow.

## What The Arduino Should Do

The Arduino firmware should:

- Open the serial port at the configured baud rate.
- Read incoming packets from the PC.
- Decode the target object data.
- Use the normalized coordinates to drive whatever hardware is attached, such as a pan/tilt mount, servo, or other
  motion system.
- Optionally send a short JSON reply back to the PC so the UI can show a live center marker.

## Outgoing Packet Format

The default Python side sends only the coordinates needed by a pan/tilt controller:

```json
{"x":0.52,"y":0.41}
```

Meaning:

- `x`, `y` - normalized target center in the current frame, from `0.0` to `1.0`

Set `serial.advanced_datapackets: true` when the receiver needs diagnostic metadata. Advanced packets add `id`, `class`,
`confidence`, and optionally `px`, `py` (predicted center).

## CRC Behavior

If `serial.crc: true`, the packet is not plain JSON on the wire. The code appends a 4-byte CRC32 value after the JSON
payload.

That means the Arduino firmware must do one of these:

- Disable CRC in the config while prototyping.
- Or read the JSON body first, then validate and strip the trailing CRC bytes before parsing.

If you want the simplest firmware parser, set `serial.crc: false` during development.

## Nonce Handshake

Before Python sends tracking telemetry to physical hardware, it sends a JSON
challenge containing a fresh random nonce:

```json
{"type":"nirt_handshake","nonce":"random-hex-value"}
```

The controller must reply before the two-second timeout with the exact same
nonce:

```json
{"type":"nirt_ready","nonce":"random-hex-value"}
```

Until that reply is received, Python refuses to send control packets. This
prevents telemetry from being sent to a stale, incompatible, or accidental
serial device. The nonce makes an old response unusable for a later connection.

It is a protocol identity check, not encryption or authentication against an
attacker who can read and write the physical serial link.

## Incoming Data

The PC side accepts JSON replies from the Arduino and uses them to update the UI center marker. It understands either of
these shapes:

```json
{"nc":[0.5,0.5]}
```

or

```json
{"x":0.5,"y":0.5}
```

This is mainly for simulation and feedback visualization.

### Shot / hit signal

When the currently locked target is hit, the Arduino can send a newline-delimited
`SHOT` message or one of these JSON messages:

```text
SHOT
```

```json
{"shot":true}
```

`HIT`, `TRIGGER`, `{"hit":true}`, and `{"event":"shot"}` are also accepted.

An optional `id` can identify the target explicitly, for example
`{"event":"shot","id":4}`.

Without an ID, the application marks the currently locked target as hit, removes it from the front of the queue, and
locks the next available target.

By default, the shot target is appended to the end, producing a P1 → P2 → P1 cycle. Set `tracking.cycle_remember: false`
to permanently exclude shot targets.

## Practical Firmware Logic

A simple firmware loop can:

1. Read a full JSON packet.
2. Parse `x` and `y`.
3. Compare the target position against the desired center.
4. Compute left/right or up/down correction.
5. Drive actuators with bounded speed or step size.
6. Send an acknowledgement or status packet if needed.

If you use the predicted point `px` and `py`, the controller can move slightly ahead of the target instead of reacting
only to the current frame.

## Example Sketch

A minimal Arduino sketch is available in [servo_controller.ino](../arduino/servo_controller.ino).
