"""Quick EventBus connectivity diagnostic"""
import websocket
import json
import time

ws = websocket.create_connection('ws://localhost:5555/eventbus', timeout=10)
print('Connected')

# Subscribe
ws.send(json.dumps({'type': 'SUBSCRIBE', 'event_types': ['L2_OUTPUT', 'L2_THINKING_STEP']}))
print('Subscribe sent')

# Send user text
msg = json.dumps({
    'type': 'PUBLISH',
    'event': {
        'type': 'USER_TEXT',
        'source': 'diag_test',
        'payload': {
            'text': 'hello',
            'session_id': 'diag_001',
            'request_id': 'diag_req_001',
            'confidence': 1.0,
        }
    }
})
ws.send(msg)
print('Published USER_TEXT')

# Wait for any responses
ws.settimeout(15)
for i in range(20):
    try:
        raw = ws.recv()
        data = json.loads(raw)
        evt_type = data.get('type', '')
        if evt_type == 'SUBSCRIBE':
            evt = data.get('event', {})
            inner = evt.get('type', '')
            src = evt.get('source', '')
            payload_keys = list(evt.get('payload', {}).keys())[:5]
            print(f'  EVENT: {inner} from={src} keys={payload_keys}')
        elif evt_type == 'ACK':
            ack_type = data.get('event_type', '')
            ack_msg = data.get('message', '')
            print(f'  ACK: {ack_type} - {ack_msg}')
        else:
            print(f'  OTHER: {evt_type} => {str(data)[:150]}')
    except Exception as e:
        print(f'  timeout/error: {e}')
        break

ws.close()
print('Done')
