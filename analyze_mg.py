import json, time

with open('data/memory_graph/memory_graph.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

nodes = data['nodes']
out = []

sessions = []
rounds = []
for nid, n in nodes.items():
    if n.get('node_type') != 'dialogue':
        continue
    sub = n.get('metadata', {}).get('sub_type', '')
    if sub == 'session':
        sessions.append(n)
    elif sub == 'round':
        rounds.append(n)

sessions.sort(key=lambda n: n.get('created_at', 0), reverse=True)
rounds.sort(key=lambda n: n.get('created_at', 0), reverse=True)

out.append('Nodes=%d, Sessions=%d, Rounds=%d' % (len(nodes), len(sessions), len(rounds)))
out.append('')
out.append('=== Latest 10 Sessions ===')
for s in sessions[:10]:
    ts = time.strftime('%H:%M:%S', time.localtime(s.get('created_at', 0)))
    sid = s['node_id']
    label = s.get('label', '')[:70]
    out.append('[%s] %s' % (ts, sid))
    out.append('  label: %s' % label)

out.append('')
out.append('=== Latest 15 Rounds ===')
for r in rounds[:15]:
    meta = r.get('metadata', {})
    ts = time.strftime('%H:%M:%S', time.localtime(r.get('created_at', 0)))
    rid = r['node_id']
    goal = meta.get('goal', '')[:70]
    sess = meta.get('session_id', '?')
    user = meta.get('user_text', '')[:60]
    bot = meta.get('bot_text', '')[:60]
    out.append('[%s] %s' % (ts, rid))
    out.append('  goal=%s' % goal)
    out.append('  session=%s' % sess)
    if user:
        out.append('  user=%s' % user)
    if bot:
        out.append('  bot=%s' % bot)
    out.append('')

out.append('=== Task Nodes ===')
tasks = [(nid, n) for nid, n in nodes.items() if n.get('node_type') == 'task']
tasks.sort(key=lambda x: x[1].get('created_at', 0), reverse=True)
out.append('Total task nodes: %d' % len(tasks))
for tid, t in tasks[:5]:
    status = t.get('metadata', {}).get('status', '?')
    ts = time.strftime('%H:%M:%S', time.localtime(t.get('created_at', 0)))
    label = t.get('label', '')[:60]
    out.append('[%s] %s [%s] %s' % (ts, tid, status, label))

with open('test_mg_analysis.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(out))
print('Done')
