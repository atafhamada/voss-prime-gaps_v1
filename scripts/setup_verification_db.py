#!/usr/bin/env python3
"""تحميل قاعدة التحقق OEIS A006880"""
import os, json, urllib.request, datetime

DB_FILE  = os.path.join(os.path.dirname(__file__), '..', 'data', 'verification_db.json')
OEIS_URL = 'https://oeis.org/A006880/b006880.txt'

def main():
    if os.path.exists(DB_FILE):
        db = json.load(open(DB_FILE))
        if len(db.get('pi_of_10_pow_n', {})) >= 10:
            print("DB already exists at " + DB_FILE)
            return
    print('Downloading from ' + OEIS_URL)
    req = urllib.request.Request(OEIS_URL, headers={'User-Agent': 'VOSS/1.0'})
    raw = urllib.request.urlopen(req, timeout=30).read().decode('utf-8')
    pi_map = {}
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith('#'): continue
        p = line.split()
        if len(p) == 2:
            try: pi_map[str(int(p[0]))] = int(p[1])
            except ValueError: pass
    db = {
        'source': 'OEIS A006880',
        'url': OEIS_URL,
        'downloaded': datetime.datetime.now().isoformat(timespec='seconds'),
        'pi_of_10_pow_n': pi_map,
    }
    os.makedirs(os.path.dirname(DB_FILE), exist_ok=True)
    json.dump(db, open(DB_FILE, 'w'), indent=2)
    print('Saved ' + str(len(pi_map)) + ' entries')

if __name__ == '__main__':
    main()
