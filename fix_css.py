with open('static/style.css', 'rb') as f:
    lines = f.readlines()
clean = []
for l in lines:
    try:
        text = l.decode('utf-8')
        if '/* ══════ CANVAS PANEL ══════ */' in text:
            break
        clean.append(l)
    except:
        break
with open('static/style.css', 'wb') as f:
    f.writelines(clean)
