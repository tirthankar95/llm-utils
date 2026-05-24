# ──────────────────────────────────────────────
# 1. EXAMPLE DATA
#    Replace with DroidCollection / SecurityEval
#    labels: 1 = insecure, 0 = secure
# ──────────────────────────────────────────────

SECURE_EXAMPLES = [
    # SQL injection — safe (parameterised query)
    'def get_user(db, user_id):\n    return db.execute("SELECT * FROM users WHERE id = ?", (user_id,))',
    # Path traversal — safe
    'import os\ndef read_file(filename):\n    safe = os.path.basename(filename)\n    with open(os.path.join("/data", safe)) as f:\n        return f.read()',
    # XSS — safe (escaped output)
    'import html\ndef render(user_input):\n    return "<p>" + html.escape(user_input) + "</p>"',
    # Command injection — safe
    'import subprocess\ndef run_cmd(filename):\n    return subprocess.run(["cat", filename], capture_output=True)',
    # Hardcoded secret — safe
    'import os\ndef get_key():\n    return os.environ.get("API_KEY")',
    # Open redirect — safe
    'from urllib.parse import urlparse\ndef redirect(url):\n    if urlparse(url).netloc == "trusted.com":\n        return url\n    return "/"',
    # Integer overflow — safe
    'def safe_add(a: int, b: int) -> int:\n    result = a + b\n    if result > 2**31 - 1:\n        raise OverflowError\n    return result',
    # Insecure deserialisation — safe
    'import json\ndef load_data(s):\n    return json.loads(s)',
]

INSECURE_EXAMPLES = [
    # SQL injection — unsafe (string concat)
    'def get_user(db, user_id):\n    return db.execute("SELECT * FROM users WHERE id = " + user_id)',
    # Path traversal — unsafe
    'def read_file(filename):\n    with open("/data/" + filename) as f:\n        return f.read()',
    # XSS — unsafe (raw output)
    'def render(user_input):\n    return "<p>" + user_input + "</p>"',
    # Command injection — unsafe
    'import os\ndef run_cmd(filename):\n    return os.system("cat " + filename)',
    # Hardcoded secret — unsafe
    'def get_key():\n    return "sk-abc123secret"',
    # Open redirect — unsafe
    'def redirect(url):\n    return url',
    # Integer overflow — unsafe
    'def unsafe_add(a, b):\n    return a + b',
    # Insecure deserialisation — unsafe
    'import pickle\ndef load_data(s):\n    return pickle.loads(s)',
]