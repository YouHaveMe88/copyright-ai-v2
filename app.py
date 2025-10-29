from flask import Flask, render_template, request, jsonify
from urllib.request import urlopen
from bs4 import BeautifulSoup
import re, os, datetime, pytz
from dotenv import load_dotenv
import traceback
import itertools
import json, urllib.request
from urllib.error import HTTPError, URLError

# === OpenAI v1 client (untuk OpenRouter) ===
from openai import OpenAI

try:
    import requests
except Exception:
    requests = None

import json, urllib.request

def http_post_raw(url, payload, headers=None, timeout=90):
    """
    POST JSON → (status_code:int, body_bytes:bytes, headers:dict).
    Menangani 'requests' dan fallback 'urllib' + body error.
    """
    body = json.dumps(payload).encode("utf-8")
    headers = headers or {}

    if requests:
        try:
            resp = requests.post(url, headers=headers, data=body, timeout=timeout)
            return resp.status_code, resp.content, dict(resp.headers or {})
        except Exception as e:
            msg = f"[requests error] {e}".encode("utf-8", "ignore")
            return -1, msg, {}
    else:
        try:
            req = urllib.request.Request(url, data=body, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=timeout) as r:
                # r.headers adalah email.message.Message; cast ke dict
                hdrs = dict(r.headers.items()) if hasattr(r.headers, "items") else {}
                return getattr(r, "status", 200), r.read(), hdrs
        except HTTPError as e:
            try:
                err_bytes = e.read()
            except Exception:
                err_bytes = str(e).encode("utf-8", "ignore")
            return e.code, err_bytes, dict(getattr(e, "headers", {}) or {})
        except URLError as e:
            msg = f"[urllib error] {e}".encode("utf-8", "ignore")
            return -1, msg, {}

load_dotenv()

app = Flask(__name__)

# === UTILITAS ===
def clean_text(html):
    soup = BeautifulSoup(html, "html.parser")
    [s.decompose() for s in soup(["script", "style"])]
    text = re.sub(r"\s+", " ", soup.get_text(separator=" ")).strip()
    return text

def fetch_article(url):
    try:
        html = urlopen(url, timeout=10).read()
        text = clean_text(html)
        return text[:12000]
    except Exception as e:
        print("Error fetch_article:", e)
        return f"[Gagal ambil artikel: {e}]"

def format_text(text):
    text = re.sub(r'\s+', ' ', text).strip()
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    paragraphs, paragraph = [], ""
    for s in sentences:
        paragraph += s + " "
        if len(paragraph) > 200:
            paragraphs.append(paragraph.strip())
            paragraph = ""
    if paragraph:
        paragraphs.append(paragraph.strip())
    return "\n".join(f"<p>{p}</p>" for p in paragraphs)

def sanitize_model_id(m: str) -> str:
    """
    Rapikan ID model: trim dan buang tanda baca/whitespace di UJUNG (.,;:"'’ spasi).
    """
    return re.sub(r'[.\s,;:"\'’]+$', '', (m or '').strip())


# === UTIL MODEL ID (tambahkan di sini) ===
def normalize_model_id(m: str) -> str:
    """
    Ganti alias salah ketik/versi lama ke ID yang benar.
    """
    alias = {
        "qwen/qwen3-72b-instruct": "qwen/qwen2.5-72b-instruct",
        "qwen/qwen3-32b-instruct": "qwen/qwen2.5-32b-instruct",
        "qwen/qwen3-14b-instruct": "qwen/qwen2.5-14b-instruct",
        "qwen/qwen3-7b-instruct":  "qwen/qwen2.5-7b-instruct",
    }
    m = sanitize_model_id(m)
    return alias.get(m, m)

# === OpenRouter client factory ===
def get_or_client(api_key=None):
    site_url  = os.getenv("OPENROUTER_SITE_URL", "")
    site_name = os.getenv("OPENROUTER_SITE_NAME", "")
    default_headers = {}
    if site_url:  default_headers["HTTP-Referer"] = site_url
    if site_name: default_headers["X-Title"] = site_name

    key = api_key or os.getenv("OPENROUTER_API_KEY") or _CURRENT_KEY
    return OpenAI(base_url="https://openrouter.ai/api/v1",
                  api_key=key,
                  default_headers=default_headers or None)

def ai_describe_image(prompt_text: str, image_url: str, model: str):
    prompt_text = (prompt_text or "").strip() or "Jelaskan gambar ini dalam bahasa Indonesia."
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": prompt_text},
            {"type": "image_url", "image_url": {"url": image_url}}
        ]
    }]
    raw = or_chat(messages=messages, model=model, temperature=0.4, max_tokens=500)
    if isinstance(raw, str) and raw.startswith("__MODEL_ERROR__"):
        return raw
    return format_text(strip_markdown(raw))

def fallback_model_id(m: str) -> str:
    """
    Jika model tidak tersedia/invalid, jatuhkan bertahap.
    """
    chain = {
        "qwen/qwen2.5-72b-instruct": "qwen/qwen2.5-32b-instruct",
        "qwen/qwen2.5-32b-instruct": "qwen/qwen2.5-14b-instruct",
        "qwen/qwen2.5-14b-instruct": "qwen/qwen2.5-7b-instruct",
    }
    return chain.get(m, m)

def _load_key_pool():
    """
    Ambil daftar API key dari:
    - OPENROUTER_API_KEYS (dipisah koma), ATAU
    - OPENROUTER_API_KEY (single).
    Otomatis trimming & mengabaikan entri kosong.
    """
    keys = []
    raw_multi  = os.getenv("OPENROUTER_API_KEYS", "").strip()
    raw_single = os.getenv("OPENROUTER_API_KEY", "").strip()
    for raw in (raw_multi, raw_single):
        if not raw:
            continue
        keys.extend([k.strip() for k in raw.split(",") if k.strip()])
    return keys

# pool & pointer global untuk rotasi key
_KEY_POOL  = _load_key_pool() or [""]       # minimal 1 elemen agar tidak NameError
_KEY_CYCLE = itertools.cycle(_KEY_POOL)
_CURRENT_KEY = next(_KEY_CYCLE)

def _should_swap_key(exc: Exception) -> bool:
    """
    Tentukan apakah error layak memicu rotasi API key.
    Termasuk: 401 Unauthorized, 402 Payment/Quota, 429 Rate limit, 529 upstream busy.
    """
    s = str(exc).lower()
    return any(tok in s for tok in [
        " 401", "unauthorized",
        " 402", "payment", "insufficient", "quota",
        " 429", "rate limit",
        " 529"
    ])

# === Helper panggil chat.completions ===
def or_chat(messages, model, temperature=0.7, max_tokens=1000):
    """
    Panggil OpenRouter dengan:
      - normalisasi & sanitasi ID model
      - fallback 72B→32B→14B→7B jika 'not a valid model id' / 'no endpoints found'
      - rotasi API key otomatis jika 401/402/429/529
    Return: string (content) atau '__MODEL_ERROR__ ...'
    """
    global _CURRENT_KEY

    attempts_keys = max(1, len(_KEY_POOL))            # berapa banyak key yang bisa dicoba
    attempts_models = 4                               # ruang untuk fallback model
    last_err = None
    tried_models = set()

    # loop gabungan: model-fallback * key-rotation
    for _ in range(attempts_keys * attempts_models):
        m = normalize_model_id(model)
        if m in tried_models:
            # sudah dicoba model ini; agar tidak infinite loop,
            # rotasi key jika perlu atau break
            pass
        tried_models.add(m)

        client = get_or_client(api_key=_CURRENT_KEY)
        try:
            resp = client.chat.completions.create(
                model=m,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content.strip()

        except Exception as e:
            last_err = e
            s = str(e).lower()
            print(f"[OpenRouter] model={m} key=****{_CURRENT_KEY[-6:]} error:", repr(e))
            traceback.print_exc()

            # 1) Model tidak valid / tidak tersedia → coba fallback model
            if ("not a valid model id" in s) or ("no endpoints found" in s):
                nm = fallback_model_id(m)
                if nm != m:
                    print(f"[OpenRouter] fallback model → {nm}")
                    model = nm
                    # lanjut retry pakai model fallback (key tetap)
                    continue

            # 2) Kredensial/kuota/ratelimit/upstream → rotasi key dan coba lagi
            if _should_swap_key(e):
                # kalau hanya 1 key, tidak ada yang bisa diratakan
                if len(_KEY_POOL) > 1:
                    _CURRENT_KEY = next(_KEY_CYCLE)
                    print("[OpenRouter] switching API key → ****" + _CURRENT_KEY[-6:])
                    # model tetap sama, coba lagi dengan key baru
                    continue

            # 3) Error lain → keluar
            break

    return f"__MODEL_ERROR__ {last_err}"

def or_images(prompt: str, model: str = None, size: str = "1024x1024"):
    """
    Generate image via OpenRouter Images API dengan beberapa bentuk payload & fallback.
    - Tambahkan Accept: application/json
    - Coba /images, lalu /images/generations
    - Coba beberapa payload shape (prompt / input / messages)
    - Jika server mengirim image/* → b64
    - Opsi: fallback terakhir ke Pollinations (tanpa API key) bila tetap gagal
    """
    key = os.getenv("OPENROUTER_API_KEY") or _CURRENT_KEY
    if not key or not key.strip():
        return {"error": "__MODEL_ERROR__ Missing OPENROUTER_API_KEY"}

    model = normalize_model_id(model or os.getenv("DEFAULT_IMAGE_MODEL", "stability-ai/sdxl"))
    prompt = (prompt or "").strip()[:4000]
    size = (size or "1024x1024").strip()

    base_headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "Accept": "application/json",  # <— paksa JSON
    }

    if os.getenv("OPENROUTER_SITE_URL", ""):
        base_headers["HTTP-Referer"] = os.getenv("OPENROUTER_SITE_URL")
    if os.getenv("OPENROUTER_SITE_NAME", ""):
        base_headers["X-Title"] = os.getenv("OPENROUTER_SITE_NAME")

    # Beberapa stack menerapkan schema yang berbeda—kita coba beberapa
    payload_variants = [
        # 1) paling umum (OpenRouter docs)
        {"model": model, "prompt": prompt, "size": size},
        # 2) beberapa gateway pakai 'input'
        {"model": model, "input": prompt, "size": size},
        # 3) beberapa pakai messages multimodal-ish
        {"model": model, "messages": [{"role": "user", "content": [{"type": "input_text", "text": prompt}]}], "size": size},
    ]
    
    def parse_json(body_bytes: bytes):
        try:
            txt = body_bytes.decode("utf-8", "ignore")
            if not txt.strip():
                return None
            data = json.loads(txt)
        except Exception:
            return None
        arr = data.get("data") or []
        if arr:
            first = arr[0]
            if first.get("b64_json"):
                return {"b64": first["b64_json"]}
            if first.get("image"):  # beberapa provider
                return {"b64": first["image"]}
            if first.get("url"):
                return {"url": first["url"]}
        return None

    def try_endpoint(url, payload):
        status, body, hdrs = http_post_raw(url, payload, base_headers)
        ctype = (hdrs.get("Content-Type") or hdrs.get("content-type") or "").lower()

        # 1) image/* langsung → bungkus jadi b64
        if status == 200 and ctype.startswith("image/") and body:
            import base64
            return {"b64": base64.b64encode(body).decode("ascii")}

        # 2) JSON → parse
        if status == 200:
            parsed = parse_json(body)
            if parsed:
                return parsed

        # 3) simpan error ringkas untuk diagnosa
        head = (body[:120].decode("utf-8", "ignore") if isinstance(body, (bytes, bytearray)) else str(body)).replace("\n", " ")
        return {"_error": f"status={status}, ctype={ctype}, head={head!r}"}

    # Coba kombinasi endpoint × payload
    endpoints = [
        "https://openrouter.ai/api/v1/images",
        "https://openrouter.ai/api/v1/images/generations",
    ]

    errors = []
    for ep in endpoints:
        for pv in payload_variants:
            r = try_endpoint(ep, pv)
            if "b64" in r or "url" in r:
                return r
            errors.append(f"{ep} {r.get('_error')} payload_keys={list(pv.keys())}")

    # === OPSI FALLBACK BEBAS KEY (komentari jika tidak mau) ===
    # Supaya tombolmu tetap “hidup”, kita bisa fallback ke Pollinations.
    # Ini tidak butuh API key; hasilnya URL gambar (tidak b64).
    try:
        from urllib.parse import quote_plus
        polli = "https://image.pollinations.ai/prompt/" + quote_plus(prompt)
        return {"url": polli}
    except Exception:
        pass

    return {"error": "__MODEL_ERROR__ Image API failed. " + " | ".join(errors[:4])}

    
def build_image_prompt_from_news(text: str) -> str:
    base = strip_markdown(text or "")
    base = re.sub(r"\s+", " ", base).strip()
    base = base[:1500]  # jaga supaya tidak terlalu panjang
    guide = (
        "Buat satu ilustrasi editorial yang relevan dengan ringkasan berita berikut. "
        "Gaya: modern editorial, kontras jelas, mudah dibaca pada feed sosial, tanpa teks panjang di dalam gambar. "
        "Fokus pada visual metaforis yang kuat dan komposisi bersih.\n\nRingkasan:\n"
    )
    return guide + base

# === AI SUMMARY/REWRITE ===
def ai_rewrite(text, action, model):
    if action == "summarize":
        prompt = f"Ringkas berita berikut agar lebih singkat, jelas, dan tetap informatif:\n\n{text}"
    elif action == "rewrite":
        prompt = f"Tulis ulang berita berikut agar tetap bermakna sama, tetapi dengan gaya baru dan tidak terdeteksi sebagai salinan:\n\n{text}"
    else:
        prompt = f"Ringkas berita berikut agar lebih singkat, jelas, dan tetap informatif:\n\n{text}"

    raw = or_chat(messages=[{"role": "user", "content": prompt}], model=model)
    # <-- tambahkan guard error:
    if isinstance(raw, str) and raw.startswith("__MODEL_ERROR__"):
        return raw
    raw = strip_markdown(raw)
    return format_text(raw)

def strip_markdown(s: str) -> str:
    if not isinstance(s, str):
        return s
    # tebal/miring: **bold**, __bold__, *i*, _i_
    s = re.sub(r'(\*\*|__)(.*?)\1', r'\2', s)
    s = re.sub(r'(\*|_)(.*?)\1', r'\2', s)
    # inline code: `code`
    s = re.sub(r'`([^`]*)`', r'\1', s)
    # heading: # Title → Title
    s = re.sub(r'^\s{0,3}#{1,6}\s+', '', s, flags=re.MULTILINE)
    # link markdown: [text](url) → text
    s = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'\1', s)
    return s

def extract_titles(raw: str, max_items: int = 5):
    """
    Ambil beberapa judul dari teks 'raw' (apa pun formatnya),
    hilangkan bullet/nomor/koma, dedup, dan kembalikan list bersih.
    """
    if not isinstance(raw, str):
        return []

    s = strip_markdown(raw)

    # Normalisasi bullet/penomoran → newline
    s = re.sub(r'(?mi)^\s*\d+[\)\.\-:]\s*', '\n', s)  # "1) " / "1. "
    s = re.sub(r'[•*]\s*', '\n', s)                  # "• " / "* "
    s = re.sub(r'\s{2,}', ' ', s)                    # spasi ganda

    # Pecah baris
    parts = []
    for line in s.split('\n'):
        line = line.strip()
        if not line:
            continue
        # Kalau satu baris masih berisi banyak judul dipisah koma/garis miring → pecah lagi
        if ('; ' in line) or (' | ' in line) or (', ' in line and len(line) > 80):
            parts.extend([p.strip() for p in re.split(r';\s*|\|\s*|,\s*', line)])
        else:
            parts.append(line)

    # Bersihkan tanda baca di ujung, filter panjang wajar, dedup (preserve order)
    cleaned = []
    seen = set()
    for p in parts:
        p = p.strip(" ,;—–-")
        if 4 <= len(p) <= 120 and p.lower() not in seen:
            cleaned.append(p)
            seen.add(p.lower())

    return cleaned[:max_items]

# === GENERATE TITLE ===
def ai_generate_title(text, model):
    prompt = (
        "Buat 1 judul berita yang menarik, padat, maksimal 12 kata, "
        "dan hindari tanda kutip. Jika terpaksa membuat beberapa opsi, "
        "tulis SATU per baris tanpa nomor/bullet.\n\n"
        + text
    )
    client = get_or_client()
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.8,
            max_tokens=80,
        )
        raw = resp.choices[0].message.content.strip()

        # Coba ekstrak beberapa judul rapi
        titles = extract_titles(raw, max_items=5)
        if not titles:
            # fallback: satu string bersih
            title = strip_markdown(raw).replace('"', '').strip(" ,;—–-")
            return title

        # Jika hanya 1 → kembalikan string tunggal
        if len(titles) == 1:
            return titles[0]

        # Jika >1 → kembalikan HTML list berurutan
        items = "".join(f"<li>{t}</li>" for t in titles)
        return f"<ol>{items}</ol>"

    except Exception as e:
        print("OpenRouter error in ai_generate_title:", repr(e))
        traceback.print_exc()
        return f"__MODEL_ERROR__ {e}"

# === GENERATE HASHTAGS ===
def ai_generate_hashtags(text, include_global, model):
    prompt = f"""
Analisis teks berita berikut dan buat dua daftar tagar.

🎵 TikTok Hashtags: (maks 10)
📸 Instagram Hashtags: (maks 10)
{"🌍 Global Hashtags: (maks 5, opsional)" if include_global else ""}

Berita:
{text}
"""
    client = get_or_client()
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.9,
            max_tokens=500,
        )
        raw = strip_markdown(resp.choices[0].message.content.strip())
        raw = raw.replace("\n", "<br>")
        return f"<div style='margin-top:10px'>{raw}</div>"
    except Exception as e:
        print("OpenRouter error in ai_generate_hashtags:", repr(e))
        traceback.print_exc()
        return f"__MODEL_ERROR__ {e}"

# === ROUTES ===
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/process", methods=["POST"])
def process():
    data = request.get_json() or {} 
    text_input = data.get("text", "")
    url_input = data.get("url", "")
    image_url    = data.get("image_url", "")
    mode = data.get("mode", "AI")
    action = data.get("action", "summarize")
    include_global = data.get("include_global", True)

    requested_model = normalize_model_id(
        data.get("model") or os.getenv("DEFAULT_MODEL", "google/gemini-2.5-flash")
    )

    if action == "describe_image":
        if not image_url:
            return jsonify({"error": "Masukkan image_url untuk deskripsi gambar."}), 400
        
        result = ai_describe_image(text_input, image_url, requested_model)
        if isinstance(result, str) and result.startswith("__MODEL_ERROR__"):
            return jsonify({"error": result.replace("__MODEL_ERROR__", "Model error:").strip()}), 502
        return jsonify({"result": result})

    if url_input:
        text_input = fetch_article(url_input)

    if not text_input:
        return jsonify({"error": "Tidak ada teks atau URL yang valid."}), 400

    # --- action khusus ---
    if action == "generate_title":
        title_out = ai_generate_title(text_input, requested_model)
        if isinstance(title_out, str) and title_out.startswith("__MODEL_ERROR__"):
            return jsonify({"error": title_out.replace("__MODEL_ERROR__", "Model error:").strip()}), 502

        # Jika sudah <ol>…</ol> berarti multi-judul; kalau bukan, bungkus <h2>
        if isinstance(title_out, str) and title_out.lstrip().startswith("<ol>"):
            return jsonify({"result": title_out})
        else:
            return jsonify({"result": f"<h2>{title_out}</h2>"})

    if action == "generate_hashtags":
        tags = ai_generate_hashtags(text_input, include_global, requested_model)
        if isinstance(tags, str) and tags.startswith("__MODEL_ERROR__"):
            return jsonify({"error": tags.replace("__MODEL_ERROR__", "Model error:").strip()}), 502
        return jsonify({"result": tags})

    # --- ringkas/rewriter ---
    if mode == "AI":
        result = ai_rewrite(text_input, action, requested_model)
    else:
        result = format_text(text_input)

    if isinstance(result, str) and result.startswith("__MODEL_ERROR__"):
        return jsonify({"error": result.replace("__MODEL_ERROR__", "Model error:").strip()}), 502

    return jsonify({"result": result})

@app.route("/image", methods=["POST"])
def image():
    data = request.get_json() or {}
    url_input   = data.get("url", "") or ""
    text_input  = data.get("text", "") or ""
    img_prompt  = (data.get("image_prompt") or "").strip()
    size        = (data.get("size") or "1024x1024").strip()
    model_img   = normalize_model_id(data.get("image_model") or os.getenv("DEFAULT_IMAGE_MODEL", "stability-ai/sdxl"))

    extra = {}
    if data.get("steps") not in (None, "", "0"):
        try: extra["steps"] = int(data["steps"])
        except: pass
    if data.get("cfg_scale") not in (None, ""):
        try: extra["cfg_scale"] = float(data["cfg_scale"])
        except: pass
    if data.get("seed") not in (None, "", "0"):
        try: extra["seed"] = int(data["seed"])
        except: pass

    if img_prompt:
        final_prompt = img_prompt
    else:
        if url_input and not text_input:
            text_input = fetch_article(url_input)
        if not text_input or text_input.startswith("[Gagal ambil artikel"):
            return jsonify({"error": "Tidak ada sumber teks/URL yang valid untuk membuat gambar."}), 400
        final_prompt = build_image_prompt_from_news(text_input)

    out = or_images(final_prompt, model=model_img, size=size)
    if "error" in out:
        return jsonify({"error": out["error"].replace('__MODEL_ERROR__', 'Model error:')}), 502

    # Kirim balik sebagai data URL kalau b64, atau langsung URL kalau disediakan provider
    if "b64" in out:
        return jsonify({"data_url": "data:image/png;base64," + out["b64"]})
    else:
        return jsonify({"image_url": out["url"]})

# === Cache Harian untuk Jadwal AI (tetap) ===
_cache_day = None
_cache_schedule = None

def ai_schedule(model="google/gemini-2.5-flash"):
    global _cache_day, _cache_schedule
    today = datetime.date.today().isoformat()
    if _cache_day == today and _cache_schedule:
        return _cache_schedule

    tz = pytz.timezone("Asia/Jakarta")
    now = datetime.datetime.now(tz)
    day = now.strftime("%A")
    time_str = now.strftime("%H:%M")

    prompt = f"""
Hari ini {day}, jam {time_str} WIB.
Berdasarkan algoritme terbaru TikTok dan Instagram tahun 2025,
buat jadwal upload konten yang berpotensi FYP dan trending hari ini.

Pertimbangkan tren mingguan umum (weekend vs weekday).
Jelaskan alasan tiap waktu secara ringkas dan gunakan emoji.
"""
    client = get_or_client()
    res = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=500,
    )
    result = res.choices[0].message.content.strip()
    _cache_day = today
    _cache_schedule = result
    return result

@app.route("/get_schedule")
def get_schedule():
    # Kamu bisa ganti model khusus jadwal via ENV kalau mau:
    model = os.getenv("SCHEDULE_MODEL", "google/gemini-2.5-flash")
    data = {"schedule": ai_schedule(model)}
    return jsonify(data)

@app.route("/schedule")
def schedule_page():
    return render_template("schedule.html")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
