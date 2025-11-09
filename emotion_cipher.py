# emotion_cipher.py (transformer version)
# from __future__ import annotations
# import base64
# import base58 
# import os
# from dataclasses import dataclass
# from typing import Dict, List, Tuple

# from cryptography.hazmat.primitives.ciphers.aead import AESGCM
# from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
# from cryptography.hazmat.primitives import hashes



# # --- NEW: transformer imports ---
# import torch
# from transformers import AutoTokenizer, AutoModelForSequenceClassification

# # --------- Config ---------
# EMOTIONS = ["JOY", "SADNESS", "ANGER", "FEAR", "DISGUST", "SURPRISE"]
# EMO_TO_ID: Dict[str, int] = {e: i for i, e in enumerate(EMOTIONS)}
# ID_TO_EMO: Dict[int, str] = {i: e for e, i in EMO_TO_ID.items()}
# VERSION = 1
# PBKDF2_ITERS = 200_000
# SALT_LEN = 8           # 128-bit
# NONCE_LEN = 8         # 96-bit (AES-GCM recommended)
# KEY_LEN = 32            # 256-bit AES key

# # Fallback lexicon (only used if transformer init/inference fails)
# LEXICON = {
#     "happy": {"JOY": 1.0}, "ecstatic": {"JOY": 1.0}, "joy": {"JOY": 1.0},
#     "glad": {"JOY": 0.7}, "love": {"JOY": 0.8},
#     "sad": {"SADNESS": 1.0}, "unhappy": {"SADNESS": 0.9},
#     "disappointed": {"SADNESS": 0.9}, "down": {"SADNESS": 0.6},
#     "cry": {"SADNESS": 0.8},
#     "angry": {"ANGER": 1.0}, "mad": {"ANGER": 0.8},
#     "furious": {"ANGER": 1.0}, "frustrated": {"ANGER": 0.8},
#     "afraid": {"FEAR": 1.0}, "scared": {"FEAR": 1.0},
#     "anxious": {"FEAR": 0.9}, "worried": {"FEAR": 0.8}, "nervous": {"FEAR": 0.8},
#     "disgusted": {"DISGUST": 1.0}, "gross": {"DISGUST": 0.8}, "nasty": {"DISGUST": 0.7},
#     "surprised": {"SURPRISE": 1.0}, "shocked": {"SURPRISE": 1.0}, "amazed": {"SURPRISE": 0.8},
# }

# # Cosmetic base64 style mapping (unchanged)
# B64_URL  = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"
# STYLE64  = "9x@T!aZkP#13qW$VbCdDeEfFgGhHiIjJmNoOpPrRsStTuUvVwWxYyZz0245678LQ"
# STYLE_FWD  = str.maketrans(B64_URL, STYLE64)
# STYLE_BACK = str.maketrans(STYLE64, B64_URL)

# # --------- Crypto utils ---------
# def pbkdf2_key(passphrase: str, salt: bytes) -> bytes:
#     kdf = PBKDF2HMAC(
#         algorithm=hashes.SHA256(), length=KEY_LEN, salt=salt, iterations=PBKDF2_ITERS
#     )
#     return kdf.derive(passphrase.encode("utf-8"))

# def crc8(data: bytes) -> int:
#     poly = 0x07
#     crc = 0
#     for b in data:
#         crc ^= b
#         for _ in range(8):
#             if crc & 0x80:
#                 crc = ((crc << 1) ^ poly) & 0xFF
#             else:
#                 crc = (crc << 1) & 0xFF
#     return crc

# @dataclass
# class EmotionResult:
#     vector: Dict[str, float]            # e.g. {"JOY": 0.71, ...}
#     top2: List[Tuple[str, float]]       # [("JOY", 0.71), ("FEAR", 0.22)]

# # --------- Transformer setup ---------
# _TRANSFORMER_OK = False
# try:
#     _MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"
#     _tok = AutoTokenizer.from_pretrained(_MODEL_NAME)
#     _mdl = AutoModelForSequenceClassification.from_pretrained(_MODEL_NAME)
#     _mdl.eval()
#     _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
#     _mdl.to(_DEVICE)
#     # Model labels (includes NEUTRAL)
#     _id2label = _mdl.config.id2label
#     _label2id = {v: int(k) for k, v in _mdl.config.id2label.items()}
#     # Map model labels to our 6 emotions
#     # Model labels are typically: anger, disgust, fear, joy, neutral, sadness, surprise
#     _map_to_six = {
#         "joy": "JOY",
#         "sadness": "SADNESS",
#         "anger": "ANGER",
#         "fear": "FEAR",
#         "disgust": "DISGUST",
#         "surprise": "SURPRISE",
#         # "neutral" intentionally omitted
#     }
#     _TRANSFORMER_OK = True
# except Exception:
#     _TRANSFORMER_OK = False

# # --------- Emotion detectors ---------
# def _detect_emotions_lexicon(text: str) -> EmotionResult:
#     totals = {e: 0.0 for e in EMOTIONS}
#     words = [w.strip(".,!?;:\"'()[]").lower() for w in text.split()]
#     for w in words:
#         if w in LEXICON:
#             for emo, wgt in LEXICON[w].items():
#                 totals[emo] += wgt
#     s = sum(totals.values())
#     if s == 0:
#         totals["JOY"] = 1e-6; s = 1e-6
#     vector = {e: totals[e] / s for e in EMOTIONS}
#     top2 = sorted(vector.items(), key=lambda kv: kv[1], reverse=True)[:2]
#     return EmotionResult(vector=vector, top2=top2)

# def detect_emotions(text: str) -> EmotionResult:
#     # Prefer transformer; fall back to lexicon if init/inference fails
#     if not _TRANSFORMER_OK:
#         return _detect_emotions_lexicon(text)
#     try:
#         with torch.inference_mode():
#             inputs = _tok(text, return_tensors="pt", truncation=True, max_length=256)
#             inputs = {k: v.to(_DEVICE) for k, v in inputs.items()}
#             logits = _mdl(**inputs).logits[0]
#             probs = torch.softmax(logits, dim=-1).cpu().tolist()

#         # Build a dict of model_label -> prob
#         model_vec = { _id2label[i].lower(): probs[i] for i in range(len(probs)) }

#         # Map to our 6 emotions (drop neutral), then renormalize
#         vec6 = {e: 0.0 for e in EMOTIONS}
#         for mlabel, p in model_vec.items():
#             if mlabel in _map_to_six:
#                 vec6[_map_to_six[mlabel]] += float(p)

#         fear_words = ["anxious", "worried", "nervous", "afraid", "scared"]
#         text_lower = text.lower()
#         if any(w in text_lower for w in fear_words):
#             vec6["FEAR"] += 0.15  
#             s = sum(vec6.values())
#             vec6 = {e: v / s for e, v in vec6.items()}
#         s = sum(vec6.values())

#         if s == 0:
#             vec6 = {e: (1.0 / len(EMOTIONS)) for e in EMOTIONS}
#         else:
#             vec6 = {e: v / s for e, v in vec6.items()}

#         top2 = sorted(vec6.items(), key=lambda kv: kv[1], reverse=True)[:2]
#         return EmotionResult(vector=vec6, top2=top2)
#     except Exception:
#         return _detect_emotions_lexicon(text)

# # --------- Signature packing ---------
# def pack_signature(top2: List[Tuple[str, float]]) -> str:
#     (e1, p1), (e2, p2) = top2 if len(top2) == 2 else (top2[0], ("JOY", 0.0))
#     b = bytearray()
#     b.append(VERSION & 0xFF)
#     b.append(EMO_TO_ID[e1] & 0xFF)
#     b.append(max(0, min(255, int(round(p1 * 255)))))
#     b.append(EMO_TO_ID[e2] & 0xFF)
#     b.append(max(0, min(255, int(round(p2 * 255)))))
#     b.append(crc8(bytes(b)))
#     sig = base64.urlsafe_b64encode(bytes(b)).decode("ascii")
#     return sig

# def unpack_signature(sig: str) -> Tuple[List[Tuple[str, float]], bool]:
#     try:
#         raw = base64.urlsafe_b64decode(sig.encode("ascii"))
#         if len(raw) != 6:
#             return [("JOY", 0.5), ("SADNESS", 0.5)], False
#         ver, e1, p1, e2, p2, c = raw
#         ok = (c == crc8(raw[:5])) and (ver == VERSION)
#         e1n = ID_TO_EMO.get(e1, "JOY")
#         e2n = ID_TO_EMO.get(e2, "SADNESS")
#         return [(e1n, p1 / 255.0), (e2n, p2 / 255.0)], ok
#     except Exception:
#         return [("JOY", 0.5), ("SADNESS", 0.5)], False

# def interleave_signature(ctxt_styled: str, sig: str) -> str:
#     s = sig
#     return (
#         f"EC{s[:3]}{ctxt_styled[:len(ctxt_styled)//3]}"
#         f"EM{s[3:6]}{ctxt_styled[len(ctxt_styled)//3:2*len(ctxt_styled)//3]}"
#         f"O{s[6:]}{ctxt_styled[2*len(ctxt_styled)//3:]}"
#     )

# def extract_signature(token: str) -> Tuple[str, str]:
#     assert token.startswith("EC"), "Invalid token prefix"
#     t = token[2:]
#     a = t[:3]; t = t[3:]
#     i = t.find("EM"); 
#     if i < 0: raise ValueError("Missing EM marker")
#     chunk1, t = t[:i], t[i+2:]
#     b = t[:3]; t = t[3:]
#     j = t.find("O")
#     if j < 0: raise ValueError("Missing O marker")
#     chunk2, t = t[:j], t[j+1:]
#     c = t[:2]
#     chunk3 = t[2:]
#     sig = a + b + c
#     ctxt_styled = chunk1 + chunk2 + chunk3
#     return sig, ctxt_styled

# # --------- Public API ---------
# def encrypt_message(plaintext: str, passphrase: str) -> Dict[str, str]:
#     emo = detect_emotions(plaintext)
#     sig = pack_signature(emo.top2)

#     salt = os.urandom(SALT_LEN)
#     key = pbkdf2_key(passphrase, salt)
#     aes = AESGCM(key)
#     nonce = os.urandom(NONCE_LEN)
#     ct = aes.encrypt(nonce, plaintext.encode("utf-8"), None)  # ciphertext|tag

#     blob = salt + nonce + ct
#     b58 = base58.b58encode(blob).decode("ascii")
#     styled = b58.translate(STYLE_FWD)

#     token = interleave_signature(styled, sig)

#     return {
#         "encrypted_text": token,
#         "detected_emotions": emo.top2,
#     }

# def decrypt_message(token: str, passphrase: str) -> Dict[str, str]:
#     if not token.startswith("EC"):
#         raise ValueError("Malformed token: missing EC prefix")
    
#     sig, styled = extract_signature(token)
#     top2, ok = unpack_signature(sig)

#     b58 = styled.translate(STYLE_BACK)
#     blob = base58.b58decode(b58.encode("ascii"))

#     salt = blob[:SALT_LEN]
#     nonce = blob[SALT_LEN:SALT_LEN+NONCE_LEN]
#     ct = blob[SALT_LEN+NONCE_LEN:]

#     key = pbkdf2_key(passphrase, salt)
#     aes = AESGCM(key)
#     plaintext = aes.decrypt(nonce, ct, None).decode("utf-8")

#     return {
#         "original_message": plaintext,
#         "detected_emotions": top2,
#         "signature_ok": ok,
#     }

# # --------- Quick demo ---------
# if __name__ == "__main__":
#     msg1 = "Feeling ecstatic about joining the new AI research team, though a bit anxious about the deadlines ahead."
#     pw = "correct horse battery staple"

#     enc = encrypt_message(msg1, pw)
#     print("Encrypted Text:", enc["encrypted_text"])
#     print("Detected Emotions:", enc["detected_emotions"])

#     dec = decrypt_message(enc["encrypted_text"], pw)
#     print("Decrypted:", dec["original_message"])
#     print("Detected (from signature):", dec["detected_emotions"], "CRC OK:", dec["signature_ok"])

from __future__ import annotations
import base64
import base58 
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes

# --- NEW: transformer imports ---
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# --------- Config ---------
EMOTIONS = ["JOY", "SADNESS", "ANGER", "FEAR", "DISGUST", "SURPRISE"]
EMO_TO_ID: Dict[str, int] = {e: i for i, e in enumerate(EMOTIONS)}
ID_TO_EMO: Dict[int, str] = {i: e for e, i in EMO_TO_ID.items()}
VERSION = 1
PBKDF2_ITERS = 200_000
SALT_LEN = 8           # 64-bit salt
NONCE_LEN = 8          # 64-bit nonce (keeping original for compatibility)
KEY_LEN = 32           # 256-bit AES key

# Fallback lexicon (only used if transformer init/inference fails)
LEXICON = {
    "happy": {"JOY": 1.0}, "ecstatic": {"JOY": 1.0}, "joy": {"JOY": 1.0},
    "glad": {"JOY": 0.7}, "love": {"JOY": 0.8},
    "sad": {"SADNESS": 1.0}, "unhappy": {"SADNESS": 0.9},
    "disappointed": {"SADNESS": 0.9}, "down": {"SADNESS": 0.6},
    "cry": {"SADNESS": 0.8},
    "angry": {"ANGER": 1.0}, "mad": {"ANGER": 0.8},
    "furious": {"ANGER": 1.0}, "frustrated": {"ANGER": 0.8},
    "afraid": {"FEAR": 1.0}, "scared": {"FEAR": 1.0},
    "anxious": {"FEAR": 0.9}, "worried": {"FEAR": 0.8}, "nervous": {"FEAR": 0.8},
    "disgusted": {"DISGUST": 1.0}, "gross": {"DISGUST": 0.8}, "nasty": {"DISGUST": 0.7},
    "surprised": {"SURPRISE": 1.0}, "shocked": {"SURPRISE": 1.0}, "amazed": {"SURPRISE": 0.8},
}

# Character translation disabled - using plain Base58
# (The original STYLE64 had duplicate characters causing decryption failures)
B64_URL  = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"
STYLE64  = B64_URL  # No styling - use original Base58 alphabet
STYLE_FWD  = str.maketrans(B64_URL, STYLE64)
STYLE_BACK = str.maketrans(STYLE64, B64_URL)

# --------- Crypto utils ---------
def pbkdf2_key(passphrase: str, salt: bytes) -> bytes:
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(), length=KEY_LEN, salt=salt, iterations=PBKDF2_ITERS
    )
    return kdf.derive(passphrase.encode("utf-8"))

def crc8(data: bytes) -> int:
    poly = 0x07
    crc = 0
    for b in data:
        crc ^= b
        for _ in range(8):
            if crc & 0x80:
                crc = ((crc << 1) ^ poly) & 0xFF
            else:
                crc = (crc << 1) & 0xFF
    return crc

@dataclass
class EmotionResult:
    vector: Dict[str, float]
    top2: List[Tuple[str, float]]

# --------- Transformer setup ---------
_TRANSFORMER_OK = False
try:
    _MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"
    _tok = AutoTokenizer.from_pretrained(_MODEL_NAME)
    _mdl = AutoModelForSequenceClassification.from_pretrained(_MODEL_NAME)
    _mdl.eval()
    _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    _mdl.to(_DEVICE)
    _id2label = _mdl.config.id2label
    _label2id = {v: int(k) for k, v in _mdl.config.id2label.items()}
    _map_to_six = {
        "joy": "JOY",
        "sadness": "SADNESS",
        "anger": "ANGER",
        "fear": "FEAR",
        "disgust": "DISGUST",
        "surprise": "SURPRISE",
    }
    _TRANSFORMER_OK = True
except Exception:
    _TRANSFORMER_OK = False

# --------- Emotion detectors ---------
def _detect_emotions_lexicon(text: str) -> EmotionResult:
    totals = {e: 0.0 for e in EMOTIONS}
    words = [w.strip(".,!?;:\"'()[]").lower() for w in text.split()]
    for w in words:
        if w in LEXICON:
            for emo, wgt in LEXICON[w].items():
                totals[emo] += wgt
    s = sum(totals.values())
    if s == 0:
        totals["JOY"] = 1e-6; s = 1e-6
    vector = {e: totals[e] / s for e in EMOTIONS}
    top2 = sorted(vector.items(), key=lambda kv: kv[1], reverse=True)[:2]
    return EmotionResult(vector=vector, top2=top2)

def detect_emotions(text: str) -> EmotionResult:
    if not _TRANSFORMER_OK:
        return _detect_emotions_lexicon(text)
    try:
        with torch.inference_mode():
            inputs = _tok(text, return_tensors="pt", truncation=True, max_length=256)
            inputs = {k: v.to(_DEVICE) for k, v in inputs.items()}
            logits = _mdl(**inputs).logits[0]
            probs = torch.softmax(logits, dim=-1).cpu().tolist()

        model_vec = { _id2label[i].lower(): probs[i] for i in range(len(probs)) }
        vec6 = {e: 0.0 for e in EMOTIONS}
        for mlabel, p in model_vec.items():
            if mlabel in _map_to_six:
                vec6[_map_to_six[mlabel]] += float(p)

        fear_words = ["anxious", "worried", "nervous", "afraid", "scared"]
        text_lower = text.lower()
        if any(w in text_lower for w in fear_words):
            vec6["FEAR"] += 0.15  
            s = sum(vec6.values())
            vec6 = {e: v / s for e, v in vec6.items()}
        s = sum(vec6.values())

        if s == 0:
            vec6 = {e: (1.0 / len(EMOTIONS)) for e in EMOTIONS}
        else:
            vec6 = {e: v / s for e, v in vec6.items()}

        top2 = sorted(vec6.items(), key=lambda kv: kv[1], reverse=True)[:2]
        return EmotionResult(vector=vec6, top2=top2)
    except Exception:
        return _detect_emotions_lexicon(text)

# --------- Signature packing ---------
def pack_signature(top2: List[Tuple[str, float]]) -> str:
    (e1, p1), (e2, p2) = top2 if len(top2) == 2 else (top2[0], ("JOY", 0.0))
    b = bytearray()
    b.append(VERSION & 0xFF)
    b.append(EMO_TO_ID[e1] & 0xFF)
    b.append(max(0, min(255, int(round(p1 * 255)))))
    b.append(EMO_TO_ID[e2] & 0xFF)
    b.append(max(0, min(255, int(round(p2 * 255)))))
    b.append(crc8(bytes(b)))
    sig = base64.urlsafe_b64encode(bytes(b)).decode("ascii")
    return sig

def unpack_signature(sig: str) -> Tuple[List[Tuple[str, float]], bool]:
    try:
        raw = base64.urlsafe_b64decode(sig.encode("ascii"))
        if len(raw) != 6:
            return [("JOY", 0.5), ("SADNESS", 0.5)], False
        ver, e1, p1, e2, p2, c = raw
        ok = (c == crc8(raw[:5])) and (ver == VERSION)
        e1n = ID_TO_EMO.get(e1, "JOY")
        e2n = ID_TO_EMO.get(e2, "SADNESS")
        return [(e1n, p1 / 255.0), (e2n, p2 / 255.0)], ok
    except Exception:
        return [("JOY", 0.5), ("SADNESS", 0.5)], False

# FIXED: Simpler interleaving that's easier to reverse
def interleave_signature(ctxt_styled: str, sig: str) -> str:
    """
    Format: EC<sig><ctxt_styled>
    Much simpler - just prepend EC and signature
    """
    return f"EC{sig}{ctxt_styled}"

def extract_signature(token: str) -> Tuple[str, str]:
    """
    FIXED: Extract signature from simplified format
    Format: EC<8-char-sig><ctxt_styled>
    """
    if not token.startswith("EC"):
        raise ValueError("Invalid token prefix")
    
    # Remove "EC" prefix
    payload = token[2:]
    
    # Signature is always 8 characters (6 bytes base64-encoded = 8 chars)
    if len(payload) < 8:
        raise ValueError("Token too short")
    
    sig = payload[:8]
    ctxt_styled = payload[8:]
    
    return sig, ctxt_styled

# --------- Public API ---------
def encrypt_message(plaintext: str, passphrase: str) -> Dict[str, str]:
    emo = detect_emotions(plaintext)
    sig = pack_signature(emo.top2)

    salt = os.urandom(SALT_LEN)
    key = pbkdf2_key(passphrase, salt)
    aes = AESGCM(key)
    nonce = os.urandom(NONCE_LEN)
    ct = aes.encrypt(nonce, plaintext.encode("utf-8"), None)

    blob = salt + nonce + ct
    b58 = base58.b58encode(blob).decode("ascii")
    # REMOVED STYLING - Direct base58 without translation
    # styled = b58.translate(STYLE_FWD)
    styled = b58  # Use raw base58

    token = interleave_signature(styled, sig)

    return {
        "encrypted_text": token,
        "detected_emotions": emo.top2,
    }

def decrypt_message(token: str, passphrase: str) -> Dict[str, str]:
    if not token.startswith("EC"):
        raise ValueError("Malformed token: missing EC prefix")
    
    sig, styled = extract_signature(token)
    top2, ok = unpack_signature(sig)

    # REMOVED STYLING - Direct base58 without translation
    # b58 = styled.translate(STYLE_BACK)
    b58 = styled  # Use raw base58
    blob = base58.b58decode(b58.encode("ascii"))

    if len(blob) < SALT_LEN + NONCE_LEN:
        raise ValueError("Token too short - corrupted data")

    salt = blob[:SALT_LEN]
    nonce = blob[SALT_LEN:SALT_LEN+NONCE_LEN]
    ct = blob[SALT_LEN+NONCE_LEN:]

    key = pbkdf2_key(passphrase, salt)
    aes = AESGCM(key)
    plaintext = aes.decrypt(nonce, ct, None).decode("utf-8")

    return {
        "original_message": plaintext,
        "detected_emotions": top2,
        "signature_ok": ok,
    }

# --------- Quick demo with diagnostics ---------
if __name__ == "__main__":
    # First, test translation reversibility
    print("=" * 60)
    print("TRANSLATION TEST")
    print("=" * 60)
    test_str = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"
    translated = test_str.translate(STYLE_FWD)
    back = translated.translate(STYLE_BACK)
    print(f"Original:    {test_str}")
    print(f"Translated:  {translated}")
    print(f"Back:        {back}")
    print(f"Reversible:  {test_str == back}")
    
    if test_str != back:
        print("\n⚠ PROBLEM FOUND: Translation is NOT reversible!")
        print("Mismatches:")
        for i, (a, b) in enumerate(zip(test_str, back)):
            if a != b:
                print(f"  Position {i}: '{a}' -> translated -> '{b}'")
    else:
        print("✓ Translation table is valid\n")
    
    msg1 = "Feeling ecstatic about joining the new AI research team, though a bit anxious about the deadlines ahead."
    pw = "correct horse battery staple"

    print("=" * 60)
    print("ENCRYPTION TEST")
    print("=" * 60)
    enc = encrypt_message(msg1, pw)
    print("Original:", msg1)
    print("\nEncrypted Token:", enc["encrypted_text"])
    print("Token Length:", len(enc["encrypted_text"]))
    print("Detected Emotions:", enc["detected_emotions"])

    print("\n" + "=" * 60)
    print("DECRYPTION TEST")
    print("=" * 60)
    
    try:
        # Add debugging
        token = enc["encrypted_text"]
        print(f"Token starts with EC: {token.startswith('EC')}")
        print(f"Token length: {len(token)}")
        
        sig, styled = extract_signature(token)
        print(f"Signature: {sig} (len={len(sig)})")
        print(f"Styled ciphertext length: {len(styled)}")
        
        # Test translation
        b58 = styled.translate(STYLE_BACK)
        print(f"Base58 after translation: {b58[:50]}...")
        
        blob = base58.b58decode(b58.encode("ascii"))
        print(f"Blob length: {len(blob)} (expected >= {SALT_LEN + NONCE_LEN})")
        print(f"Salt length: {SALT_LEN}, Nonce length: {NONCE_LEN}")
        
        dec = decrypt_message(token, pw)
        print("\n✓ Decrypted:", dec["original_message"])
        print("✓ Detected (from signature):", dec["detected_emotions"])
        print("✓ Signature Valid:", dec["signature_ok"])
        print("\n✓✓ Match:", msg1 == dec["original_message"])
        
    except Exception as e:
        print(f"\n✗ ERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()