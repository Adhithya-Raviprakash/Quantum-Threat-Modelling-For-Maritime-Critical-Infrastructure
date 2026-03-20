"""
Hybrid RSA + Post-Quantum Cryptography (ML-KEM / Kyber) Key Exchange Prototype
================================================================================
Project : Quantum Threat Modeling for Maritime Critical Infrastructure
Subject : CSE2021 – Cryptology  |  Slot: A1  |  VIT AP  |  March 2026
Authors : Nikhil (23BCB7059) & Adhithya Raviprakash (23BCB7141)

Description
-----------
This prototype demonstrates a Hybrid Key Exchange pattern that combines:
  • Classical RSA-2048 (OAEP/SHA-256)  — for backward compatibility
  • Simulated ML-KEM (Kyber-768)       — for quantum resistance (NIST FIPS 203)

The final shared secret is derived by XOR-combining both secrets and then
passing through HKDF-SHA256, so security holds as long as EITHER algorithm
is unbroken. This is the "Hybrid Package" pattern recommended for the
maritime PQC migration transition period.

Simulated ML-KEM
----------------
A production deployment would use the `oqs` (liboqs) library:
    pip install liboqs-python
    import oqs
    kem = oqs.KeyEncapsulation("Kyber768")

Since liboqs requires native C bindings, this file ships a software
simulation of ML-KEM's interface using AES-256-GCM + HKDF as a stand-in.
All API signatures match the real liboqs interface so swapping in the
genuine library requires only replacing the SimulatedKyber768 class.

Dependencies
------------
    pip install cryptography
"""

import os
import hashlib
import hmac
import struct
from typing import Tuple

from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.backends import default_backend


# ══════════════════════════════════════════════════════════════════════════════
# Simulated ML-KEM (Kyber-768) Interface
# Replace this class with `import oqs` + oqs.KeyEncapsulation("Kyber768")
# for a production / hardware deployment.
# ══════════════════════════════════════════════════════════════════════════════

class SimulatedKyber768:
    """
    Software simulation of ML-KEM (Kyber-768) key encapsulation.

    Interface mirrors liboqs Python bindings:
        generate_keypair()  → (public_key_bytes, secret_key_bytes)
        encap(public_key)   → (ciphertext_bytes, shared_secret_bytes)
        decap(ciphertext, secret_key) → shared_secret_bytes

    Security note: This is a SIMULATION for demonstration purposes.
    It uses AES-256-GCM + HKDF to mimic the KEM interface.
    Use the real liboqs library for any real deployment.
    """

    ALGORITHM_NAME    = "Kyber768 (simulated)"
    PUBLIC_KEY_SIZE   = 1184   # bytes — matches real Kyber-768
    SECRET_KEY_SIZE   = 2400   # bytes — matches real Kyber-768
    CIPHERTEXT_SIZE   = 1088   # bytes — matches real Kyber-768
    SHARED_SECRET_LEN = 32     # bytes

    @staticmethod
    def generate_keypair() -> Tuple[bytes, bytes]:
        """Generate a Kyber-768 keypair (public_key, secret_key)."""
        # Underlying secret: 32 random bytes as the true key material
        true_secret = os.urandom(32)
        # Pad to realistic Kyber-768 key sizes for interface fidelity
        public_key = _hkdf_expand(true_secret, b"kyber768-pk", SimulatedKyber768.PUBLIC_KEY_SIZE)
        # Secret key encodes the true secret at the front, padded to size
        secret_key = true_secret + _hkdf_expand(true_secret, b"kyber768-sk",
                                                  SimulatedKyber768.SECRET_KEY_SIZE - 32)
        return public_key, secret_key

    @staticmethod
    def encap(public_key: bytes) -> Tuple[bytes, bytes]:
        """
        Encapsulate a shared secret using the recipient's public key.
        Returns (ciphertext, shared_secret).
        """
        ephemeral_seed  = os.urandom(32)
        # Derive shared secret from public key + ephemeral seed
        shared_secret   = _hkdf_expand(
            public_key[:32] + ephemeral_seed,
            b"kyber768-ss", SimulatedKyber768.SHARED_SECRET_LEN
        )
        # Ciphertext = AESGCM-encrypt the ephemeral seed under a key derived from pk
        enc_key  = _hkdf_expand(public_key[:32], b"kyber768-enc", 32)
        aesgcm   = AESGCM(enc_key)
        nonce    = os.urandom(12)
        ct_core  = aesgcm.encrypt(nonce, ephemeral_seed, None)
        # Pad ciphertext to realistic Kyber-768 size
        ciphertext_raw = nonce + ct_core
        padding_needed = SimulatedKyber768.CIPHERTEXT_SIZE - len(ciphertext_raw)
        ciphertext = ciphertext_raw + _hkdf_expand(ephemeral_seed, b"ct-pad", padding_needed)
        return ciphertext, shared_secret

    @staticmethod
    def decap(ciphertext: bytes, secret_key: bytes) -> bytes:
        """
        Decapsulate the shared secret using the recipient's secret key.
        Returns shared_secret.
        """
        true_secret     = secret_key[:32]
        public_key_seed = _hkdf_expand(true_secret, b"kyber768-pk", SimulatedKyber768.PUBLIC_KEY_SIZE)
        enc_key         = _hkdf_expand(public_key_seed[:32], b"kyber768-enc", 32)
        aesgcm          = AESGCM(enc_key)
        nonce           = ciphertext[:12]
        ct_core         = ciphertext[12:12 + 16 + 32]   # nonce(12) + tag(16) + plaintext(32)
        ephemeral_seed  = aesgcm.decrypt(nonce, ct_core, None)
        shared_secret   = _hkdf_expand(
            public_key_seed[:32] + ephemeral_seed,
            b"kyber768-ss", SimulatedKyber768.SHARED_SECRET_LEN
        )
        return shared_secret


# ══════════════════════════════════════════════════════════════════════════════
# RSA-2048 Key Utilities
# ══════════════════════════════════════════════════════════════════════════════

def rsa_generate_keypair(key_size: int = 2048):
    """Generate an RSA keypair."""
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=key_size,
        backend=default_backend()
    )
    return private_key.public_key(), private_key


def rsa_encap(public_key) -> Tuple[bytes, bytes]:
    """
    RSA-OAEP encapsulation: generate a random 32-byte secret and
    encrypt it under the public key.
    Returns (ciphertext, shared_secret).
    """
    shared_secret = os.urandom(32)
    ciphertext    = public_key.encrypt(
        shared_secret,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None
        )
    )
    return ciphertext, shared_secret


def rsa_decap(ciphertext: bytes, private_key) -> bytes:
    """RSA-OAEP decapsulation. Returns shared_secret."""
    return private_key.decrypt(
        ciphertext,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None
        )
    )


# ══════════════════════════════════════════════════════════════════════════════
# Hybrid KEM Combiner
# ══════════════════════════════════════════════════════════════════════════════

def hybrid_combine(rsa_secret: bytes, kyber_secret: bytes,
                   context: bytes = b"maritime-hybrid-kem-v1") -> bytes:
    """
    Combine RSA and ML-KEM shared secrets into a single 32-byte session key.

    Method: XOR both 32-byte secrets, then feed through HKDF-SHA256 with
    a domain-separation context label. Security holds as long as EITHER
    component is unbroken.
    """
    xor_combined = bytes(a ^ b for a, b in zip(rsa_secret, kyber_secret))
    final_key = HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=None,
        info=context,
        backend=default_backend()
    ).derive(xor_combined)
    return final_key


# ══════════════════════════════════════════════════════════════════════════════
# Full Hybrid Key Exchange Protocol
# ══════════════════════════════════════════════════════════════════════════════

class HybridKeyExchange:
    """
    Full Hybrid RSA-2048 + ML-KEM (Kyber-768) key exchange protocol.

    Models a two-party exchange between a Ship (Initiator) and
    a Port Authority (Responder), as described in the project's
    SATCOM and Port OT threat mitigation sections.

    Usage:
        # On the Port Authority side (Responder):
        responder = HybridKeyExchange(role="responder")
        pub_keys  = responder.get_public_keys()           # send to ship

        # On the Ship side (Initiator):
        initiator = HybridKeyExchange(role="initiator")
        ciphertexts, ship_secret = initiator.encapsulate(pub_keys)  # send ciphertexts to port

        # Back on the Port Authority side:
        port_secret = responder.decapsulate(ciphertexts)

        # Both now share the same session key:
        assert ship_secret == port_secret
    """

    def __init__(self, role: str = "initiator"):
        assert role in ("initiator", "responder"), "role must be 'initiator' or 'responder'"
        self.role = role
        self._rsa_pub   = None
        self._rsa_priv  = None
        self._kyber_pub = None
        self._kyber_priv = None

        if role == "responder":
            self._setup_keys()

    def _setup_keys(self):
        """Generate both RSA and Kyber keypairs (Responder only)."""
        self._rsa_pub, self._rsa_priv      = rsa_generate_keypair()
        self._kyber_pub, self._kyber_priv  = SimulatedKyber768.generate_keypair()
        print(f"  [Responder] RSA-2048 keypair generated  "
              f"({len(self._rsa_pub.public_bytes(serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo))} bytes pub)")
        print(f"  [Responder] Kyber-768 keypair generated "
              f"(pub={len(self._kyber_pub)} bytes, sk={len(self._kyber_priv)} bytes)")

    def get_public_keys(self) -> dict:
        """Return serialized public keys for transmission to the Initiator."""
        assert self.role == "responder", "Only the Responder has keys to share"
        return {
            "rsa_public_key":   self._rsa_pub.public_bytes(
                                    serialization.Encoding.PEM,
                                    serialization.PublicFormat.SubjectPublicKeyInfo),
            "kyber_public_key": self._kyber_pub,
        }

    def encapsulate(self, pub_keys: dict) -> Tuple[dict, bytes]:
        """
        Initiator: encapsulate secrets under Responder's public keys.
        Returns (ciphertexts_dict, session_key).
        """
        assert self.role == "initiator", "Only the Initiator encapsulates"

        # Load RSA public key from PEM
        rsa_pub = serialization.load_pem_public_key(
            pub_keys["rsa_public_key"], backend=default_backend()
        )
        kyber_pub = pub_keys["kyber_public_key"]

        rsa_ct,   rsa_secret   = rsa_encap(rsa_pub)
        kyber_ct, kyber_secret = SimulatedKyber768.encap(kyber_pub)

        session_key = hybrid_combine(rsa_secret, kyber_secret)

        ciphertexts = {
            "rsa_ciphertext":   rsa_ct,
            "kyber_ciphertext": kyber_ct,
        }
        print(f"  [Initiator] RSA ciphertext:   {len(rsa_ct)} bytes")
        print(f"  [Initiator] Kyber ciphertext: {len(kyber_ct)} bytes")
        print(f"  [Initiator] Session key:      {session_key.hex()}")
        return ciphertexts, session_key

    def decapsulate(self, ciphertexts: dict) -> bytes:
        """
        Responder: decapsulate ciphertexts to recover the session key.
        Returns session_key.
        """
        assert self.role == "responder", "Only the Responder decapsulates"

        rsa_secret   = rsa_decap(ciphertexts["rsa_ciphertext"],   self._rsa_priv)
        kyber_secret = SimulatedKyber768.decap(ciphertexts["kyber_ciphertext"], self._kyber_priv)

        session_key = hybrid_combine(rsa_secret, kyber_secret)
        print(f"  [Responder] Session key:      {session_key.hex()}")
        return session_key


# ══════════════════════════════════════════════════════════════════════════════
# Authenticated Message Encryption (Post-Handshake)
# ══════════════════════════════════════════════════════════════════════════════

class SecureChannel:
    """
    AES-256-GCM authenticated encryption channel using a session key
    produced by HybridKeyExchange. Models encrypted AIS / SATCOM messages.
    """

    def __init__(self, session_key: bytes):
        assert len(session_key) == 32, "Session key must be 32 bytes (AES-256)"
        self._aesgcm = AESGCM(session_key)

    def encrypt(self, plaintext: bytes, associated_data: bytes = b"") -> bytes:
        """Encrypt + authenticate a message. Returns nonce || ciphertext."""
        nonce = os.urandom(12)
        ct    = self._aesgcm.encrypt(nonce, plaintext, associated_data or None)
        return nonce + ct

    def decrypt(self, payload: bytes, associated_data: bytes = b"") -> bytes:
        """Decrypt + verify a message. Raises on authentication failure."""
        nonce, ct = payload[:12], payload[12:]
        return self._aesgcm.decrypt(nonce, ct, associated_data or None)


# ══════════════════════════════════════════════════════════════════════════════
# Helper Functions
# ══════════════════════════════════════════════════════════════════════════════

def _hkdf_expand(key_material: bytes, info: bytes, length: int) -> bytes:
    """HKDF-expand helper used internally by the Kyber simulation."""
    return HKDF(
        algorithm=hashes.SHA256(),
        length=length,
        salt=None,
        info=info,
        backend=default_backend()
    ).derive(key_material)


# ══════════════════════════════════════════════════════════════════════════════
# Demo / Test
# ══════════════════════════════════════════════════════════════════════════════

def run_demo():
    """
    Full end-to-end demonstration simulating a Ship ↔ Port Authority
    hybrid key exchange followed by secure AIS message encryption.
    """
    SEP = "─" * 65

    print(SEP)
    print("  Hybrid RSA-2048 + ML-KEM (Kyber-768) Key Exchange Demo")
    print("  Maritime Context: Ship ↔ Port Authority Secure Channel")
    print(SEP)

    # ── Step 1: Port Authority generates keypairs ──────────────────────────
    print("\n[Step 1] Port Authority: Generating hybrid keypairs...")
    port = HybridKeyExchange(role="responder")
    pub_keys = port.get_public_keys()

    # ── Step 2: Ship encapsulates secrets ─────────────────────────────────
    print("\n[Step 2] Ship (Initiator): Encapsulating secrets...")
    ship = HybridKeyExchange(role="initiator")
    ciphertexts, ship_session_key = ship.encapsulate(pub_keys)

    # ── Step 3: Port Authority decapsulates ───────────────────────────────
    print("\n[Step 3] Port Authority (Responder): Decapsulating...")
    port_session_key = port.decapsulate(ciphertexts)

    # ── Step 4: Verify shared secret ─────────────────────────────────────
    print("\n[Step 4] Verifying shared session keys match...")
    match = hmac.compare_digest(ship_session_key, port_session_key)
    status = "✓ MATCH — Hybrid key exchange successful!" if match else "✗ MISMATCH — Key exchange failed!"
    print(f"  {status}")
    assert match, "Session keys do not match!"

    # ── Step 5: Encrypt a simulated AIS message ──────────────────────────
    print("\n[Step 5] Encrypting a simulated AIS broadcast message...")
    ship_channel = SecureChannel(ship_session_key)
    port_channel = SecureChannel(port_session_key)

    ais_message = (
        b"!AIVDM,1,1,,A,15M67N0Ph;G?Ui`E`FepT@3n00Sa,0*73"
        b"|MMSI:366913120|LAT:37.8199|LON:-122.4783"
        b"|SPEED:12.4kn|HEADING:045|STATUS:UnderWayUsingEngine"
    )
    aad = b"AIS-SATCOM-v1"   # Associated Authenticated Data

    encrypted = ship_channel.encrypt(ais_message, aad)
    print(f"  Plaintext  ({len(ais_message):4d} bytes): {ais_message[:60].decode()}...")
    print(f"  Encrypted  ({len(encrypted):4d} bytes): {encrypted.hex()[:60]}...")

    # ── Step 6: Decrypt and verify ────────────────────────────────────────
    print("\n[Step 6] Port Authority decrypting AIS message...")
    decrypted = port_channel.decrypt(encrypted, aad)
    print(f"  Decrypted  ({len(decrypted):4d} bytes): {decrypted[:60].decode()}...")
    assert decrypted == ais_message, "Decryption mismatch!"
    print("  ✓ Message integrity verified — AES-256-GCM authentication passed.")

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  Summary")
    print(SEP)
    print(f"  RSA-2048 ciphertext size  : {len(ciphertexts['rsa_ciphertext'])} bytes")
    print(f"  Kyber-768 ciphertext size : {len(ciphertexts['kyber_ciphertext'])} bytes")
    print(f"  Combined session key size : {len(ship_session_key) * 8} bits (AES-256)")
    print(f"  AIS payload (encrypted)   : {len(encrypted)} bytes")
    print(f"  Security model            : Hybrid (RSA ∧ Kyber) — safe if EITHER holds")
    print(f"  Quantum threat mitigated  : Shor's Algorithm (RSA), HNDL (Kyber forward secrecy)")
    print(SEP)

    return True


if __name__ == "__main__":
    run_demo()
