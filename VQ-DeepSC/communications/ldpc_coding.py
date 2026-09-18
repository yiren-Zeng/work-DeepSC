import numpy as np
import tensorflow as tf


try:
    tf.config.set_visible_devices([], "GPU")
    print("[LDPC Info] TensorFlow GPU disabled; Sionna LDPC runs on CPU.")
except RuntimeError as error:
    print(f"[LDPC Warning] Could not hide TensorFlow GPUs: {error}")

from sionna.phy.fec.ldpc import LDPC5GDecoder, LDPC5GEncoder


def get_ldpc_code(block_length, rate=0.5):
    information_length = int(block_length)
    coded_length = int(information_length / rate)
    encoder = LDPC5GEncoder(k=information_length, n=coded_length)
    decoder = LDPC5GDecoder(encoder)
    return {
        "encoder": encoder,
        "decoder": decoder,
        "k": information_length,
        "n": coded_length,
        "rate": rate,
    }


def ldpc_encode(bits, code=None):
    if code is None:
        return np.asarray(bits)
    bits = np.asarray(bits).reshape(-1)
    information_length = int(code["k"])
    num_blocks = (len(bits) + information_length - 1) // information_length
    padded = np.pad(
        bits,
        (0, num_blocks * information_length - len(bits)),
        "constant",
    )
    encoded = code["encoder"](
        tf.constant(
            padded.reshape(num_blocks, information_length), dtype=tf.float32
        )
    )
    return encoded.numpy().reshape(-1)


def ldpc_decode(received_llr, code=None):
    received_llr = np.asarray(received_llr).reshape(-1)
    if code is None:
        return (received_llr < 0).astype(int)
    coded_length = int(code["n"])
    num_blocks = (len(received_llr) + coded_length - 1) // coded_length
    padded = np.pad(
        received_llr,
        (0, num_blocks * coded_length - len(received_llr)),
        "constant",
    )
    decoded = code["decoder"](
        tf.constant(padded.reshape(num_blocks, coded_length), dtype=tf.float32)
    )
    return decoded.numpy().reshape(-1).astype(int)
