"""NumPy forward pass. The browser copy in src/lib/infer.js follows these loops."""

import numpy as np


def conv2d(x, w, b, stride, pad):
    c, height, width = x.shape
    cout, cin, kh, kw = w.shape
    hp = height + 2 * pad
    wp = width + 2 * pad
    xp = np.zeros((c, hp, wp), np.float32)
    xp[:, pad : pad + height, pad : pad + width] = x
    hout = (height + 2 * pad - kh) // stride + 1
    wout = (width + 2 * pad - kw) // stride + 1
    y = np.zeros((cout, hout, wout), np.float32)
    for oc in range(cout):
        for oy in range(hout):
            y0 = oy * stride
            for ox in range(wout):
                x0 = ox * stride
                patch = xp[:, y0 : y0 + kh, x0 : x0 + kw]
                y[oc, oy, ox] = b[oc] + np.sum(patch * w[oc])
    return y


def linear(x, w, b):
    return w @ x + b


def relu(x):
    return np.maximum(x, 0).astype(np.float32)


def up2(x):
    c, height, width = x.shape
    y = np.empty((c, height * 2, width * 2), np.float32)
    y[:, 0::2, 0::2] = x
    y[:, 0::2, 1::2] = x
    y[:, 1::2, 0::2] = x
    y[:, 1::2, 1::2] = x
    return y


def concat_channels(a, b):
    return np.concatenate([a, b], axis=0).astype(np.float32)


def decode(weights, h1, h2):
    up_h2 = up2(h2)
    joined = concat_channels(up_h2, h1)
    mid = relu(conv2d(joined, weights["up_w"], weights["up_b"], 1, 1))
    raw = conv2d(mid, weights["head_w"], weights["head_b"], 1, 0)
    prob = 1.0 / (1.0 + np.exp(-raw))
    return prob[0].astype(np.float32)


def forward(weights, x):
    c = relu(conv2d(x, weights["cls1_w"], weights["cls1_b"], 1, 1))
    c = relu(conv2d(c, weights["cls2_w"], weights["cls2_b"], 2, 1))
    c = relu(conv2d(c, weights["cls3_w"], weights["cls3_b"], 2, 1))
    flat = c.reshape(-1).astype(np.float32)
    emb = relu(linear(flat, weights["fc1_w"], weights["fc1_b"]))
    logits = linear(emb, weights["fc2_w"], weights["fc2_b"]).astype(np.float32)
    h1 = relu(conv2d(x, weights["seg1_w"], weights["seg1_b"], 1, 1))
    h2 = relu(conv2d(h1, weights["seg2_w"], weights["seg2_b"], 2, 1))
    mask = decode(weights, h1, h2)
    return logits, emb.astype(np.float32), mask
