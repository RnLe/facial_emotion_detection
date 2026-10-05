//! The operations, on one face at a time. Activations are channel-major, `data[c][y][x]`;
//! transformer tokens use the same layout with the tokens along x (`data[d][t]`), so a
//! linear layer on tokens and a 1x1 conv are the same matrix product.

use crate::model::Rows;

#[derive(Clone)]
pub struct Act {
    pub c: usize,
    pub h: usize,
    pub w: usize,
    pub data: Vec<f32>,
}

impl Act {
    fn n(&self) -> usize {
        self.h * self.w
    }
}

const PB: usize = 128;

/// out[o][j] = scale[o] * sum_k w[o][k] * x[k][j], for x with k rows of p values. Four
/// output rows at a time over blocks of PB values, so each row of x is read once per
/// four outputs and the accumulators stay in cache; the int8 weight is turned into a
/// float once per (row, k), outside the inner loop, which the compiler vectorises.
pub fn gemm(w: &Rows, x: &[f32], p: usize, out: &mut [f32]) {
    let (o, k) = (w.rows, w.cols);
    debug_assert_eq!(x.len(), k * p);
    if p == 1 {
        for r in 0..o {
            let row = &w.data[r * k..(r + 1) * k];
            let mut s = [0f32; 4];
            let mut chunks = row.chunks_exact(4).zip(x.chunks_exact(4));
            for (wc, xc) in &mut chunks {
                for i in 0..4 {
                    s[i] += wc[i] as f32 * xc[i];
                }
            }
            let tail: f32 = row[k / 4 * 4..].iter().zip(&x[k / 4 * 4..]).map(|(&a, &b)| a as f32 * b).sum();
            out[r] = (s[0] + s[1] + s[2] + s[3] + tail) * w.scale[r];
        }
        return;
    }
    let mut acc = vec![0f32; 4 * PB];
    for p0 in (0..p).step_by(PB) {
        let pn = PB.min(p - p0);
        let mut r = 0;
        while r < o {
            let nr = 4.min(o - r);
            acc.fill(0.0);
            {
                let (a0, rest) = acc.split_at_mut(PB);
                let (a1, rest) = rest.split_at_mut(PB);
                let (a2, a3) = rest.split_at_mut(PB);
                let (a0, a1, a2, a3) = (&mut a0[..pn], &mut a1[..pn], &mut a2[..pn], &mut a3[..pn]);
                let wv = |i: usize, kk: usize| if i < nr { w.data[(r + i) * k + kk] as f32 } else { 0.0 };
                for kk in 0..k {
                    let xr = &x[kk * p + p0..kk * p + p0 + pn];
                    let (w0, w1, w2, w3) = (wv(0, kk), wv(1, kk), wv(2, kk), wv(3, kk));
                    for ((((v, b0), b1), b2), b3) in xr.iter().zip(a0.iter_mut()).zip(a1.iter_mut()).zip(a2.iter_mut()).zip(a3.iter_mut()) {
                        *b0 += w0 * v;
                        *b1 += w1 * v;
                        *b2 += w2 * v;
                        *b3 += w3 * v;
                    }
                }
            }
            for i in 0..nr {
                let s = w.scale[r + i];
                let dst = &mut out[(r + i) * p + p0..(r + i) * p + p0 + pn];
                for (d, a) in dst.iter_mut().zip(&acc[i * PB..i * PB + pn]) {
                    *d = a * s;
                }
            }
            r += 4;
        }
    }
}

fn add_bias(out: &mut [f32], b: Option<&[f32]>, p: usize) {
    if let Some(b) = b {
        for (row, &bias) in out.chunks_exact_mut(p).zip(b) {
            for v in row {
                *v += bias;
            }
        }
    }
}

fn im2col(x: &Act, kh: usize, kw: usize, stride: usize, pad: usize, oh: usize, ow: usize) -> Vec<f32> {
    let mut cols = vec![0f32; x.c * kh * kw * oh * ow];
    for c in 0..x.c {
        for i in 0..kh {
            for j in 0..kw {
                let row = ((c * kh + i) * kw + j) * oh * ow;
                for oy in 0..oh {
                    let iy = (oy * stride + i) as isize - pad as isize;
                    if iy < 0 || iy >= x.h as isize {
                        continue;
                    }
                    let src = c * x.h * x.w + iy as usize * x.w;
                    let dst = row + oy * ow;
                    for ox in 0..ow {
                        let ix = (ox * stride + j) as isize - pad as isize;
                        if ix >= 0 && ix < x.w as isize {
                            cols[dst + ox] = x.data[src + ix as usize];
                        }
                    }
                }
            }
        }
    }
    cols
}

#[allow(clippy::too_many_arguments)]
pub fn conv(x: &Act, w: &Rows, b: Option<&[f32]>, kh: usize, kw: usize, stride: usize, pad: usize, groups: usize) -> Act {
    let oh = (x.h + 2 * pad - kh) / stride + 1;
    let ow = (x.w + 2 * pad - kw) / stride + 1;
    let p = oh * ow;
    let mut out = vec![0f32; w.rows * p];
    if groups == 1 {
        if kh == 1 && kw == 1 && stride == 1 && pad == 0 {
            gemm(w, &x.data, p, &mut out);
        } else {
            gemm(w, &im2col(x, kh, kw, stride, pad, oh, ow), p, &mut out);
        }
    } else {
        // depthwise: one filter per channel (ConvNeXt's 7x7 convs)
        assert!(groups == x.c && w.rows == x.c, "only plain and depthwise convs are supported");
        for c in 0..x.c {
            let f = &w.data[c * kh * kw..(c + 1) * kh * kw];
            let src = &x.data[c * x.h * x.w..(c + 1) * x.h * x.w];
            let dst = &mut out[c * p..(c + 1) * p];
            for oy in 0..oh {
                for ox in 0..ow {
                    let mut s = 0f32;
                    for i in 0..kh {
                        let iy = (oy * stride + i) as isize - pad as isize;
                        if iy < 0 || iy >= x.h as isize {
                            continue;
                        }
                        for j in 0..kw {
                            let ix = (ox * stride + j) as isize - pad as isize;
                            if ix >= 0 && ix < x.w as isize {
                                s += f[i * kw + j] as f32 * src[iy as usize * x.w + ix as usize];
                            }
                        }
                    }
                    dst[oy * ow + ox] = s * w.scale[c];
                }
            }
        }
    }
    add_bias(&mut out, b, p);
    Act { c: w.rows, h: oh, w: ow, data: out }
}

pub fn linear(x: &Act, w: &Rows, b: Option<&[f32]>) -> Act {
    let p = x.n();
    let mut out = vec![0f32; w.rows * p];
    gemm(w, &x.data, p, &mut out);
    add_bias(&mut out, b, p);
    Act { c: w.rows, h: x.h, w: x.w, data: out }
}

pub fn affine(mut x: Act, scale: &[f32], shift: Option<&[f32]>) -> Act {
    let n = x.n();
    for (c, row) in x.data.chunks_exact_mut(n).enumerate() {
        let (s, t) = (scale[c], shift.map_or(0.0, |sh| sh[c]));
        for v in row {
            *v = *v * s + t;
        }
    }
    x
}

pub fn relu(mut x: Act) -> Act {
    for v in &mut x.data {
        *v = v.max(0.0);
    }
    x
}

pub fn gelu(mut x: Act) -> Act {
    for v in &mut x.data {
        *v = 0.5 * *v * (1.0 + libm::erff(*v * std::f32::consts::FRAC_1_SQRT_2));
    }
    x
}

pub fn maxpool(x: &Act, k: usize, stride: usize, pad: usize) -> Act {
    let oh = (x.h + 2 * pad - k) / stride + 1;
    let ow = (x.w + 2 * pad - k) / stride + 1;
    let mut out = vec![f32::NEG_INFINITY; x.c * oh * ow];
    for c in 0..x.c {
        let src = &x.data[c * x.h * x.w..(c + 1) * x.h * x.w];
        for oy in 0..oh {
            for ox in 0..ow {
                let mut m = f32::NEG_INFINITY;
                for i in 0..k {
                    let iy = (oy * stride + i) as isize - pad as isize;
                    if iy < 0 || iy >= x.h as isize {
                        continue;
                    }
                    for j in 0..k {
                        let ix = (ox * stride + j) as isize - pad as isize;
                        if ix >= 0 && ix < x.w as isize {
                            m = m.max(src[iy as usize * x.w + ix as usize]);
                        }
                    }
                }
                out[c * oh * ow + oy * ow + ox] = m;
            }
        }
    }
    Act { c: x.c, h: oh, w: ow, data: out }
}

pub fn avgpool(x: &Act, k: usize) -> Act {
    let (oh, ow) = (x.h / k, x.w / k);
    let mut out = vec![0f32; x.c * oh * ow];
    let inv = 1.0 / (k * k) as f32;
    for c in 0..x.c {
        for oy in 0..oh {
            for ox in 0..ow {
                let mut s = 0f32;
                for i in 0..k {
                    for j in 0..k {
                        s += x.data[c * x.h * x.w + (oy * k + i) * x.w + ox * k + j];
                    }
                }
                out[c * oh * ow + oy * ow + ox] = s * inv;
            }
        }
    }
    Act { c: x.c, h: oh, w: ow, data: out }
}

pub fn gap(x: &Act) -> Act {
    let n = x.n();
    let data = x.data.chunks_exact(n).map(|row| row.iter().sum::<f32>() / n as f32).collect();
    Act { c: x.c, h: 1, w: 1, data }
}

/// LayerNorm over the channels, at every position (or token).
pub fn layer_norm(mut x: Act, w: &[f32], b: &[f32], eps: f32) -> Act {
    let (c, n) = (x.c, x.n());
    for j in 0..n {
        let mut mean = 0f32;
        for k in 0..c {
            mean += x.data[k * n + j];
        }
        mean /= c as f32;
        let mut var = 0f32;
        for k in 0..c {
            let d = x.data[k * n + j] - mean;
            var += d * d;
        }
        let inv = 1.0 / (var / c as f32 + eps).sqrt();
        for k in 0..c {
            let v = &mut x.data[k * n + j];
            *v = (*v - mean) * inv * w[k] + b[k];
        }
    }
    x
}

pub fn add(mut x: Act, y: &Act) -> Act {
    for (a, b) in x.data.iter_mut().zip(&y.data) {
        *a += b;
    }
    x
}

/// Channels of y appended to those of x (a DenseNet block's growing stack).
pub fn cat(mut x: Act, y: &Act) -> Act {
    x.data.extend_from_slice(&y.data);
    x.c += y.c;
    x
}

/// A class token prepended to the tokens.
pub fn cls(x: &Act, token: &[f32]) -> Act {
    let (c, t) = (x.c, x.n());
    let mut data = Vec::with_capacity(c * (t + 1));
    for k in 0..c {
        data.push(token[k]);
        data.extend_from_slice(&x.data[k * t..(k + 1) * t]);
    }
    Act { c, h: 1, w: t + 1, data }
}

pub fn pos(mut x: Act, p: &[f32]) -> Act {
    for (a, b) in x.data.iter_mut().zip(p) {
        *a += b;
    }
    let t = x.n();
    x.h = 1;
    x.w = t;
    x
}

fn softmax(v: &mut [f32]) {
    let m = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut s = 0f32;
    for x in v.iter_mut() {
        *x = (*x - m).exp();
        s += *x;
    }
    for x in v.iter_mut() {
        *x /= s;
    }
}

/// Multi-head self-attention as in torch.nn.MultiheadAttention (no masks, no dropout).
pub fn mha(x: &Act, in_w: &Rows, in_b: &[f32], out_w: &Rows, out_b: &[f32], heads: usize) -> Act {
    let (c, t) = (x.c, x.n());
    let dh = c / heads;
    let qkv = linear(x, in_w, Some(in_b));
    let scale = 1.0 / (dh as f32).sqrt();
    let mut out = vec![0f32; c * t];
    let mut q = vec![0f32; t * dh];
    let mut k = vec![0f32; t * dh];
    let mut s = vec![0f32; t];
    for hd in 0..heads {
        for d in 0..dh {
            for j in 0..t {
                q[j * dh + d] = qkv.data[(hd * dh + d) * t + j] * scale;
                k[j * dh + d] = qkv.data[(c + hd * dh + d) * t + j];
            }
        }
        for t1 in 0..t {
            let qr = &q[t1 * dh..(t1 + 1) * dh];
            for (t2, sv) in s.iter_mut().enumerate() {
                *sv = qr.iter().zip(&k[t2 * dh..(t2 + 1) * dh]).map(|(a, b)| a * b).sum();
            }
            softmax(&mut s);
            for d in 0..dh {
                let v = &qkv.data[(2 * c + hd * dh + d) * t..(2 * c + hd * dh + d + 1) * t];
                out[(hd * dh + d) * t + t1] = s.iter().zip(v).map(|(a, b)| a * b).sum();
            }
        }
    }
    linear(&Act { c, h: 1, w: t, data: out }, out_w, Some(out_b))
}

pub fn take0(x: &Act) -> Act {
    let t = x.n();
    Act { c: x.c, h: 1, w: 1, data: (0..x.c).map(|k| x.data[k * t]).collect() }
}

/// CCT's sequence pooling: a softmax weight per token from one linear map, then the
/// weighted average of the tokens.
pub fn seq_pool(x: &Act, w: &[f32], b: f32) -> Act {
    let (c, t) = (x.c, x.n());
    let mut z = vec![b; t];
    for k in 0..c {
        for (j, zj) in z.iter_mut().enumerate() {
            *zj += w[k] * x.data[k * t + j];
        }
    }
    softmax(&mut z);
    let data = (0..c).map(|k| x.data[k * t..(k + 1) * t].iter().zip(&z).map(|(a, b)| a * b).sum()).collect();
    Act { c, h: 1, w: 1, data }
}
