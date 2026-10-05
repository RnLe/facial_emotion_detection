//! A model file (written by scripts/26_web.py): the network as a list of operations and
//! the tensors they use, run on one 48x48 grayscale face.
//!
//! Format, little endian: "FERM", u32 version, f32 mean and std of the input pixels,
//! u16 name length and name, u32 tensor count and per tensor u8 kind (0 f32, 1 int8 rows
//! with one f32 scale each), u8 rank, u32 dimensions, u32 data offset (int8: then the u32
//! offset of its scales), u32 operation count and per operation u8 code, u8 argument
//! count, u32 arguments; padding to 16 bytes, then the data.

use crate::ops::{self, Act};

const NONE: u32 = u32::MAX;

pub enum Tensor {
    F32 { data: Vec<f32>, shape: Vec<usize> },
    I8 { data: Vec<i8>, scale: Vec<f32>, shape: Vec<usize> },
}

impl Tensor {
    pub fn shape(&self) -> &[usize] {
        match self {
            Tensor::F32 { shape, .. } | Tensor::I8 { shape, .. } => shape,
        }
    }

    fn f32(&self) -> &[f32] {
        match self {
            Tensor::F32 { data, .. } => data,
            Tensor::I8 { .. } => panic!("expected an f32 tensor"),
        }
    }
}

/// A weight matrix of int8 rows, each with its own scale.
pub struct Rows<'a> {
    pub data: &'a [i8],
    pub scale: &'a [f32],
    pub rows: usize,
    pub cols: usize,
}

#[derive(Clone, Copy, Debug)]
pub enum Op {
    Conv { w: u32, b: u32, stride: usize, pad: usize, groups: usize },
    Affine { scale: u32, shift: u32 },
    Relu,
    Gelu,
    MaxPool { k: usize, stride: usize, pad: usize },
    AvgPool { k: usize },
    Gap,
    Flatten,
    Linear { w: u32, b: u32 },
    LayerNorm { w: u32, b: u32, eps: f32 },
    Save(usize),
    Load(usize),
    Add(usize),
    Cat(usize),
    Cls(u32),
    Pos(u32),
    Mha { in_w: u32, in_b: u32, out_w: u32, out_b: u32, heads: usize },
    Take0,
    SeqPool { w: u32, b: u32 },
}

pub struct Model {
    pub name: String,
    mean: f32,
    std: f32,
    tensors: Vec<Tensor>,
    ops: Vec<Op>,
}

struct Reader<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
        let end = self.at.checked_add(n).filter(|&e| e <= self.bytes.len()).ok_or("the model file ends early")?;
        let out = &self.bytes[self.at..end];
        self.at = end;
        Ok(out)
    }
    fn u8(&mut self) -> Result<u8, String> {
        Ok(self.take(1)?[0])
    }
    fn u16(&mut self) -> Result<u16, String> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }
    fn u32(&mut self) -> Result<u32, String> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn f32(&mut self) -> Result<f32, String> {
        Ok(f32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
}

fn f32s(bytes: &[u8], at: usize, n: usize) -> Result<Vec<f32>, String> {
    let raw = bytes.get(at..at + 4 * n).ok_or("a tensor lies outside the file")?;
    Ok(raw.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect())
}

impl Model {
    pub fn parse(bytes: &[u8]) -> Result<Model, String> {
        let mut r = Reader { bytes, at: 0 };
        if r.take(4)? != b"FERM" {
            return Err("not a model file".into());
        }
        if r.u32()? != 1 {
            return Err("unknown model file version".into());
        }
        let (mean, std) = (r.f32()?, r.f32()?);
        let n = r.u16()? as usize;
        let name = String::from_utf8(r.take(n)?.to_vec()).map_err(|_| "bad model name")?;
        let count = r.u32()? as usize;
        let mut table = Vec::with_capacity(count);
        for _ in 0..count {
            let kind = r.u8()?;
            let rank = r.u8()? as usize;
            let shape: Vec<usize> = (0..rank).map(|_| r.u32().map(|d| d as usize)).collect::<Result<_, _>>()?;
            let off = r.u32()? as usize;
            let soff = if kind == 1 { Some(r.u32()? as usize) } else { None };
            table.push((kind, shape, off, soff));
        }
        let nops = r.u32()? as usize;
        let mut ops = Vec::with_capacity(nops);
        for _ in 0..nops {
            let code = r.u8()?;
            let argc = r.u8()? as usize;
            let a: Vec<u32> = (0..argc).map(|_| r.u32()).collect::<Result<_, _>>()?;
            let u = |i: usize| a[i] as usize;
            ops.push(match code {
                0 => Op::Conv { w: a[0], b: a[1], stride: u(2), pad: u(3), groups: u(4) },
                1 => Op::Affine { scale: a[0], shift: a[1] },
                2 => Op::Relu,
                3 => Op::Gelu,
                4 => Op::MaxPool { k: u(0), stride: u(1), pad: u(2) },
                5 => Op::AvgPool { k: u(0) },
                6 => Op::Gap,
                7 => Op::Flatten,
                8 => Op::Linear { w: a[0], b: a[1] },
                9 => Op::LayerNorm { w: a[0], b: a[1], eps: f32::from_bits(a[2]) },
                10 => Op::Save(u(0)),
                11 => Op::Load(u(0)),
                12 => Op::Add(u(0)),
                13 => Op::Cat(u(0)),
                14 => Op::Cls(a[0]),
                15 => Op::Pos(a[0]),
                16 => Op::Mha { in_w: a[0], in_b: a[1], out_w: a[2], out_b: a[3], heads: u(4) },
                17 => Op::Take0,
                18 => Op::SeqPool { w: a[0], b: a[1] },
                _ => return Err(format!("unknown operation {code}")),
            });
        }
        let data = (r.at + 15) / 16 * 16;
        let mut tensors = Vec::with_capacity(count);
        for (kind, shape, off, soff) in table {
            let numel: usize = shape.iter().product();
            let at = data + off;
            tensors.push(if kind == 0 {
                Tensor::F32 { data: f32s(bytes, at, numel)?, shape }
            } else {
                let raw = bytes.get(at..at + numel).ok_or("a tensor lies outside the file")?;
                let q = raw.iter().map(|&b| b as i8).collect();
                Tensor::I8 { data: q, scale: f32s(bytes, data + soff.unwrap(), shape[0])?, shape }
            });
        }
        Ok(Model { name, mean, std, tensors, ops })
    }

    fn f(&self, k: u32) -> Option<&[f32]> {
        (k != NONE).then(|| self.tensors[k as usize].f32())
    }

    fn rows(&self, k: u32) -> Rows<'_> {
        match &self.tensors[k as usize] {
            Tensor::I8 { data, scale, shape } => Rows { data, scale, rows: shape[0], cols: data.len() / shape[0] },
            Tensor::F32 { .. } => panic!("expected an int8 tensor"),
        }
    }

    /// Logits for the seven classes of one 48x48 face (row by row, 0 black to 255 white).
    pub fn predict(&self, pixels: &[u8]) -> Result<Vec<f32>, String> {
        if pixels.len() != 48 * 48 {
            return Err(format!("expected 2304 pixels, got {}", pixels.len()));
        }
        let mut x = Act {
            c: 1,
            h: 48,
            w: 48,
            data: pixels.iter().map(|&p| (p as f32 / 255.0 - self.mean) / self.std).collect(),
        };
        let mut slots: Vec<Option<Act>> = vec![None, None];
        for &op in &self.ops {
            x = match op {
                Op::Conv { w, b, stride, pad, groups } => {
                    let shape = self.tensors[w as usize].shape();
                    ops::conv(&x, &self.rows(w), self.f(b), shape[2], shape[3], stride, pad, groups)
                }
                Op::Affine { scale, shift } => ops::affine(x, self.f(scale).unwrap(), self.f(shift)),
                Op::Relu => ops::relu(x),
                Op::Gelu => ops::gelu(x),
                Op::MaxPool { k, stride, pad } => ops::maxpool(&x, k, stride, pad),
                Op::AvgPool { k } => ops::avgpool(&x, k),
                Op::Gap => ops::gap(&x),
                Op::Flatten => Act { c: x.c * x.h * x.w, h: 1, w: 1, data: x.data },
                Op::Linear { w, b } => ops::linear(&x, &self.rows(w), self.f(b)),
                Op::LayerNorm { w, b, eps } => ops::layer_norm(x, self.f(w).unwrap(), self.f(b).unwrap(), eps),
                Op::Save(s) => {
                    slots[s] = Some(x.clone());
                    x
                }
                Op::Load(s) => slots[s].clone().ok_or("load from an empty slot")?,
                Op::Add(s) => ops::add(x, slots[s].as_ref().ok_or("add from an empty slot")?),
                Op::Cat(s) => {
                    let joined = ops::cat(slots[s].take().ok_or("cat into an empty slot")?, &x);
                    slots[s] = Some(joined.clone());
                    joined
                }
                Op::Cls(t) => ops::cls(&x, self.f(t).unwrap()),
                Op::Pos(t) => ops::pos(x, self.f(t).unwrap()),
                Op::Mha { in_w, in_b, out_w, out_b, heads } => {
                    ops::mha(&x, &self.rows(in_w), self.f(in_b).unwrap(), &self.rows(out_w), self.f(out_b).unwrap(), heads)
                }
                Op::Take0 => ops::take0(&x),
                Op::SeqPool { w, b } => ops::seq_pool(&x, self.f(w).unwrap(), self.f(b).unwrap()[0]),
            };
        }
        Ok(x.data)
    }
}
