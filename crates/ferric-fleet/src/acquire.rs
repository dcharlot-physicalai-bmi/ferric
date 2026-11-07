//! Pinned SmolVLA base files. Small files were downloaded and hashed on 2026-11-07.
//! `model.safetensors` is the Hugging Face LFS oid at revision `5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6`.
//! This crate did not read that 906712520-byte file.

pub struct Artifact {
    pub path: &'static str,
    pub size: u64,
    pub sha256: &'static str,
    pub sha256_source: &'static str,
}

const REV: &str = "5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6";

pub fn artifacts() -> Vec<Artifact> {
    vec![
        Artifact {
            path: "config.json",
            size: 2299,
            sha256: "650584b56c104720f7a3c91d1ec6bec9e8de8ac11e60c92ba2fa82d93eda147d",
            sha256_source: "sha256 of bytes downloaded 2026-11-07",
        },
        Artifact {
            path: "policy_preprocessor.json",
            size: 1872,
            sha256: "7683d648280a72f0d51e869fb8dd1596beed90b7f84849330cf1bc26634a4bd6",
            sha256_source: "sha256 of bytes downloaded 2026-11-07",
        },
        Artifact {
            path: "policy_postprocessor.json",
            size: 660,
            sha256: "2b78bb742065288df2ec63b0ee35f97a1f6950171cf2272f7e34b1fe3873b17b",
            sha256_source: "sha256 of bytes downloaded 2026-11-07",
        },
        Artifact {
            path: "policy_preprocessor_step_5_normalizer_processor.safetensors",
            size: 640,
            sha256: "490ab239d96e263687c0b2e386a0afbc235a2eceb9857c36ed32f2f162a3e7c8",
            sha256_source: "sha256 of bytes downloaded 2026-11-07; matches the LFS oid",
        },
        Artifact {
            path: "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
            size: 640,
            sha256: "490ab239d96e263687c0b2e386a0afbc235a2eceb9857c36ed32f2f162a3e7c8",
            sha256_source: "sha256 of bytes downloaded 2026-11-07; same bytes as the preprocessor stats",
        },
        Artifact {
            path: "model.safetensors",
            size: 906_712_520,
            sha256: "7cd549ac2351fb069c0ddb3c34ad2d09cfc92b56a15dccdfc2e41467aaca01eb",
            sha256_source: "Hugging Face LFS oid at revision 5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6; full file not read here",
        },
    ]
}

pub fn url(path: &str) -> String {
    format!("https://huggingface.co/lerobot/smolvla_base/resolve/{REV}/{path}")
}

pub fn verify_bytes(path: &str, bytes: &[u8]) -> Result<&'static str, String> {
    let art = artifacts().into_iter().find(|a| a.path == path).ok_or_else(|| format!("unknown artifact {path}"))?;
    if bytes.len() as u64 != art.size {
        return Err(format!("{path}: size {} != pinned {}", bytes.len(), art.size));
    }
    let got = sha256_hex(bytes);
    if got != art.sha256 {
        return Err(format!("{path}: sha256 {got} != pinned {}", art.sha256));
    }
    Ok(art.sha256_source)
}

fn sha256_hex(data: &[u8]) -> String {
    // Compact SHA-256. Same published vectors as ferric-accept.
    hex(&Sha::new().feed(data).finish())
}

fn hex(bytes: &[u8]) -> String {
    const H: &[u8; 16] = b"0123456789abcdef";
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push(H[(b >> 4) as usize] as char);
        s.push(H[(b & 0xf) as usize] as char);
    }
    s
}

struct Sha {
    h: [u32; 8],
    buf: [u8; 64],
    n: usize,
    len: u64,
}

impl Sha {
    fn new() -> Self {
        Self {
            h: [0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19],
            buf: [0; 64],
            n: 0,
            len: 0,
        }
    }
    fn feed(mut self, mut data: &[u8]) -> Self {
        self.len += data.len() as u64;
        while !data.is_empty() {
            let take = (64 - self.n).min(data.len());
            self.buf[self.n..self.n + take].copy_from_slice(&data[..take]);
            self.n += take;
            data = &data[take..];
            if self.n == 64 {
                let block = self.buf;
                self.compress(&block);
                self.n = 0;
            }
        }
        self
    }
    fn compress(&mut self, block: &[u8; 64]) {
        const K: [u32; 64] = [
            0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
            0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
            0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
            0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
            0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
            0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
            0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
            0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
        ];
        let mut w = [0u32; 64];
        for i in 0..16 {
            w[i] = u32::from_be_bytes([block[4 * i], block[4 * i + 1], block[4 * i + 2], block[4 * i + 3]]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16].wrapping_add(s0).wrapping_add(w[i - 7]).wrapping_add(s1);
        }
        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = self.h;
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let t1 = h.wrapping_add(s1).wrapping_add(ch).wrapping_add(K[i]).wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let t2 = s0.wrapping_add(maj);
            h = g; g = f; f = e; e = d.wrapping_add(t1); d = c; c = b; b = a; a = t1.wrapping_add(t2);
        }
        for (i, v) in [a, b, c, d, e, f, g, h].into_iter().enumerate() {
            self.h[i] = self.h[i].wrapping_add(v);
        }
    }
    fn finish(mut self) -> [u8; 32] {
        let bits = self.len.wrapping_mul(8);
        self = self.feed(&[0x80]);
        while self.n != 56 {
            self = self.feed(&[0]);
        }
        self.buf[56..64].copy_from_slice(&bits.to_be_bytes());
        let block = self.buf;
        self.compress(&block);
        let mut out = [0u8; 32];
        for (i, v) in self.h.iter().enumerate() {
            out[4 * i..4 * i + 4].copy_from_slice(&v.to_be_bytes());
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::sha256_hex;

    #[test]
    fn sha256_matches_the_empty_and_abc_vectors() {
        assert_eq!(sha256_hex(b""), "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
        assert_eq!(sha256_hex(b"abc"), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    }
}
