//! Every model against the numpy reference interpreter in scripts/26_web.py (which matches
//! PyTorch): 16 test faces per model. Run `python scripts/26_web.py models` first.

use std::path::Path;

use fer_wasm::model::Model;

const MODELS: [&str; 7] = ["cnn", "vgg", "resnet", "densenet", "vit", "convnext", "cct"];

#[test]
fn matches_the_reference() {
    let web = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../runs/web");
    if !web.join("models").exists() {
        eprintln!("no exported models in {}, skipped", web.display());
        return;
    }
    for name in MODELS {
        let model = Model::parse(&std::fs::read(web.join(format!("models/{name}.ferm"))).unwrap()).unwrap();
        let vectors = std::fs::read(web.join(format!("vectors/{name}.bin"))).unwrap();
        let n = 16;
        let (pixels, logits) = vectors.split_at(n * 48 * 48);
        let expected: Vec<f32> = logits.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
        let mut worst = 0f32;
        for i in 0..n {
            let got = model.predict(&pixels[i * 2304..(i + 1) * 2304]).unwrap();
            for (g, e) in got.iter().zip(&expected[i * 7..(i + 1) * 7]) {
                worst = worst.max((g - e).abs());
            }
        }
        println!("{name}: largest logit difference {worst:.2e}");
        assert!(worst < 2e-3, "{name}: {worst}");
    }
}
