//! The facial emotion study's seven networks in the browser: a small interpreter for the
//! model files written by scripts/26_web.py (a list of operations and int8 weights),
//! compiled to WebAssembly. Nothing leaves the page: a model is built from bytes the page
//! already holds and returns seven logits for a 48x48 face.

pub mod model;
pub mod ops;

use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct Network(model::Model);

#[wasm_bindgen]
impl Network {
    /// Builds a network from the bytes of its model file.
    #[wasm_bindgen(constructor)]
    pub fn new(bytes: &[u8]) -> Result<Network, JsError> {
        model::Model::parse(bytes).map(Network).map_err(|e| JsError::new(&e))
    }

    pub fn name(&self) -> String {
        self.0.name.clone()
    }

    /// Logits of the seven classes (angry, disgust, fear, happy, neutral, sad, surprise)
    /// for one 48x48 grayscale face, row by row.
    pub fn predict(&self, pixels: &[u8]) -> Result<Vec<f32>, JsError> {
        self.0.predict(pixels).map_err(|e| JsError::new(&e))
    }
}
