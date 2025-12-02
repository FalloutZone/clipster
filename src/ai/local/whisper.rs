// src/stt/whisper
use std::error::Error;
use whisper_rs::{WhisperContext, WhisperContextParameters, FullParams, SamplingStrategy};
use crate::println_raw;

pub struct WhisperModel {
    ctx: WhisperContext,
}

impl WhisperModel {
    pub fn new(model_path: &str) -> Result<Self, Box<dyn Error>> {
        println_raw!("Loading Whisper model from: {}", model_path);

        let ctx = WhisperContext::new_with_params(
            model_path,
            WhisperContextParameters::default()
        ).map_err(|e| format!("Failed to load model: {}", e))?;

        println_raw!("✓ Whisper model loaded");
        Ok(Self { ctx })
    }

    pub fn transcribe(&mut self, samples: &[f32]) -> Result<String, Box<dyn Error>> {
        if samples.is_empty() {
            return Ok(String::new());
        }

        //println_raw!("Processing {} samples ({:.2}s)", samples.len(), samples.len() as f32 / 16000.0);

        // Create parameters
        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });

        params.set_print_progress(false);
        params.set_print_special(false);
        params.set_print_realtime(false);
        params.set_print_timestamps(false);
        params.set_language(Some("en"));
        params.set_n_threads(4);
        params.set_translate(false);

        // Create state
        let mut state = self.ctx.create_state()
            .map_err(|e| format!("Failed to create state: {}", e))?;

        // Run transcription
        state.full(params, samples)
            .map_err(|e| format!("Transcription failed: {}", e))?;

        // Use the iterator to get all segments
        let mut transcription = String::new();
        for segment in state.as_iter() {
            transcription.push_str(&segment.to_str_lossy()?);
            transcription.push(' ');
        }

        let result = transcription.trim().to_string();

        if result.is_empty() {
            println_raw!("⚠ No transcription generated (silence detected)");
        } else {
            //println_raw!("✓ Transcription: {}", result);
        }

        Ok(result)
    }
}