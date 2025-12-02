use std::error::Error;
use whisper_rs::{WhisperContext, WhisperContextParameters, FullParams, SamplingStrategy};

pub struct WhisperModel {
    ctx: WhisperContext,
}

impl WhisperModel {
    pub fn new(model_path: &str) -> Result<Self, Box<dyn Error>> {
        println!("Loading Whisper model from: {}", model_path);

        let ctx = WhisperContext::new_with_params(
            model_path,
            WhisperContextParameters::default()
        ).map_err(|e| format!("Failed to load model: {}", e))?;

        println!("Whisper model loaded");
        Ok(Self { ctx })
    }

    pub fn transcribe(&mut self, samples: &[f32]) -> Result<String, Box<dyn Error>> {
        if samples.is_empty() {
            return Ok(String::new());
        }

        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });

        params.set_print_progress(false);
        params.set_print_special(false);
        params.set_print_realtime(false);
        params.set_print_timestamps(false);
        params.set_language(Some("en"));
        params.set_n_threads(4);
        params.set_translate(false);

        let mut state = self.ctx.create_state()
            .map_err(|e| format!("Failed to create state: {}", e))?;

        // transcription
        state.full(params, samples)
            .map_err(|e| format!("Transcription failed: {}", e))?;

        // get all segments
        let mut transcription = String::new();
        for segment in state.as_iter() {
            transcription.push_str(&segment.to_str_lossy()?);
            transcription.push(' ');
        }

        let result = transcription.trim().to_string();

        if result.is_empty() {
            println!("No transcription generated (silence detected)");
        }

        Ok(result)
    }
}