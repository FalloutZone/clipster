use rubato::{
    Resampler, SincFixedIn, SincInterpolationType, SincInterpolationParameters, WindowFunction,
};
use std::error::Error;

pub fn resample_to_16khz(samples: &[f32], original_rate: u32) -> Result<Vec<f32>, Box<dyn Error>> {
    if original_rate == 16000 {
        return Ok(samples.to_vec());
    }

    let params = SincInterpolationParameters {
        sinc_len: 256,
        f_cutoff: 0.95,
        interpolation: SincInterpolationType::Linear,
        oversampling_factor: 256,
        window: WindowFunction::BlackmanHarris2,
    };

    let mut resampler = SincFixedIn::<f32>::new(
        16000 as f64 / original_rate as f64,
        2.0,
        params,
        samples.len(),
        1,
    )?;

    let waves_in = vec![samples.to_vec()];
    let waves_out = resampler.process(&waves_in, None)?;

    Ok(waves_out[0].clone())
}

pub fn normalize_audio(samples: &[f32]) -> Vec<f32> {
    let max_amplitude = samples
        .iter()
        .map(|&s| s.abs())
        .fold(0.0f32, f32::max);

    if max_amplitude > 0.0 {
        samples.iter().map(|&s| s / max_amplitude).collect()
    } else {
        samples.to_vec()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_audio_scales_to_one() {
        let samples = vec![0.5, -0.8, 0.3, -0.4];
        let normalized = normalize_audio(&samples);

        let max_amplitude = normalized.iter().map(|&s| s.abs()).fold(0.0f32, f32::max);
        assert!((max_amplitude - 1.0).abs() < 1e-6, "Max amplitude should be 1.0");
    }

    #[test]
    fn test_normalize_audio_preserves_ratios() {
        let samples = vec![0.2, 0.4, 0.6, 0.8];
        let normalized = normalize_audio(&samples);

        // Ratios should be preserved
        assert!((normalized[0] / normalized[1] - 0.5).abs() < 1e-6);
        assert!((normalized[1] / normalized[2] - 2.0/3.0).abs() < 1e-6);
        assert!((normalized[2] / normalized[3] - 0.75).abs() < 1e-6);
    }

    #[test]
    fn test_normalize_audio_handles_empty() {
        let samples: Vec<f32> = vec![];
        let normalized = normalize_audio(&samples);
        assert_eq!(normalized.len(), 0);
    }

    #[test]
    fn test_normalize_audio_handles_zeros() {
        let samples = vec![0.0, 0.0, 0.0];
        let normalized = normalize_audio(&samples);
        assert_eq!(normalized, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_normalize_audio_handles_negative_values() {
        let samples = vec![-0.5, -1.0, -0.25];
        let normalized = normalize_audio(&samples);

        let max_amplitude = normalized.iter().map(|&s| s.abs()).fold(0.0f32, f32::max);
        assert!((max_amplitude - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_resample_to_16khz_no_change_when_already_16khz() {
        let samples = vec![0.1, 0.2, 0.3, 0.4];
        let result = resample_to_16khz(&samples, 16000).unwrap();
        assert_eq!(result, samples);
    }

    #[test]
    fn test_resample_to_16khz_changes_sample_count() {
        let samples = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
        let result = resample_to_16khz(&samples, 48000).unwrap();

        // When downsampling from 48kHz to 16kHz (3:1 ratio), should have ~1/3 samples
        // Allow some margin due to resampling algorithm
        assert!(result.len() < samples.len());
    }
}