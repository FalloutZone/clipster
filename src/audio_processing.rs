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