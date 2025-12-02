use rodio::cpal::{self, traits::{DeviceTrait, HostTrait, StreamTrait}, Sample, SizedSample};
use std::error::Error;
use std::sync::{Arc, Mutex};

pub struct AudioRecorder {
    samples: Arc<Mutex<Vec<f32>>>,
    stream: Option<cpal::Stream>,
    sample_rate: u32,
}

impl AudioRecorder {
    pub fn new() -> Result<Self, Box<dyn Error>> {
        Ok(Self {
            samples: Arc::new(Mutex::new(Vec::new())),
            stream: None,
            sample_rate: 0,
        })
    }

    pub fn start_recording(&mut self) -> Result<(), Box<dyn Error>> {
        let host = cpal::default_host();
        let device = host
            .default_input_device()
            .ok_or("No input device available")?;

        let config = device.default_input_config()?;
        self.sample_rate = config.sample_rate().0;

        self.samples.lock().unwrap().clear();

        let samples = Arc::clone(&self.samples);
        let channels = config.channels() as usize;

        let stream = match config.sample_format() {
            cpal::SampleFormat::F32 => self.build_stream::<f32>(&device, &config.into(), samples, channels)?,
            cpal::SampleFormat::I16 => self.build_stream::<i16>(&device, &config.into(), samples, channels)?,
            cpal::SampleFormat::U16 => self.build_stream::<u16>(&device, &config.into(), samples, channels)?,
            _ => return Err("Unsupported sample format".into()),
        };

        stream.play()?;
        self.stream = Some(stream);

        Ok(())
    }

    fn build_stream<T>(
        &self,
        device: &cpal::Device,
        config: &cpal::StreamConfig,
        samples: Arc<Mutex<Vec<f32>>>,
        channels: usize,
    ) -> Result<cpal::Stream, Box<dyn Error>>
    where
        T: Sample + SizedSample,
        f32: cpal::FromSample<T>,
    {
        let err_fn = |err| eprintln!("Error in audio stream: {}", err);

        let stream = device.build_input_stream(
            config,
            move |data: &[T], _: &cpal::InputCallbackInfo| {
                let mut samples = samples.lock().unwrap();

                // Convert to mono by averaging channels
                for frame in data.chunks(channels) {
                    let mono_sample: f32 = frame
                        .iter()
                        .map(|&s| f32::from_sample(s))
                        .sum::<f32>()
                        / channels as f32;
                    samples.push(mono_sample);
                }
            },
            err_fn,
            None,
        )?;

        Ok(stream)
    }

    pub fn stop_recording(&mut self) -> Result<Vec<f32>, Box<dyn Error>> {
        if let Some(stream) = self.stream.take() {
            drop(stream);
        }

        let samples = self.samples.lock().unwrap().clone();

        Ok(samples)
    }

    pub fn get_sample_rate(&self) -> u32 {
        self.sample_rate
    }

    #[allow(dead_code)]
    pub fn is_recording(&self) -> bool {
        self.stream.is_some()
    }
}
