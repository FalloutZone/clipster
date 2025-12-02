mod ai;
mod audio_processing;
mod audio_recorder;

use arboard::Clipboard;
use global_hotkey::{
    GlobalHotKeyManager,
    GlobalHotKeyEvent,
    hotkey::{Code, Modifiers, HotKey}
};
use std::sync::Arc;
use tokio::sync::Mutex;
use std::collections::HashMap;

use crate::ai::local::whisper::WhisperModel;
use crate::ai::remote::anthropic::AnthropicAI;
use crate::ai::remote::openai::OpenAI;
use crate::ai::remote::traits::{Message, AI};
use crate::audio_recorder::AudioRecorder;

const BANNER: &'static str = "\
▄▖▜ ▘    ▗
▌ ▐ ▌▛▌▛▘▜▘█▌▛▘
▙▖▐▖▌▙▌▄▌▐▖▙▖▌
     ▌         ";

const SYSTEM_PROMPT: &str = "\
You are a code assistant.
ONLY respond with the requested code, command, or snippet.
NO explanations.
NO markdown (unless it was specifically asked for).
NO unnecessary quotes around response.
BE CONCISE and immediately usable.

Correct example:
User: \"Regex for email\"
Response: \"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}\"

Incorrect example:
User: \"Regex for email\"\
Response: \"```text
`^[^\\s@]+@[^\\s@]+\\.[^\\s@]+$
```

Or more comprehensive:

```text
^[a-zA-Z0-9.!#$%&'*+/=?^_`{|}~-]+@[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?(?: ...\"
```
";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum AIProvider {
    Anthropic,
    OpenAI,
    XAI,
}

impl AIProvider {
    fn name(&self) -> &str {
        match self {
            AIProvider::Anthropic => "Anthropic (Claude)",
            AIProvider::OpenAI => "OpenAI (GPT)",
            AIProvider::XAI => "xAI (Grok)",
        }
    }

    fn hotkey_combo(&self) -> (Option<Modifiers>, Code) {
        match self {
            AIProvider::Anthropic => (Some(Modifiers::CONTROL | Modifiers::SHIFT), Code::Space),
            AIProvider::OpenAI => (Some(Modifiers::CONTROL | Modifiers::ALT), Code::Space),
            AIProvider::XAI => (Some(Modifiers::CONTROL | Modifiers::SHIFT), Code::KeyX),
        }
    }

    fn hotkey_display(&self) -> &str {
        match self {
            AIProvider::Anthropic => "Ctrl+Shift+Space",
            AIProvider::OpenAI => "Ctrl+Alt+Space",
            AIProvider::XAI => "Ctrl+Shift+X",
        }
    }
}

struct AIConfig {
    provider: AIProvider,
    ai: Arc<Box<dyn AI>>,
}

fn setup_ais() -> Result<Vec<AIConfig>, Box<dyn std::error::Error>> {
    let mut configs = Vec::new();

    // Check Anthropic
    if std::env::var("ANTHROPIC_API_KEY").is_ok() {
        match AnthropicAI::from_env() {
            Ok(ai) => {
                configs.push(AIConfig {
                    provider: AIProvider::Anthropic,
                    ai: Arc::new(Box::new(ai.with_temperature(0.8).with_max_tokens(500))),
                });
            }
            Err(e) => eprintln!("Anthropic key found but failed to initialize: {}", e),
        }
    }

    // Check OpenAI
    if std::env::var("OPENAI_API_KEY").is_ok() {
        match OpenAI::openai_5() {
            Ok(ai) => {
                configs.push(AIConfig {
                    provider: AIProvider::OpenAI,
                    ai: Arc::new(Box::new(ai.with_temperature(0.8).with_max_tokens(500))),
                });
            }
            Err(e) => eprintln!("OpenAI key found but failed to initialize: {}", e),
        }
    }

    // Check xAI
    if std::env::var("XAI_API_KEY").is_ok() {
        match OpenAI::grok() {
            Ok(ai) => {
                configs.push(AIConfig {
                    provider: AIProvider::XAI,
                    ai: Arc::new(Box::new(ai.with_temperature(0.8).with_max_tokens(500))),
                });
            }
            Err(e) => eprintln!("xAI key found but failed to initialize: {}", e),
        }
    }

    if configs.is_empty() {
        return Err("No AI API keys found. Please set ANTHROPIC_API_KEY, OPENAI_API_KEY, or XAI_API_KEY".into());
    }

    Ok(configs)
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ai_configs = setup_ais()?;

    let recorder = Arc::new(Mutex::new(AudioRecorder::new()?));
    let whisper = Arc::new(Mutex::new(
        WhisperModel::new("models/ggml-tiny.en.bin")?
    ));
    let clipboard = Arc::new(Mutex::new(Clipboard::new()?));

    let hotkey_manager = GlobalHotKeyManager::new()?;
    let mut hotkey_map: HashMap<u32, AIConfig> = HashMap::new();

    println!("{}", BANNER.to_string());
    println!("Clipster AI Assistant Ready!\n");
    println!("Available AI providers:");

    for config in ai_configs {
        let (mods, code) = config.provider.hotkey_combo();
        let hotkey = HotKey::new(mods, code);

        hotkey_manager.register(hotkey)?;
        let hotkey_id = hotkey.id();

        println!("  {} - {}", config.provider.hotkey_display(), config.provider.name());

        hotkey_map.insert(hotkey_id, AIConfig {
            provider: config.provider,
            ai: config.ai.clone(),
        });
    }

    println!("\nHold hotkey to record, release to process");
    println!("Press Ctrl+C to exit\n");

    let receiver = GlobalHotKeyEvent::receiver();
    let mut active_recording: Option<(AIProvider, Arc<Box<dyn AI>>)> = None;

    loop {
        if let Ok(event) = receiver.recv() {
            if let Some(config) = hotkey_map.get(&event.id) {
                match event.state {
                    global_hotkey::HotKeyState::Pressed => {
                        println!("Recording for {}...", config.provider.name());
                        recorder.lock().await.start_recording()?;
                        active_recording = Some((config.provider, config.ai.clone()));
                    }
                    global_hotkey::HotKeyState::Released => {
                        if let Some((provider, ai)) = active_recording.take() {
                            println!("Processing with {}...", provider.name());

                            let samples = recorder.lock().await.stop_recording()?;
                            let sample_rate = recorder.lock().await.get_sample_rate();

                            if !samples.is_empty() {
                                // Resample and normalize
                                let resampled = audio_processing::resample_to_16khz(&samples, sample_rate)?;
                                let normalized = audio_processing::normalize_audio(&resampled);

                                // Transcribe
                                match whisper.lock().await.transcribe(&normalized) {
                                    Ok(text) => {
                                        println!("You said: {}", text);

                                        // Get AI response
                                        match get_ai_response(&ai, &text).await {
                                            Ok(response) => {
                                                // Copy to clipboard
                                                clipboard.lock().await.set_text(&response)?;
                                                println!("Copied to clipboard via {}!", provider.name());

                                                #[cfg(not(target_os = "windows"))]
                                                let _ = notify_rust::Notification::new()
                                                    .summary(&format!("AI Assistant ({})", provider.name()))
                                                    .body("Response copied! Ready to paste.")
                                                    .show();

                                                let preview = if response.len() > 100 {
                                                    format!("{}...", &response[..100])
                                                } else {
                                                    response.clone()
                                                };
                                                println!("Preview: {}\n", preview);
                                            }
                                            Err(e) => eprintln!("AI Error: {}", e),
                                        }
                                    }
                                    Err(e) => eprintln!("Transcription Error: {}", e),
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

async fn get_ai_response(
    ai: &Arc<Box<dyn AI>>,
    user_prompt: &str
) -> Result<String, Box<dyn std::error::Error>> {
    let messages = vec![
        Message {
            role: "system".to_string(),
            content: SYSTEM_PROMPT.to_string(),
        },
        Message {
            role: "user".to_string(),
            content: user_prompt.to_string(),
        },
    ];

    let response = ai.chat(messages).await?;

    let cleaned = clean_response(&response);

    Ok(cleaned)
}

fn clean_response(response: &str) -> String {
    let mut cleaned = response.trim();

    // Remove markdown code fences with language specifiers
    if cleaned.starts_with("```") {
        if let Some(newline_pos) = cleaned.find('\n') {
            cleaned = &cleaned[newline_pos + 1..];
        }
    }

    cleaned = cleaned.trim_end_matches("```").trim();

    cleaned.to_string()
}
