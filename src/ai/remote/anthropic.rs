use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::error::Error;

use crate::ai::remote::traits::{AI, Message};

#[derive(Debug, Clone)]
pub struct AnthropicAI {
    client: reqwest::Client,
    api_key: String,
    model: String,
    max_tokens: u32,
    temperature: f32,
}

#[derive(Debug, Serialize)]
struct AnthropicRequest {
    model: String,
    messages: Vec<AnthropicMessage>,
    max_tokens: u32,
    temperature: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct AnthropicMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Deserialize)]
struct AnthropicResponse {
    content: Vec<ContentBlock>,
}

#[derive(Debug, Deserialize)]
struct ContentBlock {
    text: String,
}

impl AnthropicAI {
    pub fn new(api_key: String, model: &str) -> Self {
        Self {
            client: reqwest::Client::new(),
            api_key,
            model: model.to_string(),
            max_tokens: 1000,
            temperature: 0.7,
        }
    }

    pub fn from_env() -> Result<Self, Box<dyn Error>> {
        let api_key = std::env::var("ANTHROPIC_API_KEY".to_string())?;
        Ok(Self::new(api_key, "claude-haiku-4-5-20251001"))
    }

    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = temperature.clamp(0.0, 1.0);
        self
    }

    #[allow(dead_code)]
    pub fn with_model(mut self, model: &str) -> Self {
        self.model = model.to_string();
        self
    }

    pub async fn chat_internal(
        &self,
        messages: Vec<AnthropicMessage>,
        system: Option<String>,
    ) -> Result<String, Box<dyn Error>> {
        let request = AnthropicRequest {
            model: self.model.clone(),
            messages,
            max_tokens: self.max_tokens,
            temperature: self.temperature,
            system,
        };

        let response = self
            .client
            .post("https://api.anthropic.com/v1/messages")
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await?;
            return Err(format!("Anthropic API error {}: {}", status, error_text).into());
        }

        let anthropic_response: AnthropicResponse = response.json().await?;

        anthropic_response
            .content
            .first()
            .map(|block| block.text.clone())
            .ok_or_else(|| "No response from Anthropic API".into())
    }
}

#[async_trait]
impl AI for AnthropicAI {
    async fn chat(&self, messages: Vec<Message>) -> Result<String, Box<dyn Error>> {
        let mut system_content: Option<String> = None;
        let mut anthropic_messages = Vec::new();

        for msg in messages {
            if msg.role == "system" {
                system_content = Some(msg.content);
            } else {
                anthropic_messages.push(AnthropicMessage {
                    role: msg.role,
                    content: msg.content,
                });
            }
        }

        self.chat_internal(anthropic_messages, system_content).await
    }
}
