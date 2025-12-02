use crate::ai_trait::{AI, Message};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::error::Error;

#[derive(Debug, Clone)]
pub struct OpenAI {
    client: reqwest::Client,
    api_key: String,
    base_url: String,
    model: String,
    max_tokens: u32,
    temperature: f32,
}

#[derive(Debug, Serialize)]
struct ChatRequest {
    model: String,
    messages: Vec<Message>,
    max_tokens: u32,
    temperature: f32,
}

#[derive(Debug, Deserialize)]
struct ChatResponse {
    choices: Vec<Choice>,
}

#[derive(Debug, Deserialize)]
struct Choice {
    message: Message,
}

impl OpenAI {
    pub fn new(api_key: String, base_url: &str, model: &str) -> Self {
        Self {
            client: reqwest::Client::new(),
            api_key,
            base_url: base_url.trim_end_matches('/').to_string(),
            model: model.to_string(),
            max_tokens: 1000,
            temperature: 0.7,
        }
    }

    pub fn openai() -> Result<Self, Box<dyn Error>> {
        let api_key = std::env::var("OPENAI_API_KEY")?;
        Ok(Self::new(api_key, "https://api.openai.com/v1", "gpt-4o-mini"))
    }

    pub fn grok() -> Result<Self, Box<dyn Error>> {
        let api_key = std::env::var("XAI_API_KEY")?;
        Ok(Self::new(api_key, "https://api.x.ai/v1", "grok-beta"))
    }

    pub fn anthropic_openrouter() -> Result<Self, Box<dyn Error>> {
        let api_key = std::env::var("OPENROUTER_API_KEY")?;
        Ok(Self::new(
            api_key,
            "https://openrouter.ai/api/v1",
            "anthropic/claude-3.5-sonnet",
        ))
    }

    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = temperature;
        self
    }

    pub fn with_model(mut self, model: &str) -> Self {
        self.model = model.to_string();
        self
    }

    pub async fn generate(&self, prompt: &str) -> Result<String, Box<dyn Error>> {
        let messages = vec![Message {
            role: "user".to_string(),
            content: prompt.to_string(),
        }];
        self.chat(messages).await
    }

    pub async fn chat(&self, messages: Vec<Message>) -> Result<String, Box<dyn Error>> {
        let request = ChatRequest {
            model: self.model.clone(),
            messages,
            max_tokens: self.max_tokens,
            temperature: self.temperature,
        };

        let response = self
            .client
            .post(format!("{}/chat/completions", self.base_url))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await?;
            return Err(format!("API error {}: {}", status, error_text).into());
        }

        let chat_response: ChatResponse = response.json().await?;

        chat_response
            .choices
            .first()
            .map(|choice| choice.message.content.clone())
            .ok_or_else(|| "No response from API".into())
    }

    pub async fn generate_with_system(
        &self,
        system_prompt: &str,
        user_prompt: &str,
    ) -> Result<String, Box<dyn Error>> {
        let messages = vec![
            Message {
                role: "system".to_string(),
                content: system_prompt.to_string(),
            },
            Message {
                role: "user".to_string(),
                content: user_prompt.to_string(),
            },
        ];
        self.chat(messages).await
    }
}

#[async_trait]
impl AI for OpenAI {
    async fn chat(&self, messages: Vec<Message>) -> Result<String, Box<dyn Error>> {
        let messages: Vec<Message> = messages
            .into_iter()
            .map(|m| Message {
                role: m.role,
                content: m.content,
            })
            .collect();
        self.chat(messages).await
    }
}
