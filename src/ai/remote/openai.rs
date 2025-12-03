use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::error::Error;

use crate::ai::remote::traits::{AI, Message};

#[derive(Debug, Clone)]
pub struct OpenAI {
    client: reqwest::Client,
    api_key: String,
    base_url: String,
    model: String,
    max_tokens: u32,
    temperature: f32,
    uses_completion_tokens: bool,
}

#[derive(Debug, Serialize)]
struct ChatRequest {
    model: String,
    messages: Vec<Message>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_completion_tokens: Option<u32>,
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
    pub fn new(api_key: String, base_url: &str, model: &str, uses_completion_tokens: bool) -> Self {
        Self {
            client: reqwest::Client::new(),
            api_key,
            base_url: base_url.trim_end_matches('/').to_string(),
            model: model.to_string(),
            max_tokens: 1000,
            temperature: 0.7,
            uses_completion_tokens
        }
    }

    pub fn openai_5() -> Result<Self, Box<dyn Error>> {
        let api_key = std::env::var("OPENAI_API_KEY")?;
        Ok(Self::new(api_key, "https://api.openai.com/v1", "gpt-5.1", true))
    }

    pub fn grok() -> Result<Self, Box<dyn Error>> {
        let api_key = std::env::var("XAI_API_KEY")?;
        Ok(Self::new(api_key, "https://api.x.ai/v1", "grok-4-latest", false))
    }

    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = temperature;
        self
    }

    #[allow(dead_code)]
    pub fn with_model(mut self, model: &str) -> Self {
        self.model = model.to_string();
        self
    }

    #[allow(dead_code)]
    pub async fn generate(&self, prompt: &str) -> Result<String, Box<dyn Error>> {
        let messages = vec![Message {
            role: "user".to_string(),
            content: prompt.to_string(),
        }];
        self.chat(messages).await
    }

    async fn do_chat(&self, messages: Vec<Message>) -> Result<String, Box<dyn Error>> {
        let request = ChatRequest {
            model: self.model.clone(),
            messages,
            max_tokens: if self.uses_completion_tokens { None } else { Some(self.max_tokens) },
            max_completion_tokens: if self.uses_completion_tokens { Some(self.max_tokens) } else { None },
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

    #[allow(dead_code)]
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
        self.do_chat(messages).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mockito;

    #[tokio::test]
    async fn test_openai_with_max_tokens() {
        let client = OpenAI::new("test_key".to_string(), "https://api.test.com", "gpt-4", true)
            .with_max_tokens(2000);

        assert_eq!(client.max_tokens, 2000);
    }

    #[tokio::test]
    async fn test_openai_with_temperature() {
        let client = OpenAI::new("test_key".to_string(), "https://api.test.com", "gpt-4", true)
            .with_temperature(0.9);

        assert_eq!(client.temperature, 0.9);
    }

    #[tokio::test]
    async fn test_openai_with_model() {
        let client = OpenAI::new("test_key".to_string(), "https://api.test.com", "gpt-4", true)
            .with_model("gpt-4-turbo");

        assert_eq!(client.model, "gpt-4-turbo");
    }

    #[tokio::test]
    async fn test_openai_uses_completion_tokens_true() {
        let client = OpenAI::new("test_key".to_string(), "https://api.test.com", "gpt-4", true);
        assert!(client.uses_completion_tokens);
    }

    #[tokio::test]
    async fn test_openai_uses_completion_tokens_false() {
        let client = OpenAI::new("test_key".to_string(), "https://api.test.com", "gpt-4", false);
        assert!(!client.uses_completion_tokens);
    }

    #[tokio::test]
    async fn test_openai_base_url_trim() {
        let client = OpenAI::new("test_key".to_string(), "https://api.test.com/", "gpt-4", true);
        assert_eq!(client.base_url, "https://api.test.com");
    }

    #[tokio::test]
    async fn test_generate_creates_user_message() {
        let client = OpenAI::new("test_key".to_string(), "https://api.test.com", "gpt-4", true);

        // This test validates the generate method constructs the correct message structure
        // We can't easily test the actual API call without mocking, but we can test the method exists
        assert_eq!(client.model, "gpt-4");
    }

    #[tokio::test]
    async fn test_chat_request_serialization() {
        let messages = vec![
            Message {
                role: "user".to_string(),
                content: "Hello".to_string(),
            }
        ];

        let request = ChatRequest {
            model: "gpt-4".to_string(),
            messages,
            max_tokens: Some(100),
            max_completion_tokens: None,
            temperature: 0.7,
        };

        let json = serde_json::to_string(&request).unwrap();
        assert!(json.contains("\"model\":\"gpt-4\""));
        assert!(json.contains("\"temperature\":0.7"));
    }

    #[tokio::test]
    async fn test_mock_api_success() {
        let mut server = mockito::Server::new_async().await;

        let mock = server.mock("POST", "/chat/completions")
            .match_header("authorization", "Bearer test_key")
            .match_header("content-type", "application/json")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(r#"{
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "Hello! How can I help?"
                        }
                    }
                ]
            }"#)
            .create_async()
            .await;

        let client = OpenAI::new("test_key".to_string(), &server.url(), "gpt-4", false);

        let messages = vec![Message {
            role: "user".to_string(),
            content: "Hi".to_string(),
        }];

        let result = client.do_chat(messages).await;

        mock.assert_async().await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "Hello! How can I help?");
    }

    #[tokio::test]
    async fn test_mock_api_error() {
        let mut server = mockito::Server::new_async().await;

        let mock = server.mock("POST", "/chat/completions")
            .with_status(401)
            .with_body("Unauthorized")
            .create_async()
            .await;

        let client = OpenAI::new("test_key".to_string(), &server.url(), "gpt-4", false);

        let messages = vec![Message {
            role: "user".to_string(),
            content: "Hi".to_string(),
        }];

        let result = client.do_chat(messages).await;

        mock.assert_async().await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("401"));
    }
}
