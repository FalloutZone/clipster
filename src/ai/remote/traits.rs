use async_trait::async_trait;
use std::error::Error;
use std::fmt::Debug;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Message {
    pub role: String,
    pub content: String,
}

#[async_trait]
pub trait AI: Send + Sync + Debug {
    async fn chat(&self, messages: Vec<Message>) -> Result<String, Box<dyn Error>>;

    async fn generate(&self, prompt: &str) -> Result<String, Box<dyn Error>> {
        let messages = vec![Message {
            role: "user".to_string(),
            content: prompt.to_string(),
        }];
        self.chat(messages).await
    }
}
