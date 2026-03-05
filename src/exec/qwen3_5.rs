//! Qwen3.5 exec implementation for CLI `run` subcommand

use std::time::Instant;

use anyhow::Result;

use crate::exec::ExecModel;
use crate::models::GenerateModel;
use crate::models::qwen3_5::generate::Qwen3_5GenerateModel;
use crate::utils::get_file_path;

pub struct Qwen3_5Exec;

impl ExecModel for Qwen3_5Exec {
    fn run(input: &[String], output: Option<&str>, weight_path: &str) -> Result<()> {
        let input_text = &input[0];
        let target_text = if input_text.starts_with("file://") {
            let path = get_file_path(input_text)?;
            std::fs::read_to_string(path)?
        } else {
            input_text.clone()
        };

        let i_start = Instant::now();
        let mut model = Qwen3_5GenerateModel::init(weight_path, None, None)?;
        let i_duration = i_start.elapsed();
        println!("Time elapsed in load model is: {:?}", i_duration);
        let url = &input[1];
        let input_url = if url.starts_with("http://")
            || url.starts_with("https://")
            || url.starts_with("file://")
        {
            url.clone()
        } else {
            format!("file://{}", url)
        };
        let message = if input_url.ends_with("mp4") {
            format!(
                r#"{{
            "model": "qwen3.5",
            "messages": [
                {{
                    "role": "user",
                    "content": [
                        {{
                            "type": "video",
                            "video_url": 
                            {{
                                "url": "{}"
                            }}
                        }},
                        {{
                            "type": "text", 
                            "text": "{}"
                        }}
                    ]
                }}
            ]
        }}"#,
                input_url, target_text
            )
        } else {
            format!(
                r#"{{
            "model": "qwen2.5",
            "messages": [
                {{
                    "role": "user",
                    "content": [
                        {{
                            "type": "image",
                            "image_url": {{
                                "url": "{}"
                            }}
                        }},
                        {{
                            "type": "text", 
                            "text": "{}"
                        }}
                    ]
                }}
            ]
        }}"#,
                input_url, target_text
            )
        };
        let mes = serde_json::from_str(&message)?;

        let i_start = Instant::now();
        let result = model.generate(mes)?;
        let i_duration = i_start.elapsed();
        println!("Time elapsed in generate is: {:?}", i_duration);

        println!("Result: {:?}", result);

        if let Some(out) = output {
            std::fs::write(out, format!("{:?}", result))?;
            println!("Output saved to: {}", out);
        }

        Ok(())
    }
}
