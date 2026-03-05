use candle_nn::Activation;

use crate::models::qwen3vl::config::Qwen3VLVisionConfig;

#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
pub struct RopeParameters {
    pub mrope_interleaved: bool,
    pub mrope_section: Vec<usize>,
    pub rope_type: String,
    pub rope_theta: f32,
    pub partial_rotary_factor: f32,
}

#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
pub struct Qwen3_5TextConfig {
    pub attention_bias: bool,
    pub attention_dropout: f32,
    pub attn_output_gate: bool,
    pub dtype: String,
    pub eos_token_id: u32,
    pub full_attention_interval: usize,
    pub head_dim: usize,
    pub hidden_act: Activation,
    pub hidden_size: usize,
    pub initializer_range: f32,
    pub intermediate_size: usize,
    pub layer_types: Vec<String>,
    pub linear_conv_kernel_dim: usize,
    pub linear_key_head_dim: usize,
    pub linear_num_key_heads: usize,
    pub linear_num_value_heads: usize,
    pub linear_value_head_dim: usize,
    pub max_position_embeddings: usize,
    pub mlp_only_layers: Vec<usize>,
    pub mtp_num_hidden_layers: usize,
    pub mtp_use_dedicated_embeddings: bool,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub num_key_value_heads: usize,
    pub rms_norm_eps: f64,
    pub tie_word_embeddings: bool,
    pub use_cache: bool,
    pub vocab_size: usize,
    pub mamba_ssm_dtype: String,
    pub rope_parameters: RopeParameters,
}

#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
pub struct Qwen3_5Config {
    pub image_token_id: u32,
    pub text_config: Qwen3_5TextConfig,
    pub tie_word_embeddings: bool,
    pub video_token_id: u32,
    pub vision_config: Qwen3VLVisionConfig,
    pub vision_end_token_id: u32,
    pub vision_start_token_id: u32,
}
