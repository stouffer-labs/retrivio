//! Local hash backend: deterministic feature-hashed vectors for tests and offline use.

use std::collections::HashMap;

use sha1::{Digest, Sha1};

use crate::embed::{embed_metric_texts, Embedder};
use crate::util::{vector_norm, word_tokens};

/// The `embed_backend = "hash"` model name. The backend is [`LocalHashEmbedder`]: offline,
/// deterministic, unit-length feature-hash vectors of `local_embed_dim` dimensions. For tests
/// and smoke checks only; the vectors are not semantic. Its model key is `hash:<dim>`, so a
/// store embedded with it is its own embedding space (a switch to or from it is a re-embed).
pub(crate) const HASH_BACKEND_MODEL: &str = "local-hash";

/// Model key of the hash backend at `dim` dimensions (the same value [`LocalHashEmbedder`]
/// reports, so config-derived and embedder-derived keys agree).
pub(crate) fn hash_model_key(dim: i64) -> String {
    format!(
        "hash:{}",
        LocalHashEmbedder::effective_dim(dim.max(0) as usize)
    )
}

pub(crate) struct LocalHashEmbedder {
    dim: usize,
    synonym_map: HashMap<String, Vec<String>>,
}

impl LocalHashEmbedder {
    /// The dimension actually used for a requested one (never below 64).
    pub(crate) fn effective_dim(dim: usize) -> usize {
        dim.max(64)
    }

    pub(crate) fn new(dim: usize) -> Self {
        let use_dim = Self::effective_dim(dim);
        let groups: &[&[&str]] = &[
            &[
                "semantic", "meaning", "ontology", "taxonomy", "model", "schema", "layer",
            ],
            &["api", "service", "endpoint", "backend"],
            &["ui", "frontend", "interface", "ux"],
            &["storage", "database", "db", "persistence"],
            &["auth", "authentication", "login", "identity"],
            &["agent", "assistant", "automation"],
        ];
        let mut synonym_map: HashMap<String, Vec<String>> = HashMap::new();
        for group in groups {
            for token in *group {
                let mut list = Vec::new();
                for other in *group {
                    if other != token {
                        list.push((*other).to_string());
                    }
                }
                synonym_map.insert((*token).to_string(), list);
            }
        }
        Self {
            dim: use_dim,
            synonym_map,
        }
    }

    /// `hash:<dim>`, the key `embed_backend = "hash"` stores vectors under (see
    /// [`hash_model_key`]).
    fn model_key_local(&self) -> String {
        format!("hash:{}", self.dim)
    }

    pub(crate) fn embed_one_local(&self, text: &str) -> Vec<f32> {
        let mut vec = vec![0.0f32; self.dim];
        let normalized = text.to_lowercase();
        let tokens = word_tokens(&normalized);
        if tokens.is_empty() {
            return vec;
        }

        for token in &tokens {
            self.add_feature(&mut vec, &format!("t:{}", token), 1.0);
            if let Some(expanded) = self.synonym_map.get(token) {
                for syn in expanded {
                    self.add_feature(&mut vec, &format!("s:{}", syn), 0.35);
                }
            }
        }
        for pair in tokens.windows(2) {
            self.add_feature(&mut vec, &format!("b:{}_{}", pair[0], pair[1]), 0.8);
        }

        let compact: String = normalized.chars().filter(|c| !c.is_whitespace()).collect();
        let compact_chars: Vec<char> = compact.chars().collect();
        if compact_chars.len() >= 3 {
            for tri in compact_chars.windows(3) {
                let trigram: String = tri.iter().collect();
                self.add_feature(&mut vec, &format!("c:{}", trigram), 0.15);
            }
        }

        let norm = vector_norm(&vec);
        if norm > 0.0 {
            for v in &mut vec {
                *v = (*v as f64 / norm) as f32;
            }
        }
        vec
    }

    fn add_feature(&self, vec: &mut [f32], feature: &str, weight: f32) {
        let mut hasher = Sha1::new();
        hasher.update(feature.as_bytes());
        let digest = hasher.finalize();
        let mut first = [0u8; 8];
        first.copy_from_slice(&digest[..8]);
        let idx = (u64::from_le_bytes(first) as usize) % self.dim;
        let sign = if (digest[8] & 1) == 0 { 1.0 } else { -1.0 };
        vec[idx] += sign * weight;
    }
}

impl Embedder for LocalHashEmbedder {
    fn model_key(&self) -> String {
        self.model_key_local()
    }

    /// Every vector is scaled to unit length before it is returned.
    fn normalizes_output(&self) -> bool {
        true
    }

    fn embed_many(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, String> {
        let out: Vec<Vec<f32>> = texts.iter().map(|t| self.embed_one_local(t)).collect();
        embed_metric_texts(out.len());
        Ok(out)
    }

    fn embed_one(&self, text: &str) -> Result<Vec<f32>, String> {
        embed_metric_texts(1);
        Ok(self.embed_one_local(text))
    }
}

#[cfg(test)]
mod hash_backend_tests {
    use super::*;
    use crate::bench::default_bench_model_key;
    use crate::config::{config_enum_options, config_set_value, ConfigValues};
    use crate::embed::{build_embedder, ensure_native_embed_backend, model_key_for_cfg};
    use crate::util::vector_norm;
    use std::collections::HashMap;

    #[test]
    fn hash_backend_is_accepted_offline_and_keyed_by_its_dimension() {
        let mut map: HashMap<String, String> = HashMap::new();
        map.insert("embed_backend".into(), "HASH".into());
        map.insert("local_embed_dim".into(), "10".into());
        let cfg = ConfigValues::from_map(map);
        assert_eq!(cfg.embed_backend, "hash");
        assert_eq!(cfg.embed_model, HASH_BACKEND_MODEL);
        // The requested 10 dimensions are raised to the embedder's minimum of 64, and the
        // config-derived key equals the key the embedder reports.
        assert_eq!(model_key_for_cfg(&cfg), "hash:64");
        let embedder = build_embedder(&cfg).expect("hash embedder needs no service");
        assert_eq!(embedder.model_key(), "hash:64");
        assert!(embedder.normalizes_output());
        assert!(ensure_native_embed_backend(&cfg, "test").is_ok());

        let a = embedder.embed_one("otter habitat budget").unwrap();
        let b = embedder.embed_one("otter habitat budget").unwrap();
        let c = embedder.embed_one("zebra migration checklist").unwrap();
        assert_eq!(a, b, "deterministic");
        assert_ne!(a, c);
        assert_eq!(a.len(), 64);
        assert!(
            (vector_norm(&a) - 1.0).abs() < 1e-4,
            "unit length: {}",
            vector_norm(&a)
        );
        let many = embedder
            .embed_many(&["x".to_string(), "otter habitat budget".to_string()])
            .unwrap();
        assert_eq!(many[1], a);

        // Config plumbing: the setter accepts it, unknown names are rejected, an unknown
        // backend in a config file falls back to ollama, and the option list shows it.
        let mut cfg2 = cfg.clone();
        assert!(config_set_value(&mut cfg2, "embed_backend", "falkor").is_err());
        config_set_value(&mut cfg2, "embed_backend", "ollama").unwrap();
        assert_eq!(
            cfg2.embed_model, "qwen3-embedding",
            "default model follows the backend"
        );
        config_set_value(&mut cfg2, "embed_backend", "Hash").unwrap();
        assert_eq!(cfg2.embed_backend, "hash");
        assert_eq!(cfg2.embed_model, HASH_BACKEND_MODEL);
        let mut bad: HashMap<String, String> = HashMap::new();
        bad.insert("embed_backend".into(), "falkor".into());
        assert_eq!(ConfigValues::from_map(bad).embed_backend, "ollama");
        assert!(config_enum_options("embed_backend")
            .unwrap()
            .contains(&"hash"));
        assert_eq!(default_bench_model_key(&cfg), "hash:64");
        // Its own embedding space: the same dimension under ollama is a different key.
        let mut ollama = cfg.clone();
        ollama.embed_backend = "ollama".into();
        ollama.embed_model = "test-local".into();
        assert_ne!(model_key_for_cfg(&ollama), model_key_for_cfg(&cfg));
    }
}
