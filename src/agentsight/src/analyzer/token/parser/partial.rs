//! Recover usage fields without treating echoed schemas or message text as billing data.

use serde::de::{DeserializeSeed, IgnoredAny, MapAccess, Visitor};
use serde_json::Value;
use std::collections::HashMap;
use std::fmt;

const COUNTERS: &[&str] = &[
    "input_tokens",
    "output_tokens",
    "prompt_tokens",
    "completion_tokens",
    "cache_creation_input_tokens",
    "cache_read_input_tokens",
    "cached_tokens",
    "promptTokenCount",
    "candidatesTokenCount",
    "thoughtsTokenCount",
    "toolUsePromptTokenCount",
    "cachedContentTokenCount",
    "prompt_token_count",
    "candidates_token_count",
    "thoughts_token_count",
    "tool_use_prompt_token_count",
];

#[derive(Clone, Copy)]
enum Scope {
    Root,
    Envelope,
    Usage,
    Details,
}

struct Fields<'a> {
    scope: Scope,
    counters: &'a mut HashMap<String, u64>,
    found_usage: &'a mut bool,
}

impl<'de> DeserializeSeed<'de> for Fields<'_> {
    type Value = ();

    fn deserialize<D: serde::Deserializer<'de>>(self, deserializer: D) -> Result<(), D::Error> {
        deserializer.deserialize_map(self)
    }
}

impl<'de> Visitor<'de> for Fields<'_> {
    type Value = ();

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a usage object or its protocol envelope")
    }

    fn visit_map<M: MapAccess<'de>>(self, mut map: M) -> Result<(), M::Error> {
        while let Some(key) = map.next_key::<String>()? {
            let nested = match (self.scope, key.as_str()) {
                (Scope::Root, "response" | "message") => Some(Scope::Envelope),
                (Scope::Root | Scope::Envelope, "usage" | "usageMetadata") => {
                    // A real envelope supersedes bare-fragment compatibility.
                    if !*self.found_usage {
                        self.counters.clear();
                        *self.found_usage = true;
                    }
                    Some(Scope::Usage)
                }
                (Scope::Usage, "input_tokens_details" | "prompt_tokens_details") => {
                    Some(Scope::Details)
                }
                (Scope::Root, "input_tokens_details" | "prompt_tokens_details")
                    if !*self.found_usage =>
                {
                    Some(Scope::Details)
                }
                _ => None,
            };
            if let Some(scope) = nested {
                map.next_value_seed(Fields {
                    scope,
                    counters: self.counters,
                    found_usage: self.found_usage,
                })?;
            } else if (matches!(self.scope, Scope::Usage)
                || (matches!(self.scope, Scope::Details)
                    && matches!(
                        key.as_str(),
                        "cached_tokens" | "cache_creation_input_tokens" | "cache_read_input_tokens"
                    ))
                || (matches!(self.scope, Scope::Root) && !*self.found_usage))
                && COUNTERS.contains(&key.as_str())
            {
                if let Some(value) = map.next_value::<Value>()?.as_u64() {
                    self.counters.entry(key).or_insert(value);
                }
            } else {
                // IgnoredAny validates nesting and escaped strings without
                // materializing large tool schemas or arbitrary response data.
                map.next_value::<IgnoredAny>()?;
            }
        }
        Ok(())
    }
}

pub(super) fn counters(data: &str) -> HashMap<String, u64> {
    let data = data.trim_start();
    let payload = if data.starts_with('{') {
        data
    } else {
        data.lines()
            .find_map(|line| line.strip_prefix("data:").map(str::trim_start))
            .unwrap_or("")
    };
    let mut counters = HashMap::new();
    let mut found_usage = false;
    // A truncated suffix is expected. Keep only fields parsed successfully
    // before the error, with serde's normal nesting and string validation.
    let _ = Fields {
        scope: Scope::Root,
        counters: &mut counters,
        found_usage: &mut found_usage,
    }
    .deserialize(&mut serde_json::Deserializer::from_str(payload));
    counters
}

#[cfg(test)]
mod tests {
    use super::super::TokenParser;

    #[test]
    fn partial_usage_boundaries_skip_nested_examples_and_detail_echoes() {
        for (data, expected) in [
            (
                r#"{"usage":{"input_tokens":57,"output_tokens":3,"x_details":[{"input_tokens":999,"cached_tokens":888}]},"cut":""#,
                (57, 3, None),
            ),
            (
                r#"{"message":{"content":[{"input_tokens":9000}],"usage":{"input_tokens":57,"output_tokens":3,"cache_read_input_tokens":2}},"cut":""#,
                (57, 3, Some(2)),
            ),
            (
                r#"{"choices":[{"message":{"content":{"usage":{"prompt_tokens":9000}}}}],"usage":{"prompt_tokens":57,"completion_tokens":3,"prompt_tokens_details":{"cached_tokens":2}},"cut":""#,
                (57, 3, Some(2)),
            ),
            (
                r#"{"candidates":[{"content":{"parts":[{"functionCall":{"args":{"promptTokenCount":9000}}}]}}],"usageMetadata":{"promptTokenCount":57,"candidatesTokenCount":3,"cachedContentTokenCount":2},"cut":""#,
                (57, 3, Some(2)),
            ),
            (
                r#"{"usage":{"input_tokens_details":{"output_tokens":9000,"cached_tokens":2},"input_tokens":57,"output_tokens":3},"cut":""#,
                (57, 3, Some(2)),
            ),
        ] {
            let usage = TokenParser::new().parse_data(data).unwrap();
            assert_eq!(
                (
                    usage.input_tokens,
                    usage.output_tokens,
                    usage.cache_read_input_tokens
                ),
                expected,
                "{data}"
            );
        }
    }

    #[test]
    fn partial_usage_boundaries_reject_non_usage_and_non_integer_values() {
        for data in [
            r#"{"response":{"metadata":{"usage":{"input_tokens":9000,"output_tokens":8000}},"cut":""#,
            r#"{"tools":[{"parameters":{"default":{"usageMetadata":{"promptTokenCount":9000}}}}],"cut":""#,
            r#"{"usage":{"input_tokens":1.5,"output_tokens":-2},"cut":""#,
            r#"{"usage":{"x_details":[{"input_tokens":9000}]},"cut":""#,
        ] {
            assert!(TokenParser::new().parse_data(data).is_none(), "{data}");
        }
    }
}
