use anyhow::Result;
use candle_core::Device;
use regex::Regex;

use crate::eval::generate::generate_text;
use crate::model::arch::MoeModel;

/// Embedded reasoning canary questions with expected answer tokens.
///
/// Categories:
/// - Basic arithmetic (original 10)
/// - Logic and comparisons
/// - Unit conversions and world knowledge
/// - Word problems
/// - Sequences and patterns
pub const CANARY_QUESTIONS: &[(&str, &str)] = &[
    // -- Basic arithmetic --
    ("What is 2 + 2?", "4"),
    ("What is 15 * 3?", "45"),
    ("What is 100 / 5?", "20"),
    ("What is 7 - 3?", "4"),
    ("What is the square root of 144?", "12"),
    ("What is 2^10?", "1024"),
    ("If x = 5 and y = 3, what is x + y?", "8"),
    ("What is 99 + 1?", "100"),
    ("What is 50% of 200?", "100"),
    ("How many minutes in 2 hours?", "120"),
    // -- More arithmetic --
    ("What is 17 + 28?", "45"),
    ("What is 256 / 16?", "16"),
    ("What is 13 * 7?", "91"),
    ("What is 1000 - 387?", "613"),
    ("What is 3^4?", "81"),
    // -- Logic and comparisons --
    ("Which is larger, 3/4 or 2/3?", "3/4"),
    ("Is 17 a prime number? Answer yes or no.", "yes"),
    ("Is 15 a prime number? Answer yes or no.", "no"),
    ("What is the next prime after 7?", "11"),
    ("How many sides does a hexagon have?", "6"),
    // -- Unit conversions and world knowledge --
    ("How many seconds in one minute?", "60"),
    ("How many days in a leap year?", "366"),
    ("How many centimeters in one meter?", "100"),
    ("How many bytes in a kilobyte?", "1024"),
    ("How many degrees in a right angle?", "90"),
    // -- Word problems --
    ("If a train travels at 60 mph for 3 hours, how many miles does it travel?", "180"),
    ("A store sells apples for $2 each. How much do 7 apples cost?", "14"),
    ("If you have 3 dozen eggs, how many eggs do you have?", "36"),
    ("A rectangle has a width of 5 and a length of 8. What is its area?", "40"),
    ("If you flip a fair coin, what is the probability of heads? Express as a fraction.", "1/2"),
    // -- Sequences and patterns --
    ("What is the 5th Fibonacci number (starting from 1, 1, 2, 3, ...)?", "5"),
    ("What comes next in the sequence: 2, 4, 8, 16, ?", "32"),
    ("What is the sum of the first 5 positive integers?", "15"),
    ("What is 10 factorial divided by 9 factorial?", "10"),
    ("If a sequence starts at 3 and each term adds 5, what is the 4th term?", "18"),
];

/// Number of tokens to generate for each reasoning canary question.
const CANARY_GEN_TOKENS: usize = 32;

/// Run reasoning canary: simple math questions to detect catastrophic damage.
///
/// For each question, generates up to `CANARY_GEN_TOKENS` tokens greedily and
/// checks whether the expected answer appears anywhere in the generated text.
/// This handles instruct models that use preambles like "The answer is..." or
/// thinking tokens before answering.
pub fn run_reasoning_canary(
    model: &dyn MoeModel,
    tokenizer: &tokenizers::Tokenizer,
    device: &Device,
) -> Result<(usize, usize)> {
    let mut passed = 0;
    let total = CANARY_QUESTIONS.len();

    for (question, expected_answer) in CANARY_QUESTIONS {
        let formatted = model.format_chat_prompt(question);
        let encoding = tokenizer
            .encode(formatted.as_str(), false)
            .map_err(|e| anyhow::anyhow!("Tokenize failed: {e}"))?;

        let text = generate_text(model, tokenizer, encoding.get_ids(), CANARY_GEN_TOKENS, device)?;

        // Word-boundary match to avoid "4" matching "14", "2048", etc.
        let pattern = format!(r"\b{}\b", regex::escape(expected_answer));
        let re = Regex::new(&pattern).unwrap();
        if re.is_match(&text) {
            passed += 1;
        } else {
            tracing::debug!(
                "Canary failed: '{}' -> '{}' (expected '{}')",
                question,
                text.chars().take(80).collect::<String>(),
                expected_answer,
            );
        }
    }

    tracing::info!("Reasoning canary: {}/{} passed", passed, total);
    Ok((passed, total))
}
