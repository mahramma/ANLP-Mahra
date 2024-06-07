import ssl
import time
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
from datasets import load_metric
import nltk
from nltk.tokenize import sent_tokenize
import os  # Import the os module

# Set the environment variable to disable parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "false"

ssl._create_default_https_context = ssl._create_unverified_context
nltk.download('punkt')

# Timing wrapper function
def timing(f):
    def wrap(*args):
        start_time = time.time()
        result = f(*args)
        end_time = time.time()
        print(f"Function {f.__name__} took {end_time - start_time:.2f} seconds")
        return result

    return wrap


# Load summarization pipeline with a smaller model
print("Loading summarization model...")
summarizer = pipeline("summarization", model="t5-small")
print("Summarization model loaded.")

# Load NER model and tokenizer
print("Loading NER model...")
ner_model_name = "dbmdz/bert-large-cased-finetuned-conll03-english"
ner_tokenizer = AutoTokenizer.from_pretrained(ner_model_name)
ner_model = AutoModelForTokenClassification.from_pretrained(ner_model_name)
print("NER model loaded.")

# Load ROUGE metric
print("Loading ROUGE metric...")
rouge = load_metric("rouge", process_id=0, num_process=1, trust_remote_code=True)
print("ROUGE metric loaded.")


@timing
def extract_key_sentences(long_form_answer):
    print("Extracting key sentences...")
    # Tokenize the long-form answer into sentences
    sentences = sent_tokenize(long_form_answer)
    # Summarize the entire long-form answer
    input_length = len(long_form_answer)
    max_summary_length = min(input_length // 2, 150)  # Adjust the maximum summary length dynamically
    summarized_text = summarizer(long_form_answer, max_length=max_summary_length, min_length=30, do_sample=False)[0]['summary_text']
    # Tokenize the summary into sentences
    summary_sentences = sent_tokenize(summarized_text)
    print("Key sentences extracted.")
    return summary_sentences


@timing
def decontextualize_sentences(sentences):
    print("Decontextualizing sentences...")
    decontextualized_sentences = []
    for sentence in sentences:
        # Apply NER
        ner_inputs = ner_tokenizer(sentence, return_tensors="pt")
        ner_outputs = ner_model(**ner_inputs).logits
        tokens = ner_tokenizer.convert_ids_to_tokens(ner_inputs["input_ids"].squeeze().tolist())
        ner_tags = torch.argmax(ner_outputs, dim=2).squeeze().tolist()

        decontext_sentence = sentence
        for token, tag in zip(tokens, ner_tags):
            if tag != 0 and token not in ["[CLS]", "[SEP]"]:
                decontext_sentence = decontext_sentence.replace(token, token.capitalize())

        # Simple replacement for pronouns (can be expanded with more sophisticated methods)
        decontext_sentence = decontext_sentence.replace(" this ", " the ").replace(" it ", " the information ").replace(
            " these ", " the elements ")
        decontextualized_sentences.append(decontext_sentence)

    print("Sentences decontextualized.")
    return decontextualized_sentences


@timing
def generate_summary(question, long_form_answer):
    key_sentences = extract_key_sentences(long_form_answer)
    decontextualized_sentences = decontextualize_sentences(key_sentences)
    summary = '. '.join(decontextualized_sentences)
    return summary


@timing
def evaluate_summary(reference, generated_summary):
    rouge_output = rouge.compute(predictions=[generated_summary], references=[reference])
    rouge_scores = {}
    for metric, results in rouge_output.items():
        if metric.startswith("rouge"):
            precision = results.mid.precision
            recall = results.mid.recall
            f1_score = results.mid.fmeasure
            exact_match = results.mid.fmeasure
            rouge_scores[metric] = {
                "precision": precision,
                "recall": recall,
                "f1": f1_score,
                "exact_match": exact_match,
            }
    return rouge_scores






# Function for error analysis
def analyze_errors(reference, generated_summary):
    reference_tokens = reference.split()
    generated_tokens = generated_summary.split()

    # Find common tokens
    common_tokens = set(reference_tokens).intersection(generated_tokens)

    # Find missing tokens
    missing_tokens = set(reference_tokens) - common_tokens

    # Find extra tokens
    extra_tokens = set(generated_tokens) - common_tokens

    return common_tokens, missing_tokens, extra_tokens


# Example usage
question = "Why does car sickness seem to hit the hardest when you look down at your phone, book, etc.?"
long_form_answer = ("The brain perceived motion because it receives information from the eyes, ears, and muscles. "
                    "When these parts send conflicting information, the brain doesn’t know which is right and which is wrong, "
                    "and this is what causes motion sickness. An example of this is when reading a book while you are in a "
                    "moving car. To your eyes, the book is stationary while your inner ear and the rest of your body can "
                    "feel a sense of motion. This would likely cause car sickness.")
reference_summary = (
    "Motion sickness occurs when the brain receives conflicting signals from the eyes, ears, and muscles. "
    "Reading a book in a moving car, for example, can cause the brain to get confused and result in sickness.")

print("Generating summary for question 1...")
generated_summary = generate_summary(question, long_form_answer)
print("Summary generated for question 1.")

print("Evaluating summary for question 1...")
rouge_scores = evaluate_summary(reference_summary, generated_summary)
print("Summary evaluated for question 1.")

# Print ROUGE scores in a formatted way
print("\nROUGE Scores for question 1:")
for metric, scores in rouge_scores.items():
    print(f"{metric}:")
    for score_type, value in scores.items():
        print(f"{score_type.capitalize()}: {value}")

# Error analysis for question 1
common_tokens, missing_tokens, extra_tokens = analyze_errors(reference_summary, generated_summary)
print("\nError Analysis for Question 1:")
print("Common Tokens:", common_tokens)
print("Missing Tokens:", missing_tokens)
print("Extra Tokens:", extra_tokens)
