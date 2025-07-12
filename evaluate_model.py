import os
import torch
from datasets import load_dataset
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from evaluate import load

# Step 1: Load the CNN/DailyMail dataset
def load_data():
    print("Loading CNN/DailyMail dataset...")
    dataset = load_dataset("cnn_dailymail", "3.0.0")
    return dataset

# Step 2: Load the pre-trained summarization model
def load_model():
    print("Loading pre-trained summarization model...")
    model_name = "facebook/bart-large-cnn"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    summarization_pipeline = pipeline("summarization", model=model, tokenizer=tokenizer)
    return summarization_pipeline

# Step 3: Generate summaries for a subset of 100 examples
def generate_summaries(dataset, summarization_pipeline):
    print("Generating summaries for 100 examples...")
    subset = dataset["test"][:100]
    articles = [example["article"] for example in subset]
    summaries = summarization_pipeline(articles, max_length=130, min_length=30, do_sample=False)
    return summaries, subset

# Step 4: Compute BLEU scores
def compute_bleu_scores(references, predictions):
    print("Computing BLEU scores...")
    bleu = load("bleu")
    results = {
        "BLEU-1": bleu.compute(predictions=predictions, references=references, max_order=1),
        "BLEU-2": bleu.compute(predictions=predictions, references=references, max_order=2),
        "BLEU-3": bleu.compute(predictions=predictions, references=references, max_order=3),
    }
    return results

# Step 5: Compute BERTScore
def compute_bertscore(references, predictions):
    print("Computing BERTScore...")
    bertscore = load("bertscore")
    results = bertscore.compute(predictions=predictions, references=references, lang="en")
    return results

# Step 6: Compute Perplexity
def compute_perplexity(model, tokenizer, predictions):
    print("Computing Perplexity...")
    perplexities = []
    for prediction in predictions:
        inputs = tokenizer(prediction, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            perplexity = torch.exp(outputs.loss)
            perplexities.append(perplexity.item())
    return sum(perplexities) / len(perplexities)

# Main function
def main():
    dataset = load_data()
    summarization_pipeline = load_model()
    summaries, subset = generate_summaries(dataset, summarization_pipeline)

    references = [example["highlights"] for example in subset]
    predictions = [summary["summary_text"] for summary in summaries]

    bleu_scores = compute_bleu_scores(references, predictions)
    bertscore_results = compute_bertscore(references, predictions)
    perplexity = compute_perplexity(summarization_pipeline.model, summarization_pipeline.tokenizer, predictions)

    print("\nEvaluation Results:")
    print("BLEU Scores:", bleu_scores)
    print("BERTScore:", bertscore_results)
    print("Perplexity:", perplexity)

if __name__ == "__main__":
    main()
