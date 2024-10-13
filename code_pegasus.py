from transformers import PegasusForConditionalGeneration, PegasusTokenizer
from transformers import pipeline

# Replace *Path* with the model path
model_path = "./fine_tune_model_pegasus_50"

# Replace **Path** with the tokenizer path
tokenizer_path = "./fine_tune_tokenizer_pegasus_50"

# Load the fine-tuned model
model = PegasusForConditionalGeneration.from_pretrained(model_path, ignore_mismatched_sizes=True)

# Load the tokenizer associated with the fine-tuned model
tokenizer = PegasusTokenizer.from_pretrained(tokenizer_path)

# Ensure the model is in evaluation mode
model.eval()

# Create a custom summarization pipeline
gen_kwargs = {"length_penalty": 1.0, "num_beams": 8, "max_length": 700}
custom_summarization_pipeline = pipeline('summarization', model=model, tokenizer=tokenizer, **gen_kwargs)


file_path = "./9.txt" # Replace with the conversation file path to check others
with open(file_path, 'r') as file:
    text = file.read()

# Call the custom summarization pipeline
summary = custom_summarization_pipeline(text)

print('Summary:\n',summary[0]['summary_text'])
