# Offline-QA-Chatbot
Work offline with Open source LLM

Choose a Text Generation Model with HuggingFace

If you're using the HuggingFace library for the first time to generate text, it will download a model for you. After that, it will use the downloaded model for future runs.

# Parameters:

device_type: This is where you choose if you want to use your computer's regular power (CPU) or its faster power (GPU, if available).
model_id: This is like a name for the model you want to use. It helps HuggingFace find the right model for you.
model_basename (optional): If you're using a specific type of model, you can mention it here.
What It Does:
This function helps you set up a tool to create text using a specific model. It's like having a writing assistant that follows the rules of a particular writer.

The code works with special HuggingFace models that have names ending in "GPTQ" and also have certain words like ".no-act.order" or ".safetensors" in their name on the HuggingFace website.
It is also designed to work with any HuggingFace models that have names ending in "-HF" or have a ".bin" file in their HuggingFace repository.

# main() function performs a specific task for getting information:
It gets a special type of model, either "HuggingFaceInstructEmbeddings" or "HuggingFaceEmbeddings".
It loads a set of stored vectors from a previous step.
It also loads a local language model (LLM) using a function called "load_model". You can choose different LLMs.
It sets up a process for retrieving question and answer pairs.
It then uses this setup to find answers to questions.

# load the LLM for generating Natural Language responses

    # for HF models
    # model_id = "TheBloke/vicuna-7B-1.1-HF"
    # model_basename = None
    # model_id = "TheBloke/Wizard-Vicuna-7B-Uncensored-HF"
    # model_id = "TheBloke/guanaco-7B-HF"
    # model_id = 'NousResearch/Nous-Hermes-13b' # Requires ~ 23GB VRAM. Using STransformers
    # alongside will 100% create OOM on 24GB cards.
    # llm = load_model(device_type, model_id=model_id)

    # for GPTQ (quantized) models
    # model_id = "TheBloke/Nous-Hermes-13B-GPTQ"
    # model_basename = "nous-hermes-13b-GPTQ-4bit-128g.no-act.order"
    # model_id = "TheBloke/WizardLM-30B-Uncensored-GPTQ"
    # model_basename = "WizardLM-30B-Uncensored-GPTQ-4bit.act-order.safetensors" # Requires
    # ~21GB VRAM. Using STransformers alongside can potentially create OOM on 24GB cards.
    # model_id = "TheBloke/wizardLM-7B-GPTQ"
    # model_basename = "wizardLM-7B-GPTQ-4bit.compat.no-act-order.safetensors"
    # model_id = "TheBloke/WizardLM-7B-uncensored-GPTQ"
    # model_basename = "WizardLM-7B-uncensored-GPTQ-4bit-128g.compat.no-act-order.safetensors"

    # for GGML (quantized cpu+gpu+mps) models - check if they support llama.cpp
    # model_id = "TheBloke/wizard-vicuna-13B-GGML"
    # model_basename = "wizard-vicuna-13B.ggmlv3.q4_0.bin"
    # model_basename = "wizard-vicuna-13B.ggmlv3.q6_K.bin"
    # model_basename = "wizard-vicuna-13B.ggmlv3.q2_K.bin"
    # model_id = "TheBloke/orca_mini_3B-GGML"
    # model_basename = "orca-mini-3b.ggmlv3.q4_0.bin"

# environment setup

Install anaconda environment. 
Anaconda3-2023.03-1-Windows-x86_64 was used.

pip install -r requirements.txt
pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.3.1/en_core_web_sm-2.3.1.tar.gz

Optional if it was not installed
pip install sentencepiece
pip install protobuf==3.19.0

# Run code

Copy your text files to DATA directory.

Run the following cmd
    python ingest.py
    python generate_questions.py question_count=10
    python generate_answers.py

# Available file types
    .txt, .md, .py, .pdf, .csv, .docx, .doc"

    For text file, ANSI encoding is default.

HTML reading feature will come soon