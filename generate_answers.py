import logging

import click
import torch

from auto_gptq import AutoGPTQForCausalLM
from huggingface_hub import hf_hub_download

from langchain.chains import RetrievalQA
from langchain.embeddings.huggingface import HuggingFaceInstructEmbeddings
from langchain.llms import HuggingFacePipeline, LlamaCpp
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Chroma

import chromadb

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    LlamaForCausalLM,
    LlamaTokenizer,
    pipeline,
)

from constants import (
    EMBEDDING_MODEL_NAME, 
    PERSIST_PATH,
    QUESTION_FILE_PATH,
    CHROMA_SETTINGS
)


def load_model(device_type, model_id, model_basename=None):
    logging.info(f"Loading Model: {model_id} ...")
    logging.info(f"Working on {device_type} ...")
    logging.info("Processing ...")

    if model_basename is not None:
        if ".ggml" in model_basename:
            logging.info("Using Llamacpp for GGML quantized models ...")

            model_path = hf_hub_download(repo_id=model_id, filename=model_basename)
            max_ctx_size = 2048
            kwargs = {
                "model_path": model_path,
                "n_ctx": max_ctx_size,
                "max_tokens": max_ctx_size,
            }
            
            if device_type.lower() == "mps":
                kwargs["n_gpu_layers"] = 1000
            if device_type.lower() == "cuda":
                kwargs["n_gpu_layers"] = 1000
                kwargs["n_batch"] = max_ctx_size
            
            return LlamaCpp(**kwargs)

        else:
            logging.info("Using AutoGPTQForCausalLM for quantized models ...")

            if ".safetensors" in model_basename:
                model_basename = model_basename.replace(".safetensors", "")

            logging.info("Loading Tokenizer ...")

            tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
            
            logging.info("Loaded")

            model = AutoGPTQForCausalLM.from_quantized(
                model_id,
                model_basename=model_basename,
                use_safetensors=True,
                trust_remote_code=True,
                device="cuda:0",
                use_triton=False,
                quantize_config=None
            )
    elif (
        device_type.lower() == "cuda"
    ):  
        logging.info("Using AutoModelForCausalLM for full models")
        logging.info("Loading Tokenizer ...")

        tokenizer = AutoTokenizer.from_pretrained(model_id)

        logging.info("Loaded")

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            max_memory={0: "15GB"} # Uncomment when CUDA out of memory errors occur
        )
        model.tie_weights()
    else:
        logging.info("Using LlamaTokenizer ...")

        tokenizer = LlamaTokenizer.from_pretrained(model_id)
        model = LlamaForCausalLM.from_pretrained(model_id)

    generation_config = GenerationConfig.from_pretrained(model_id)

    logging.info("Creating a pipeline for text generation ...")

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=2048,
        temperature=0,
        top_p=0.95,
        repetition_penalty=1.15,
        generation_config=generation_config,
    )

    logging.info("Loading LLM ...")

    local_llm = HuggingFacePipeline(pipeline=pipe)

    logging.info("Loaded")

    return local_llm

@click.command()
@click.option(
    "--device_type",
    default="cuda" if torch.cuda.is_available() else "cpu",
    type=click.Choice(
        [
            "cpu",
            "cuda",
            "ipu",
            "xpu",
            "mkldnn",
            "opengl",
            "opencl",
            "ideep",
            "hip",
            "ve",
            "fpga",
            "ort",
            "xla",
            "lazy",
            "vulkan",
            "mps",
            "meta",
            "hpu",
            "mtia",
        ],
    ),
    help="Device to run on. (Default is cuda)",
)
@click.option(
    "--show_sources",
    "-s",
    is_flag=True,
    help="Show sources along with answers (Default is False)",
)
def main(device_type, show_sources):
    logging.info(f"Running on: {device_type}")
    logging.info(f"Show sources: {show_sources}")

    embeddings = HuggingFaceInstructEmbeddings(model_name=EMBEDDING_MODEL_NAME, model_kwargs={"device": device_type})

    logging.info("Loading DB ...")

    client = chromadb.PersistentClient(settings=CHROMA_SETTINGS , path=PERSIST_PATH)

    db = Chroma(
        client=client,
        embedding_function=embeddings,
    )
    retriever = db.as_retriever()

    model_id = "TheBloke/Llama-2-7B-Chat-GGML"
    model_basename = "llama-2-7b-chat.ggmlv3.q4_0.bin"

    template = """Use the following pieces of context to answer the question at the end. If you don't know the answer,\
                just say that you don't know, don't try to make up an answer.

                {context}

                {history}
                Question: {question}
                Helpful Answer:"""

    prompt = PromptTemplate(input_variables=["history", "context", "question"], template=template)
    memory = ConversationBufferMemory(input_key="question", memory_key="history")

    llm = load_model(device_type, model_id=model_id, model_basename=model_basename)

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt, "memory": memory},
    )

    # while True:
    #     query = input("\nEnter a query: ")

    with open(QUESTION_FILE_PATH, "r", encoding='utf-8') as questions_file:
        for line in questions_file:
            query = line.strip()

            if query == "exit":
                break

            res = qa(query)
            answer, docs = res["result"], res["source_documents"]

            print("\n\n> Question:")
            print(query)
            print("\n> Answer:")
            print(answer)

            if show_sources:  
                print("--------  SOURCE DOCUMENTS   --------")
                for document in docs:
                    print("\n> " + document.metadata["source"] + ":")
                    print(document.page_content)
                print("--------  SOURCE DOCUMENTS   --------")


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(message)s", level=logging.INFO
    )
    main()
