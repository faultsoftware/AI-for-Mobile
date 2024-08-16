# Install dependencies

Optimum allows us to easily export Hugging Face models to the ONNX format.

We can then run the model on our devices!


```python
!python -m pip install optimum
```


Optimum can export to various ecosystems, but we will use ONNX and its family of tools


```python
!pip install onnx onnxruntime onnxruntime-extensions
```

# Export model to ONNX format

Ensure Optimum is installed and review the export help section

Optimum has a number of arguments we will configure below


```python
!optimum-cli export onnx --help
```

Most importantly we will supply a model from HuggingFace


```python
hugging_face_model = "google-bert/bert-base-uncased"
```

The directory we will output the model to


```python
onnx_output_directory = "onnx_output"
```

We could let Optimum infer the task

But we will pass it in explicitly since we will use it for processing

See help output above for all task options


```python
task_value = "question-answering"
task_option = "--task" + " " + task_value
```

Optimization is optional

Not all models support optimization

See help output above for all optimize options


```python
optimize_value = "O1"
optimize_option = ""
# Uncomment the below to pass an optimize option
# optimize_option = "--optimize" + " " + optimize_value"
```

Once complete, you will see a model.onnx file in your ONNX output directory


```python
!optimum-cli export onnx {onnx_output_directory} --model {hugging_face_model} {task_option} {optimize_option}
```

# Quantitize the model

Ensure onnxruntime is installed and review the quantize help section

Optimum has a number of arguments we will configure below


```python
!optimum-cli onnxruntime quantize --help
```

The directory we will output the quantized model to


```python
onnx_quantize_output_directory = "onnx_quantize_output"
```

The architecture we will target

arm64 will work for most iOS and Android devices

You may also want to run again with avx2 to suport the much rarer x86 Android devices

See help output above for all architecture options


```python
architecture_value = "arm64"
architecture_option = "--" + architecture_value
```

Quantize the ONNX model

Once complete, you will see a model.onnx file in your ONNX quantize output directory


```python
!optimum-cli onnxruntime quantize {architecture_option} --onnx_model {onnx_output_directory} --output {onnx_quantize_output_directory}
```

# Run pre/post processing

Ensure onnxruntime extenion tools are installed and review the `add_pre_post_processing_to_model` help section

This command has a number of arguments we will configure below


```python
!python -m onnxruntime_extensions.tools.add_pre_post_processing_to_model --help
```

In this case since we have a Bert model, we will use the `transformers_and_bert` function directly.

This function automatically creates a pipeline for pre/post processing.

Take a [look at the implementation](https://github.com/microsoft/onnxruntime-extensions/blob/main/onnxruntime_extensions/tools/add_pre_post_processing_to_model.py#L322), it has lots of helpful pipeline logic.


```python
model_type_value = "transformers"
model_type_option = "--model_type" + " " + model_type_value
```

Now we set the tokenizer type

This is only used for the transformers option


```python
tokenizer_type_value = "BertTokenizer"
tokenizer_type_option = "--tokenizer_type" + " " + tokenizer_type_value
```

Now we set the NLP task type

This is only used for the transformers option



```python
nlp_task_type_value = "QuestionAnswering"
nlp_task_type_option = "--nlp_task_type" + " " + nlp_task_type_value
```

We now set the vocab file.

For preprocessing, this is a tokenizer model file that will be supplied to the tokenizer we specified above (for example, `BertTokenizer`).

For postprocessing, in the case of a QuestionAnswering task type, it will also be supplied to the `BertTokenizerQADecoder`.

Lucky for us, when we export a model with onnx, the vocab file is in the same folder as the model as `vocab.txt`.


```python
vocab_file_value = onnx_quantize_output_directory + "/" + "vocab.txt"
vocab_file_option = "--vocab_file" + " " + vocab_file_value
```

Lastly, we set the path to the actual quantitized model we will process.


```python
model_path = onnx_quantize_output_directory + "/" + "model_quantized.onnx"
```

Now let's actually run `add_pre_post_processing_to_model` on our quantitized model.

We will call `transformers_and_bert` directly which handles the pipeline creation.


```python
# The current add_pre_post_processing_to_model main has a bug with the argument order https://github.com/microsoft/onnxruntime-extensions/pull/782
# !python -m onnxruntime_extensions.tools.add_pre_post_processing_to_model {model_type_option} {tokenizer_type_option} {nlp_task_type_option} {vocab_file_option} {model_path}

# Invoke pre/post processing with transformers_and_bert directly
from pathlib import Path
import transformers
from onnxruntime_extensions.tools import add_pre_post_processing_to_model
from contextlib import contextmanager

quantized_model_path = Path(model_path)
tokenizer = transformers.AutoTokenizer.from_pretrained(hugging_face_model)

@contextmanager
def temp_vocab_file():
    vocab_file = quantized_model_path.parent / "vocab.txt"
    yield vocab_file
    vocab_file.unlink()

    output_model_path = quantized_model_path.with_name(quantized_model_path.stem+'_with_pre_post_processing').with_suffix(quantized_model_path.suffix)

with temp_vocab_file() as vocab_file:
    import json
    with open(str(vocab_file), 'w') as f:
        f.write(json.dumps(tokenizer.vocab))
    add_pre_post_processing_to_model.transformers_and_bert(quantized_model_path, output_model_path, vocab_file_value, tokenizer_type_value, nlp_task_type_value)
quantized_model_path.unlink()
```
