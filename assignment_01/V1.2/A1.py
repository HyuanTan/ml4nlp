import torch, nltk, pickle
from torch import nn
from collections import Counter
from transformers import BatchEncoding, PretrainedConfig, PreTrainedModel
from torch.utils.data import DataLoader, Subset
from datasets import load_dataset
import numpy as np
import sys, time, os

# THIS IS ONLY FOR RESOLVING ERRORS:
# check NLTK 'punkt' is available
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    print("NLTK 'punkt' model not found. Downloading...")
    nltk.download("punkt", quiet=True)

# check NLTK 'punkt_tab' is available
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    print("NLTK 'punkt_tab' model not found. Downloading...")
    nltk.download("punkt_tab", quiet=True)


###
### Part 1. Tokenization.
###
def lowercase_tokenizer(text):
    return [t.lower() for t in nltk.word_tokenize(text)]


def build_tokenizer(
    train_file,
    tokenize_fun=lowercase_tokenizer,
    max_voc_size=None,
    model_max_length=None,
    pad_token="<PAD>",
    unk_token="<UNK>",
    bos_token="<BOS>",
    eos_token="<EOS>",
):
    """Build a tokenizer from the given file.

    Args:
         train_file:        The name of the file containing the training texts.
         tokenize_fun:      The function that maps a text to a list of string tokens.
         max_voc_size:      The maximally allowed size of the vocabulary.
         model_max_length:  Truncate texts longer than this length.
         pad_token:         The dummy string corresponding to padding.
         unk_token:         The dummy string corresponding to out-of-vocabulary tokens.
         bos_token:         The dummy string corresponding to the beginning of the text.
         eos_token:         The dummy string corresponding to the end the text.
    """
    print(f"Building vocabulary from {train_file}...")

    # 1. go thru file and get token frequencies
    counter = Counter()
    with open(train_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:  # empty lines
                tokens = tokenize_fun(line)
                counter.update(tokens)

    print(f"Found {len(counter)} unique raw tokens.")

    # 2. create vocabulary mappings
    str_to_int = {}
    int_to_str = {}

    special_tokens = [bos_token, eos_token, unk_token, pad_token]
    for i, token in enumerate(special_tokens):  # add special symbols
        str_to_int[token] = i
        int_to_str[i] = token

    # 3. voc_size truncation (at most (max_voc_size - 4) words)
    num_words_to_keep = -1  # default = keep all
    if max_voc_size is not None:
        num_words_to_keep = max_voc_size - len(special_tokens)

        # handle error
        if num_words_to_keep < 0:
            print(
                f"Warning: max_voc_size ({max_voc_size}) is smaller than number of special tokens ({len(special_tokens)})."
            )
            num_words_to_keep = 0

    # get most common words
    if num_words_to_keep != -1:
        # truncate to n most common tokens
        most_common_word_freqs = counter.most_common(num_words_to_keep)
        most_common_words = [word for word, freq in most_common_word_freqs]
        print(
            f"Truncating vocabulary to {num_words_to_keep} most common words + {len(special_tokens)} special tokens."
        )
    else:
        # keep all unique tokens
        most_common_words = list(counter.keys())
        print(
            f"Keeping all {len(most_common_words)} unique words + {len(special_tokens)} special tokens."
        )

    # 4. add og words
    current_index = len(special_tokens)  # start from index 4
    for word in most_common_words:
        # make sure words donot overwrite special token strings
        if word not in str_to_int:
            str_to_int[word] = current_index
            int_to_str[current_index] = word
            current_index += 1

    print(f"Final vocabulary size: {len(str_to_int)}")

    # 5. return a Tokenizer object
    return A1Tokenizer(
        str_to_int=str_to_int,
        int_to_str=int_to_str,
        tokenize_fun=tokenize_fun,
        model_max_length=model_max_length,
        pad_token=pad_token,
        unk_token=unk_token,
        bos_token=bos_token,
        eos_token=eos_token,
    )


# changed the order here to define A1Tokenizer first
class A1Tokenizer:
    """A minimal implementation of a tokenizer similar to tokenizers in the HuggingFace library."""

    def __init__(
        self,
        str_to_int,
        int_to_str,
        tokenize_fun,
        model_max_length,
        pad_token,
        unk_token,
        bos_token,
        eos_token,
    ):
        """Initialize the tokenizer.

        Args:
            str_to_int: Mapping from token string to integer ID.
            int_to_str: Mapping from integer ID to token string.
            tokenize_fun: Function to split text into string tokens.
            model_max_length: Default max length for truncation.
            pad_token: String for the padding token.
            unk_token: String for the unknown token.
            bos_token: String for the beginning-of-sequence token.
            eos_token: String for the end-of-sequence token.
        """
        self.str_to_int = str_to_int
        self.int_to_str = int_to_str
        self.tokenize_fun = tokenize_fun

        # special token strings
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.bos_token = bos_token
        self.eos_token = eos_token

        # special token IDs
        self.pad_token_id = self.str_to_int[self.pad_token]
        self.unk_token_id = self.str_to_int[self.unk_token]
        self.bos_token_id = self.str_to_int[self.bos_token]
        self.eos_token_id = self.str_to_int[self.eos_token]

        # set max length
        self.model_max_length = int(model_max_length)

    def __call__(self, texts, truncation=False, padding=False, return_tensors=None):
        """Tokenize the given texts and return a BatchEncoding containing the integer-encoded tokens.

        Args:
          texts:           The texts to tokenize.
          truncation:      Whether the texts should be truncated to model_max_length.
          padding:         Whether the tokenized texts should be padded on the right side.
          return_tensors:  If None, then return lists; if 'pt', then return PyTorch tensors.

        Returns:
          A BatchEncoding where the field `input_ids` stores the integer-encoded texts and !OPTIONAL BUT I IMPLEMENTED! `attention_mask` stores the corresponding attention mask.
        """
        if return_tensors and return_tensors != "pt":
            raise ValueError("return_tensors must be None or 'pt'")

        if isinstance(texts, str):  # if single string input
            texts = [texts]

        # check max length
        effective_max_length = self.model_max_length
        if truncation and not effective_max_length:
            raise ValueError("Truncation set to True, but no max_length provided.")

        # 1. encode all texts
        all_input_ids = []
        for text in texts:
            # split word
            tokens = self.tokenize_fun(text)

            # convert to IDs (also handles unknown tokens)
            ids = [self.str_to_int.get(token, self.unk_token_id) for token in tokens]

            # add special tokens
            ids = [self.bos_token_id] + ids + [self.eos_token_id]

            # apply truncation if needed
            if truncation and effective_max_length and len(ids) > effective_max_length:
                ids = ids[:effective_max_length]
                # make sure the last token is EOS if we truncate it off
                if ids[-1] != self.eos_token_id:
                    ids[-1] = self.eos_token_id

            all_input_ids.append(ids)

        # 2. padding
        batch_input_ids = []
        batch_attention_mask = []

        if not padding:
            batch_input_ids = all_input_ids
        else:
            # find the longest sequence in the batch
            max_len_in_batch = max(len(ids) for ids in all_input_ids)

            # apply padding and create attention masks
            for ids in all_input_ids:
                pad_len = max_len_in_batch - len(ids)

                padded_ids = ids + [self.pad_token_id] * pad_len
                attention_mask = [1] * len(ids) + [0] * pad_len

                batch_input_ids.append(padded_ids)
                batch_attention_mask.append(attention_mask)

        # 3. convert to tensors if needed
        if return_tensors == "pt":
            batch_input_ids = torch.tensor(batch_input_ids, dtype=torch.long)
            if padding:
                batch_attention_mask = torch.tensor(
                    batch_attention_mask, dtype=torch.long
                )

        # 4. return BatchEncoding
        batch_encoding_dict = {"input_ids": batch_input_ids}
        if padding:
            batch_encoding_dict["attention_mask"] = batch_attention_mask

        return BatchEncoding(batch_encoding_dict)

    def __len__(self):
        """Return the size of the vocabulary."""
        return len(self.str_to_int)

    def save(self, filename):
        """Save the tokenizer to the given file."""
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def from_file(filename):
        """Load a tokenizer from the given file."""
        with open(filename, "rb") as f:
            return pickle.load(f)


###
### Part 3. Defining the model.
###


class A1RNNModelConfig(PretrainedConfig):
    """Configuration object that stores hyperparameters that define the RNN-based language model."""

    def __init__(
        self, vocab_size=None, embedding_size=None, hidden_size=None, **kwargs
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size


class A1RNNModel(PreTrainedModel):
    """The neural network model that implements a RNN-based language model."""

    config_class = A1RNNModelConfig

    def __init__(self, config):
        super().__init__(config)
        assert (
            config.vocab_size is not None
            and config.embedding_size is not None
            and config.hidden_size is not None
        )

        self.embedding = nn.Embedding(
            num_embeddings=config.vocab_size, embedding_dim=config.embedding_size
        )
        self.rnn = nn.GRU(
            input_size=config.embedding_size,
            hidden_size=config.hidden_size,
            batch_first=True,
        )
        self.unembedding = nn.Linear(
            in_features=config.hidden_size, out_features=config.vocab_size
        )

    def forward(self, X):
        """The forward pass of the RNN-based language model.

        Args:
          X:  The input tensor (2D), consisting of a batch of integer-encoded texts.
        Returns:
          The output tensor (3D), consisting of logits for all token positions for all vocabulary items.
        """
        embedded = self.embedding(X)
        rnn_out, _ = self.rnn(embedded)
        out = self.unembedding(rnn_out)
        return out


# ###
# ### Part 4. Training the language model.
# ###

# ## Hint: the following TrainingArguments hyperparameters may be relevant for your implementation:
# #
# # - optim:            What optimizer to use. You can assume that this is set to 'adamw_torch',
# #                     meaning that we use the PyTorch AdamW optimizer.
# # - eval_strategy:    You can assume that this is set to 'epoch', meaning that the model should
# #                     be evaluated on the validation set after each epoch
# # - use_cpu:          Force the trainer to use the CPU; otherwise, CUDA or MPS should be used.
# #                     (In your code, you can just use the provided method select_device.)
# # - learning_rate:    The optimizer's learning rate.
# # - num_train_epochs: The number of epochs to use in the training loop.
# # - per_device_train_batch_size:
# #                     The batch size to use while training.
# # - per_device_eval_batch_size:
# #                     The batch size to use while evaluating.
# # - output_dir:       The directory where the trained model will be saved.

# class A1Trainer:
#     """A minimal implementation similar to a Trainer from the HuggingFace library."""

#     def __init__(self, model, args, train_dataset, eval_dataset, tokenizer):
#         """Set up the trainer.

#            Args:
#              model:          The model to train.
#              args:           The training parameters stored in a TrainingArguments object.
#              train_dataset:  The dataset containing the training documents.
#              eval_dataset:   The dataset containing the validation documents.
#              eval_dataset:   The dataset containing the validation documents.
#              tokenizer:      The tokenizer.
#         """
#         self.model = model
#         self.args = args
#         self.train_dataset = train_dataset
#         self.eval_dataset = eval_dataset
#         self.tokenizer = tokenizer

#         assert(args.optim == 'adamw_torch')
#         assert(args.eval_strategy == 'epoch')

#     def select_device(self):
#         """Return the device to use for training, depending on the training arguments and the available backends."""
#         if self.args.use_cpu:
#             return torch.device('cpu')
#         if not self.args.no_cuda and torch.cuda.is_available():
#             return torch.device('cuda')
#         if torch.mps.is_available():
#             return torch.device('mps')
#         return torch.device('cpu')

#     def train(self):
#         """Train the model."""
#         args = self.args

#         device = self.select_device()
#         print('Device:', device)
#         self.model.to(device)

#         loss_func = torch.nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)

#         # TODO: Relevant arguments: at least args.learning_rate, but you can optionally also consider
#         # other Adam-related hyperparameters here.
#         optimizer = torch.optim.AdamW(...)

#         # TODO: Relevant arguments: args.per_device_train_batch_size, args.per_device_eval_batch_size
#         train_loader = DataLoader(...)
#         val_loader = DataLoader(...)

#         # TODO: Your work here is to implement the training loop.
#         #
#         # for each training epoch (use args.num_train_epochs here):
#         #   for each batch B in the training set:
#         #
#         #       PREPROCESSING AND FORWARD PASS:
#         #       input_ids = apply your tokenizer to B
# 	      #       X = all columns in input_ids except the last one
# 	      #       Y = all columns in input_ids except the first one
# 	      #       put X and Y onto the GPU (or whatever device you use)
#         #       apply the model to X
#         #   	compute the loss for the model output and Y
#         #
#         #       BACKWARD PASS AND MODEL UPDATE:
#         #       optimizer.zero_grad()
#         #       loss.backward()
#         #       optimizer.step()

#         print(f'Saving to {args.output_dir}.')
#         self.model.save_pretrained(args.output_dir)
