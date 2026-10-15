import random # For sampling from probability distributions.
from collections import Counter, defaultdict # For counting n-grams.
import pandas as pd # For constructing and visualizing tables.
import requests

from io import StringIO

from urllib.parse import urlparse
from pathlib import Path

def classify(text) -> bool | None:
    text = text.strip()

    parsed = urlparse(text)
    if parsed.scheme in ("http", "https", "ftp") and parsed.netloc:
        return True

    if Path(text).exists():
        return False

    return None


def space_tokenize(text: str) -> list[str]:
    """Splits a string into a list of words (tokens).

    Splits text on space.

    Args:
        text: The input text.

    Returns:
        A list of tokens. Returns empty list if text is empty or all spaces.
    """
    tokens = text.split(" ")
    return tokens

def generate_ngrams(text: str, n: int) -> list[tuple[str]]:
    """Generates n-grams from a given text.

    Args:
        text: The input text string.
        n: The size of the n-grams (e.g., 2 for bigrams, 3 for trigrams).

    Returns:
        A list of n-grams, each represented as a list of tokens.
    """

    # Tokenize text.
    tokens = space_tokenize(text)  # Add your code here.

    # Construct the list of n-grams.
    ngrams = []

    # Add your code here.
    for i in range(len(tokens) - n + 1):
        ngrams.append(tuple(tokens[i:i+n]))

    return ngrams

def get_ngram_counts(dataset: list[str], n: int) -> dict[str, Counter]:
    """Computes the n-gram counts from a dataset.

    This function takes a list of text strings (paragraphs or sentences) as
    input, constructs n-grams from each text, and creates a dictionary where:

    * Keys represent n-1 token long contexts `context`.
    * Values are a Counter object `counts` such that `counts[next_token]` is the
      count of `next_token` following `context`.

    Args:
        dataset: The list of text strings in the dataset.
        n: The size of the n-grams to generate (e.g., 2 for bigrams, 3 for
            trigrams).

    Returns:
        A dictionary where keys are (n-1)-token contexts and values are Counter
        objects storing the counts of each next token for that context.

    """

    # Define the dictionary as a defaultdict that is automatically initialized
    # with an empty Counter object. This allows you to access and set the value
    # of ngram_counts[context][next_token] without initializing
    # ngram_counts[context] or ngram_counts[context][next_token] first.
    # Reference
    # https://docs.python.org/3/library/collections.html#collections.Counter and
    # https://docs.python.org/3/library/collections.html#collections.defaultdict
    # for more information on how to use defaultdict and Counter types.
    ngram_counts = defaultdict(Counter)

    for paragraph in dataset:
        # Add your code here.
        ngram = generate_ngrams(paragraph, n)
        for x in ngram:
            context = " ".join(x[:-1])      # the first n-1 tokens
            next_word = x[-1]     # the last token
            ngram_counts[context][next_word] += 1
            # ngram_counts[" ".join(x[:-1])][Counter(x[-1])]

    return dict(ngram_counts)

def build_ngram_model(
    dataset: list[str],
    n: int
) -> dict[str, dict[str, float]]:
    """Builds an n-gram language model.

    This function takes a list of text strings (paragraphs or sentences) as
    input, generates n-grams from each text using the function get_ngram_counts
    and converts them into probabilities.  The resulting model is a dictionary,
    where keys are (n-1)-token contexts and values are dictionaries mapping
    possible next tokens to their conditional probabilities given the context.

    Args:
        dataset: A list of text strings representing the dataset.
        n: The size of the n-grams (e.g., 2 for a bigram model).

    Returns:
        A dictionary representing the n-gram language model, where keys are
        (n-1)-tokens contexts and values are dictionaries mapping possible next
        tokens to their conditional probabilities.
    """

    # A dictionary to store P(B | A).
    # ngram_model[context][token] should store P(token | context).
    ngram_model = {}

    # Use the ngram_counts as computed by the get_ngram_counts function.
    ngram_counts = get_ngram_counts(dataset, n)

    # Loop through the possible contexts. `context` is a string
    # and `next_tokens` is a dictionary mapping possible next tokens to their
    # counts of following `context`.
    for context, next_tokens in ngram_counts.items():

        # Compute Count(A) and P(B | A) here.
        # Add your code here.
        ngram_count_context = sum(ngram_counts[context].values())
        ngram_model[context] = {x:next_tokens[x]/ngram_count_context for x in next_tokens}

    return ngram_model

def generate_next_n_tokens(
    n: int,
    ngram_model: dict[str, dict[str, float]],
    prompt: str,
    num_tokens_to_generate: int,
) -> str:
    """Generates `num_tokens_to_generate` tokens following a given prompt using
    an n-gram language model.

    This function takes an n-gram model and uses it to predict the most
    likely next token for the given prompt. The generation process
    continues iteratively, appending predicted tokens to the prompt until the
    desired number of tokens is generated or a context is
    encountered for which the model has no predictions.

    Args:
        n: The size of the n-grams to use (e.g., 2 for a bigram model).
        ngram_model: A dictionary representing the n-gram language model.
        prompt: The starting text prompt for generating the next tokens.
        num_tokens_to_generate: The number of words to generate following
            the prompt.

    Returns:
        A string containing the original prompt followed by the generated
        tokens. If no valid continuation is found for a given context, the
        function will return the text generated up to that point and print a
        message indicating that no continuation could be found.
    """

    # Split prompt into individual tokens.
    generated_words = space_tokenize(prompt); tokens_count = 0

    for _ in range(num_tokens_to_generate):
        # Get last (n-1) tokens as context.
        context = generated_words[-(n - 1):]
        context = " ".join(context)
        if context in ngram_model:
            # Sample next word based on probabilities.
            next_word = random.choices(
                list(ngram_model[context].keys()),
                weights=ngram_model[context].values() # type: ignore
            )[0]

            generated_words.append(next_word); tokens_count += 1
            if tokens_count >= int(num_tokens_to_generate/4):
                if tuple(next_word.strip())[-1] == ".": break
        else:
            print(
                "⚠️ No valid continuation found. Change the prompt or"
                " try sampling another continuation.\n"
            )
            break
    else: generated_words.append("...")

    return " ".join(generated_words)

def download_data(url: str) -> pd.DataFrame | None:
    """Downloads the Africa Galore dataset from the provided URL.

    Args:
        url: The URL to download the dataset from.

    Returns:
        A pandas DataFrame representing the downloaded dataset.
    """
    try: 
        response = requests.get(url)
        data =  pd.read_json(StringIO(response.text))
        return data
    except requests.exceptions.ConnectionError:
        print("⚠️ The dataset was unable to download, please check your internet connection and try again.\n")
        return None

def sort_data(data):
    names = data["name"].tolist()
    category = tuple(set(data["category"].tolist()))
    category_subset = {x:tuple(y[1] for y in zip(data["category"].tolist(), data["name"].tolist()) if y[0] == x) for x in category}
    dataset: list[str] = data["description"]  # type: ignore
    datasubset = {x[0]:x[1] for x in zip(names, dataset)}
    return {'category':category, 'category_subset':category_subset, 'dataset':dataset, 'datasubset':datasubset}

def get_data(path: str = "https://storage.googleapis.com/dm-educational/assets/ai_foundations/africa_galore.json"):
    """Reads a dataset from a local file or URL.

    Args:
        path: The path to the dataset file or the URL to download it from.

    Returns:
        A pandas DataFrame representing the dataset.
    """

    try:
        if classify(path):
            return download_data(path), path
        elif classify(path) is None:
            print("⚠️ The provided path is neither a valid URL nor a local file path. Please provide a valid path or URL.\n")
        else:
            try: return pd.read_json(path), path
            except FileNotFoundError: print("⚠️ The specified file was not found. Please check the file path and try again.\n")
    except ValueError as e:
        if str(e).strip().lower() == 'trailing data': 
            try: pd.read_json(path, lines=True)
            except ValueError: pass
        print("Please check that the file or URL content is in the correct JSON format")

def display_data(data: pd.DataFrame, default: bool = True) -> list[str]:
    temp = sort_data(data)
    category = temp['category']
    category_subset = temp['category_subset']
    dataset = temp['dataset']
    datasubset = temp['datasubset']

    txt = "default" if default else "provided"
    while True:
        print(f"The {txt} dataset contains {len(dataset)} entries in the following categories:")
        for cat in range(len(category)):
            print(f"{cat}  - {category[cat]}")
        while True:
            prompt = input("Select a category to view the dataset or press enter to continue: ")
            if not prompt.strip(): return dataset
            if not prompt.strip().isdecimal(): print("wrong input!!!"); continue
            try: 
                cat = category[int(prompt.strip())]
                temp = category_subset[cat] #type: ignore
                print(f"DATA UNDER CATEGORY {cat.upper()} ({len(temp)}):")
                for x in range(len(temp)): print(f"{x} - {temp[x]}")
                tempA = input("Select one of the above to view it's data or press enter to go to homepage or type 'up' to go back to category selection page: ")

                if tempA.strip() == 'up': break
                elif not tempA.strip():  return dataset
                else:
                    while not tempA.strip().isdecimal():
                        print("wrong input!!!")
                        tempA = input("-> ")
                    try:
                        nam = temp[int(tempA.strip())]
                        print(f"DATA FOR {nam}:")
                        print(datasubset[nam])
                        input("Press enter to go back to the previous page...")
                    except KeyError: print("Invalid selection!!!")
            except KeyError: print("Invalid selection!!!")