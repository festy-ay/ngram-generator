from main import *

print("""
This is an N-grams AI model built on the foundations of the Google deepmind AI research. It is designed to generate text based on the statistical properties of n-grams, which are contiguous sequences of n items from a given sample of text or speech.
    
When you input a prompt, you can either select a dataset from the provided list (or a random one would be used) that the AI uses to generate text, or you provide one (could be a comprehension passage, an essay, article or something) and the AI generates it's text within the context of your provided data set.

This model is a simple model and does keep track of your prompt and it's generated text.
    
    NB: Any dataset you provide should be in the same format as the defalt dataset, you can view it in: https://storage.googleapis.com/dm-educational/assets/ai_foundations/dataset.json
        Or to put it straight, it should be a list of dictionaries with the following keys: 'category', 'name', and 'description' or else the Ngrams AI might not function as expected.
        No downloaded data is saved!!!
""")

dataset = []
data = get_data()
default = data is not None
if default: dataset = display_data(data[0])

while True:
    print("1 - View Dataset") if default else print("1 - Input Dataset from file, enter URL or download default dataset")
    prompt = input("""2 - Change Dataset\n3 - Input prompt\n4 - Exit\n""")

    if prompt.strip() == "4": break

    elif prompt == "1" or prompt == "2":
        if default and prompt == "1":
            dataset = display_data(data[0])
        else:
            if dataset: print(f"Current dataset: {data[1]}")
            prompt = input("Press enter to download default dataset, enter a file path or enter a URL to download a dataset from the web: ")
            if not prompt.strip(): data = get_data(); default = data is not None

    elif default and prompt == "3":
        prompt = input("""
Select Ngrams Model:
    1 - Unigrams
    2 - Bigrams
    3 - Trigrams
    4 - Enter N manually
    5 - Generate random text
""")
        n: int; ngram_model: dict[str, dict[str, float]]

        if 0 < int(prompt.strip()) <= 3: n = int(prompt.strip()); ngram_model = build_ngram_model(dataset, n)

        elif int(prompt.strip()) == 4:
            prompt = input("Enter a number from 1 - 10: ")
            while int(prompt.strip()) > 10:
                print("Invalid selection. Please enter a number from 1 - 10.")
                prompt = input("-> ")
            n = int(prompt.strip()); ngram_model = build_ngram_model(dataset, n)

        elif int(prompt.strip()) == 5: n = random.randint(1, 10); ngram_model = build_ngram_model(dataset, n)

        else: print("Invalid selection. Please enter a number only."); continue


        prompt = input("-> ")
        while not prompt.strip() or len(space_tokenize(prompt)) < n:
            if not prompt.strip():
                print("Your prompt cannot be empty. Please enter a valid prompt.")
            else:
                print(f"Your prompt is too short. Please enter a prompt with at least {n} words.")
            prompt = input("-> ")
        print(f"Generating text based on the prompt: '{prompt}' using a {n}-gram model...")
        print(generate_next_n_tokens(n, ngram_model, prompt, 50))

    else: print("No Dataset selected, please download or input one to continue (Select 2 to do that).")
