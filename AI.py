from main import *

print("""
This is an N-grams AI model built on the foundations of the Google deepmind AI research. It is designed to generate text based on the statistical properties of n-grams, which are contiguous sequences of n items from a given sample of text or speech.
    
When you input a prompt, you can either select a dataset from the provided list (or a random one would be used) that the AI uses to generate text, or you provide one (could be a comprehension passage, an essay, article or something) and the AI generates it's text within the context of your provided data set.

This model is a simple model and does keep track of your prompt and it's generated text.
    
    NB: Any dataset you provide should be in the same format as the dataset dtatset, you can view it in: https://storage.googleapis.com/dm-educational/assets/ai_foundations/dataset.json
        Or to put it straight, it should be a list of dictionaries with the following keys: 'category', 'name', and 'description'.
""")

dataset = []
data = download_data()
default = data is not None
if default:
    names = data["name"].tolist()
    category = tuple(set(data["category"].tolist()))
    category_subset = {x:tuple(y[1] for y in zip(data["category"].tolist(), data["name"].tolist()) if y[0] == x) for x in category}
    dataset = data["description"]  # type: ignore
    datasubset = {x[0]:x[1] for x in zip(names, dataset)}


while True:
    print("1 - View Dataset") if default else print("1 - Input Dataset from file or enter URL")
    prompt = input("""2 - Change Dataset
3 - Input prompt
4 - Exit
    """)

    if prompt == "4": break

    elif prompt == "1":
        if default:
            hp = True
            while hp:
                print(f"The default dataset contains {len(dataset)} entries in the following categories:")
                for cat in range(len(category)):
                    print(f"{cat}  - {category[cat]}")
                cp = True
                while cp:
                    prompt = input("Select a category to view the dataset: ")
                    if not prompt.strip().isdecimal(): print("wrong input!!!"); continue
                    try: 
                        cat = category[int(prompt.strip())]
                        temp = category_subset[cat] #type: ignore
                        print(f"DATA UNDER CATEGORY {cat.upper()} ({len(temp)}):")
                        for x in range(len(temp)): print(f"{x} - {temp[x]}")
                        tempA = input("Select one of the above to view it's data or press enter to go to homepage or type 'up' to go back to category selection page: ")
                        while not tempA.strip().isdecimal():
                            print("wrong input!!!")
                            tempA = input("-> ")

                        if tempA.strip() == 'up': break
                        elif not tempA.strip(): hp = False; break
                        else:
                            try:
                                nam = temp[int(tempA.strip())]
                                print(f"DATA FOR {nam}:")
                                print(datasubset[nam])
                                input("Press enter to go back to the previous page...")
                            except KeyError: print("Invalid selection!!!")
                    except KeyError: print("Invalid selection!!!")
        else:
            if not dataset:
                prompt = input

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
