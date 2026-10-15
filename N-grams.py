from main import *

africa_galore = pd.read_json("/mnt/d/My SD Card/Chromebook/Projects/vs. code/N Grams AI model/africa_galore.json")
dataset: list[str] = africa_galore["description"] # type: ignore
#print(f"The dataset consists of {dataset.shape[0]} paragraphs.")

#for paragraph in dataset[:10]:
#    # textwrap automatically adds line breaks to make long texts more readable.
#    formatted_paragraph = textwrap.fill(paragraph)
#    print(f"{formatted_paragraph}\n")


# Tokenize an example text with the `space_tokenize` function.
#space_tokenize(dataset[0])


# all_unigrams = []
# all_bigrams = []
# all_trigrams = []


# for paragraph in dataset:
#     # Calling `generate_ngrams` with n=1 constructs a list of unigrams.
#     all_unigrams.extend(generate_ngrams(paragraph, n=1))
#     # Calling `generate_ngrams` with n=2 constructs a list of bigrams (2-grams).
#     all_bigrams.extend(generate_ngrams(paragraph, n=2))
#     # Calling `generate_ngrams` with n=2 constructs a list of trigram (3-grams).
#     all_trigrams.extend(generate_ngrams(paragraph, n=3))

# print("First 10 Unigrams:", all_unigrams[:10])
# print("First 10 Bigrams:", all_bigrams[:10])
# print("First 10 Trigrams:", all_trigrams[:10])

# # Use the Python Counter data type for computing the counts of all bigrams.
# # See: https://docs.python.org/3/library/collections.html#collections.Counter
# bigram_counts = Counter(all_bigrams)

# # Print the ten most common bigrams.
# print("Most common bigrams:")
# for bigram, count in bigram_counts.most_common(10):
#     print(f"  ({bigram}, {count})")

# # Use the Python Counter data type for computing the counts of all trigrams.
# trigram_counts = Counter(all_trigrams)

# # Print the ten most common trigrams.
# print("\n\nMost common trigrams:")
# for trigram, count in trigram_counts.most_common(10):
#     print(f"  ({trigram}, {count})")



# # Example usage of the function.
# example_data = [
#     "This is an example sentence.",
#     "Another example sentence.",
#     "Split a sentence."
# ]
# ngram_counts = get_ngram_counts(example_data, 2)

# # Print the bigram counts dictionary for the dataset consisting of the
# # three example sentences.
# print("Bigram counts dictionary:\n")
# print("{")
# for context, counter in ngram_counts.items():
#     print(f"  '{context}': {counter},")
# print("}")

# bigram_counts = get_ngram_counts(dataset, n=2)

# # Use the pandas library to display the counts in a table.
# bigram_counts_matrix = {
#     context: dict(counts) for context, counts in bigram_counts.items()
# }
# bigram_data_frame = pd.DataFrame.from_dict(
#     bigram_counts_matrix, orient="index").fillna(0)

# display(bigram_data_frame)

# zero_count = (bigram_data_frame == 0).sum().sum()
# print(
#     f"Number of bigrams with a count of 0: {zero_count:,}"
#     f" ({zero_count/bigram_data_frame.size * 100:.2f}%)"
# )

# trigram_counts = get_ngram_counts(dataset, n=3)

# # Use the pandas library to display the counts in a table.
# trigram_counts_matrix = {
#     context: dict(counts) for context, counts in trigram_counts.items()
# }
# trigram_data_frame = pd.DataFrame.from_dict(
#     trigram_counts_matrix, orient="index").fillna(0)

# display(trigram_data_frame)

# zero_count = (trigram_data_frame == 0).sum().sum()
# print(
#     f"Number of trigrams with a count of 0: {zero_count:,}"
#     f" ({zero_count/trigram_data_frame.size * 100:.2f}%)"
# )

# context = "a staple"
# trigram_counts[context]

# context = "a staple"
# # Compute the bigram count for "a staple" with sum().
# bigram_count_a_staple = sum(trigram_counts[context].values())

# print(
#     'Bigram count of "a staple" computed indirectly from trigram counts: ',
#     bigram_count_a_staple,
# )

# # Extract the bigram count for "a staple" from bigram_counts.
# print('Bigram count of "a staple" computed directly: ',
#       bigram_counts["a"]["staple"])



# # Test the method above by bulding a simple trigram model.
# test_dataset = ["Table Mountain is tall.", "Table Mountain is beautiful."]
# test_trigram_model = build_ngram_model(test_dataset, n=3)
# print(test_trigram_model)

# # @title Run this cell to test your implementation.
# ngrams.test_build_ngram_model(build_ngram_model, get_ngram_counts)

# trigram_model = build_ngram_model(dataset, n=3)
trigram_model = build_ngram_model([
    '''jollof rice is a beloved one-pot rice dish from west africa , characterized by its vibrant reddish-orange color and rich , often slightly spicy , tomato-based flavor infused with onions , peppers , garlic , ginger , curry powder , and thyme , all cooked in stock.'''
], n=3)

# print(f"P(B | \"as it\") = {trigram_model['as it']}")
# print(f"P(B | \"as they\") = {trigram_model['as they']}")

# context = "The name"
# print(trigram_model[context])
# context = "looking for"
# tokens = trigram_model[context]
# candidate_tokens = [*tokens.keys()]
# candidate_tokens_probabilities = [*tokens.values()]

# trigram_counts = get_ngram_counts([
#     '''jollof rice is a beloved one-pot rice dish from west africa , characterized by its vibrant reddish-orange color and rich , often slightly spicy , tomato-based flavor infused with onions , peppers , garlic , ginger , curry powder , and thyme , all cooked in stock.'''
# ], n=3)
# Use the pandas library to display the counts in a table.
# print(trigram_counts)
# trigram_counts_matrix = {
#     context: dict(counts) for context, counts in trigram_counts.items()
# }
# trigram_data_frame = pd.DataFrame.from_dict(
#     trigram_counts_matrix, orient="index").fillna(0)

# zero_count = (trigram_data_frame == 0).sum().sum()
# print(
#     f"Number of trigrams with a count of 0:", end=" "
# )

# tokens = '''jollof rice is a beloved one-pot rice dish from west africa , characterized by its vibrant reddish-orange color and rich , often slightly spicy , tomato-based flavor infused with onions , peppers , garlic , ginger , curry powder , and thyme , all cooked in stock .'''.split()          # your lowercase, punctuation-separated text
# vocab = sorted(set(tokens))    # 38 unique words

# full_index = pd.MultiIndex.from_product([vocab, vocab])
# full_frame = trigram_data_frame.reindex(index=full_index, columns=vocab).fillna(0)

# display(full_frame)
# zero_count = (full_frame == 0).sum().sum()
# print(zero_count)   # 54826
# # Extract candidate tokens and associated probabilities from `trigram_model`.
# # Add your code here.


# print(f"Candidate tokens: {candidate_tokens}")
# print(f"Candidate token probabilities: {candidate_tokens_probabilities}")

# # Sample from the list of candidate tokens according to the
# # associated probabilities.
# next_token = random.choices(candidate_tokens,
#                             weights=candidate_tokens_probabilities)[0]

# print("\n\nSampled next token:")
# print(context, next_token)


# #Bigram Model
# prompt = "Jide was hungry so she went looking for"

# # Construct a bigram model using the Africa Galore dataset.
# bigram_model = build_ngram_model(dataset, n=2)

# n = 2  # Bigram.
# num_tokens_to_generate = 10  # Generate next n words.
# generate_next_n_tokens(
#     n=n,
#     ngram_model=bigram_model,
#     prompt=prompt,
#     num_tokens_to_generate=num_tokens_to_generate,
# )

# #Trigram Model
# prompt = "Jide was hungry so she went looking for"

# n = 3  # Trigram.
# num_tokens_to_generate = 10  # Generate next n words.
# generate_next_n_tokens(
#     n=n,
#     ngram_model=trigram_model,
#     prompt=prompt,
#     num_tokens_to_generate=num_tokens_to_generate,
# )

