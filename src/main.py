# !pip install openai pinecone-client transformers

# Import necessary libraries
import openai  # For interacting with OpenAI's API
from pinecone import (
    Pinecone,
    ServerlessSpec,
)  # For vector database operations using Pinecone
from transformers import (
    GPT2Tokenizer,
    GPT2Model,
)  # For text tokenization and embedding generation using GPT-2
import torch  # For numerical operations and tensor manipulation

# Set your OpenAI and Pinecone API keys
OPENAI_API_KEY = "OPENAI_API_KEY"  # Replace with your actual OpenAI API key
PINECONE_API_KEY = "PINECONE_API_KEY"  # Replace with your actual Pinecone API key
PINECONE_CLOUD = "aws"  # Specify the cloud provider for your Pinecone instance
PINECONE_REGION = "us-east-1"  # Specify the region for your Pinecone instance

# Initialize OpenAI client
openai.api_key = OPENAI_API_KEY

# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_REGION)

# Set the maximum number of tokens for the prompt
MAX_TOKENS = 225
# Define the name for the Pinecone index
PINECONE_INDEX_NAME = "batchot"

# Sample conversation history for demonstration
history = [
    "1: User: Hi there! How are you doing today? | Bot: Hello! I'm doing great, thank you! How can I assist you today?",
    "2: User: What's the weather like today in New York? | Bot: Today in New York, it's sunny with a slight chance of rain.",
    "3: User: Great! Do you have any good lunch suggestions? | Bot: Sure! How about trying a new salad recipe?",
    "4: User: That sounds healthy. Any specific recipes? | Bot: You could try a quinoa salad with avocado and chicken.",
    "5: User: Sounds delicious! I'll try it. What about dinner? | Bot: For dinner, you could make grilled salmon with vegetables.",
    "6: User: Thanks for the suggestions! Any dessert ideas? | Bot: How about a simple fruit salad or yogurt with honey?",
    "7: User: Perfect! Now, what are some good exercises? | Bot: You can try a mix of cardio and strength training exercises.",
    "8: User: Any specific recommendations for cardio? | Bot: Running, cycling, and swimming are all excellent cardio exercises.",
    "9: User: I'll start with running. Can you recommend any books? | Bot: 'Atomic Habits' by James Clear is a highly recommended book.",
    "10: User: I'll check it out. What hobbies can I take up? | Bot: You could explore painting, hiking, or learning a new instrument.",
    "11: User: Hiking sounds fun! Any specific trails? | Bot: There are great trails in the Rockies and the Appalachian Mountains.",
    "12: User: I'll plan a trip. What about indoor activities? | Bot: Indoor activities like reading, cooking, or playing board games.",
    "13: User: Nice! Any good board games? | Bot: Settlers of Catan and Ticket to Ride are both excellent choices.",
    "14: User: I'll try them out. Any movie recommendations? | Bot: 'Inception' and 'The Matrix' are must-watch movies.",
    "15: User: I love those movies! Any TV shows? | Bot: 'Breaking Bad' and 'Stranger Things' are very popular.",
    "16: User: Great choices! What about podcasts? | Bot: 'How I Built This' and 'The Daily' are very informative.",
    "17: User: Thanks! What are some good travel destinations? | Bot: Paris, Tokyo, and Bali are amazing travel spots.",
    "18: User: I'll add them to my list. Any packing tips? | Bot: Roll your clothes to save space and use packing cubes.",
    "19: User: That's helpful! What about travel insurance? | Bot: Always get travel insurance for safety and peace of mind.",
    "20: User: Thanks for the tips! Any last advice? | Bot: Just enjoy your journey and make the most out of your experiences.",
]


# Define a function to add embeddings to Pinecone index
def add_embeddings_to_pinecone(history, index_name=PINECONE_INDEX_NAME):
    """
    Adds embeddings for each message in the history to the specified Pinecone index.

    Args:
        history (list): A list of conversation messages.
        index_name (str, optional): The name of the Pinecone index. Defaults to PINECONE_INDEX_NAME.

    The function does the following:
    1. Initializes the GPT-2 tokenizer and model
    2. Creates a Pinecone index if it doesn't exist
    3. Encodes each message in the history and creates an embedding
    4. Upserts the embedding and associated metadata to Pinecone
    """

    # Initialize the GPT-2 tokenizer and model for text processing and embedding generation
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2Model.from_pretrained("gpt2")

    # Check if the specified index exists in Pinecone
    if index_name not in pc.list_indexes():
        print(f"Existing indexes: {pc.list_indexes()}")
        try:
            # Create a new index in Pinecone if it doesn't exist
            pc.create_index(
                name=index_name,  # Name of the index
                dimension=768,  # Dimensionality of the embeddings (768 for GPT-2)
                metric="cosine",  # Distance metric for similarity search (cosine similarity)
                spec=ServerlessSpec(
                    cloud="aws",  # Cloud provider for the index
                    region="us-east-1",  # Region for the index
                ),
            )
            print(f"Index {index_name} created successfully.")

        except Exception as e:
            print(f"Error creating index: {e}")
            return

    # Connect to the Pinecone index
    index = pc.Index(index_name)

    # Iterate through each message in the history
    for i, message in enumerate(history):
        # Tokenize the message using the GPT-2 tokenizer
        inputs = tokenizer(
            message, return_tensors="pt", truncation=True, max_length=512
        )
        # Generate embeddings for the tokenized message using the GPT-2 model
        with torch.no_grad():
            outputs = model(**inputs)
        # Extract the embedding vector as a list
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze().tolist()

        try:
            # Upsert the embedding vector along with the message text into the Pinecone index
            index.upsert(vectors=[(str(i), embedding, {"text": message})])
            print(f"Inserted message {i}: {message}")
        except Exception as e:
            print(f"Error upserting vector {i}: {e}")


# Call the function to add embeddings for the sample conversation history
add_embeddings_to_pinecone(history)


# Define a function to retrieve relevant history from Pinecone index
def retrieve_relevant_history(query, index_name=PINECONE_INDEX_NAME):
    """
    Retrieves relevant history messages from the Pinecone index based on the query.

    Args:
        query (str): The user's query.
        index_name (str, optional): The name of the Pinecone index. Defaults to PINECONE_INDEX_NAME.

    Returns:
        list: A list of relevant history messages.

    The function does the following:
    1. Encodes the query using GPT-2
    2. Searches for similar vectors in Pinecone
    3. Returns the text of the most similar entries
    """

    # Initialize the GPT-2 tokenizer and model for text processing and embedding generation
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2Model.from_pretrained("gpt2")

    # Tokenize the query using the GPT-2 tokenizer
    inputs = tokenizer(query, return_tensors="pt", truncation=True, max_length=512)
    # Generate embeddings for the tokenized query using the GPT-2 model
    with torch.no_grad():
        outputs = model(**inputs)
    # Extract the query embedding vector as a list
    query_embedding = outputs.last_hidden_state.mean(dim=1).squeeze().tolist()

    # Connect to the Pinecone index
    index = pc.Index(index_name)
    # Query the Pinecone index for the top 3 most similar messages to the query embedding
    results = index.query(vector=query_embedding, top_k=3, include_metadata=True)

    # Extract and return the text content of the retrieved messages
    return [result.metadata["text"] for result in results.matches]


# Define a function to prepare the prompt for the OpenAI model
def prepare_prompt(test_prompt, history, index_name=PINECONE_INDEX_NAME):
    """
    Prepares the prompt for the OpenAI model by retrieving relevant history and combining it with the test prompt.

    Args:
        test_prompt (str): The user's test prompt.
        history (list): A list of conversation messages.
        index_name (str, optional): The name of the Pinecone index. Defaults to PINECONE_INDEX_NAME.

    Returns:
        tuple: A tuple containing the combined prompt, the context referred to, and the tokenizer.

    The function does the following:
    1. Retrieves relevant history based on the query
    2. Combines the relevant history with the user's query
    3. Truncates the prompt if it exceeds the maximum token limit
    """

    # Retrieve relevant history messages based on the test prompt
    relevant_messages = retrieve_relevant_history(test_prompt, index_name)
    # Combine the relevant messages into a single string
    context = "\n".join(relevant_messages)
    # Construct the combined prompt by appending the context, test prompt, and "Bot:"
    combined_prompt = f"{context}\nUser: {test_prompt}\nBot:"

    # Initialize the GPT-2 tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    # Encode the combined prompt using the tokenizer
    tokens = tokenizer.encode(combined_prompt)

    # Truncate the combined prompt if it exceeds the maximum token limit
    if len(tokens) > MAX_TOKENS:
        tokens = tokens[-MAX_TOKENS:]
        combined_prompt = tokenizer.decode(tokens)

    # Return the combined prompt, context referred to, and the tokenizer
    return combined_prompt, context, tokenizer


# Define a function to test the final prompt
def test_final_prompt():
    """
    This function tests the entire pipeline by generating a response to a test prompt.

    The function does the following:
    1. Defines a test prompt
    2. Prepares the prompt using the prepare_prompt function
    3. Sends the prepared prompt to the OpenAI API
    4. Prints the results, including the context referred and the model's response
    """

    # Define the final test prompt
    final_test_prompt = "Do you think it will help me stay fit?"

    # Prepare the prompt for the OpenAI model
    prepared_prompt, context_referred, tokenizer = prepare_prompt(
        final_test_prompt, history
    )
    print(prepared_prompt)
    print("##########################")

    # Generate a response from the OpenAI model
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",  # Specify the OpenAI model to use
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant.",
            },  # System message to set the assistant's behavior
            {
                "role": "user",
                "content": prepared_prompt,
            },  # User message containing the prepared prompt
        ],
        max_tokens=MAX_TOKENS
        - len(tokenizer.encode(prepared_prompt)),  # Limit the response length
    )

    # Print the final test prompt, context referred to, and the model's response
    print(f"Final Test Prompt: {final_test_prompt}")
    print(f"Context Referred: {context_referred}")
    print(
        f"Final Test Prompt Response: {response.choices[0].message['content'].strip()}"
    )


# Call the test function to generate the Final Test Prompt Response
test_final_prompt()
