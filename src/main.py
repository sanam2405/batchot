# !pip install openai pinecone-client transformers

import openai
from pinecone import Pinecone, ServerlessSpec
from transformers import GPT2Tokenizer, GPT2Model
import torch

OPENAI_API_KEY = "OPENAI_API_KEY"
PINECONE_API_KEY = "PINCECONE_API_KEY"
PINECONE_CLOUD = "aws"
PINECONE_REGION = "us-east-1"

# Initialize OpenAI client
openai.api_key = OPENAI_API_KEY

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_REGION)

MAX_TOKENS = 225
PINECONE_INDEX_NAME = "batchot"

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


def add_embeddings_to_pinecone(history, index_name=PINECONE_INDEX_NAME):
    # Initialize the tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2Model.from_pretrained("gpt2")

    # Create Pinecone index if it doesn't exist
    if index_name not in pc.list_indexes():
        print(f"Existing indexes: {pc.list_indexes()}")
        try:
            pc.create_index(
                name=index_name,
                dimension=768,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
            print(f"Index {index_name} created successfully.")

        except Exception as e:
            print(f"Error creating index: {e}")
            return

    index = pc.Index(index_name)

    # Encode and upsert each message
    for i, message in enumerate(history):
        inputs = tokenizer(
            message, return_tensors="pt", truncation=True, max_length=512
        )
        with torch.no_grad():
            outputs = model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze().tolist()

        try:
            index.upsert(vectors=[(str(i), embedding, {"text": message})])
            print(f"Inserted message {i}: {message}")
        except Exception as e:
            print(f"Error upserting vector {i}: {e}")


add_embeddings_to_pinecone(history)


def retrieve_relevant_history(query, index_name=PINECONE_INDEX_NAME):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2Model.from_pretrained("gpt2")

    inputs = tokenizer(query, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    query_embedding = outputs.last_hidden_state.mean(dim=1).squeeze().tolist()

    index = pc.Index(index_name)
    results = index.query(vector=query_embedding, top_k=3, include_metadata=True)

    return [result.metadata["text"] for result in results.matches]


def prepare_prompt(test_prompt, history, index_name=PINECONE_INDEX_NAME):
    relevant_messages = retrieve_relevant_history(test_prompt, index_name)
    context = "\n".join(relevant_messages)
    combined_prompt = f"{context}\nUser: {test_prompt}\nBot:"

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokens = tokenizer.encode(combined_prompt)

    if len(tokens) > MAX_TOKENS:
        tokens = tokens[-MAX_TOKENS:]
        combined_prompt = tokenizer.decode(tokens)

    return combined_prompt, context, tokenizer


def test_final_prompt():
    final_test_prompt = "Do you think it will help me stay fit?"

    prepared_prompt, context_referred, tokenizer = prepare_prompt(
        final_test_prompt, history
    )

    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        # model="gpt-3.5-turbo-1106",
        # model="gpt-4o-mini",
        # model="text-embedding-3-small",
        # model="gpt-4"
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prepared_prompt},
        ],
        max_tokens=MAX_TOKENS - len(tokenizer.encode(prepared_prompt)),
    )

    print(f"Final Test Prompt: {final_test_prompt}")
    print(f"Context Referred: {context_referred}")
    print(
        f"Final Test Prompt Response: {response.choices[0].message['content'].strip()}"
    )


# Call the test function to generate the Final Test Prompt Response
test_final_prompt()
