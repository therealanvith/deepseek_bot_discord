import discord
import aiohttp
import asyncio
import re
import os
from discord.ext import commands
from PIL import Image
import pytesseract
from io import BytesIO

# --- CONFIGURATION ---
DISCORD_BOT_TOKEN = os.getenv('BOT_TOKEN')  # Replace with your actual token
OPENROUTER_API_KEY = os.getenv('API_KEY')  # Replace with your actual OpenRouter API key
API_URL = "https://openrouter.ai/api/v1/chat/completions"

# Using your model name
MODEL = "deepseek/deepseek-r1:free"

# --- BOT SETUP ---
intents = discord.Intents.default()
intents.message_content = True  # Enable "Message Content Intent" in Developer Portal
bot = commands.Bot(command_prefix="!", intents=intents)

# --- GLOBAL VARIABLES --- (you can store them in a database if needed)
message_context = {}  # A dictionary to store conversation context per channel

# --- HELPER FUNCTIONS ---
def chunk_text(text: str, max_length: int = 2000):
    """Splits text into chunks of up to max_length characters."""
    return [text[i : i + max_length] for i in range(0, len(text), max_length)]

async def fetch_referenced_message(message: discord.Message):
    """Safely fetch the message that the user is replying to."""
    if not message.reference:
        return None
    ref = message.reference.resolved
    if ref:
        return ref
    try:
        return await message.channel.fetch_message(message.reference.message_id)
    except Exception:
        return None

async def process_image(image_url: str):
    """Uses OCR to extract text from an image URL."""
    async with aiohttp.ClientSession() as session:
        async with session.get(image_url) as resp:
            img_data = await resp.read()
            img = Image.open(BytesIO(img_data))
            text = pytesseract.image_to_string(img)
            return text.strip()

async def get_ai_response(user_prompt: str) -> tuple[str, str]:
    """
    Calls the AI model via OpenRouter's API.
    Uses a system message to demand that the response include 'Reason:' and 'Answer:' sections.
    Returns (answer, reason). If the model doesn't comply, returns (full_response, "N/A").
    """
    system_instructions = (
        "Always include two sections in your response:\n"
        "1) 'Reason:' - your chain-of-thought or your thinking like that of a standard reasoning bot.\n"
        "2) 'Answer:' - your final answer in proper sentences like a textbot.\n"
        "Even for simple prompts, please include both sections."
    )

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    data = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system_instructions},
            {"role": "user", "content": user_prompt}
        ],
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(API_URL, headers=headers, json=data) as resp:
                resp.raise_for_status()
                response_json = await resp.json()
                content = response_json["choices"][0]["message"]["content"]

                if "Reason:" in content and "Answer:" in content:
                    reason_part = content.split("Answer:")[0].split("Reason:")[-1].strip()
                    answer_part = content.split("Answer:")[-1].strip()
                else:
                    return content.strip(), "N/A"

                # Clean up: remove extra blank lines and bold markers
                reason_part = re.sub(r'\n\s*\n+', '\n', reason_part).replace("**", "").strip()
                answer_part = re.sub(r'\n\s*\n+', '\n', answer_part).replace("**", "").strip()

                return answer_part, reason_part

    except Exception as e:
        print(f"Error calling AI API: {str(e)}")
        return "I'm having trouble responding right now. Please try again later.", "Seems like issue connecting to API"


# --- BOT EVENT HANDLERS ---
@bot.event
async def on_ready():
    print(f"Logged in as {bot.user}")


@bot.event
async def on_message(message: discord.Message):
    """
    Processes incoming messages:
    - If replying to the bot or mentioning the bot, builds a prompt.
    - Retrieves the AI response with 'Reason:' and 'Answer:'.
    - Sends only the first message as a reply; all subsequent messages are sent normally.
    """
    if message.author == bot.user:
        return

    # --- Handling Image OCR ---
    if message.attachments:
        # Look for image attachments
        for attachment in message.attachments:
            if attachment.content_type.startswith("image"):
                image_text = await process_image(attachment.url)
                prompt = f"User sent an image with text: {image_text}\nPlease respond with 'Reason:' and 'Answer:' sections."
                break
        else:
            # If no images are present, continue as normal
            prompt = message.content
    else:
        # Handle non-image messages
        prompt = message.content

    # --- Keep track of conversation context ---
    if message.channel.id not in message_context:
        message_context[message.channel.id] = []

    message_context[message.channel.id].append({"author": message.author.name, "content": message.content})

    # If there's a lot of context, you may want to trim older messages
    # This is optional based on how much context you want to provide to the AI
    if len(message_context[message.channel.id]) > 20:  # Limit to 20 messages
        message_context[message.channel.id].pop(0)

    # Combine the context and the current message to create a prompt
    full_context = "\n".join([f"{msg['author']}: {msg['content']}" for msg in message_context[message.channel.id]])

    prompt = f"Context of conversation:\n{full_context}\n\nUser's current message:\n{prompt}\nPlease respond with 'Reason:' and 'Answer:' sections."

    # --- Process AI Response ---
    async with message.channel.typing():
        answer, reason = await get_ai_response(prompt)
        # Format the reasoning with a "Reasoning: " label
        reasoning_message = f"(Reasoning: {reason})"
        answer_message = answer

        # Send reasoning first
        reasoning_chunks = chunk_text(reasoning_message)
        if reasoning_chunks:
            await message.channel.send(reasoning_chunks[0], reference=message, mention_author=False)
            for chunk in reasoning_chunks[1:]:
                await message.channel.send(chunk)

        # Send answer
        for chunk in chunk_text(answer_message):
            await message.channel.send(chunk)

    # Allow commands to be processed after handling the message
    await bot.process_commands(message)


# --- RUN THE BOT ---
bot.run(DISCORD_BOT_TOKEN)
