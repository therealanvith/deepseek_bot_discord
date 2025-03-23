import discord
import aiohttp
import asyncio
import re
import os
from discord.ext import commands
from PIL import Image
import io
import pytesseract

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

# --- GLOBAL VARIABLES ---
activated_channels = {}  # To track which channels are activated for full responses
MAX_CONTEXT_MESSAGES = 20  # Max number of messages to fetch from the past

# --- HELPER FUNCTIONS ---
def chunk_text(text: str, max_length: int = 2000):
    """Splits text into chunks of up to max_length characters."""
    return [text[i: i + max_length] for i in range(0, len(text), max_length)]

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

async def extract_text_from_image(image_url: str) -> str:
    """Extracts text from an image using OCR."""
    async with aiohttp.ClientSession() as session:
        async with session.get(image_url) as response:
            if response.status == 200:
                img_data = await response.read()
                img = Image.open(io.BytesIO(img_data))
                text = pytesseract.image_to_string(img)
                return text
            else:
                return "Could not retrieve the image."

# --- BOT COMMAND HANDLERS ---

@bot.command()
async def activate(ctx):
    """Activates the bot to respond to all messages in the current channel."""
    activated_channels[ctx.channel.id] = True
    await ctx.send(f"The bot is now activated and will respond to all messages in this channel!")

@bot.command()
async def deactivate(ctx):
    """Deactivates the bot from responding to all messages in the current channel."""
    if ctx.channel.id in activated_channels:
        del activated_channels[ctx.channel.id]
        await ctx.send(f"The bot is now deactivated and will only respond to mentions and replies in this channel.")
    else:
        await ctx.send(f"The bot is not activated in this channel.")

@bot.command()
async def ping(ctx):
    """Test command to check if the bot is responding."""
    await ctx.send('Pong!')


# --- BOT EVENT HANDLERS ---

@bot.event
async def on_ready():
    print(f"Logged in as {bot.user}")

@bot.event
async def on_message(message: discord.Message):
    """
    Handles all messages:
    - Ignores messages from other bots.
    - Responds to mentions of the bot (@bot), even when not activated.
    - Responds to replies to its own messages.
    - Responds to all messages in activated channels.
    """
    if message.author == bot.user or message.author.bot:
        return  # Ignore bot's own messages and other bots' messages

    # Respond to mentions (@bot), even when not activated
    if bot.user.mentioned_in(message):
        prompt = message.content

        # Fetch the last MAX_CONTEXT_MESSAGES from the channel's message history
        full_context = ""
        async for msg in message.channel.history(limit=MAX_CONTEXT_MESSAGES):
            full_context = f"{msg.author.name}: {msg.content}\n" + full_context

        # Combine the context and the current message to create a prompt
        prompt = f"Context of conversation:\n{full_context}\n\nUser's current message:\n{prompt}\nPlease respond with 'Reason:' and 'Answer:' sections."

        # Process the AI response
        async with message.channel.typing():
            answer, reason = await get_ai_response(prompt)
            reasoning_message = f"(Reasoning: {reason})"
            answer_message = answer

            # Send reasoning first
            reasoning_chunks = chunk_text(reasoning_message)
            if reasoning_chunks:
                await message.channel.send(reasoning_chunks[0], reference=message, mention_author=False)
                for chunk in reasoning_chunks[1:]:
                    await message.channel.send(chunk)

            # Send the final answer
            for chunk in chunk_text(answer_message):
                await message.channel.send(chunk)

    # Handle replies to the bot's own message
    elif message.reference and message.reference.resolved.author == bot.user:
        referenced_msg = message.reference.resolved
        prompt = f"Previous bot message:\n{referenced_msg.content}\n\nUser's new message:\n{message.content}\nPlease respond with 'Reason:' and 'Answer:' sections."

        async with message.channel.typing():
            answer, reason = await get_ai_response(prompt)
            reasoning_message = f"(Reasoning: {reason})"
            answer_message = answer

            # Send reasoning first
            reasoning_chunks = chunk_text(reasoning_message)
            if reasoning_chunks:
                await message.channel.send(reasoning_chunks[0], reference=message, mention_author=False)
                for chunk in reasoning_chunks[1:]:
                    await message.channel.send(chunk)

            # Send the final answer
            for chunk in chunk_text(answer_message):
                await message.channel.send(chunk)

    # Handle activated channels (respond to all messages)
    elif message.channel.id in activated_channels:
        prompt = message.content

        # Fetch the last MAX_CONTEXT_MESSAGES from the channel's message history
        full_context = ""
        async for msg in message.channel.history(limit=MAX_CONTEXT_MESSAGES):
            full_context = f"{msg.author.name}: {msg.content}\n" + full_context

        # Combine the context and the current message to create a prompt
        prompt = f"Context of conversation:\n{full_context}\n\nUser's current message:\n{prompt}\nPlease respond with 'Reason:' and 'Answer:' sections."

        # Process the AI response
        async with message.channel.typing():
            answer, reason = await get_ai_response(prompt)
            reasoning_message = f"(Reasoning: {reason})"
            answer_message = answer

            # Send reasoning first
            reasoning_chunks = chunk_text(reasoning_message)
            if reasoning_chunks:
                await message.channel.send(reasoning_chunks[0], reference=message, mention_author=False)
                for chunk in reasoning_chunks[1:]:
                    await message.channel.send(chunk)

            # Send the final answer
            for chunk in chunk_text(answer_message):
                await message.channel.send(chunk)

    # Check for image attachments and perform OCR if present
    elif message.attachments:
        for attachment in message.attachments:
            if attachment.content_type.startswith("image/"):  # Only process images
                text_from_image = await extract_text_from_image(attachment.url)

                # Make sure the bot understands this is OCR-derived text
                prompt = f"Here is the text extracted from an image using OCR:\n{text_from_image}\nPlease respond with 'Reason:' and 'Answer:' sections based on the extracted text from the image."

                async with message.channel.typing():
                    answer, reason = await get_ai_response(prompt)
                    reasoning_message = f"(Reasoning: {reason})"
                    answer_message = answer

                    # Send reasoning first
                    reasoning_chunks = chunk_text(reasoning_message)
                    if reasoning_chunks:
                        await message.channel.send(reasoning_chunks[0], reference=message, mention_author=False)
                        for chunk in reasoning_chunks[1:]:
                            await message.channel.send(chunk)

                    # Send the final answer
                    for chunk in chunk_text(answer_message):
                        await message.channel.send(chunk)

                return  # Stop after processing the image and sending the response

    # Allow commands to be processed after handling the message
    await bot.process_commands(message)


# --- RUN THE BOT ---
bot.run(DISCORD_BOT_TOKEN)
