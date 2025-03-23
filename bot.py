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

# --- GLOBAL VARIABLES ---
activated_channels = {}  # To track which channels are activated for full responses
MAX_CONTEXT_MESSAGES = 20  # Max number of messages to fetch from the past

# --- HELPER FUNCTIONS ---
def chunk_text(text: str, max_length: int = 2000):
    """Splits text into chunks of up to max_length characters."""
    return [text[i : i + max_length] for i in range(0, len(text), max_length)]

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
        await ctx.send(f"The bot is now deactivated and will only respond to mentions in this channel.")
    else:
        await ctx.send(f"The bot is not activated in this channel.")

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
    - If the bot is activated, responds to all messages.
    """
    if message.author == bot.user:
        return

    # If the bot is activated for the channel, respond to all messages
    if message.channel.id in activated_channels:
        prompt = message.content

        # Fetch the last MAX_CONTEXT_MESSAGES from the channel's message history
        full_context = ""
        async for msg in message.channel.history(limit=MAX_CONTEXT_MESSAGES):
            full_context = f"{msg.author.name}: {msg.content}\n" + full_context

        # Combine the context and the current message to create a prompt
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
