# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# IMPORTS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import discord
import aiohttp
import asyncio
import os
import logging
from discord.ext import commands
from PIL import Image
import io
import pytesseract
import numpy as np
import cv2

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# CONFIGURATION
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
DISCORD_BOT_TOKEN = os.getenv('BOT_TOKEN')
OPENROUTER_API_KEY = os.getenv('API_KEY')
API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "deepseek/deepseek-r1:free"

TESSERACT_BINARY_PATH = os.path.join(os.getenv('GITHUB_WORKSPACE', ''), 'tesseract-local', 'usr', 'bin', 'tesseract')
pytesseract.pytesseract.tesseract_cmd = TESSERACT_BINARY_PATH
TESSERACT_CONFIG = '--oem 1 --psm 3 -l eng+osd'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

if not os.path.exists(TESSERACT_BINARY_PATH):
    logger.error(f"Tesseract binary not found at {TESSERACT_BINARY_PATH}")
else:
    logger.info(f"Tesseract binary set to {TESSERACT_BINARY_PATH}")

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# BOT SETUP
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# GLOBAL VARIABLES
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
activated_channels = {}
MAX_CONTEXT_MESSAGES = 10

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# HELPER FUNCTIONS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
async def get_ai_response(user_prompt: str) -> tuple[str, str]:
    system_instructions = (
        "You are a helpful Discord bot that solves problems and answers questions. "
        "Respond in two sections:\n"
        "1) Reason: (step-by-step reasoning)\n"
        "2) Answer: (concise final answer)"
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

                # Log the raw response
                logger.info(f"Raw AI API response: {content}")

                # Extract Reason and Answer
                if "Reason:" in content and "Answer:" in content:
                    reason = content.split("Answer:")[0].split("Reason:")[-1].strip()
                    answer = content.split("Answer:")[-1].strip()
                else:
                    reason = content
                    answer = "Answer: Formatting issue."

                return answer, reason
    except Exception as e:
        logger.error(f"AI API error: {e}")
        return "Answer: AI error.", "Reason: API issue"

def preprocess_image(img: Image.Image) -> list:
    try:
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        contrasted = cv2.convertScaleAbs(gray, alpha=2.0, beta=10)
        _, binary = cv2.threshold(contrasted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return [Image.fromarray(binary)]
    except Exception as e:
        logger.error(f"Image preprocessing error: {e}")
        return [img]

async def extract_text_from_image(image_url: str) -> str:
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(image_url) as response:
                if response.status == 200:
                    img_data = await response.read()
                    original_img = Image.open(io.BytesIO(img_data))
                    processed_images = preprocess_image(original_img)
                    best_text = ""

                    for img in processed_images:
                        text = pytesseract.image_to_string(img, config=TESSERACT_CONFIG)
                        if text.strip():
                            best_text = text
                            break

                    return best_text.strip() if best_text.strip() else "No text detected in the image."
                else:
                    logger.error(f"Image fetch failed, status: {response.status}")
                    return "Could not retrieve the image."
    except Exception as e:
        logger.error(f"OCR error: {e}")
        return f"Error: {e}"

async def get_conversation_context(channel: discord.TextChannel, limit: int = MAX_CONTEXT_MESSAGES) -> str:
    full_context = ""
    async for msg in channel.history(limit=limit):
        full_context = f"{msg.author.name}: {msg.content}\n" + full_context
    return full_context

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# BOT COMMAND HANDLERS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
@bot.command()
async def activate(ctx):
    activated_channels[ctx.channel.id] = True
    await ctx.send("Bot activated in this channel.")

@bot.command()
async def deactivate(ctx):
    activated_channels.pop(ctx.channel.id, None)
    await ctx.send("Bot deactivated in this channel.")

@bot.command()
async def ping(ctx):
    await ctx.send("Pong!")

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# BOT EVENT HANDLERS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
@bot.event
async def on_ready():
    logger.info(f"Logged in as {bot.user}")

@bot.event
async def on_message(message: discord.Message):
    if message.author == bot.user or message.author.bot:
        return

    logger.info(f"Received message from {message.author.name}: {message.content}")
    await bot.process_commands(message)

    full_context = await get_conversation_context(message.channel)

    if bot.user.mentioned_in(message):
        prompt = f"{full_context}\n\nUser said: {message.content}"
        async with message.channel.typing():
            answer, reason = await get_ai_response(prompt)
            await message.channel.send(f"Reason: {reason}", reference=message, mention_author=False)
            await message.channel.send(f"Answer: {answer}")
        return

    if message.channel.id in activated_channels:
        if message.attachments:
            for attachment in message.attachments:
                if attachment.content_type and attachment.content_type.startswith("image/"):
                    async with message.channel.typing():
                        text = await extract_text_from_image(attachment.url)
                        prompt = (
                            f"{full_context}\n\n"
                            f"Message: {message.content}\n\n"
                            f"Extracted text: {text}"
                        )
                        answer, reason = await get_ai_response(prompt)
                        await message.channel.send(f"Reason: {reason}", reference=message, mention_author=False)
                        await message.channel.send(f"Answer: {answer}")
                    return

        prompt = f"{full_context}\n\nUser said: {message.content}"
        async with message.channel.typing():
            answer, reason = await get_ai_response(prompt)
            await message.channel.send(f"Reason: {reason}", reference=message, mention_author=False)
            await message.channel.send(f"Answer: {answer}")

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# RUN THE BOT
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if __name__ == "__main__":
    bot.run(DISCORD_BOT_TOKEN)
