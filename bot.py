# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# IMPORTS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import discord
import aiohttp
import asyncio
import os
import logging
import io
from discord.ext import commands
from PIL import Image
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

# Tesseract configuration
TESSERACT_BINARY_PATH = os.path.join(os.getenv('GITHUB_WORKSPACE', ''), 'tesseract-local', 'usr', 'bin', 'tesseract')
pytesseract.pytesseract.tesseract_cmd = TESSERACT_BINARY_PATH
TESSERACT_CONFIG = '--oem 1 --psm 3 -l eng+osd'

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('bot.log'), logging.StreamHandler()]
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

activated_channels = {}
MAX_CONTEXT_MESSAGES = 10

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# HELPER FUNCTIONS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def chunk_text(text: str, max_length: int = 2000) -> list:
    return [text[i:i + max_length] for i in range(0, len(text), max_length)]

async def get_ai_response(user_prompt: str) -> tuple[str, str]:
    system_instructions = (
        "You are a helpful Discord bot. Structure your response into two sections:\n"
        "1) 'Reason:' - Your step-by-step logic.\n"
        "2) 'Answer:' - One concise sentence.\n"
        "Plain text only."
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
                    reason_part = content
                    answer_part = "Answer: Response formatting error."
                return answer_part, reason_part
    except Exception as e:
        logger.error(f"AI error: {e}")
        return "Answer: Unable to respond right now.", "Reason: API error"

def preprocess_image(img: Image.Image) -> list:
    try:
        img = img.convert("RGB")
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img_cv = cv2.resize(img_cv, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        denoised = cv2.fastNlMeansDenoising(gray, None, 15, 7, 21)
        contrasted = cv2.convertScaleAbs(denoised, alpha=2.0, beta=10)
        _, binary = cv2.threshold(contrasted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return [Image.fromarray(binary), Image.fromarray(contrasted)]
    except Exception as e:
        logger.error(f"Preprocessing error: {e}")
        return [img]

async def extract_text_from_image(image_url: str) -> str:
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(image_url) as response:
                if response.status != 200:
                    return "Could not retrieve the image."
                img_data = await response.read()
                original_img = Image.open(io.BytesIO(img_data)).convert("RGB")
                processed_images = preprocess_image(original_img)

                best_text = ""
                best_conf = 0
                for img in processed_images:
                    text = pytesseract.image_to_string(img, config=TESSERACT_CONFIG)
                    if text.strip():
                        return text.strip()
                return "No text detected in the image."
    except Exception as e:
        logger.error(f"OCR error: {e}")
        return "Error processing image."

async def get_conversation_context(channel: discord.TextChannel, limit: int = MAX_CONTEXT_MESSAGES) -> str:
    context = ""
    async for msg in channel.history(limit=limit):
        context = f"{msg.author.name}: {msg.content}\n" + context
    return context

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# COMMANDS
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
# EVENTS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

@bot.event
async def on_ready():
    logger.info(f"Logged in as {bot.user}")

@bot.event
async def on_message(message: discord.Message):
    if message.author == bot.user or message.author.bot:
        return

    await bot.process_commands(message)
    full_context = await get_conversation_context(message.channel)

    # Handle OCR from images
    for attachment in message.attachments:
        logger.info(f"Attachment: {attachment.filename}, type: {attachment.content_type}")
        if attachment.filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp")):
            async with message.channel.typing():
                text = await extract_text_from_image(attachment.url)
                prompt = (
                    f"Context:\n{full_context}\n\n"
                    f"User message: {message.content}\n"
                    f"OCR extracted text: {text}\n\n"
                    "Respond with 'Reason:' and 'Answer:' sections."
                )
                answer, reason = await get_ai_response(prompt)
                await message.channel.send(f"Reason: {reason}", reference=message, mention_author=False)
                await message.channel.send(f"Answer: {answer}")
            return

    if "#search" in message.content.lower():
        await message.channel.send("🔍 Web search has been disabled. Ask your question directly.")
        return

    if bot.user.mentioned_in(message) or message.channel.id in activated_channels:
        prompt = f"Context:\n{full_context}\n\nUser message:\n{message.content}"
        async with message.channel.typing():
            answer, reason = await get_ai_response(prompt)
            await message.channel.send(f"Reason: {reason}", reference=message, mention_author=False)
            await message.channel.send(f"Answer: {answer}")

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# RUN BOT
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if __name__ == "__main__":
    bot.run(DISCORD_BOT_TOKEN)
