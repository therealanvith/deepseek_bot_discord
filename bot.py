# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# IMPORTS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import discord
import aiohttp
import asyncio
import os
import logging
import random
from discord.ext import commands
from PIL import Image
import io
import pytesseract
import numpy as np
import cv2
import json

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# CONFIGURATION
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
DISCORD_BOT_TOKEN = os.getenv('BOT_TOKEN')
OPENROUTER_API_KEY = os.getenv('API_KEY')
API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "deepseek/deepseek-r1:free"  # Updated model name

# Tesseract configuration
TESSERACT_BINARY_PATH = os.path.join(os.getenv('GITHUB_WORKSPACE', ''), 'tesseract-local', 'usr', 'bin', 'tesseract')
pytesseract.pytesseract.tesseract_cmd = TESSERACT_BINARY_PATH
TESSERACT_CONFIG = '--oem 1 --psm 3 -l eng+osd'

# File to persist activated channels
ACTIVATED_CHANNELS_FILE = "activated_channels.json"

# Logging setup with file handler
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bot.log'),  # Log to a file
        logging.StreamHandler()  # Also log to console
    ]
)
logger = logging.getLogger(__name__)

# Validate Tesseract binary path
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

# Load activated channels from file or start fresh
activated_channels = {}
if os.path.exists(ACTIVATED_CHANNELS_FILE):
    try:
        with open(ACTIVATED_CHANNELS_FILE, "r") as f:
            data = json.load(f)
            activated_channels = {int(k): v for k, v in data.items()}
            logger.info(f"Loaded activated channels from file: {activated_channels}")
    except Exception as e:
        logger.error(f"Error loading activated channels: {e}")

MAX_CONTEXT_MESSAGES = 10

def save_activated_channels():
    try:
        with open(ACTIVATED_CHANNELS_FILE, "w") as f:
            json.dump({str(k): v for k, v in activated_channels.items()}, f)
    except Exception as e:
        logger.error(f"Error saving activated channels: {e}")

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# HELPER FUNCTIONS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def chunk_text(text: str, max_length: int = 2000) -> list:
    """Splits text into chunks smaller than max_length for Discord's message limit."""
    return [text[i:i + max_length] for i in range(0, len(text), max_length)]

async def get_ai_response(user_prompt: str) -> tuple[str, str]:
    """Fetches a response from the DeepSeek API and returns raw response and reason part."""
    system_instructions = (
        "You are a helpful Discord bot that solves problems and answers questions. "
        "Your response must always be structured with exactly two sections:\n"
        "1) 'Reason:' - Explain your chain-of-thought or reasoning step-by-step.\n"
        "2) 'Answer:' - Provide your final answer in a single, concise sentence.\n"
        "Do not use any special formatting, code blocks, or LaTeX. Respond with plain text only."
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
                logger.info(f"Raw API response: {content}")

                # Minimal processing to split Reason and Answer:
                if "Reason:" in content and "Answer:" in content:
                    reason_part = content.split("Answer:")[0].split("Reason:")[-1].strip()
                    answer_part = content.split("Answer:")[-1].strip()
                else:
                    reason_part = content
                    answer_part = "Answer not found in response."

                return answer_part, reason_part

    except Exception as e:
        logger.error(f"Error calling AI API: {str(e)}")
        return "Answer: I'm having trouble responding right now. Please try again later.", "Reason: API connection issue"

def preprocess_image(img: Image.Image) -> list:
    """Preprocess an image for better OCR accuracy."""
    try:
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        original_height, original_width = img_cv.shape[:2]

        scale_factor = 3.0 if (original_height < 1000 or original_width < 1000) else 1.0
        img_cv = cv2.resize(img_cv, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)

        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        mean_value = np.mean(gray)
        if mean_value < 127:
            gray = cv2.bitwise_not(gray)
            alpha = 2.0
        else:
            alpha = 1.8
        beta = 10

        denoised = cv2.fastNlMeansDenoising(gray, None, 15, 7, 21)
        contrasted = cv2.convertScaleAbs(denoised, alpha=alpha, beta=beta)
        _, binary = cv2.threshold(contrasted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        adaptive = cv2.adaptiveThreshold(contrasted, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

        kernel = np.ones((2, 2), np.uint8)
        dilated_binary = cv2.dilate(binary, kernel, iterations=1)
        dilated_adaptive = cv2.dilate(adaptive, kernel, iterations=1)

        return [Image.fromarray(dilated_binary), Image.fromarray(dilated_adaptive), Image.fromarray(contrasted)]

    except Exception as e:
        logger.error(f"Image preprocessing error: {str(e)}")
        return [img]

async def extract_text_from_image(image_url: str) -> str:
    """Extracts text from an image using OCR."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(image_url) as response:
                if response.status == 200:
                    img_data = await response.read()
                    original_img = Image.open(io.BytesIO(img_data))

                    processed_images = preprocess_image(original_img)
                    best_text = ""
                    best_confidence = 0

                    for img in processed_images:
                        for psm in [1, 3, 4, 6, 7, 8, 11]:
                            config = f'--oem 1 --psm {psm} -l eng+osd'
                            ocr_data = pytesseract.image_to_data(img, config=config, output_type=pytesseract.Output.DICT)
                            logger.info(f"OCR data for PSM {psm}: {ocr_data['text']}")

                            conf_values = [float(conf) for conf in ocr_data['conf'] if conf != '-1']
                            avg_conf = sum(conf_values) / len(conf_values) if conf_values else 0
                            text = pytesseract.image_to_string(img, config=config)

                            logger.info(f"Average confidence for PSM {psm}: {avg_conf:.2f}")

                            if text.strip() and avg_conf > best_confidence:
                                best_text = text
                                best_confidence = avg_conf

                    if not best_text.strip():
                        best_text = pytesseract.image_to_string(original_img, config=TESSERACT_CONFIG)
                        logger.info(f"Fallback OCR on original image: {best_text}")

                    if best_text.strip():
                        logger.info(f"Best OCR text selected with confidence {best_confidence:.2f}: {best_text}")

                    result = best_text.strip() if best_text.strip() else "No text detected in the image."
                    logger.info(f"Final OCR result: {result}")
                    return result
                else:
                    logger.error(f"Failed to fetch image, status: {response.status}")
                    return "Could not retrieve the image."
    except Exception as e:
        logger.error(f"OCR error: {str(e)}")
        return f"Error processing image: {str(e)}"

async def get_conversation_context(channel: discord.TextChannel, limit: int = MAX_CONTEXT_MESSAGES) -> str:
    """Gathers recent conversation context from the channel."""
    full_context = ""
    async for msg in channel.history(limit=limit):
        full_context = f"{msg.author.name}: {msg.content}\n" + full_context
    return full_context

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# BOT EVENTS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

@bot.event
async def on_ready():
    logger.info(f"Logged in as {bot.user.name} ({bot.user.id})")
    print(f"Logged in as {bot.user.name} ({bot.user.id})")

@bot.event
async def on_message(message):
    # Ignore messages from the bot itself
    if message.author == bot.user:
        return

    # Only respond if channel is activated
    if message.channel.id not in activated_channels:
        return

    # Log raw message content
    logger.info(f"Received message in channel {message.channel.id} from {message.author}: {message.content}")

    # If the message has attachments, try OCR on the first image attachment
    ocr_text = ""
    if message.attachments:
        for attachment in message.attachments:
            if any(attachment.filename.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.gif']):
                ocr_text = await extract_text_from_image(attachment.url)
                if ocr_text:
                    break

    # Gather context and add OCR text if any
    context_text = await get_conversation_context(message.channel)
    if ocr_text:
        context_text += f"\nExtracted text from image:\n{ocr_text}"

    # Compose prompt for AI
    prompt = f"{context_text}\nUser said: {message.content}"

    # Get AI response (answer, reason)
    answer, reason = await get_ai_response(prompt)

    # Send the answer chunked to Discord to avoid hitting message size limit
    for chunk in chunk_text(answer):
        await message.channel.send(chunk)

    # Log raw API response (reason) - not sent to channel, but logged
    logger.info(f"API raw response (reason): {reason}")

    await bot.process_commands(message)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# BOT COMMANDS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

@bot.command(name="activate")
async def activate(ctx):
    activated_channels[ctx.channel.id] = True
    save_activated_channels()
    await ctx.send("Bot activated in this channel.")
    logger.info(f"Activated channel {ctx.channel.id}")

@bot.command(name="deactivate")
async def deactivate(ctx):
    if ctx.channel.id in activated_channels:
        activated_channels.pop(ctx.channel.id)
        save_activated_channels()
        await ctx.send("Bot deactivated in this channel.")
        logger.info(f"Deactivated channel {ctx.channel.id}")
    else:
        await ctx.send("Bot is not activated in this channel.")

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# RUN BOT
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if __name__ == "__main__":
    bot.run(DISCORD_BOT_TOKEN)
