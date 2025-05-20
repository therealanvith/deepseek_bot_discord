# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# IMPORTS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import discord
import aiohttp
import asyncio
import re
import os
import logging
import random
from discord.ext import commands
from PIL import Image, ImageEnhance, ImageFilter
import io
import pytesseract
import numpy as np
import cv2
from bs4 import BeautifulSoup  # For web scraping (not used now, kept for potential future use)
import uuid
import time
import json

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# CONFIGURATION
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
DISCORD_BOT_TOKEN = os.getenv('BOT_TOKEN')
OPENROUTER_API_KEY = os.getenv('API_KEY')
API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "deepseek/deepseek-r1:free"  # Updated model name

# List of SearXNG instances to try
searxng_instances = [
    "https://search.brave.com",
    "https://searx.space",
    "https://search.disroot.org",
    "https://searx.nixnet.services",
    "https://search.rhscz.eu",
    "https://searx.tuxcloud.net",
    "https://searx.be",
]

# Tesseract configuration
TESSERACT_BINARY_PATH = os.path.join(os.getenv('GITHUB_WORKSPACE', ''), 'tesseract-local', 'usr', 'bin', 'tesseract')
pytesseract.pytesseract.tesseract_cmd = TESSERACT_BINARY_PATH
TESSERACT_CONFIG = '--oem 1 --psm 3 -l eng+osd'

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
activated_channels = {}
MAX_CONTEXT_MESSAGES = 10

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# HELPER FUNCTIONS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def chunk_text(text: str, max_length: int = 2000) -> list:
    """Splits text into chunks smaller than max_length for Discord's message limit."""
    return [text[i:i + max_length] for i in range(0, len(text), max_length)]

async def fetch_referenced_message(message: discord.Message) -> discord.Message:
    """Fetches the message being replied to, if any."""
    if not message.reference:
        return None
    ref = message.reference.resolved
    if ref:
        return ref
    try:
        return await message.channel.fetch_message(message.reference.message_id)
    except Exception as e:
        logger.error(f"Error fetching referenced message: {e}")
        return None

async def perform_search_with_searxng(query: str, max_retries: int = 3, retry_delay: int = 5) -> str:
    """Performs a search using a SearXNG instance with fallback and rate-limit handling."""
    logger.info(f"Starting SearXNG search for query: '{query}'")
    
    if not query:
        logger.error("No search query provided.")
        return "Error: Missing search query. Falling back to internal knowledge."

    # Randomize instance order to distribute load
    instances = random.sample(searxng_instances, len(searxng_instances))
    
    for instance in instances:
        for attempt in range(max_retries):
            try:
                async with aiohttp.ClientSession() as session:
                    url = f"{instance}/search"
                    params = {
                        "q": query,
                        "format": "json",
                        "language": "en",
                        "safesearch": "1"
                    }
                    
                    logger.info(f"Attempt {attempt + 1}/{max_retries}: Trying {instance}")
                    
                    async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=10)) as response:
                        # Handle rate limiting
                        if response.status == 429:
                            wait_time = int(response.headers.get('Retry-After', retry_delay))
                            logger.warning(f"Rate limited on {instance}. Waiting {wait_time} seconds...")
                            await asyncio.sleep(wait_time)
                            continue
                            
                        if response.status != 200:
                            logger.error(f"Bad status from {instance}: {response.status}")
                            break
                            
                        try:
                            data = await response.json()
                        except Exception as e:
                            logger.error(f"JSON decode error from {instance}: {str(e)}")
                            break
                            
                        if not data.get('results'):
                            logger.info(f"No results from {instance}")
                            break
                            
                        # Format the top 3 results
                        results = []
                        for i, result in enumerate(data['results'][:3]):
                            title = result.get('title', 'No title')
                            url = result.get('url', 'No URL')
                            content = result.get('content', 'No description')
                            results.append(
                                f"🔍 **Result {i+1}**: {title}\n"
                                f"🔗 {url}\n"
                                f"📝 {content}\n"
                            )
                        
                        return "\n".join(results) if results else "No results found."
            
            except asyncio.TimeoutError:
                logger.warning(f"Timeout on {instance}, attempt {attempt + 1}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay)
                continue
                
            except Exception as e:
                logger.error(f"Error with {instance}: {str(e)}")
                break
    
    return "Could not get search results from any SearXNG instance. The instances may be down or rate-limiting us."

async def get_ai_response(user_prompt: str) -> tuple[str, str]:
    """Fetches a response from the DeepSeek API and formats it with Reason: and Answer: sections."""
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
                logger.info(f"AI API response: {content}")

                # Minimal processing: just separate Reason: and Answer:
                if "Reason:" in content and "Answer:" in content:
                    reason_part = content.split("Answer:")[0].split("Reason:")[-1].strip()
                    answer_part = content.split("Answer:")[-1].strip()
                else:
                    # Fallback if the API doesn't format correctly
                    if "So," in content:
                        parts = content.rsplit("So,", 1)
                        reason_part = parts[0].strip()
                        answer_part = "So, " + parts[1].strip()
                    elif "Hence" in content:
                        parts = content.rsplit("Hence", 1)
                        reason_part = parts[0].strip()
                        answer_part = "Hence " + parts[1].strip()
                    elif "Therefore" in content:
                        parts = content.rsplit("Therefore", 1)
                        reason_part = parts[0].strip()
                        answer_part = "Therefore " + parts[1].strip()
                    else:
                        sentences = content.split(". ")
                        if len(sentences) > 1:
                            reason_part = ". ".join(sentences[:-1]).strip()
                            answer_part = sentences[-1].strip()
                        else:
                            reason_part = content
                            answer_part = "I couldn't determine a clear answer due to formatting issues."
                    if not answer_part.startswith("Answer:"):
                        answer_part = "Answer: " + answer_part

                return answer_part, reason_part

    except Exception as e:
        logger.error(f"Error calling AI API: {str(e)}")
        return "Answer: I'm having trouble responding right now. Please try again later.", "Reason: API connection issue"

def preprocess_image(img: Image.Image) -> list:
    """Preprocesses an image for better OCR accuracy."""
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
# BOT COMMAND HANDLERS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

@bot.command()
async def activate(ctx: commands.Context):
    """Activates the bot to respond to all messages in the channel."""
    activated_channels[ctx.channel.id] = True
    await ctx.send("The bot is now activated and will respond to all messages in this channel!")

@bot.command()
async def deactivate(ctx: commands.Context):
    """Deactivates the bot from responding to all messages in the channel."""
    if ctx.channel.id in activated_channels:
        del activated_channels[ctx.channel.id]
        await ctx.send("The bot is now deactivated and will only respond to mentions and replies.")
    else:
        await ctx.send("The bot is not activated in this channel.")

@bot.command()
async def ping(ctx: commands.Context):
    """Simple ping command to test bot responsiveness."""
    await ctx.send('Pong!')

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# BOT EVENT HANDLERS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

@bot.event
async def on_ready():
    """Logs when the bot is ready."""
    logger.info(f"Logged in as {bot.user}")

@bot.event
async def on_message(message: discord.Message):
    """Handles incoming messages and triggers responses."""
    if message.author == bot.user or message.author.bot:
        return
    
    # Log the received message
    logger.info(f"Received message from {message.author.name}: {message.content}")
    
    await bot.process_commands(message)
    full_context = await get_conversation_context(message.channel)
    
    # Check if the message is a trigger in a deactivated channel
    is_triggered = (
        bot.user.mentioned_in(message) or
        (message.reference and message.reference.resolved and message.reference.resolved.author == bot.user) or
        (message.reference and message.reference.resolved and message.reference.resolved.author != bot.user and bot.user.mentioned_in(message))
    )
    
    # Handle replies to the bot (works in both activated and deactivated channels)
    if message.reference and message.reference.resolved and message.reference.resolved.author == bot.user:
        referenced_msg = message.reference.resolved
        prompt = f"Previous bot message:\n{referenced_msg.content}\n\nUser's new message:\n{message.content}\nPlease respond with 'Reason:' and 'Answer:' sections."
        async with message.channel.typing():
            answer, reason = await get_ai_response(prompt)
            await message.channel.send(f"Reason: {reason}", reference=message, mention_author=False)
            await message.channel.send(f"Answer: {answer}")
        return
    
    # Handle replies to another person's message with a bot mention (works in both activated and deactivated channels)
    if message.reference and message.reference.resolved and message.reference.resolved.author != bot.user and bot.user.mentioned_in(message):
        referenced_msg = message.reference.resolved
        prompt = (
            f"Context of conversation:\n{full_context}\n\n"
            f"The user replied to another person's message: {referenced_msg.author.name}: {referenced_msg.content}\n"
            f"User's reply mentioning the bot: {message.content}\n"
            "Please respond with 'Reason:' and 'Answer:' sections."
        )
        async with message.channel.typing():
            answer, reason = await get_ai_response(prompt)
            await message.channel.send(f"Reason: {reason}", reference=message, mention_author=False)
            await message.channel.send(f"Answer: {answer}")
        return
    
    # Handle bot mentions (works in both activated and deactivated channels)
    if bot.user.mentioned_in(message):
        prompt = f"Context of conversation:\n{full_context}\n\nUser's current message:\n{message.content}\nPlease respond with 'Reason:' and 'Answer:' sections."
        async with message.channel.typing():
            answer, reason = await get_ai_response(prompt)
            await message.channel.send(f"Reason: {reason}", reference=message, mention_author=False)
            await message.channel.send(f"Answer: {answer}")
        return
    
    # Handle #search and image attachments only in activated channels or if triggered in deactivated channels
    if message.channel.id in activated_channels or is_triggered:
        # Handle #search keyword
        if "#search" in message.content.lower():
            search_query = message.content.lower().split("#search", 1)[1].strip()
            if not search_query:
                await message.channel.send("Please provide a search query after #search, e.g., `#search python tutorial`")
                return
            
            async with message.channel.typing():
                try:
                    # Perform the search
                    search_results = await perform_search_with_searxng(search_query)
                    
                    # Check if we got actual results
                    if "Could not get search results" in search_results:
                        # Fallback to just using the AI
                        prompt = (
                            f"User requested a search for: {search_query}\n"
                            "But we couldn't get live search results.\n"
                            "Please answer using your knowledge.\n"
                            "Respond with 'Reason:' and 'Answer:' sections."
                        )
                    else:
                        # Combine search results with AI processing
                        prompt = (
                            f"Context of conversation:\n{full_context}\n\n"
                            f"User requested a search for: {search_query}\n\n"
                            f"Here are the search results:\n{search_results}\n\n"
                            "Please analyze these results and provide a concise answer.\n"
                            "Respond with 'Reason:' and 'Answer:' sections."
                        )
                    
                    answer, reason = await get_ai_response(prompt)
                    
                    # Send the response in chunks if too long
                    for chunk in chunk_text(f"Reason: {reason}\n\nAnswer: {answer}"):
                        await message.channel.send(chunk)
                        
                except Exception as e:
                    logger.error(f"Search error: {str(e)}")
                    await message.channel.send("⚠️ Couldn't complete your search right now. This might be due to high load on search services. You can try again later or ask your question directly.")
            return
        
        # Handle image attachments (OCR without search)
        for attachment in message.attachments:
            if attachment.content_type and attachment.content_type.startswith("image/"):
                async with message.channel.typing():
                    text_from_image = await extract_text_from_image(attachment.url)
                    prompt = (
                        f"Context of conversation:\n{full_context}\n\n"
                        f"User's current message:\n{message.content}\n\n"
                        f"Text from the image has been extracted using OCR:\n\n{text_from_image}\n\n"
                        "If the OCR text is 'No text detected in the image,' explain in the 'Reason:' section that the image text couldn't be extracted and suggest the user provide a clearer image or type the text manually. "
                        "Otherwise, use the extracted text to solve the problem or answer the user's request. "
                        "Please respond with 'Reason:' and 'Answer:' sections."
                    )
                    
                    answer, reason = await get_ai_response(prompt)
                    await message.channel.send(f"Reason: {reason}", reference=message, mention_author=False)
                    await message.channel.send(f"Answer: {answer}")
                return
    
    # Handle messages in activated channels (for regular messages)
    if message.channel.id in activated_channels:
        prompt = f"Context of conversation:\n{full_context}\n\nUser's current message:\n{message.content}\nPlease respond with 'Reason:' and 'Answer:' sections."
        async with message.channel.typing():
            answer, reason = await get_ai_response(prompt)
            await message.channel.send(f"Reason: {reason}", reference=message, mention_author=False)
            await message.channel.send(f"Answer: {answer}")

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# RUN THE BOT
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if __name__ == "__main__":
    bot.run(DISCORD_BOT_TOKEN)
