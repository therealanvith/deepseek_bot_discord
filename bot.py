# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# IMPORTS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import discord
import aiohttp
import asyncio
import re
import os
import logging
from discord.ext import commands
from PIL import Image, ImageEnhance, ImageFilter
import io
import pytesseract
import numpy as np
import cv2
from bs4 import BeautifulSoup  # For web scraping Google search results

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# CONFIGURATION
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
DISCORD_BOT_TOKEN = os.getenv('BOT_TOKEN')
OPENROUTER_API_KEY = os.getenv('API_KEY')
API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "deepseek/deepseek-r1:free"  # Updated model name
GOOGLE_SEARCH_URL = "https://www.google.com/search?q="  # Base URL for Google search

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

async def perform_google_search(query: str) -> str:
    """Performs a Google search and returns the top results as a string."""
    logger.info(f"Starting Google search for query: '{query}'")
    try:
        async with aiohttp.ClientSession() as session:
            encoded_query = "+".join(query.split())
            url = f"{GOOGLE_SEARCH_URL}{encoded_query}"
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            logger.info(f"Sending request to URL: {url}")
            async with session.get(url, headers=headers, timeout=5) as response:
                logger.info(f"Received response with status: {response.status}")
                if response.status != 200:
                    logger.error(f"Google search failed with status: {response.status}")
                    return "Error: Unable to perform Google search."
                
                html = await response.text()
                logger.info("Successfully fetched HTML content")
                soup = BeautifulSoup(html, 'html.parser')
                
                # Try to extract standard search results
                results = []
                # Updated selector for modern Google search results
                search_results = soup.find_all('div', class_='MjjYud') or soup.find_all('div', class_='g') or soup.find_all('div', class_='tF2Cxc')
                logger.info(f"Found {len(search_results)} potential search result elements using standard selectors")
                
                for g in search_results[:3]:  # Limit to top 3 results
                    title = g.find('h3') or g.find('div', role='heading')
                    snippet = g.find('div', class_='VwiC3b') or g.find('span', class_='aCOpRe') or g.find('div', class_='IsZvec')
                    if title and snippet:
                        results.append(f"Title: {title.text}\nSnippet: {snippet.text}\n")
                        logger.info(f"Added result - Title: {title.text}")
                
                # If no standard results, try to extract from featured snippet or knowledge panel
                if not results:
                    logger.info("No standard search results found, trying featured snippet or knowledge panel")
                    # Featured snippet (often in div.kCrYT or div.xpd)
                    featured_snippet = soup.find('div', class_='kCrYT') or soup.find('div', class_='xpd')
                    if featured_snippet:
                        snippet_text = featured_snippet.get_text(strip=True)
                        if snippet_text:
                            results.append(f"Featured Snippet: {snippet_text}\n")
                            logger.info(f"Added featured snippet: {snippet_text[:100]}...")
                    
                    # Knowledge panel (often in div.kp-wholepage or div.knowledge-panel)
                    knowledge_panel = soup.find('div', class_='kp-wholepage') or soup.find('div', class_='knowledge-panel')
                    if knowledge_panel:
                        panel_text = knowledge_panel.get_text(strip=True)
                        if panel_text:
                            results.append(f"Knowledge Panel: {panel_text[:200]}...\n")
                            logger.info(f"Added knowledge panel: {panel_text[:100]}...")
                
                # If still no results, extract raw text as a last resort
                if not results:
                    logger.info("No structured results found, extracting raw text as fallback")
                    # Remove scripts and styles to clean up the text
                    for script in soup(["script", "style"]):
                        script.decompose()
                    raw_text = soup.get_text(separator=' ', strip=True)
                    # Limit to first 500 characters to avoid overwhelming the AI
                    if raw_text:
                        results.append(f"Raw Text (first 500 chars): {raw_text[:500]}...\n")
                        logger.info(f"Added raw text: {raw_text[:100]}...")
                
                if not results:
                    logger.warning("No search results found after all attempts")
                    return "No search results found."
                
                search_result = "\n".join(results)
                logger.info(f"Search results for query '{query}':\n{search_result}")
                return search_result
    except asyncio.TimeoutError:
        logger.error(f"Google search timed out for query '{query}'")
        return "Error: Search timed out. Please try a simpler query."
    except Exception as e:
        logger.error(f"Error during Google search for query '{query}': {str(e)}")
        return f"Error during Google search: {str(e)}"

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
            # Add timeout to prevent API hangs
            async with session.post(API_URL, headers=headers, json=data, timeout=10) as resp:
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

    except asyncio.TimeoutError:
        logger.error("AI API request timed out")
        return "Answer: I'm having trouble responding right now. Please try again later.", "Reason: API request timed out"
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
            # Add timeout to prevent long delays
            async with session.get(image_url, timeout=5) as response:
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
    except asyncio.TimeoutError:
        logger.error("Image fetch timed out")
        return "Error: Image fetch timed out."
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
    
    # Handle replies to the bot (works in both activated and deactivated channels)
    if message.reference and message.reference.resolved and message.reference.resolved.author == bot.user:
        referenced_msg = message.reference.resolved
        prompt = f"Previous bot message:\n{referenced_msg.content}\n\nUser's new message:\n{message.content}\nPlease respond with 'Reason:' and 'Answer:' sections."
        async with message.channel.typing():
            answer, reason = await get_ai_response(prompt)
            await message.channel.send(f"Reason: {reason}", reference=message, mention_author=False)
            await
