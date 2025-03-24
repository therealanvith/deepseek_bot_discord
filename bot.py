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
from bs4 import BeautifulSoup  # For web scraping Perplexity AI search results

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# CONFIGURATION
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
DISCORD_BOT_TOKEN = os.getenv('BOT_TOKEN')
OPENROUTER_API_KEY = os.getenv('API_KEY')
API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "deepseek/deepseek-r1:free"  # Updated model name
PERPLEXITY_SEARCH_URL = "https://www.perplexity.ai/search?q="  # Base URL for Perplexity AI search

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
# Update the USER_AGENTS list with more recent browser versions (as of 2025)
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) Gecko/20100101 Firefox/125.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_5) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.0 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/123.0.2420.97",
]

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

async def perform_perplexity_search(query: str) -> str:
    """Performs a search using Perplexity AI with advanced anti-detection techniques."""
    logger.info(f"Starting Perplexity AI search for query: '{query}'")
    
    # Random delay before request (0.5-2.5 seconds)
    await asyncio.sleep(random.uniform(0.5, 2.5))
    
    try:
        # Use a headless browser approach with playwright instead of direct HTTP requests
        async with aiohttp.ClientSession() as session:
            encoded_query = query.replace(" ", "+")
            
            # Use a different URL format that's less likely to be blocked
            # Try the API endpoint instead of the search page
            url = f"https://www.perplexity.ai/api/search?q={encoded_query}"
            
            # Create a more realistic browser fingerprint
            user_agent = random.choice(USER_AGENTS)
            
            # Add extensive browser-like headers to appear more human
            headers = {
                "User-Agent": user_agent,
                "Accept": "application/json, text/plain, */*",
                "Accept-Language": "en-US,en;q=0.9,es;q=0.8",
                "Accept-Encoding": "gzip, deflate, br",
                "Referer": "https://www.perplexity.ai/",
                "Origin": "https://www.perplexity.ai",
                "Connection": "keep-alive",
                "Sec-Fetch-Dest": "empty",
                "Sec-Fetch-Mode": "cors",
                "Sec-Fetch-Site": "same-origin",
                "Sec-Ch-Ua": '"Chromium";v="118", "Google Chrome";v="118"',
                "Sec-Ch-Ua-Mobile": "?0",
                "Sec-Ch-Ua-Platform": '"Windows"',
                "Cache-Control": "no-cache",
                "Pragma": "no-cache",
                "Priority": "u=1, i",
                "DNT": "1",
            }
            
            # Create a more realistic cookie set
            cookies = {
                "perplexity_uuid": f"{uuid.uuid4().hex}",
                "perplexity_session_id": f"{uuid.uuid4().hex}",
                "perplexity_jwt_production": f"perplexity.{uuid.uuid4().hex}.{uuid.uuid4().hex}",
                "__Host-next-auth.csrf-token": f"{uuid.uuid4().hex}%7C{uuid.uuid4().hex}",
                "_ga": f"GA1.1.{random.randint(1000000, 9999999)}.{random.randint(1000000, 9999999)}",
                "_ga_MK7GNVB0D1": f"GS1.1.{int(time.time())}.1.0.{int(time.time())}.0.0.0",
            }
            
            logger.info(f"Sending request to URL: {url} with User-Agent: {user_agent}")
            
            # First, make a GET request to the main site to get legitimate cookies
            async with session.get("https://www.perplexity.ai/", 
                                  headers={"User-Agent": user_agent},
                                  timeout=10) as initial_response:
                
                if initial_response.status != 200:
                    logger.warning(f"Initial site request failed with status: {initial_response.status}")
                
                # Extract and use any cookies set by the initial request
                site_cookies = initial_response.cookies
                for cookie_name in site_cookies:
                    cookies[cookie_name] = site_cookies[cookie_name].value
            
            # Add a random delay to mimic human behavior
            await asyncio.sleep(random.uniform(1.0, 3.0))
            
            # Try the direct search approach first
            try:
                # Use a direct API approach
                api_payload = {
                    "query": query,
                    "conversationId": str(uuid.uuid4()),
                    "options": {
                        "source": "search",
                        "searchOptions": {"type": "keyword"},
                    }
                }
                
                async with session.post("https://www.perplexity.ai/api/search",
                                      headers=headers,
                                      cookies=cookies,
                                      json=api_payload,
                                      timeout=15) as response:
                    
                    logger.info(f"Received API response with status: {response.status}")
                    
                    if response.status == 200:
                        try:
                            result_json = await response.json()
                            logger.info("Successfully parsed JSON response")
                            
                            # Extract information from the JSON response
                            results = []
                            
                            # Extract the main answer if available
                            if "answer" in result_json:
                                results.append(f"Answer: {result_json['answer']}\n")
                            
                            # Extract sources if available
                            if "sources" in result_json and isinstance(result_json["sources"], list):
                                for i, source in enumerate(result_json["sources"][:3]):
                                    if "title" in source and "url" in source:
                                        results.append(f"Source {i+1}: {source['title']} - {source['url']}\n")
                            
                            if results:
                                return "\n".join(results)
                        except Exception as e:
                            logger.error(f"Error parsing JSON: {str(e)}")
            except Exception as api_error:
                logger.error(f"API approach failed: {str(api_error)}")
            
            # Fallback to the web page approach if API approach fails
            fallback_url = f"https://www.perplexity.ai/search?q={encoded_query}"
            
            # Add another random delay before fallback
            await asyncio.sleep(random.uniform(1.0, 2.0))
            
            # Try a different request method for the fallback
            async with session.get(fallback_url,
                                 headers=headers,
                                 cookies=cookies,
                                 timeout=15,
                                 allow_redirects=True) as response:
                
                logger.info(f"Fallback: Received response with status: {response.status}")
                
                if response.status != 200:
                    logger.error(f"Perplexity AI search failed with status: {response.status}")
                    return "Error: Unable to perform Perplexity AI search. Falling back to internal knowledge."
                
                html = await response.text()
                logger.info("Successfully fetched HTML content")
                
                # Check for common block indicators
                if "too many requests" in html.lower() or "verify you are not a bot" in html.lower():
                    logger.warning("Perplexity AI detected the request as a bot or hit rate limits")
                    return "Error: Perplexity AI detected the request as a bot or hit rate limits. Falling back to internal knowledge."
                
                # Try to extract JSON data embedded in the HTML (Perplexity uses Next.js)
                json_match = re.search(r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>', html)
                if json_match:
                    try:
                        next_data = json.loads(json_match.group(1))
                        # Look for search results in the Next.js data structure
                        if "props" in next_data and "pageProps" in next_data["props"]:
                            page_props = next_data["props"]["pageProps"]
                            if "searchResults" in page_props:
                                search_data = page_props["searchResults"]
                                
                                results = []
                                # Extract the main answer
                                if "answer" in search_data:
                                    results.append(f"Answer: {search_data['answer'][:1000]}...\n")
                                
                                # Extract sources
                                if "sources" in search_data and isinstance(search_data["sources"], list):
                                    for i, source in enumerate(search_data["sources"][:3]):
                                        if "title" in source and "url" in source:
                                            results.append(f"Source {i+1}: {source['title']} - {source['url']}\n")
                                
                                if results:
                                    return "\n".join(results)
                    except json.JSONDecodeError:
                        logger.error("Failed to parse embedded JSON")
                
                # Fallback to BeautifulSoup parsing if JSON extraction fails
                soup = BeautifulSoup(html, 'html.parser')
                
                # Extract the main answer from Perplexity AI
                results = []
                
                # Try multiple selectors that might contain the answer
                answer_selectors = [
                    'div.prose', 
                    'div.default-answer',
                    'div[class*="answer"]',
                    'div[class*="result-content"]',
                    'main article'
                ]
                
                for selector in answer_selectors:
                    answer_section = soup.select_one(selector)
                    if answer_section:
                        answer_text = answer_section.get_text(strip=True)
                        if answer_text and len(answer_text) > 20:  # Ensure it's a substantial answer
                            results.append(f"Answer: {answer_text[:1000]}...\n")
                            logger.info(f"Added main answer: {answer_text[:100]}...")
                            break
                
                # Try different source selectors
                source_selectors = [
                    'a.source-link',
                    'a[href^="http"][class*="source"]',
                    'div[class*="source"] a',
                    'div[class*="citation"] a'
                ]
                
                sources_found = False
                for selector in source_selectors:
                    sources = soup.select(selector)
                    if sources:
                        for i, source in enumerate(sources[:3]):
                            source_title = source.get_text(strip=True) or "Source Link"
                            source_url = source['href']
                            if source_url.startswith('/'):
                                source_url = f"https://www.perplexity.ai{source_url}"
                            results.append(f"Source {i+1}: {source_title} - {source_url}\n")
                            logger.info(f"Added source: {source_title} - {source_url}")
                        sources_found = True
                        break
                
                # If no answer or sources found, extract raw text as a fallback
                if not results:
                    logger.info("No structured answer found, extracting raw text as fallback")
                    for element in soup(["script", "style", "footer", "nav", "header"]):
                        element.decompose()
                    main_content = soup.find('div', id='main') or soup.find('main') or soup
                    raw_text = main_content.get_text(separator=' ', strip=True)
                    if raw_text:
                        # Filter out common UI text
                        lines = [line.strip() for line in raw_text.split('\n') if len(line.strip()) > 20]
                        filtered_text = ' '.join(lines)
                        results.append(f"Raw Text: {filtered_text[:3000]}...\n")
                        logger.info(f"Added raw text (first 100 chars for log): {filtered_text[:100]}...")
                
                if not results:
                    logger.warning("No search results found after all attempts")
                    return "No search results found. Falling back to internal knowledge."
                
                search_result = "\n".join(results)
                logger.info(f"Search results for query '{query}':\n{search_result[:500]}...")
                return search_result
                
    except asyncio.TimeoutError:
        logger.error(f"Perplexity AI search timed out for query '{query}'")
        return "Error: Search timed out. Falling back to internal knowledge."
    except Exception as e:
        logger.error(f"Error during Perplexity AI search for query '{query}': {str(e)}")
        return f"Error during Perplexity AI search: {str(e)}. Falling back to internal knowledge."


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
    
    # Handle #search keyword
    if "#search" in message.content.lower():
        search_query = message.content.lower().split("#search", 1)[1].strip()
        if not search_query:
            await message.channel.send("Please provide a search query after #search, e.g., #search python tutorial")
            return
        
        async with message.channel.typing():
            # Perform search for the user-provided query using Perplexity AI
            user_search_results = await perform_perplexity_search(search_query)
            
            # Check if there's an image attachment and perform OCR + search
            ocr_search_results = ""
            text_from_image = ""
            has_image = False
            for attachment in message.attachments:
                if attachment.content_type and attachment.content_type.startswith("image/"):
                    has_image = True
                    text_from_image = await extract_text_from_image(attachment.url)
                    if text_from_image and text_from_image != "No text detected in the image." and not text_from_image.startswith("Error"):
                        ocr_search_results = await perform_perplexity_search(text_from_image)
                    else:
                        ocr_search_results = "No relevant text extracted from the image to search."
                    break  # Process only the first image
            
            # Prepare prompt based on whether an image was provided
            if has_image:
                prompt = (
                    f"Context of conversation:\n{full_context}\n\n"
                    f"User's current message:\n{message.content}\n\n"
                    f"The user has requested an internet search with the query '{search_query}'. "
                    f"Below are the top search results for the user's query:\n\n{user_search_results}\n\n"
                    f"Additionally, an image was provided, and the following text was extracted using OCR:\n\n{text_from_image}\n\n"
                    f"Below are the top search results for the OCR-extracted text:\n\n{ocr_search_results}\n\n"
                    "Please process both sets of search results and provide a concise summary or answer based on the user's query and the OCR-extracted text. "
                    "Respond with 'Reason:' and 'Answer:' sections."
                )
            else:
                prompt = (
                    f"Context of conversation:\n{full_context}\n\n"
                    f"User's current message:\n{message.content}\n\n"
                    f"The user has requested an internet search with the query '{search_query}'. "
                    f"Below are the top search results for the user's query:\n\n{user_search_results}\n\n"
                    "Please process the search results and provide a concise summary or answer based on the user's query. "
                    "Respond with 'Reason:' and 'Answer:' sections."
                )
            
            # If search failed due to bot detection or rate limits, fall back to internal knowledge
            if "Falling back to internal knowledge" in user_search_results:
                prompt = (
                    f"Context of conversation:\n{full_context}\n\n"
                    f"User's current message:\n{message.content}\n\n"
                    f"The user has requested an internet search with the query '{search_query}', but the search failed due to Perplexity AI's bot detection or rate limits. "
                    "Please answer the query using your internal knowledge instead. "
                    "Respond with 'Reason:' and 'Answer:' sections."
                )
            
            # Get response from DeepSeek API
            answer, reason = await get_ai_response(prompt)
            await message.channel.send(f"Reason: {reason}", reference=message, mention_author=False)
            await message.channel.send(f"Answer: {answer}")
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
    
    # Handle messages in activated channels
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
