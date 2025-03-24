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
MODEL = "deepseek/deepseek-r1-zero:free"
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

def reformat_equation(text: str) -> str:
    """Reformats equations from LaTeX-like syntax to plain text with proper spacing."""
    text = re.sub(r'\\frac\{([^}]+)\}\{([^}]+)\}', r'(\1/\2)', text)
    text = text.replace("\\Rightarrow", "=>").replace("\\times", "*")
    text = text.replace("\\text", "").replace("\\left", "").replace("\\right", "")
    text = text.replace("\\", "")
    text = re.sub(r'(\d|\w)([+\-*/=])', r'\1 \2', text)
    text = re.sub(r'([+\-*/=])(\d|\w)', r'\1 \2', text)
    text = re.sub(r'(\w+)(x_\d+)', r'\1 * \2', text)
    text = re.sub(r'\((\d+/\d+)\)(x_\d+)', r'(\1) * \2', text)
    return text

def wrap_equations_in_code_blocks(text: str) -> str:
    """Wraps detected equations in Discord code blocks for readability."""
    lines = text.split('\n')
    result = []
    for line in lines:
        if (re.search(r'[=+\-*/]|=>', line) or re.search(r'x_\d+', line)) and not line.strip().startswith('```'):
            line = line.replace('$$  ', '').replace('$', '').strip()
            line = reformat_equation(line)
            result.append(f"```\n{line}\n```")
        else:
            result.append(line)
    return '\n'.join(result)

async def perform_google_search(query: str) -> str:
    """Performs a Google search and returns the top results as a string."""
    try:
        async with aiohttp.ClientSession() as session:
            encoded_query = "+".join(query.split())
            url = f"{GOOGLE_SEARCH_URL}{encoded_query}"
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            async with session.get(url, headers=headers) as response:
                if response.status != 200:
                    logger.error(f"Google search failed with status: {response.status}")
                    return "Error: Unable to perform Google search."
                
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                
                results = []
                for g in soup.find_all('div', class_='g')[:5]:  # Top 5 results
                    title = g.find('h3')
                    snippet = g.find('div', class_='VwiC3b')
                    if title and snippet:
                        results.append(f"Title: {title.text}\nSnippet: {snippet.text}\n")
                
                if not results:
                    return "No search results found."
                
                search_result = "\n".join(results)
                logger.info(f"Search results for query '{query}':\n{search_result}")
                return search_result
    except Exception as e:
        logger.error(f"Error during Google search for query '{query}': {str(e)}")
        return f"Error during Google search: {str(e)}"

async def get_ai_response(user_prompt: str) -> tuple[str, str]:
    """Fetches a response from the DeepSeek API and formats it with Reason: and Answer: sections."""
    system_instructions = (
        "You are a helpful Discord bot that solves problems and answers questions. Your response will be sent directly to a Discord channel, so you must format it specifically for Discord's rendering. "
        "Your response must always be structured with exactly two sections:\n"
        "1) 'Reason:' - Explain your chain-of-thought or reasoning step-by-step. Do not use LaTeX (e.g., $x_1$,   $$...$$  , \\frac{5}{k}, \\text, \\Rightarrow, \\left, \\right). Instead, use plain text for variables (e.g., x_1, 5/k) and symbols (e.g., => for implies, ( ) for parentheses). All mathematical equations must be written in plain text and wrapped in Discord code blocks (```) for clarity, with proper spacing for readability. For example, write '5 = k * x_1' inside a code block like this:\n"
        "```\n5 = k * x_1\n```\n"
        "For complex expressions, add spaces around operators (e.g., '5 * x_1 - 2 * (7/5) * x_1' instead of '5x_1-2*(7/5)x_1'). Do not use quotation marks around equations.\n"
        "2) 'Answer:' - Provide your final answer in a single, concise sentence. Use plain text for variables and wrap any equations in Discord code blocks (```) if needed. For example, 'Answer: The tension is 11 N, so the answer is C.'\n"
        "Do not use parentheses around 'Reasoning' or other variations—use 'Reason:' and 'Answer:' exactly as specified. "
        "Do not use bold markers (**) or any other Markdown formatting except for code blocks (```). "
        "Ensure all code blocks are properly closed with ```. "
        "Even if the OCR text is incomplete or an error message, provide both sections, explaining the issue in 'Reason:' and suggesting a solution in 'Answer:'. "
        "Ensure all brackets are properly closed to avoid formatting errors."
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

                content = content.replace("**", "").replace("  $$", "").replace("$", "").replace("\\text", "").replace("\\frac", "").replace("\\Rightarrow", "=>").replace("\\times", "*").replace("\\left", "(").replace("\\right", ")").replace("\\", "")
                content = re.sub(r'\{[^}]*\}', lambda m: m.group(0).replace('{', '').replace('}', ''), content).replace("{", "").replace("}", "")
                content = re.sub(r'$$ [^ $$]*\]', lambda m: m.group(0).replace('[', '').replace(']', ''), content).replace("[", "").replace("]", "")
                if content.count("```") % 2 != 0:
                    content += "\n```"
                content = re.sub(r'"([^"]*=[^"]*)"', r'\1', content)
                content = content.replace("5xl - 2x2", "5x_1 - 2x_2").replace("step - by - step", "step-by-step")
                content = re.sub(r'\s+', ' ', content).strip()
                content = wrap_equations_in_code_blocks(content)

                if "Reason:" in content and "Answer:" in content:
                    reason_part = content.split("Answer:")[0].split("Reason:")[-1].strip()
                    answer_part = content.split("Answer:")[-1].strip()
                else:
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

                reason_part = reason_part.replace("**", "").replace("$", "")
                answer_part = answer_part.replace("**", "").replace("$", "")
                reason_part = re.sub(r'\n\s*\n+', '\n', reason_part).strip()
                answer_part = re.sub(r'\n\s*\n+', '\n', answer_part).strip()

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
                            
                            if text.strip() and avg_conf > best_confidence:
                                best_text = text
                                best_confidence = avg_conf
                    
                    if not best_text.strip():
                        best_text = pytesseract.image_to_string(original_img, config=TESSERACT_CONFIG)
                        logger.info(f"Fallback OCR on original image: {best_text}")
                    
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
    
    # Handle replies to the bot (works in both activated and deactivated channels)
    if message.reference and message.reference.resolved and message.reference.resolved.author == bot.user:
        referenced_msg = message.reference.resolved
        prompt = f"Previous bot message:\n{referenced_msg.content}\n\nUser's new message:\n{message.content}\nPlease respond with 'Reason:' and 'Answer:' sections."
        async with message.channel.typing():
            answer, reason = await get_ai_response(prompt)
            reasoning_message = f"Reason:\n{reason}"
            answer_message = f"Answer:\n{answer}"
            
            reasoning_chunks = chunk_text(reasoning_message)
            if reasoning_chunks:
                await message.channel.send(reasoning_chunks[0], reference=message, mention_author=False)
                for chunk in reasoning_chunks[1:]:
                    await message.channel.send(chunk)
            
            for chunk in chunk_text(answer_message):
                await message.channel.send(chunk)
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
            reasoning_message = f"Reason:\n{reason}"
            answer_message = f"Answer:\n{answer}"
            
            reasoning_chunks = chunk_text(reasoning_message)
            if reasoning_chunks:
                await message.channel.send(reasoning_chunks[0], reference=message, mention_author=False)
                for chunk in reasoning_chunks[1:]:
                    await message.channel.send(chunk)
            
            for chunk in chunk_text(answer_message):
                await message.channel.send(chunk)
        return
    
    # Handle bot mentions (works in both activated and deactivated channels)
    if bot.user.mentioned_in(message):
        prompt = f"Context of conversation:\n{full_context}\n\nUser's current message:\n{message.content}\nPlease respond with 'Reason:' and 'Answer:' sections."
        async with message.channel.typing():
            answer, reason = await get_ai_response(prompt)
            reasoning_message = f"Reason:\n{reason}"
            answer_message = f"Answer:\n{answer}"
            
            reasoning_chunks = chunk_text(reasoning_message)
            if reasoning_chunks:
                await message.channel.send(reasoning_chunks[0], reference=message, mention_author=False)
                for chunk in reasoning_chunks[1:]:
                    await message.channel.send(chunk)
            
            for chunk in chunk_text(answer_message):
                await message.channel.send(chunk)
        return
    
    # Handle #search keyword
    # Note: If a message contains both a mention and #search (e.g., "@BotName #search python"),
    # the mention will take precedence. If you want #search to take precedence, swap this block with the mention block above.
    if "#search" in message.content.lower():
        search_query = message.content.lower().split("#search", 1)[1].strip()
        if not search_query:
            await message.channel.send("Please provide a search query after #search, e.g., #search python tutorial")
            return
        
        async with message.channel.typing():
            # Perform search for the user-provided query
            user_search_results = await perform_google_search(search_query)
            
            # Check if there's an image attachment and perform OCR + search
            ocr_search_results = ""
            text_from_image = ""
            has_image = False
            for attachment in message.attachments:
                if attachment.content_type and attachment.content_type.startswith("image/"):
                    has_image = True
                    text_from_image = await extract_text_from_image(attachment.url)
                    if text_from_image and text_from_image != "No text detected in the image." and not text_from_image.startswith("Error"):
                        ocr_search_results = await perform_google_search(text_from_image)
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
            
            # Get response from DeepSeek API
            answer, reason = await get_ai_response(prompt)
            reasoning_message = f"Reason:\n{reason}"
            answer_message = f"Answer:\n{answer}"
            
            # Send the response in chunks if necessary
            reasoning_chunks = chunk_text(reasoning_message)
            if reasoning_chunks:
                await message.channel.send(reasoning_chunks[0], reference=message, mention_author=False)
                for chunk in reasoning_chunks[1:]:
                    await message.channel.send(chunk)
            
            for chunk in chunk_text(answer_message):
                await message.channel.send(chunk)
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
                reasoning_message = f"Reason:\n{reason}"
                answer_message = f"Answer:\n{answer}"
                
                reasoning_chunks = chunk_text(reasoning_message)
                if reasoning_chunks:
                    await message.channel.send(reasoning_chunks[0], reference=message, mention_author=False)
                    for chunk in reasoning_chunks[1:]:
                        await message.channel.send(chunk)
                
                for chunk in chunk_text(answer_message):
                    await message.channel.send(chunk)
            return
    
    # Handle messages in activated channels
    if message.channel.id in activated_channels:
        prompt = f"Context of conversation:\n{full_context}\n\nUser's current message:\n{message.content}\nPlease respond with 'Reason:' and 'Answer:' sections."
        async with message.channel.typing():
            answer, reason = await get_ai_response(prompt)
            reasoning_message = f"Reason:\n{reason}"
            answer_message = f"Answer:\n{answer}"
            
            reasoning_chunks = chunk_text(reasoning_message)
            if reasoning_chunks:
                await message.channel.send(reasoning_chunks[0], reference=message, mention_author=False)
                for chunk in reasoning_chunks[1:]:
                    await message.channel.send(chunk)
            
            for chunk in chunk_text(answer_message):
                await message.channel.send(chunk)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# RUN THE BOT
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if __name__ == "__main__":
    bot.run(DISCORD_BOT_TOKEN)
