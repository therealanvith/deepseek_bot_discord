import discord
import aiohttp
import asyncio
import re
import os
import logging
from discord.ext import commands
from PIL import Image
import io
import pytesseract
import numpy as np
import cv2
from bs4 import BeautifulSoup

DISCORD_BOT_TOKEN = os.getenv('BOT_TOKEN')
OPENROUTER_API_KEY = os.getenv('API_KEY')
API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "deepseek/deepseek-r1-zero:free"
GOOGLE_SEARCH_URL = "https://www.google.com/search?q="
TESSERACT_BINARY_PATH = os.path.join(os.getenv('GITHUB_WORKSPACE', ''), 'tesseract-local', 'usr', 'bin', 'tesseract')
pytesseract.pytesseract.tesseract_cmd = TESSERACT_BINARY_PATH
TESSERACT_CONFIG = '--oem 1 --psm 3 -l eng+osd'

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.FileHandler('bot.log'), logging.StreamHandler()])
logger = logging.getLogger(__name__)

if not os.path.exists(TESSERACT_BINARY_PATH):
    logger.error(f"Tesseract binary not found at {TESSERACT_BINARY_PATH}")
else:
    logger.info(f"Tesseract binary set to {TESSERACT_BINARY_PATH}")

intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)

activated_channels = {}
MAX_CONTEXT_MESSAGES = 10

def chunk_text(text: str, max_length: int = 2000) -> list:
    return [text[i:i + max_length] for i in range(0, len(text), max_length)]

async def perform_google_search(query: str) -> str:
    try:
        async with aiohttp.ClientSession() as session:
            # Add a timeout to prevent long delays
            async with session.get(
                f"{GOOGLE_SEARCH_URL}{'+'.join(query.split())}",
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"},
                timeout=5  # 5-second timeout
            ) as response:
                if response.status != 200:
                    logger.error(f"Google search failed with status: {response.status}")
                    return "Error: Unable to perform Google search."
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                # Reduce the number of results to speed up parsing
                results = []
                for g in soup.find_all('div', class_='g')[:3]:  # Limit to top 3 results
                    title = g.find('h3')
                    snippet = g.find('div', class_='VwiC3b')
                    if title and snippet:
                        results.append(f"Title: {title.text}\nSnippet: {snippet.text}\n")
                if not results:
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
    system_instructions = (
        "You are a helpful Discord bot that solves problems and answers questions. "
        "Your response must always be structured with exactly two sections:\n"
        "1) 'Reason:' - Explain your chain-of-thought or reasoning step-by-step.\n"
        "2) 'Answer:' - Provide your final answer in a single, concise sentence.\n"
        "Do not use any special formatting, code blocks, or LaTeX. Respond with plain text only."
    )
    headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}
    data = {"model": MODEL, "messages": [{"role": "system", "content": system_instructions}, {"role": "user", "content": user_prompt}]}
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(API_URL, headers=headers, json=data, timeout=10) as resp:
                resp.raise_for_status()
                response_json = await resp.json()
                content = response_json["choices"][0]["message"]["content"]
                logger.info(f"AI API response: {content}")
                # Minimal processing: just ensure Reason: and Answer: are separated
                if "Reason:" in content and "Answer:" in content:
                    reason_part = content.split("Answer:")[0].split("Reason:")[-1].strip()
                    answer_part = content.split("Answer:")[-1].strip()
                else:
                    # Fallback if the API doesn't format correctly
                    sentences = content.split(". ")
                    if len(sentences) > 1:
                        reason_part = ". ".join(sentences[:-1]).strip()
                        answer_part = sentences[-1].strip()
                    else:
                        reason_part = content
                        answer_part = "I couldn't determine a clear answer."
                    answer_part = "Answer: " + answer_part
                return answer_part, reason_part
    except asyncio.TimeoutError:
        logger.error("AI API request timed out")
        return "Answer: I'm having trouble responding right now. Please try again later.", "Reason: API request timed out"
    except Exception as e:
        logger.error(f"Error calling AI API: {str(e)}")
        return "Answer: I'm having trouble responding right now. Please try again later.", "Reason: API connection issue"

def preprocess_image(img: Image.Image) -> list:
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
    full_context = ""
    async for msg in channel.history(limit=limit):
        full_context = f"{msg.author.name}: {msg.content}\n" + full_context
    return full_context

@bot.command()
async def activate(ctx: commands.Context):
    activated_channels[ctx.channel.id] = True
    await ctx.send("The bot is now activated and will respond to all messages in this channel!")

@bot.command()
async def deactivate(ctx: commands.Context):
    if ctx.channel.id in activated_channels:
        del activated_channels[ctx.channel.id]
        await ctx.send("The bot is now deactivated and will only respond to mentions and replies.")
    else:
        await ctx.send("The bot is not activated in this channel.")

@bot.command()
async def ping(ctx: commands.Context):
    await ctx.send('Pong!')

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
    if message.reference and message.reference.resolved and message.reference.resolved.author == bot.user:
        referenced_msg = message.reference.resolved
        prompt = f"Previous bot message:\n{referenced_msg.content}\n\nUser's new message:\n{message.content}\nPlease respond with 'Reason:' and 'Answer:' sections."
        async with message.channel.typing():
            answer, reason = await get_ai_response(prompt)
            await message.channel.send(f"Reason: {reason}", reference=message, mention_author=False)
            await message.channel.send(f"Answer: {answer}")
        return
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
    if bot.user.mentioned_in(message):
        prompt = f"Context of conversation:\n{full_context}\n\nUser's current message:\n{message.content}\nPlease respond with 'Reason:' and 'Answer:' sections."
        async with message.channel.typing():
            answer, reason = await get_ai_response(prompt)
            await message.channel.send(f"Reason: {reason}", reference=message, mention_author=False)
            await message.channel.send(f"Answer: {answer}")
        return
    if "#search" in message.content.lower():
        search_query = message.content.lower().split("#search", 1)[1].strip()
        if not search_query:
            await message.channel.send("Please provide a search query after #search, e.g., #search python tutorial")
            return
        async with message.channel.typing():
            user_search_results = await perform_google_search(search_query)
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
                    break
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
            answer, reason = await get_ai_response(prompt)
            await message.channel.send(f"Reason: {reason}", reference=message, mention_author=False)
            await message.channel.send(f"Answer: {answer}")
        return
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
    if message.channel.id in activated_channels:
        prompt = f"Context of conversation:\n{full_context}\n\nUser's current message:\n{message.content}\nPlease respond with 'Reason:' and 'Answer:' sections."
        async with message.channel.typing():
            answer, reason = await get_ai_response(prompt)
            await message.channel.send(f"Reason: {reason}", reference=message, mention_author=False)
            await message.channel.send(f"Answer: {answer}")

if __name__ == "__main__":
    bot.run(DISCORD_BOT_TOKEN)
