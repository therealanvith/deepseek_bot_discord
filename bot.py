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

# --- CONFIGURATION ---
DISCORD_BOT_TOKEN = os.getenv('BOT_TOKEN')
OPENROUTER_API_KEY = os.getenv('API_KEY')
API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "deepseek/deepseek-r1:free"  # Update this if you've changed the model

# Tesseract configuration
TESSERACT_BINARY_PATH = os.path.join(os.getenv('GITHUB_WORKSPACE', ''), 'tesseract-local', 'usr', 'bin', 'tesseract')
pytesseract.pytesseract.tesseract_cmd = TESSERACT_BINARY_PATH
TESSERACT_CONFIG = '--oem 1 --psm 3 -l eng+osd'

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

if not os.path.exists(TESSERACT_BINARY_PATH):
    logger.error(f"Tesseract binary not found at {TESSERACT_BINARY_PATH}")
else:
    logger.info(f"Tesseract binary set to {TESSERACT_BINARY_PATH}")

# --- BOT SETUP ---
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)

# --- GLOBAL VARIABLES ---
activated_channels = {}
MAX_CONTEXT_MESSAGES = 10

# --- HELPER FUNCTIONS ---
def chunk_text(text: str, max_length: int = 2000):
    return [text[i: i + max_length] for i in range(0, len(text), max_length)]

async def fetch_referenced_message(message: discord.Message):
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

async def get_ai_response(user_prompt: str) -> tuple[str, str]:
    system_instructions = (
        "You are a helpful Discord bot that solves problems and answers questions. "
        "Your response must always be structured with exactly two sections:\n"
        "1) 'Reason:' - Explain your chain-of-thought or reasoning step-by-step. Do not use LaTeX (e.g., $x_1$, \\frac{5}{k}, \\text, \\Rightarrow). Instead, use plain text for variables (e.g., x_1, 5/k) and symbols (e.g., => for implies). Wrap mathematical equations in Discord code blocks (```) for clarity. For example, write '5 = k * x_1' inside a code block like this:\n"
        "```\n5 = k * x_1\n```\n"
        "2) 'Answer:' - Provide your final answer in clear, proper sentences. Use plain text for variables and wrap any equations in Discord code blocks (```) if needed.\n"
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

                # Clean up the AI response
                # Remove all bold markers
                content = content.replace("**", "")
                # Remove LaTeX symbols
                content = content.replace("\\text", "").replace("\\frac", "").replace("\\Rightarrow", "=>").replace("\\times", "*").replace("\\", "")
                # Remove stray brackets
                content = re.sub(r'\{[^}]*\}', lambda m: m.group(0).replace('{', '').replace('}', ''), content)
                content = content.replace("{", "").replace("}", "")
                content = re.sub(r'$$ [^ $$]*\]', lambda m: m.group(0).replace('[', '').replace(']', ''), content)
                content = content.replace("[", "").replace("]", "")
                # Fix incomplete code blocks
                if content.count("```") % 2 != 0:  # If there's an odd number of ```, add a closing one
                    content += "\n```"
                # Normalize whitespace
                content = re.sub(r'\s+', ' ', content).strip()

                # Try to parse Reason: and Answer: sections
                if "Reason:" in content and "Answer:" in content:
                    reason_part = content.split("Answer:")[0].split("Reason:")[-1].strip()
                    answer_part = content.split("Answer:")[-1].strip()
                else:
                    # Fallback: split on the last "So," if present, or assume the last sentence is the answer
                    if "So," in content:
                        parts = content.rsplit("So,", 1)
                        reason_part = parts[0].strip()
                        answer_part = "So, " + parts[1].strip()
                    else:
                        sentences = content.split(". ")
                        if len(sentences) > 1:
                            reason_part = ". ".join(sentences[:-1]).strip()
                            answer_part = sentences[-1].strip()
                        else:
                            reason_part = content
                            answer_part = "I couldn't determine a clear answer due to formatting issues."

                # Final cleanup: ensure no bold markers, LaTeX, or stray brackets remain
                reason_part = reason_part.replace("**", "").replace("{", "").replace("}", "").replace("[", "").replace("]", "").replace("$", "")
                reason_part = reason_part.replace("\\text", "").replace("\\frac", "").replace("\\Rightarrow", "=>").replace("\\times", "*").replace("\\", "")
                answer_part = answer_part.replace("**", "").replace("{", "").replace("}", "").replace("[", "").replace("]", "").replace("$", "")
                answer_part = answer_part.replace("\\text", "").replace("\\frac", "").replace("\\Rightarrow", "=>").replace("\\times", "*").replace("\\", "")
                reason_part = re.sub(r'\n\s*\n+', '\n', reason_part).strip()
                answer_part = re.sub(r'\n\s*\n+', '\n', answer_part).strip()

                return answer_part, reason_part

    except Exception as e:
        logger.error(f"Error calling AI API: {str(e)}")
        return "I'm having trouble responding right now. Please try again later.", "API connection issue"

def preprocess_image(img):
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

async def get_conversation_context(channel, limit=MAX_CONTEXT_MESSAGES):
    full_context = ""
    async for msg in channel.history(limit=limit):
        full_context = f"{msg.author.name}: {msg.content}\n" + full_context
    return full_context

# --- BOT COMMAND HANDLERS ---
@bot.command()
async def activate(ctx):
    activated_channels[ctx.channel.id] = True
    await ctx.send("The bot is now activated and will respond to all messages in this channel!")

@bot.command()
async def deactivate(ctx):
    if ctx.channel.id in activated_channels:
        del activated_channels[ctx.channel.id]
        await ctx.send("The bot is now deactivated and will only respond to mentions and replies.")
    else:
        await ctx.send("The bot is not activated in this channel.")

@bot.command()
async def ping(ctx):
    await ctx.send('Pong!')

# --- BOT EVENT HANDLERS ---
@bot.event
async def on_ready():
    logger.info(f"Logged in as {bot.user}")

@bot.event
async def on_message(message: discord.Message):
    if message.author == bot.user or message.author.bot:
        return
    
    await bot.process_commands(message)
    full_context = await get_conversation_context(message.channel)
    
    for attachment in message.attachments:
        if attachment.content_type and attachment.content_type.startswith("image/"):
            text_from_image = await extract_text_from_image(attachment.url)
            prompt = (
                f"Context of conversation:\n{full_context}\n\n"
                f"User's current message:\n{message.content}\n\n"
                f"Text from the image has been extracted using OCR:\n\n{text_from_image}\n\n"
                "If the OCR text is 'No text detected in the image,' explain in the 'Reason:' section that the image text couldn't be extracted and suggest the user provide a clearer image or type the text manually. "
                "Otherwise, use the extracted text to solve the problem or answer the user's request. "
                "Please respond with 'Reason:' and 'Answer:' sections."
            )
            
            async with message.channel.typing():
                answer, reason = await get_ai_response(prompt)
                # Ensure no bold markers or LaTeX in final output
                reason = reason.replace("**", "").replace("$", "")
                answer = answer.replace("**", "").replace("$", "")
                reasoning_message = f"(Reasoning: {reason})"
                answer_message = answer
                
                reasoning_chunks = chunk_text(reasoning_message)
                if reasoning_chunks:
                    await message.channel.send(reasoning_chunks[0], reference=message, mention_author=False)
                    for chunk in reasoning_chunks[1:]:
                        await message.channel.send(chunk)
                
                for chunk in chunk_text(answer_message):
                    await message.channel.send(chunk)
            return
    
    if bot.user.mentioned_in(message):
        prompt = f"Context of conversation:\n{full_context}\n\nUser's current message:\n{message.content}\nPlease respond with 'Reason:' and 'Answer:' sections."
        async with message.channel.typing():
            answer, reason = await get_ai_response(prompt)
            reasoning_message = f"(Reasoning: {reason})"
            answer_message = answer
            
            reasoning_chunks = chunk_text(reasoning_message)
            if reasoning_chunks:
                await message.channel.send(reasoning_chunks[0], reference=message, mention_author=False)
                for chunk in reasoning_chunks[1:]:
                    await message.channel.send(chunk)
            
            for chunk in chunk_text(answer_message):
                await message.channel.send(chunk)
    
    elif message.reference and message.reference.resolved and message.reference.resolved.author == bot.user:
        referenced_msg = message.reference.resolved
        prompt = f"Previous bot message:\n{referenced_msg.content}\n\nUser's new message:\n{message.content}\nPlease respond with 'Reason:' and 'Answer:' sections."
        async with message.channel.typing():
            answer, reason = await get_ai_response(prompt)
            reasoning_message = f"(Reasoning: {reason})"
            answer_message = answer
            
            reasoning_chunks = chunk_text(reasoning_message)
            if reasoning_chunks:
                await message.channel.send(reasoning_chunks[0], reference=message, mention_author=False)
                for chunk in reasoning_chunks[1:]:
                    await message.channel.send(chunk)
            
            for chunk in chunk_text(answer_message):
                await message.channel.send(chunk)
    
    elif message.channel.id in activated_channels:
        prompt = f"Context of conversation:\n{full_context}\n\nUser's current message:\n{message.content}\nPlease respond with 'Reason:' and 'Answer:' sections."
        async with message.channel.typing():
            answer, reason = await get_ai_response(prompt)
            reasoning_message = f"(Reasoning: {reason})"
            answer_message = answer
            
            reasoning_chunks = chunk_text(reasoning_message)
            if reasoning_chunks:
                await message.channel.send(reasoning_chunks[0], reference=message, mention_author=False)
                for chunk in reasoning_chunks[1:]:
                    await message.channel.send(chunk)
            
            for chunk in chunk_text(answer_message):
                await message.channel.send(chunk)

# --- RUN THE BOT ---
bot.run(DISCORD_BOT_TOKEN)
