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
async def get_ai_response(user_prompt: str, ocr_text: str = None) -> tuple[str, str]:
    system_instructions = (
        "You are a helpful Discord bot that solves problems and answers questions. "
        "When given OCR text, analyze it carefully. "
        "Respond in two sections:\n"
        "1) Reason: (step-by-step reasoning)\n"
        "2) Answer: (concise final answer)\n"
        "Don't use formatting symbols like * | or ~"
    )

    messages = [
        {"role": "system", "content": system_instructions},
        {"role": "user", "content": f"USER PROMPT: {user_prompt}"}
    ]
    
    if ocr_text and ocr_text != "No text detected in the image." and not ocr_text.startswith("Error:"):
        messages.append({
            "role": "user", 
            "content": f"OCR TEXT FROM IMAGE:\n{ocr_text}"
        })

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    data = {
        "model": MODEL,
        "messages": messages,
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(API_URL, headers=headers, json=data) as resp:
                logger.info(f"AI API request sent with payload: {data}")
                resp.raise_for_status()
                response_json = await resp.json()
                content = response_json["choices"][0]["message"]["content"]
                logger.info(f"Full AI response: {content}")

                if "Reason:" in content and "Answer:" in content:
                    reason = content.split("Answer:")[0].split("Reason:")[-1].strip()
                    answer = content.split("Answer:")[-1].strip()
                else:
                    reason = content
                    answer = "Answer: Formatting issue."

                return answer, reason
    except Exception as e:
        logger.error(f"AI API error: {e}", exc_info=True)
        return "Answer: AI error.", "Reason: API issue"

async def extract_text_from_image(image_url: str) -> str:
    try:
        logger.info(f"Starting OCR process for image: {image_url}")
        async with aiohttp.ClientSession() as session:
            async with session.get(image_url) as response:
                logger.info(f"Image fetch status: {response.status}")
                if response.status == 200:
                    img_data = await response.read()
                    logger.info(f"Received image data: {len(img_data)} bytes")
                    
                    img = Image.open(io.BytesIO(img_data))
                    logger.info("Image opened successfully")
                    
                    text = pytesseract.image_to_string(img, config=TESSERACT_CONFIG)
                    logger.info(f"OCR completed. Text length: {len(text)} characters")
                    
                    if text.strip():
                        logger.info(f"Sample extracted text: {text[:500]}...")
                        return text.strip()
                    else:
                        logger.warning("No text detected in image")
                        return "No text detected in the image."
                else:
                    logger.error(f"Failed to fetch image: {response.status}")
                    return "Could not retrieve the image."
    except Exception as e:
        logger.error(f"OCR failed: {str(e)}", exc_info=True)
        return f"Error: {str(e)}"

async def get_conversation_context(channel: discord.TextChannel) -> str:
    full_context = ""
    async for msg in channel.history(limit=MAX_CONTEXT_MESSAGES):
        full_context = f"{msg.author.name}: {msg.content}\n" + full_context
    logger.info(f"Conversation context: {full_context}")
    return full_context

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# BOT COMMANDS
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
# MESSAGE HANDLING
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
@bot.event
async def on_ready():
    logger.info(f"Logged in as {bot.user} (ID: {bot.user.id})")

@bot.event
async def on_message(message: discord.Message):
    if message.author == bot.user or message.author.bot:
        return

    logger.info(f"New message from {message.author}: {message.content}")
    if message.attachments:
        logger.info(f"Attachments detected: {len(message.attachments)}")
        for att in message.attachments:
            logger.info(f" - {att.filename} ({att.content_type}) - {att.url}")

    await bot.process_commands(message)

    # Handle images in activated channels
    if message.channel.id in activated_channels and message.attachments:
        image_attachments = [
            att for att in message.attachments 
            if att.content_type and att.content_type.startswith("image/")
        ]
        
        if image_attachments:
            logger.info(f"Processing {len(image_attachments)} image(s)")
            for attachment in image_attachments:
                async with message.channel.typing():
                    # Step 1: Extract text from image
                    extracted_text = await extract_text_from_image(attachment.url)
                    
                    # Step 2: Handle OCR errors
                    if extracted_text == "No text detected in the image.":
                        await message.channel.send("Answer: No text found in the image.")
                        continue
                    elif extracted_text.startswith("Error:"):
                        await message.channel.send(f"Answer: {extracted_text}")
                        continue
                    
                    # Step 3: Get context and send to AI
                    context = await get_conversation_context(message.channel)
                    prompt = f"{context}\nUser asked: {message.content}"
                    
                    answer, reason = await get_ai_response(
                        user_prompt=prompt,
                        ocr_text=extracted_text
                    )
                    
                    # Step 4: Send responses
                    await message.channel.send(f"Reason: {reason}")
                    await message.channel.send(f"Answer: {answer}")
            return

    # Handle regular messages in activated channels
    if message.channel.id in activated_channels:
        async with message.channel.typing():
            context = await get_conversation_context(message.channel)
            prompt = f"{context}\nUser said: {message.content}"
            
            answer, reason = await get_ai_response(prompt)
            await message.channel.send(f"Reason: {reason}")
            await message.channel.send(f"Answer: {answer}")

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# RUN THE BOT
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if __name__ == "__main__":
    bot.run(DISCORD_BOT_TOKEN)
