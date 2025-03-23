import discord
import aiohttp
import asyncio
import re
import os
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
MODEL = "deepseek/deepseek-r1:free"

# Tesseract configuration
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Uncomment on Windows
TESSERACT_CONFIG = '--oem 1 --psm 3 -l eng+osd'  # OCR Engine Mode 1, Page Segmentation Mode 3, English + orientation detection

# --- BOT SETUP ---
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)

# --- GLOBAL VARIABLES ---
activated_channels = {}
MAX_CONTEXT_MESSAGES = 20

# --- HELPER FUNCTIONS ---
def chunk_text(text: str, max_length: int = 2000):
    """Splits text into chunks of up to max_length characters."""
    return [text[i: i + max_length] for i in range(0, len(text), max_length)]

async def fetch_referenced_message(message: discord.Message):
    """Safely fetch the message that the user is replying to."""
    if not message.reference:
        return None
    ref = message.reference.resolved
    if ref:
        return ref
    try:
        return await message.channel.fetch_message(message.reference.message_id)
    except Exception:
        return None

async def get_ai_response(user_prompt: str) -> tuple[str, str]:
    """
    Calls the AI model via OpenRouter's API.
    Uses a system message to demand that the response include 'Reason:' and 'Answer:' sections.
    Returns (answer, reason). If the model doesn't comply, returns (full_response, "N/A").
    """
    system_instructions = (
        "Always include two sections in your response:\n"
        "1) 'Reason:' - your chain-of-thought or your thinking like that of a standard reasoning bot.\n"
        "2) 'Answer:' - your final answer in proper sentences like a textbot.\n"
        "Even for simple prompts, please include both sections."
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
                    return content.strip(), "N/A"

                # Clean up: remove extra blank lines and bold markers
                reason_part = re.sub(r'\n\s*\n+', '\n', reason_part).replace("**", "").strip()
                answer_part = re.sub(r'\n\s*\n+', '\n', answer_part).replace("**", "").strip()

                return answer_part, reason_part

    except Exception as e:
        print(f"Error calling AI API: {str(e)}")
        return "I'm having trouble responding right now. Please try again later.", "Seems like issue connecting to API"

def preprocess_image(img):
    """Apply advanced preprocessing techniques to improve OCR accuracy"""
    try:
        # Convert PIL Image to OpenCV format
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        
        # Get original dimensions for later reference
        original_height, original_width = img_cv.shape[:2]
        
        # Upscale image if it's small (this helps with OCR for low-res images)
        # Don't upscale images that are already large to avoid performance issues
        if original_height < 1000 or original_width < 1000:
            scale_factor = 2.0  # Double the size
            img_cv = cv2.resize(img_cv, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
        
        # Grayscale conversion
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        
        # Check if the image needs inversion (white text on dark background)
        # Calculate the mean value of the image
        mean_value = np.mean(gray)
        if mean_value < 127:  # If image is predominantly dark
            gray = cv2.bitwise_not(gray)  # Invert the image
        
        # Noise removal
        denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        
        # Increase contrast
        alpha = 1.5  # Contrast control (1.0-3.0)
        beta = 0  # Brightness control (0-100)
        contrasted = cv2.convertScaleAbs(denoised, alpha=alpha, beta=beta)
        
        # Binarization/thresholding with Otsu's method
        _, binary = cv2.threshold(contrasted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Alternative: Adaptive thresholding (try both and use the better one)
        adaptive = cv2.adaptiveThreshold(contrasted, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                    cv2.THRESH_BINARY, 11, 2)
        
        # Dilation to make text more prominent
        kernel = np.ones((1, 1), np.uint8)
        dilated_binary = cv2.dilate(binary, kernel, iterations=1)
        dilated_adaptive = cv2.dilate(adaptive, kernel, iterations=1)
        
        # Convert back to PIL Images
        processed_binary = Image.fromarray(dilated_binary)
        processed_adaptive = Image.fromarray(dilated_adaptive)
        
        return [processed_binary, processed_adaptive, Image.fromarray(contrasted)]
    
    except Exception as e:
        print(f"Image preprocessing error: {str(e)}")
        return [img]  # Return original image if preprocessing fails

async def extract_text_from_image(image_url: str) -> str:
    """Extracts text from an image using OCR with multiple preprocessing methods."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(image_url) as response:
                if response.status == 200:
                    img_data = await response.read()
                    original_img = Image.open(io.BytesIO(img_data))
                    
                    # Try different preprocessing techniques
                    processed_images = preprocess_image(original_img)
                    
                    # Try OCR on each processed image
                    best_text = ""
                    best_confidence = 0
                    
                    for img in processed_images:
                        # Try different PSM (Page Segmentation Mode) values
                        for psm in [3, 4, 6]:  # 3=auto, 4=single column, 6=single block
                            config = f'--oem 1 --psm {psm} -l eng+osd'
                            
                            # Get OCR data including confidence scores
                            ocr_data = pytesseract.image_to_data(img, config=config, output_type=pytesseract.Output.DICT)
                            
                            # Calculate average confidence for words with confidence > 0
                            conf_values = [conf for conf in ocr_data['conf'] if conf > 0]
                            if conf_values:
                                avg_conf = sum(conf_values) / len(conf_values)
                            else:
                                avg_conf = 0
                                
                            # Extract text
                            text = pytesseract.image_to_string(img, config=config)
                            
                            # Keep the result with highest confidence
                            if text.strip() and avg_conf > best_confidence:
                                best_text = text
                                best_confidence = avg_conf
                    
                    # If no good text was found, try one more time with the original image
                    if not best_text.strip():
                        best_text = pytesseract.image_to_string(original_img, config=TESSERACT_CONFIG)
                    
                    return best_text.strip() if best_text.strip() else "No text detected in the image."
                else:
                    return "Could not retrieve the image."
    except Exception as e:
        print(f"OCR error: {str(e)}")
        return f"Error processing image: {str(e)}"

async def get_conversation_context(channel, limit=MAX_CONTEXT_MESSAGES):
    """Fetch conversation context from channel history"""
    full_context = ""
    async for msg in channel.history(limit=limit):
        full_context = f"{msg.author.name}: {msg.content}\n" + full_context
    return full_context

# --- BOT COMMAND HANDLERS ---

@bot.command()
async def activate(ctx):
    """Activates the bot to respond to all messages in the current channel."""
    activated_channels[ctx.channel.id] = True
    await ctx.send(f"The bot is now activated and will respond to all messages in this channel!")

@bot.command()
async def deactivate(ctx):
    """Deactivates the bot from responding to all messages in the current channel."""
    if ctx.channel.id in activated_channels:
        del activated_channels[ctx.channel.id]
        await ctx.send(f"The bot is now deactivated and will only respond to mentions and replies in this channel.")
    else:
        await ctx.send(f"The bot is not activated in this channel.")

@bot.command()
async def ping(ctx):
    """Test command to check if the bot is responding."""
    await ctx.send('Pong!')


# --- BOT EVENT HANDLERS ---

@bot.event
async def on_ready():
    print(f"Logged in as {bot.user}")

@bot.event
async def on_message(message: discord.Message):
    """
    Handles all messages:
    - Ignores messages from other bots.
    - Responds to mentions of the bot (@bot), even when not activated.
    - Responds to replies to its own messages.
    - Responds to all messages in activated channels.
    - Processes images with OCR when present.
    """
    if message.author == bot.user or message.author.bot:
        return  # Ignore bot's own messages and other bots' messages
    
    # Process commands first
    await bot.process_commands(message)
    
    # Get conversation context for use in multiple cases
    full_context = await get_conversation_context(message.channel)
    
    # Check for image attachments and perform OCR if present
    for attachment in message.attachments:
        if attachment.content_type and attachment.content_type.startswith("image/"):  # Only process images
            text_from_image = await extract_text_from_image(attachment.url)
            
            # Construct the prompt specifically for OCR - maintaining original prompt structure
            prompt = f"Context of conversation:\n{full_context}\n\nUser's current message:\n{message.content}\n\nText from the image has been extracted using OCR. When the user asks for extracted text just say this:\n\n{text_from_image}\n\n or you can use this text from ocr to do asked calculations. We all know you cant analyze images on your own so we have set up a separate api for ocr and the result is provided above. Use its context to get a message to be complete for this discord bot. Whenever user asks about image just use this {text_from_image}\n\nPlease analyze the following text and respond with 'Reason:' and 'Answer:' sections."
            
            async with message.channel.typing():
                answer, reason = await get_ai_response(prompt)
                reasoning_message = f"(Reasoning: {reason})"
                answer_message = answer
                
                # Send reasoning first
                reasoning_chunks = chunk_text(reasoning_message)
                if reasoning_chunks:
                    await message.channel.send(reasoning_chunks[0], reference=message, mention_author=False)
                    for chunk in reasoning_chunks[1:]:
                        await message.channel.send(chunk)
                
                # Send the final answer
                for chunk in chunk_text(answer_message):
                    await message.channel.send(chunk)
            
            return  # Stop after processing the image and sending the response
    
    # Respond to mentions (@bot), even when not activated
    if bot.user.mentioned_in(message):
        prompt = message.content
        
        # Combine the context and the current message to create a prompt
        prompt = f"Context of conversation:\n{full_context}\n\nUser's current message:\n{prompt}\nPlease respond with 'Reason:' and 'Answer:' sections."
        
        # Process the AI response
        async with message.channel.typing():
            answer, reason = await get_ai_response(prompt)
            reasoning_message = f"(Reasoning: {reason})"
            answer_message = answer
            
            # Send reasoning first
            reasoning_chunks = chunk_text(reasoning_message)
            if reasoning_chunks:
                await message.channel.send(reasoning_chunks[0], reference=message, mention_author=False)
                for chunk in reasoning_chunks[1:]:
                    await message.channel.send(chunk)
            
            # Send the final answer
            for chunk in chunk_text(answer_message):
                await message.channel.send(chunk)
    
    # Handle replies to the bot's own message
    elif message.reference and message.reference.resolved and message.reference.resolved.author == bot.user:
        referenced_msg = message.reference.resolved
        prompt = f"Previous bot message:\n{referenced_msg.content}\n\nUser's new message:\n{message.content}\nPlease respond with 'Reason:' and 'Answer:' sections."
        
        async with message.channel.typing():
            answer, reason = await get_ai_response(prompt)
            reasoning_message = f"(Reasoning: {reason})"
            answer_message = answer
            
            # Send reasoning first
            reasoning_chunks = chunk_text(reasoning_message)
            if reasoning_chunks:
                await message.channel.send(reasoning_chunks[0], reference=message, mention_author=False)
                for chunk in reasoning_chunks[1:]:
                    await message.channel.send(chunk)
            
            # Send the final answer
            for chunk in chunk_text(answer_message):
                await message.channel.send(chunk)
    
    # Handle activated channels (respond to all messages)
    elif message.channel.id in activated_channels:
        prompt = message.content
        
        # Combine the context and the current message to create a prompt
        prompt = f"Context of conversation:\n{full_context}\n\nUser's current message:\n{prompt}\nPlease respond with 'Reason:' and 'Answer:' sections."
        
        # Process the AI response
        async with message.channel.typing():
            answer, reason = await get_ai_response(prompt)
            reasoning_message = f"(Reasoning: {reason})"
            answer_message = answer
            
            # Send reasoning first
            reasoning_chunks = chunk_text(reasoning_message)
            if reasoning_chunks:
                await message.channel.send(reasoning_chunks[0], reference=message, mention_author=False)
                for chunk in reasoning_chunks[1:]:
                    await message.channel.send(chunk)
            
            # Send the final answer
            for chunk in chunk_text(answer_message):
                await message.channel.send(chunk)

# --- RUN THE BOT ---
bot.run(DISCORD_BOT_TOKEN)
