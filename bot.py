import discord
import aiohttp
import asyncio
import re
from discord.ext import commands
import os

# --- CONFIGURATION ---
DISCORD_BOT_TOKEN =  os.getenv('BOT_TOKEN') # Replace with your actual token
OPENROUTER_API_KEY = "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"  # Replace with your actual OpenRouter API key
API_URL = "https://openrouter.ai/api/v1/chat/completions"

# Using your model name
MODEL = "deepseek/deepseek-r1-zero:free"

# --- BOT SETUP ---
intents = discord.Intents.default()
intents.message_content = True  # Enable "Message Content Intent" in Developer Portal
bot = commands.Bot(command_prefix="!", intents=intents)

# --- HELPER FUNCTIONS ---
def chunk_text(text: str, max_length: int = 2000):
    """Splits text into chunks of up to max_length characters."""
    return [text[i : i + max_length] for i in range(0, len(text), max_length)]

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
        "You are a helpful assistant. Always include two sections in your response:\n"
        "1) 'Reason:' - your chain-of-thought or explanation.\n"
        "2) 'Answer:' - your final answer.\n"
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
        return "I'm having trouble responding right now. Please try again later.", "N/A"

# --- BOT EVENT HANDLERS ---
@bot.event
async def on_ready():
    print(f"Logged in as {bot.user}")
    # Optional: stop after one hour. Remove these lines for indefinite uptime.
    bot.loop.create_task(stop_after_one_hour())

async def stop_after_one_hour():
    """Stops the bot after one hour."""
    await asyncio.sleep(3600)
    print("Shutting down the bot after one hour...")
    await bot.close()

@bot.event
async def on_message(message: discord.Message):
    """
    Processes incoming messages:
    - If replying to the bot or mentioning the bot, builds a prompt.
    - Retrieves the AI response with a 'Reason:' and 'Answer:'.
    - Sends only the first message as a reply; all subsequent messages are sent normally.
    """
    if message.author == bot.user:
        return

    prompt = ""
    referenced_msg = await fetch_referenced_message(message)

    if referenced_msg and referenced_msg.author == bot.user:
        # Condition 1: User replied to the bot's message
        old_bot_text = referenced_msg.content
        new_user_text = message.content
        prompt = (
            f"Previous bot message:\n{old_bot_text}\n\n"
            f"User's new message:\n{new_user_text}\n\n"
            "Provide your response with 'Reason:' and 'Answer:' sections."
        )
    elif referenced_msg and (bot.user in message.mentions):
        # Condition 2: User replied to another message but mentioned the bot
        referenced_text = referenced_msg.content
        user_text = message.content.replace(bot.user.mention, "").strip()
        prompt = (
            f"The user is replying to this message:\n{referenced_text}\n\n"
            f"Now, the user is asking:\n{user_text}\n\n"
            "Provide your response with 'Reason:' and 'Answer:' sections."
        )
    elif bot.user in message.mentions:
        # Condition 3: Direct mention (no reply)
        user_text = message.content.replace(bot.user.mention, "").strip()
        prompt = (
            f"{user_text}\nProvide your response with 'Reason:' and 'Answer:' sections."
        )
    else:
        await bot.process_commands(message)
        return

    if prompt:
        async with message.channel.typing():
            answer, reason = await get_ai_response(prompt)
            # Format the reasoning with a "Reasoning:" label inside parentheses
            reasoning_message = f"(Reasoning: {reason})"
            answer_message = answer

            # Combine all chunks and only use reply reference for the very first message.
            # Send reasoning messages first.
            reasoning_chunks = chunk_text(reasoning_message)
            if reasoning_chunks:
                # Send the first chunk as a reply.
                await message.channel.send(reasoning_chunks[0], reference=message, mention_author=False)
                for chunk in reasoning_chunks[1:]:
                    await message.channel.send(chunk)

            # Send the answer messages (none of these are replies).
            for chunk in chunk_text(answer_message):
                await message.channel.send(chunk)

    await bot.process_commands(message)

# --- RUN THE BOT ---
bot.run(DISCORD_BOT_TOKEN)
