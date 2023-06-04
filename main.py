import telebot
from decouple import config
import os
# from DealingWithModel import test_9_emotions
import time
import random
import subprocess
import sys
# pcm_s16le ([1][0][0][0] / 0x0001), 48000 Hz, mono, s16, 768 kb/s
# {'content_type': 'document', 'id': 2781, 'message_id': 2781, 'from_user': {'id': 5011846159, 'is_bot': False, 'first_name': 'Abdul Rahman', 'username': 'abdulator0', 'last_name': 'Souda', 'language_code': 'en', 'can_join_groups': None, 'can_read_all_group_messages': None, 'supports_inline_queries': None, 'is_premium': None, 'added_to_attachment_menu': None}, 'date': 1685809096, 'chat': {'id': 5011846159, 'type': 'private', 'title': None, 'username': 'abdulator0', 'first_name': 'Abdul Rahman', 'last_name': 'Souda', 'is_forum': None, 'photo': None, 'bio': None, 'join_to_send_messages': None, 'join_by_request': None, 'has_private_forwards': None, 'has_restricted_voice_and_video_messages': None, 'description': None, 'invite_link': None, 'pinned_message': None, 'permissions': None, 'slow_mode_delay': None, 'message_auto_delete_time': None, 'has_protected_content': None, 'sticker_set_name': None, 'can_set_sticker_set': None, 'linked_chat_id': None, 'location': None, 'active_usernames': None, 'emoji_status_custom_emoji_id': None, 'has_hidden_members': None, 'has_aggressive_anti_spam_enabled': None}, 'sender_chat': None, 'forward_from': None, 'forward_from_chat': None, 'forward_from_message_id': None, 'forward_signature': None, 'forward_sender_name': None, 'forward_date': None, 'is_automatic_forward': None, 'reply_to_message': None, 'via_bot': None, 'edit_date': None, 'has_protected_content': None, 'media_group_id': None, 'author_signature': None, 'text': None, 'entities': None, 'caption_entities': None, 'audio': None, 'document': {'file_id': 'BQACAgQAAxkBAAIK3WR7Z8htBDvgqqq9n0eWRnCkydxvAAKgDQAC1WnYU6hq9EE9QQ6zLwQ', 'file_unique_id': 'AgADoA0AAtVp2FM', 'thumb': None, 'file_name': '01_01_01_02_dogs-sitting_angry.wav', 'mime_type': 'audio/x-wav', 'file_size': 358882}, 'photo': None, 'sticker': None, 'video': None, 'video_note': None, 'voice': None, 'caption': None, 'contact': None, 'location': None, 'venue': None, 'animation': None, 'dice': None, 'new_chat_member': None, 'new_chat_members': None, 'left_chat_member': None, 'new_chat_title': None, 'new_chat_photo': None, 'delete_chat_photo': None, 'group_chat_created': None, 'supergroup_chat_created': None, 'channel_chat_created': None, 'migrate_to_chat_id': None, 'migrate_from_chat_id': None, 'pinned_message': None, 'invoice': None, 'successful_payment': None, 'connected_website': None, 'reply_markup': None, 'message_thread_id': None, 'is_topic_message': None, 'forum_topic_created': None, 'forum_topic_closed': None, 'forum_topic_reopened': None, 'has_media_spoiler': None, 'forum_topic_edited': None, 'general_forum_topic_hidden': None, 'general_forum_topic_unhidden': None, 'write_access_allowed': None, 'json': {'message_id': 2781, 'from': {'id': 5011846159, 'is_bot': False, 'first_name': 'Abdul Rahman', 'last_name': 'Souda', 'username': 'abdulator0', 'language_code': 'en'}, 'chat': {'id': 5011846159, 'first_name': 'Abdul Rahman', 'last_name': 'Souda', 'username': 'abdulator0', 'type': 'private'}, 'date': 1685809096, 'document': {'file_name': '01_01_01_02_dogs-sitting_angry.wav', 'mime_type': 'audio/x-wav', 'file_id': 'BQACAgQAAxkBAAIK3WR7Z8htBDvgqqq9n0eWRnCkydxvAAKgDQAC1WnYU6hq9EE9QQ6zLwQ', 'file_unique_id': 'AgADoA0AAtVp2FM', 'file_size': 358882}}}
# {'content_type': 'voice', 'id': 2772, 'message_id': 2772, 'from_user': {'id': 5011846159, 'is_bot': False, 'first_name': 'Abdul Rahman', 'username': 'abdulator0', 'last_name': 'Souda', 'language_code': 'en', 'can_join_groups': None, 'can_read_all_group_messages': None, 'supports_inline_queries': None, 'is_premium': None, 'added_to_attachment_menu': None}, 'date': 1685809030, 'chat': {'id': 5011846159, 'type': 'private', 'title': None, 'username': 'abdulator0', 'first_name': 'Abdul Rahman', 'last_name': 'Souda', 'is_forum': None, 'photo': None, 'bio': None, 'join_to_send_messages': None, 'join_by_request': None, 'has_private_forwards': None, 'has_restricted_voice_and_video_messages': None, 'description': None, 'invite_link': None, 'pinned_message': None, 'permissions': None, 'slow_mode_delay': None, 'message_auto_delete_time': None, 'has_protected_content': None, 'sticker_set_name': None, 'can_set_sticker_set': None, 'linked_chat_id': None, 'location': None, 'active_usernames': None, 'emoji_status_custom_emoji_id': None, 'has_hidden_members': None, 'has_aggressive_anti_spam_enabled': None}, 'sender_chat': None, 'forward_from': None, 'forward_from_chat': None, 'forward_from_message_id': None, 'forward_signature': None, 'forward_sender_name': None, 'forward_date': None, 'is_automatic_forward': None, 'reply_to_message': None, 'via_bot': None, 'edit_date': None, 'has_protected_content': None, 'media_group_id': None, 'author_signature': None, 'text': None, 'entities': None, 'caption_entities': None, 'audio': None, 'document': None, 'photo': None, 'sticker': None, 'video': None, 'video_note': None, 'voice': {'file_id': 'AwACAgQAAxkBAAIK1GR7Z4YfQO1m0jPreha4U3fs5Y9oAAKeDQAC1WnYUyviF8aWSeDkLwQ', 'file_unique_id': 'AgADng0AAtVp2FM', 'duration': 1, 'performer': None, 'title': None, 'file_name': None, 'mime_type': 'audio/ogg', 'file_size': 5443, 'thumb': None}, 'caption': None, 'contact': None, 'location': None, 'venue': None, 'animation': None, 'dice': None, 'new_chat_member': None, 'new_chat_members': None, 'left_chat_member': None, 'new_chat_title': None, 'new_chat_photo': None, 'delete_chat_photo': None, 'group_chat_created': None, 'supergroup_chat_created': None, 'channel_chat_created': None, 'migrate_to_chat_id': None, 'migrate_from_chat_id': None, 'pinned_message': None, 'invoice': None, 'successful_payment': None, 'connected_website': None, 'reply_markup': None, 'message_thread_id': None, 'is_topic_message': None, 'forum_topic_created': None, 'forum_topic_closed': None, 'forum_topic_reopened': None, 'has_media_spoiler': None, 'forum_topic_edited': None, 'general_forum_topic_hidden': None, 'general_forum_topic_unhidden': None, 'write_access_allowed': None, 'json': {'message_id': 2772, 'from': {'id': 5011846159, 'is_bot': False, 'first_name': 'Abdul Rahman', 'last_name': 'Souda', 'username': 'abdulator0', 'language_code': 'en'}, 'chat': {'id': 5011846159, 'first_name': 'Abdul Rahman', 'last_name': 'Souda', 'username': 'abdulator0', 'type': 'private'}, 'date': 1685809030, 'voice': {'duration': 1, 'mime_type': 'audio/ogg', 'file_id': 'AwACAgQAAxkBAAIK1GR7Z4YfQO1m0jPreha4U3fs5Y9oAAKeDQAC1WnYUyviF8aWSeDkLwQ', 'file_unique_id': 'AgADng0AAtVp2FM', 'file_size': 5443}}}
# from modules.ClassifiersManagement.SVCClassifier  import get_classifier
# from SVCClassifier import get_classifier
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.getcwd()+"\modules\ClassifiersManagement")
from modules.ClassifiersManagement.SVCClassifier import get_classifier,predict
# from modules.ClassifiersManagement.GradientBoostingClassifier import get_classifier,predict
sys.path.insert(0, os.getcwd()+"\modules\FeatuersManagement")
# from modules.FeatuersManagement.Feature import Feature

def getWavDetail(path):
    import wave

    # Open the WAV file
    with wave.open(path, "rb") as wav_file:
        # Get the file metadata
        # num_channels = wav_file()
        getparams = wav_file.getparams()
        compname = wav_file.getcompname()
        comptype = wav_file.getcomptype()
        num_channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        num_frames = wav_file.getnframes()
        duration = num_frames / float(sample_rate)

    # Print the metadata to the console
    print("getparams:", getparams)
    print("compname:", compname)
    print("comptype:", comptype)
    print("Number of channels:", num_channels)
    print("Sample width (bytes):", sample_width)
    print("Sample rate (Hz):", sample_rate)
    print("Number of frames:", num_frames)
    print("Duration (seconds):", duration)

    

# Function to convert ogg to wav
def ogg2wav(src_filename, dest_filename):
    process = subprocess.run(['ffmpeg', '-i', src_filename, dest_filename])
    print(src_filename,dest_filename)
    if process.returncode != 0:
        raise Exception("Error converting file")


# Initialize the bot
BOT_TOKEN = config('BOT_TOKEN')
bot = telebot.TeleBot(BOT_TOKEN)

# Load the model and print "ready" message
# rec = test_9_emotions()
rec = get_classifier()
print('Model is ready')

# Function to process voice messages


def process_voice_message(message,destination_file_path):
    getWavDetail(destination_file_path)
    emotion = predict(destination_file_path)
    # print(emotion)
    # return
    score = 10 # , rec.test_score()
    # Rename the file based on the predicted emotion and the user's username (if available)

    # new_filename = f"{message.from_user.username}_{emotion}_{destination_file_path}" if message.from_user.username else f"{emotion}_{destination_file_path}"
    # os.rename(destination_file_path,os.path.join('temp', new_filename))

    # Reply to the user with the predicted emotion and score
    bot.reply_to(message, f"The emotion is: {emotion}")
    bot.reply_to(message, f"The score is: {score}")

    # Respond with an emoji based on the predicted emotion
    emoji_dict = {
        'neutral': '🙂',
        'happy': '😄',
        'sad': '🥺',
        'angry': '😡',
        'ps': '😲',
        'calm': '😌',
        'fear': '😱',
        'disgust': '🤢',
        'boredom': '🥱'
    }
    # default to neutral face if emotion is unknown
    emoji = emoji_dict.get(emotion, '😐')
    bot.reply_to(message, emoji)


# Handle voice and audio messages
@bot.message_handler(func=lambda message: message.voice.mime_type == 'audio/ogg',content_types=['voice'])
def handle_audio_message(message):
    bot.reply_to(message,message)
    file_info = bot.get_file(message.voice.file_id)

    # Download the voice file
    bot.reply_to(message, 'The voice is being processed. Please wait.')
    downloaded_file = bot.download_file(file_info.file_path)
    # bot.reply_to(message, '1')

    # Save the file to disk
    received_file_path      = os.path.join(os.path.dirname(__file__), 'temp', f"{message.voice.file_id}.ogg")
    destination_file_path   = os.path.join(os.path.dirname(__file__), 'temp', f"{int(time.time())}_{int(random.random() * 9999999999)}.wav")
    with open(received_file_path, 'wb') as new_file:
        new_file.write(downloaded_file)

    # Convert the file to WAV format
    from pydub import AudioSegment
    ogg_audio = AudioSegment.from_file(received_file_path, format="ogg")
    wav_audio = ogg_audio.set_sample_width(2)
    wav_audio.export(destination_file_path, format="wav").close()

    # process_voice_message(message,received_file_path)
    process_voice_message(message,destination_file_path)


@bot.message_handler(func=lambda message: message.document.mime_type == 'audio/x-wav',content_types=['document'])
def handle_wav_file(message):
    # Process the WAV file here
    bot.reply_to(message,message)
    file_info = bot.get_file(message.document.file_id)

    # Download the voice file
    bot.reply_to(message, 'The voice is being processed. Please wait.')
    downloaded_file = bot.download_file(file_info.file_path)

    received_file_path      = os.path.join(os.path.dirname(__file__), 'temp', f"{message.document.file_name}")
    # destination_file_path   = os.path.join(os.path.dirname(__file__), 'temp', f"{int(time.time())}_{int(random.random() * 9999999999)}.wav")
    with open(received_file_path, 'wb') as new_file:
        new_file.write(downloaded_file)

    process_voice_message(message,received_file_path)


# Handle document messages
@bot.message_handler(content_types=['document'])
def handle_document_message(message):
    bot.reply_to(message,message)
    bot.reply_to(message, 'Only audio files are supported at this time.')


# {'content_type': 'document', 'id': 834, 'message_id': 834, 'from_user': {'id': 5011846159, 'is_bot': False, 'first_name': 'Abdul Rahman', 'username': 'abdulator0', 'last_name': 'Souda', 'language_code': 'en', 'can_join_groups': None, 'can_read_all_group_messages': None, 'supports_inline_queries': None, 'is_premium': None, 'added_to_attachment_menu': None}, 'date': 1680125080, 'chat': {'id': 5011846159, 'type': 'private', 'title': None, 'username': 'abdulator0', 'first_name': 'Abdul Rahman', 'last_name': 'Souda', 'is_forum': None, 'photo': None, 'bio': None, 'join_to_send_messages': None, 'join_by_request': None, 'has_private_forwards': None, 'has_restricted_voice_and_video_messages': None, 'description': None, 'invite_link': None, 'pinned_message': None, 'permissions': None, 'slow_mode_delay': None, 'message_auto_delete_time': None, 'has_protected_content': None, 'sticker_set_name': None, 'can_set_sticker_set': None, 'linked_chat_id': None, 'location': None, 'active_usernames': None, 'emoji_status_custom_emoji_id': None, 'has_hidden_members': None, 'has_aggressive_anti_spam_enabled': None}, 'sender_chat': None, 'forward_from': None, 'forward_from_chat': None, 'forward_from_message_id': None, 'forward_signature': None, 'forward_sender_name': None, 'forward_date': None, 'is_automatic_forward': None, 'reply_to_message': None, 'via_bot': None, 'edit_date': None, 'has_protected_content': None, 'media_group_id': None, 'author_signature': None, 'text': None, 'entities': None, 'caption_entities': None, 'audio': None, 'document': {'file_id': 'BQACAgQAAxkBAAIDQmQkrJgJr4AAAdDMpueN4eXNQjYzvAACJxEAAs1bKVFZXk1u-fMgAS8E', 'file_unique_id': 'AgADJxEAAs1bKVE', 'thumb': None, 'file_name': '03-02-02-02-02-02-08_calm.wav', 'mime_type': 'audio/x-wav', 'file_size': 490214}, 'photo': None, 'sticker': None, 'video': None, 'video_note': None, 'voice': None, 'caption': None, 'contact': None, 'location': None, 'venue': None, 'animation': None, 'dice': None, 'new_chat_member': None, 'new_chat_members': None, 'left_chat_member': None, 'new_chat_title': None, 'new_chat_photo': None, 'delete_chat_photo': None, 'group_chat_created': None, 'supergroup_chat_created': None, 'channel_chat_created': None, 'migrate_to_chat_id': None, 'migrate_from_chat_id': None, 'pinned_message': None, 'invoice': None, 'successful_payment': None, 'connected_website': None, 'reply_markup': None, 'message_thread_id': None, 'is_topic_message': None, 'forum_topic_created': None, 'forum_topic_closed': None, 'forum_topic_reopened': None, 'has_media_spoiler': None, 'forum_topic_edited': None, 'general_forum_topic_hidden': None, 'general_forum_topic_unhidden': None, 'write_access_allowed': None, 'json': {'message_id': 834, 'from': {'id': 5011846159, 'is_bot': False, 'first_name': 'Abdul Rahman', 'last_name': 'Souda', 'username': 'abdulator0', 'language_code': 'en'}, 'chat': {'id': 5011846159, 'first_name': 'Abdul Rahman', 'last_name': 'Souda', 'username': 'abdulator0', 'type': 'private'}, 'date': 1680125080, 'document': {'file_name': '03-02-02-02-02-02-08_calm.wav', 'mime_type': 'audio/x-wav', 'file_id': 'BQACAgQAAxkBAAIDQmQkrJgJr4AAAdDMpueN4eXNQjYzvAACJxEAAs1bKVFZXk1u-fMgAS8E', 'file_unique_id': 'AgADJxEAAs1bKVE', 'file_size': 490214}}}
# {'content_type': 'voice', 'id': 844, 'message_id': 844, 'from_user': {'id': 5011846159, 'is_bot': False, 'first_name': 'Abdul Rahman', 'username': 'abdulator0', 'last_name': 'Souda', 'language_code': 'en', 'can_join_groups': None, 'can_read_all_group_messages': None, 'supports_inline_queries': None, 'is_premium': None, 'added_to_attachment_menu': None}, 'date': 1680125483, 'chat': {'id': 5011846159, 'type': 'private', 'title': None, 'username': 'abdulator0', 'first_name': 'Abdul Rahman', 'last_name': 'Souda', 'is_forum': None, 'photo': None, 'bio': None, 'join_to_send_messages': None, 'join_by_request': None, 'has_private_forwards': None, 'has_restricted_voice_and_video_messages': None, 'description': None, 'invite_link': None, 'pinned_message': None, 'permissions': None, 'slow_mode_delay': None, 'message_auto_delete_time': None, 'has_protected_content': None, 'sticker_set_name': None, 'can_set_sticker_set': None, 'linked_chat_id': None, 'location': None, 'active_usernames': None, 'emoji_status_custom_emoji_id': None, 'has_hidden_members': None, 'has_aggressive_anti_spam_enabled': None}, 'sender_chat': None, 'forward_from': None, 'forward_from_chat': None, 'forward_from_message_id': None, 'forward_signature': None, 'forward_sender_name': None, 'forward_date': None, 'is_automatic_forward': None, 'reply_to_message': None, 'via_bot': None, 'edit_date': None, 'has_protected_content': None, 'media_group_id': None, 'author_signature': None, 'text': None, 'entities': None, 'caption_entities': None, 'audio': None, 'document': None, 'photo': None, 'sticker': None, 'video': None, 'video_note': None, 'voice': {'file_id': 'AwACAgQAAxkBAAIDTGQkrislX9t7yjQAAR4T2bMnb_gTqgACKxEAAs1bKVHOyc8I_l52iC8E', 'file_unique_id': 'AgADKxEAAs1bKVE', 'duration': 1, 'performer': None, 'title': None, 'file_name': None, 'mime_type': 'audio/ogg', 'file_size': 503, 'thumb': None}, 'caption': None, 'contact': None, 'location': None, 'venue': None, 'animation': None, 'dice': None, 'new_chat_member': None, 'new_chat_members': None, 'left_chat_member': None, 'new_chat_title': None, 'new_chat_photo': None, 'delete_chat_photo': None, 'group_chat_created': None, 'supergroup_chat_created': None, 'channel_chat_created': None, 'migrate_to_chat_id': None, 'migrate_from_chat_id': None, 'pinned_message': None, 'invoice': None, 'successful_payment': None, 'connected_website': None, 'reply_markup': None, 'message_thread_id': None, 'is_topic_message': None, 'forum_topic_created': None, 'forum_topic_closed': None, 'forum_topic_reopened': None, 'has_media_spoiler': None, 'forum_topic_edited': None, 'general_forum_topic_hidden': None, 'general_forum_topic_unhidden': None, 'write_access_allowed': None, 'json': {'message_id': 844, 'from': {'id': 5011846159, 'is_bot': False, 'first_name': 'Abdul Rahman', 'last_name': 'Souda', 'username': 'abdulator0', 'language_code': 'en'}, 'chat': {'id': 5011846159, 'first_name': 'Abdul Rahman', 'last_name': 'Souda', 'username': 'abdulator0', 'type': 'private'}, 'date': 1680125483, 'voice': {'duration': 1, 'mime_type': 'audio/ogg', 'file_id': 'AwACAgQAAxkBAAIDTGQkrislX9t7yjQAAR4T2bMnb_gTqgACKxEAAs1bKVHOyc8I_l52iC8E', 'file_unique_id': 'AgADKxEAAs1bKVE', 'file_size': 503}}}
# Start polling for messages
bot.polling()