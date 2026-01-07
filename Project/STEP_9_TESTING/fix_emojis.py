#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Fix corrupted emojis in index.html

file_path = r'C:\Users\rucha\Downloads\Project\STEP_8_APPLICATION\templates\index.html'

# Read file
with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
    content = f.read()

# Replace corrupted emojis with proper ones
# The � character (U+FFFD) is the replacement character for invalid UTF-8
content = content.replace('� Yellow = AI Recommended Sites', '🟡 Yellow = AI Recommended Sites')
content = content.replace("'� Purple = Success", "'🟣 Purple = Success")

# Write back
with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)

print('✅ Fixed emojis in index.html!')
print('🟡 Yellow emoji added')
print('🟣 Purple emoji added')
