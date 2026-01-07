#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Fix emojis by finding and replacing the entire line

file_path = r'C:\Users\rucha\Downloads\Project\STEP_8_APPLICATION\templates\index.html'

# Read file as bytes to see what's actually there
with open(file_path, 'rb') as f:
    content_bytes = f.read()

# Try to decode and see what we have
try:
    content = content_bytes.decode('utf-8', errors='replace')
    print(f"Original content length: {len(content)} characters")
    
    # Find the line
    if 'Yellow = AI Recommended Sites' in content:
        print("✓ Found the line to fix")
        
        # Replace with proper emojis - use unicode escape for reliability
        # Yellow circle: \U0001F7E1
        # Purple circle: \U0001F7E3
        
        old_line = "� Yellow = AI Recommended Sites | ${result.existing_borewells && result.existing_borewells.length > 0 ? '� Purple = Success | 🔴 Red = Failure' : 'No existing borewells in this area'}"
        new_line = "🟡 Yellow = AI Recommended Sites | ${result.existing_borewells && result.existing_borewells.length > 0 ? '🟣 Purple = Success | 🔴 Red = Failure' : 'No existing borewells in this area'}"
        
        content = content.replace(old_line, new_line)
        
        # Also try replacing any � characters before "Yellow" and "Purple"
        import re
        content = re.sub(r'[^\x00-\x7F]+\s+Yellow =', '🟡 Yellow =', content)
        content = re.sub(r"'[^\x00-\x7F]+\s+Purple =", "'🟣 Purple =", content)
        
        # Write back as UTF-8
        with open(file_path, 'w', encoding='utf-8', newline='') as f:
            f.write(content)
        
        print('✅ Emojis fixed!')
    else:
        print("❌ Line not found")
        
except Exception as e:
    print(f"❌ Error: {e}")
