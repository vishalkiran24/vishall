import os
import pandas as pd
from docx import Document

folder_path = 'C:\Users\DELL\Desktop\ai resume\Resume'  # Change to your folder name
data = []

for filename in os.listdir(folder_path):
    if filename.endswith('.docx'):
        doc_path = os.path.join(folder_path, filename)
        doc = Document(doc_path)
        full_text = []
        for para in doc.paragraphs:
            full_text.append(para.text)
        resume_text = '\n'.join(full_text)
        # If you know the category, set it here; else leave blank or auto-detect later
        data.append({'resume_text': resume_text, 'category': ''})

df = pd.DataFrame(data)
df.to_csv('Resume.csv', index=False)