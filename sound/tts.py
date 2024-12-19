import os,re
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import torch
from transformers import VitsTokenizer, VitsModel, set_seed
import sounddevice as sd

class TTS:
    def __init__(self, model_name,text):
        self.model_name=model_name
        self.text=text
        cl_text = self.clean_text(self.text)

        tokenizer = VitsTokenizer.from_pretrained(self.model_name)
        model = VitsModel.from_pretrained(self.model_name)

        inputs = tokenizer(text=cl_text, return_tensors="pt")

        # ตั้งค่า seed เพื่อให้ผลลัพธ์ reproducible
        set_seed(555)

        # ประมวลผลด้วยโมเดล
        with torch.no_grad():
            outputs = model(**inputs)

        # ดึงข้อมูล waveform
        waveform = outputs.waveform[0].cpu().numpy()
        sampling_rate = model.config.sampling_rate

        # เล่นเสียง
        print("กำลังเล่นเสียง...")
        sd.play(waveform, samplerate=sampling_rate)
        sd.wait()  # รอให้เสียงเล่นจนจบ
        print("เสียงเล่นเสร็จสิ้น")

    def clean_text(self,text):
    # ใช้ regex เพื่อลบข้อความตั้งแต่ตัว 'q' ถึง '.png'
        cleaned_text = re.sub(r'q.*?\.png', '', text, flags=re.IGNORECASE)
        return cleaned_text
