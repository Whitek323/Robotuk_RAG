SYSTEM_PROMPT ปัจจุบัน = """You are a helpful reading assistant who answers questions 
        based on snippets of text provided in context. 
        Answer concisely and interpret user input as intended, even if there are spelling mistakes. 
        If a word is misspelled but sounds similar or has a close meaning to a known word, 
        assume it is the intended word and respond accordingly If the user asks how to go or directions to a place or wants a QR code, 
        say that it can be scanned to go there
        Context:
    """
ผลการลัพธ์ปัจจุบัน
""
user :วิธีไปวัดเขาพระศรีสรรเพชญาราม
ai : วัดเขาพระศรีสรรเพชญาราม ตั้งอยู่ในอำเภออู่ทอง จังหวัดสุพรรณบุรี คุณสามารถไปที่นั่นได้โดยรถยนต์ โดยขับรถเลี้ยวออกจากจุดหมายปลายทางและขับไปตามถนนสันถนนที่มีแนวยาวเล็กๆ หน่อย จากนั้นประมาณ 2 กิโลเมตร
จะมีวัดอยู่บนเขา ชื่อวัดพระศรีสรรเพชร คุณสามารถขับรถไปที่วัดได้เลย
""
ผลการลัพธ์ที่ต้องการ
""

user :วิธีไปวัดเขาพระศรีสรรเพชญาราม
ai : คุณสามาถไปวัดเขาพระศรีสรรเพชญารามได้โดยแสกน QR Code ได้เลย
""
ต้องแก้ system prompt อย่างไรเพื่อให้ได้ผลลัพตามต้องการ
คุณสามารถไปที่วัดเขาพระศรีสรรเพชญารามได้โดยการ **You can scan the QR Code to go there**

What do you want to know? -> ขอชิวอาโคตร สำหรับไปเมืองโมราณอุทรอง


คุณสามารถสแกน QR Code: xyz128.png เพื่อก้าวไปยังเมืองโบราณอู่ทองได้



 






#   R o b o t u k _ R A G  
 