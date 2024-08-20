from fastapi import FastAPI, HTTPException, UploadFile, File
import requests
import logging
import json
from openai import OpenAI

# FastAPI 인스턴스 생성
app = FastAPI()

class ClovaSpeechClient:
    invoke_url = 'https://clovaspeech-gw.ncloud.com/external/v1/8848/0abffb803a07105bc06ce85abade2bb08733f5d5979e047d0553ad0005961aa2'
    secret = 'de4d7ac3790f4b7c8a947f61637637bb'

    def req_upload(self, file, completion, callback=None, userdata=None, forbiddens=None, boostings=None,
                   wordAlignment=True, fullText=True, diarization=None, sed=None):
        request_body = {
            'language': 'ko-KR',
            'completion': completion,
            'callback': callback,
            'userdata': userdata,
            'wordAlignment': wordAlignment,
            'fullText': fullText,
            'forbiddens': forbiddens,
            'boostings': boostings,
            'diarization': diarization,
            'sed': sed,
        }
        headers = {
            'Accept': 'application/json;UTF-8',
            'X-CLOVASPEECH-API-KEY': self.secret
        }
        files = {
            'media': file,
            'params': (None, json.dumps(request_body, ensure_ascii=False).encode('UTF-8'), 'application/json')
        }
        response = requests.post(headers=headers, url=self.invoke_url + '/recognizer/upload', files=files)
        return response

def perform_stt_on_audio(file):
    try:
        logging.info(f"Performing STT on audio file.")
        client = ClovaSpeechClient()
        response = client.req_upload(file=file, completion='sync')
        logging.info(f"HTTP Status Code: {response.status_code}")
        logging.info(f"Response Text: {response.text}")

        response.raise_for_status()
        result = response.json()

        logging.info(f"Full STT Response: {json.dumps(result, indent=2, ensure_ascii=False)}")
        segments = result.get('segments', [])
        logging.info(f"STT result: {segments}")
        if not segments:
            logging.error("No segments returned from STT")
        return segments
    except requests.RequestException as e:
        logging.error(f"Error during STT request: {str(e)}")
        raise HTTPException(status_code=500, detail=f"STT request failed: {str(e)}")
    except ValueError as e:
        logging.error(f"Error parsing STT response: {str(e)}")
        raise HTTPException(status_code=500, detail=f"STT response parsing failed: {str(e)}")

def summarizer(text):
    completion = client.chat.completions.create(
        model = "gpt-4o",
        messages=[
        {"role": "system", "content": "내가 주는 텍스트는 강의 텍스트야, 요약본을 보고 공부할 수 있게 (텍스트를 요약해줘, 요약하자면, 설명하면, 본 내용은) 등의 표현 없이 요약 본론부터 바로 말해줘, 모든 문장은 ~~입니다 형식으로 끝내줘"},
        {"role": "user", "content": text}
        ],
        max_tokens=4096,  
        temperature=0.3    
    )
    summary = completion.choices[0].message.content
    return summary

def extract_text_from_segments(segments):
    return " ".join([segment['text'] for segment in segments])

@app.post("/stt-summary/")
async def stt_summary(file: UploadFile = File(...)):
    # 오디오 파일을 STT 수행
    segments = perform_stt_on_audio(file.file)
    
    if not segments:
        raise HTTPException(status_code=500, detail="STT processing failed")
    
    # STT 결과 텍스트 추출
    full_text = extract_text_from_segments(segments)
    
    # OpenAI API 키 설정
    global client
    client = OpenAI(
        organization = "org-g3vt6yWLrE8ylL2eRGf9ykkU",
        api_key = "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"
    )

    # 텍스트 요약
    summary = summarizer(full_text)

    return {"summary": summary}

