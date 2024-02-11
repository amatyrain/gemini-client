import google.generativeai as genai
import PIL.Image


class GeminiClient:
    def __init__(self, api_key):
        genai.configure(api_key=api_key)

    def generate_content(self, text: str, image_path: str | None = None):
        # return output
        if image_path is None:
            model = genai.GenerativeModel("gemini-pro")
            content = [text]
        else:
            model = genai.GenerativeModel("gemini-pro-vision")
            image = PIL.Image.open(image_path)
            content = [text, image]

        response = model.generate_content(content, stream=True)
        output = ""
        for chunk in response:
            output += chunk.text

        return output
