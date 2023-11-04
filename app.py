import gradio as gr

def greet(name, is_morning, temp):
salutatuin = "Gm" if is_morning else "Ge"
greeting = f"{salutatuin} {name}. It is {temp} degress today"
celsius = (temp - 32) * 5 /9
return greeting, round(celsius, 2)


demo = gr.Interface(fn=greet, inputs=["text", "checkbox", gr.Slider(0,50)], outputs=["text", "number"])

demo.launch()