import os, random
from PIL import Image

folder = "./out"
city = random.choice(os.listdir(folder))
img = random.choice(os.listdir(f"{folder}/{city}"))


img = Image.open(f'{folder}/{city}/{img}') 
img.show()

input("")
print(f"The answer was {city}")