
from captcha.image import ImageCaptcha
import random

list = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
        'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

width, height = 170, 80
generator = ImageCaptcha(width=width, height=height)
for i in range(10000):

    random_str = ''.join(random.choice(list) for j in range(4))
    img = generator.generate_image(random_str)
    generator.create_noise_dots(img, '#000000', 4, 40)
    generator.create_noise_curve(img, '#000000')


    file_name = 'F:\Captcha_datasets\Captcha_datasets' +'_'+ random_str +'_'+str(i)+'.jpg'
    img.save(file_name)
