## Declare file path
STYLE_BAMBOO = 'static/img/bamboo.jpg'
STYLE_MISTY_MOOD = 'static/img/misty_mood.jpg'
STYLE_WAVE = 'static/img/wave.jpg'
STYLE_CHINESE = 'static/img/chinese.jpg'
STYLE_CHINGMIN = 'static/img/chingmin.jpg'
STYLE_DES_GLANEUSES = 'static/img/des_glaneuses.jpg'
STYLE_MIRROR = 'static/img/mirror.jpg'
STYLE_MOUNTAIN_WATER = 'static/img/mountain_water.jpg'
STYLE_SEATED_NUDE = 'static/img/seated_nude.jpg'
STYLE_STARRY_NIGHT = 'static/img/starry_night.jpg'
STYLE_UDNIE = 'static/img/udnie.jpg'
STYLE_WAVE_CROP = 'static/img/wave_crop.jpg'
STYLE_LA_MUSE = 'static/img/la_muse.jpg'
STYLE_ZHANGDAQIAN = 'static/img/zhangdaqian.jpg'
STYLE_MONALISA = 'static/img/monalisa.jpg'

style_dicts = {'bamboo': STYLE_BAMBOO, 'misty_mood': STYLE_MISTY_MOOD,
               'wave': STYLE_WAVE, 'chinese':STYLE_CHINESE,
               'chingmin': STYLE_CHINGMIN, 'des_glaneuses': STYLE_DES_GLANEUSES,
               'mirror': STYLE_MIRROR, 'mountain_water': STYLE_MOUNTAIN_WATER,
               'seated_nude': STYLE_SEATED_NUDE, 'starry_night': STYLE_STARRY_NIGHT,
               'udnie': STYLE_UDNIE, 'wave_crop': STYLE_WAVE_CROP,
               'zhangdaqian': STYLE_ZHANGDAQIAN, 'la_muse': STYLE_LA_MUSE, 'monalisa': STYLE_MONALISA}

def get_style_path(style):
    if style in style_dicts:
        return style_dicts[style]
    else:
        return False
