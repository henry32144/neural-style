## Declare file path
STYLE_AURORA= 'static/img/aurora.jpg'
STYLE_CHINESE = 'static/img/chinese.jpg'
STYLE_CHINGMIN = 'static/img/chingmin.jpg'
STYLE_COLORFUL = 'static/img/colorful.jpg'
STYLE_CRYSTAL = 'static/img/crystal.jpg'
STYLE_DES_GLANEUSES = 'static/img/des_glaneuses.jpg'
STYLE_FIRE = 'static/img/fire.jpg'
STYLE_JINGDO = 'static/img/jingdo.jpg'
STYLE_LA_MUSE = 'static/img/la_muse.jpg'
STYLE_MOSAIC = 'static/img/mosaic.jpg'
STYLE_MONALISA = 'static/img/monalisa.jpg'
STYLE_MOUNTAIN_WATER = 'static/img/mountain_water.jpg'
STYLE_PICASSOSELF = 'static/img/picassoself.jpg'
STYLE_SKY = 'static/img/sky.jpg'
STYLE_STARRY_NIGHT = 'static/img/starry_night.jpg'
STYLE_TIGER = 'static/img/tiger.jpg'
STYLE_UDNIE = 'static/img/udnie.jpg'
STYLE_WATER = 'static/img/water.jpg'
STYLE_WAVE_CROP = 'static/img/wave_crop.jpg'
STYLE_YBFIRE = "static/img/ybfire.jpg"
STYLE_ZHANGDAQIAN = 'static/img/zhangdachien.jpg'


style_dicts = {'aurora':STYLE_AURORA, 'chinese':STYLE_CHINESE,
                'chingmin': STYLE_CHINGMIN, 'colorful':STYLE_COLORFUL, 'crystal':STYLE_CRYSTAL,
                'des_glaneuses': STYLE_DES_GLANEUSES, 'fire': STYLE_FIRE, 'jingdo': STYLE_JINGDO,
               'la_muse': STYLE_LA_MUSE, 'mosaic': STYLE_MOSAIC, 'monalisa': STYLE_MONALISA, 'mountain_water': STYLE_MOUNTAIN_WATER,
               'picassoself': STYLE_PICASSOSELF, 'sky': STYLE_SKY, 'starry_night': STYLE_STARRY_NIGHT,
               'tiger': STYLE_TIGER, 'udnie': STYLE_UDNIE, 'water': STYLE_WATER, 'wave_crop': STYLE_WAVE_CROP, 'ybfire': STYLE_YBFIRE,
               'zhangdachien': STYLE_ZHANGDAQIAN}

def get_style_path(style):
    if style in style_dicts:
        return style_dicts[style]
    else:
        return False
