# -*- coding: utf-8 -*-
"""
Texelith — interface strings, Russian and English.
Строки интерфейса, русский и английский.

Каждая запись — пара (ru, en). Обращение через t('key'); если ключа нет,
возвращается сам ключ, поэтому пропуск виден сразу, а не превращается
в пустую подпись.
"""

import json
import os

BUILD = 25

LANGS = ('ru', 'en')
LANG_NAMES = {'ru': 'Русский', 'en': 'English'}


def _settings_paths():
    """Settings folder and language file, moving the old ones over.

    Up to build 22 the program was called Tessera and kept its settings,
    downloaded models and cache in ``~/.tessera`` and ``~/.tessera.json``.
    On the first start under the new name they are renamed to
    ``~/.texelith`` / ``~/.texelith.json``, so nothing is downloaded again.
    If the rename fails (a file is open elsewhere), the old folder keeps
    being used. Returns (folder, language file).
    """
    home = os.path.expanduser('~')
    new_dir, old_dir = os.path.join(home, '.texelith'), os.path.join(home, '.tessera')
    new_conf, old_conf = os.path.join(home, '.texelith.json'), os.path.join(home, '.tessera.json')
    for old, new in ((old_dir, new_dir), (old_conf, new_conf)):
        if os.path.exists(old) and not os.path.exists(new):
            try:
                os.rename(old, new)
            except OSError:
                pass
    d = new_dir if os.path.exists(new_dir) or not os.path.exists(old_dir) else old_dir
    c = new_conf if os.path.exists(new_conf) or not os.path.exists(old_conf) else old_conf
    return d, c


# папка настроек: модели, кэш, выбор onnxruntime, SuperMat
APP_DIR, CONF = _settings_paths()


class Lang:
    """Current interface language code ('ru' or 'en')."""
    cur = 'ru'


def load():
    """Read the saved language from the settings file; returns the code in use."""
    try:
        with open(CONF, 'r', encoding='utf-8') as f:
            v = json.load(f).get('lang')
        if v in LANGS:
            Lang.cur = v
    except Exception:
        pass
    return Lang.cur


def save(code):
    """Switch the language to code and remember it in the settings file."""
    Lang.cur = code if code in LANGS else 'ru'
    try:
        with open(CONF, 'w', encoding='utf-8') as f:
            json.dump({'lang': Lang.cur}, f)
    except Exception:
        pass


def t(key, *args):
    """UI string for key in the current language, %-formatted with args; the
    key itself if there is no such string."""
    v = S.get(key)
    if v is None:
        return key
    s = v[LANGS.index(Lang.cur)]
    return (s % args) if args else s


def name_of(code):
    """Human name of a language code, for the language menu."""
    return LANG_NAMES.get(code, code)


S = {
    # ------------------------------------------------------------ общее ----
    'app.title': ('Texelith — карты для игровых текстур',
                  'Texelith — texture map authoring'),
    'app.sub': ('Нормали, высота, затенение и материалы из готовой текстуры',
                'Normals, height, occlusion and materials from a finished texture'),
    'top.open': ('Открыть текстуру…', 'Open texture…'),
    'top.drop': ('перетащи файл в окно', 'drop a file onto the window'),
    'top.nofile': ('файл не выбран', 'no file loaded'),
    'top.fit': ('Вписать', 'Fit'),
    'top.flip': ('перевёрнутым', 'flip vertically'),
    'top.lang': ('Язык', 'Language'),
    'top.guide': ('Руководство', 'Guide'),
    'top.guide.info': ('Открывает руководство в браузере: что каждая карта значит, '
                       'в каком порядке крутить настройки и как всё это выглядит '
                       'в движке.',
                       'Opens the manual in your browser: what each map is for, the '
                       'order to work in, and how it all behaves in an engine.'),
    'tab.foliage': ('Листва: нормали', 'Foliage: normals'),
    'tab.leafdiff': ('Листва: свет и AO', 'Foliage: light and AO'),
    'tab.material': ('Материал', 'Material'),
    'tab.fix': ('Правка нормалей', 'Normal map repair'),
    'com.show': ('Показывать', 'View'),
    'com.split': ('две панели', 'split view'),
    'com.split.info': ('Правая панель для сравнения. Кисть работает только по левой.',
                       'Right pane for comparison. Painting works on the left pane only.'),
    'com.spin': ('вращать', 'rotate light'),
    'com.dxt': ('сжимать в DXT5', 'compress to DXT5'),
    'com.save.sel': ('Сохранить выбранное', 'Save selected maps'),
    'com.batch': ('Обработать всю папку…', 'Process a whole folder…'),
    'com.brush.size': ('Размер кисти, пикс.', 'Brush size, px'),
    'com.clear': ('Очистить', 'Clear'),
    'com.load': ('Загрузить', 'Load'),
    'com.save': ('Сохранить', 'Save'),
    'com.invert': ('инвертировать', 'invert'),
    'com.none': ('нет', 'none'),
    'com.computing': ('считаю…', 'working…'),
    'com.ready': ('готово', 'done'),
    'com.saved': ('сохранено: %s', 'saved: %s'),
    'com.saved.n': ('сохранено карт: %d', 'maps saved: %d'),
    'com.nothing': ('Нечего сохранять', 'Nothing to save'),
    'com.nothing.b': ('Отметь хотя бы одну карту.', 'Tick at least one map.'),
    'com.notopen': ('Не открылось', 'Could not open'),
    'com.where': ('Куда сложить карты', 'Where to put the maps'),
    'com.folder': ('Папка с текстурами', 'Folder with textures'),
    'com.zoomhint': ('%d%%   колесо — зум, двойной клик — вписать',
                     '%d%%   wheel to zoom, double click to fit'),
    'com.light': ('%d°  высота %d°', '%d°  elevation %d°'),
    'com.light.back': ('сзади', 'behind'),
    'ed.tab.leaves': ('Листья', 'Leaves'),
    'ed.tab.nets': ('Сети', 'Networks'),
    'ed.tab.mix': ('Смешивание', 'Mixing'),
    'ed.tab.normal': ('Нормаль', 'Normal'),
    'ed.tab.shape': ('Форма', 'Shape'),
    'ed.tab.orient': ('Наклон', 'Tilt'),
    'ed.tab.clump': ('Объём', 'Volume'),
    'ed.tab.ao': ('AO', 'AO'),
    'ed.tab.sss': ('Просвет', 'Translucency'),
    'ed.tab.other': ('Прочее', 'Other'),
    'np.foliage.hand': ('ветки вручную', 'branches by hand'),
    'np.out.alpha': ('альфа', 'alpha'),
    'np.set': ('набор', 'set'),
    'np.set.info': ('«Обычный» — в папку вывода. «PBR» — в папку рядом с ней, с «_PBR» в конце '
                    'имени (например, out → out_PBR): там лежат версии тех же файлов для шейдера '
                    'OpenMW-PBR, и обычные _spec и диффуз их не затирают. Какую папку подключить '
                    'в игре — решается в списке data= openmw.cfg.',
                    '"Main" — into the output folder. "PBR" — into the folder next to it, with '
                    '"_PBR" added to the name (out → out_PBR, for example): versions of the same '
                    'files for the OpenMW-PBR shader go there, so the usual _spec and diffuse are '
                    'not overwritten. Which folder the game uses is set in the data= list of '
                    'openmw.cfg.'),
    'nv.main': ('обычный', 'main'),
    'nv.pbr': ('PBR (папка _PBR)', 'PBR (_PBR folder)'),
    'nv.a': ('альфа-канал', 'alpha channel'),
    'np.out_pack.a_fill': ('A без входа — заполнить числом', 'A without input — fill with a number'),
    'np.out_pack.a_fill.info': ('Выключено: если на A ничего не подано, у файла нет своей альфы. '
                                'Включено: альфа заполняется числом ниже.',
                                'Off: with nothing connected to A the file has no alpha of its '
                                'own. On: the alpha is filled with the number below.'),
    'np.out_pack.a_val': ('A, если не подключено', 'A when not connected'),
    'nd.sss_map': ('SSS (подповерхностное рассеяние)', 'SSS (subsurface scattering)'),
    'nd.sss_map.info': ('Сила SSS по физике материала: насколько далеко свет расходится под '
                        'поверхностью. Выход «альфа для _spec» = 1 − SSS — его подавайте на A '
                        '«Выхода по каналам» для шейдера OpenMW-PBR.',
                        'SSS strength from the physics of the material: how far light spreads '
                        'under the surface. The «alpha for _spec» output = 1 − SSS — feed it to '
                        'A of «Output by channels» for the OpenMW-PBR shader.'),
    'np.sss_map.mask': ('где материал', 'where the material is'),
    'np.sss_map.sss': ('SSS', 'SSS'),
    'np.sss_map.alpha': ('альфа для _spec', 'alpha for _spec'),
    'np.sss_map.color': ('цвет рассеяния', 'scattering colour'),
    'np.sss_map.rgba': ('цвет и SSS (_sss)', 'colour and SSS (_sss)'),
    'np.sss_map.col_r': ('Цвет рассеяния: R', 'Scattering colour: R'),
    'np.sss_map.col_g': ('Цвет рассеяния: G', 'Scattering colour: G'),
    'np.sss_map.col_b': ('Цвет рассеяния: B', 'Scattering colour: B'),
    'np.sss_map.col_r.info': ('Какой свет уходит под поверхностью дальше всего: длина рассеяния '
                              'каждого канала, делённая на самую большую. Выбор материала ставит '
                              'её по измерениям. У кожи и мяса дальше всего уходит красный, у '
                              'снега — синий.',
                              'Which light goes furthest under the surface: the scattering length of '
                              'each channel divided by the largest. Picking a material sets it from '
                              'the measurements. In skin and meat red goes furthest, in snow blue.'),
    'np.sss_map.material': ('Материал', 'Material'),
    'np.sss_map.material.info': ('Измеренные материалы (Jensen и др., SIGGRAPH 2001; снег — France '
                                 'и др., 2011, цвет снега — по поглощению льда, Picard и др., 2016). '
                                 'Выбор ставит длину и цвет рассеяния. «Непрозрачное» — камень, '
                                 'металл, дерево: SSS 0.',
                                 'Measured materials (Jensen et al., SIGGRAPH 2001; snow — France '
                                 'et al., 2011, its colour from the absorption of ice, Picard et al., '
                                 '2016). Picking one sets the scattering length and colour. «Opaque» '
                                 '— stone, metal, wood: SSS 0.'),
    'np.sss_map.length': ('Длина рассеяния, мм', 'Scattering length, mm'),
    'np.sss_map.length.info': ('На каком расстоянии свет, вошедший в материал, ослабевает в e раз '
                               '(1/σtr, σtr = √(3·σa·(σa + σs\'))), среднее по R, G, B. Мрамор '
                               '5,7 мм, кожа 1,5…2,1 мм, снег ~100 мм.',
                               'The distance over which light that entered the material falls '
                               'e times (1/σtr, σtr = √(3·σa·(σa + σs\'))), averaged over R, G, '
                               'B. Marble 5.7 mm, skin 1.5…2.1 mm, snow ~100 mm.'),
    'np.sss_map.full_at': ('Полный SSS при длине, мм', 'Full SSS at length, mm'),
    'np.sss_map.full_at.info': ('SSS = длина / это число, не больше 1. У SSS в шейдере нет своей '
                                'единицы, поэтому этот масштаб — единственный выбор на глаз; '
                                'соотношения между материалами остаются физическими.',
                                'SSS = length / this number, at most 1. The shader\'s SSS has no '
                                'unit of its own, so this scale is the one choice by eye; the '
                                'ratios between materials stay physical.'),
    'nv.sss_marble': ('мрамор', 'marble'),
    'nv.sss_skin2': ('кожа, светлее (Skin2)', 'skin, lighter (Skin2)'),
    'nv.sss_skin1': ('кожа, темнее (Skin1)', 'skin, darker (Skin1)'),
    'nv.sss_meat': ('мясо (курица)', 'meat (chicken)'),
    'nv.sss_apple': ('яблоко (сочная мякоть)', 'apple (juicy flesh)'),
    'nv.sss_potato': ('картофель (плотная мякоть)', 'potato (firm flesh)'),
    'nv.sss_wholemilk': ('молоко цельное', 'whole milk'),
    'nv.sss_skimmilk': ('молоко обезжиренное', 'skim milk'),
    'nv.sss_cream': ('сливки', 'cream'),
    'nv.sss_ketchup': ('кетчуп (густая красная масса)', 'ketchup (thick red paste)'),
    'nv.sss_snow': ('снег', 'snow'),
    'nv.sss_opaque': ('непрозрачное', 'opaque'),
    'ni.sss': ('Средний SSS', 'Mean SSS'),
    'ni.length': ('Длина рассеяния, мм', 'Scattering length, mm'),
    'nd.out_pack': ('Выход по каналам', 'Output by channels'),
    'nd.out_pack.info': ('Файл, у которого каналы R, G, B и (по желанию) A собираются из разных '
                         'карт. У каждого канала выбирается, что из поданной карты в него идёт: '
                         'яркость или один её канал. Неподключённый R, G или B заполняется '
                         'числом; без A файл будет без своей альфы.',
                         'A file whose R, G, B and (optionally) A channels are taken from '
                         'different maps. For each channel you pick what of the connected map '
                         'goes into it: its brightness or one of its channels. An unconnected R, '
                         'G or B is filled with a number; without A the file has no alpha of its '
                         'own.'),
    'np.out_pack.r': ('R', 'R'), 'np.out_pack.g': ('G', 'G'),
    'np.out_pack.b': ('B', 'B'), 'np.out_pack.a': ('A', 'A'),
    'np.out_pack.r_ch': ('R: что брать', 'R: take'),
    'np.out_pack.g_ch': ('G: что брать', 'G: take'),
    'np.out_pack.b_ch': ('B: что брать', 'B: take'),
    'np.out_pack.a_ch': ('A: что брать', 'A: take'),
    'np.out_pack.r_val': ('R, если не подключено', 'R when not connected'),
    'np.out_pack.g_val': ('G, если не подключено', 'G when not connected'),
    'np.out_pack.b_val': ('B, если не подключено', 'B when not connected'),
    'np.out_pack.r_ch.info': ('Из серой карты канал берётся как есть. Из цветной — её яркость '
                              'или один её канал (у карты без альфы «альфа-канал» = 1). Нормаль '
                              'сперва переводится в 0…1, как она хранится в файле.',
                              'From a grey map the channel is taken as it is. From a colour one — '
                              'its brightness or one of its channels (alpha channel = 1 if it has '
                              'none). A normal is first brought to 0…1, as a file stores it.'),
    'np.out_pack.r_val.info': ('Чем заполнить канал, если на его вход ничего не подано (0…1). '
                               'Для _spec шейдера OpenMW-PBR пустой B (AO) ставьте 1 — иначе AO '
                               'сделает всё чёрным.',
                               'What fills the channel when nothing is connected to its input '
                               '(0…1). For the OpenMW-PBR shader\'s _spec: an empty B (AO) '
                               'should be 1, else the AO makes everything black.'),
    'out_pack.size': ('Канал %s: карта %d×%d, а первая подключённая — %d×%d. Размеры должны '
                      'совпадать.',
                      'Channel %s: the map is %d×%d, the first connected one %d×%d. The sizes '
                      'must match.'),
    'ni.R': ('Среднее в R', 'Mean of R'), 'ni.G': ('Среднее в G', 'Mean of G'),
    'ni.B': ('Среднее в B', 'Mean of B'), 'ni.A': ('Среднее в A', 'Mean of A'),
    'out.alpha_replaced': ('Альфа текстуры заменена той, что подана на вход «альфа». В файл '
                           'запишется поданная.',
                           'The texture\'s alpha is replaced by the one fed into the «alpha» '
                           'input. The file gets the fed one.'),
    'out.alpha_ignored': ('Вход «альфа» действует только на выходе «diffuse»; здесь он не '
                          'используется (у этой карты альфа значит другое).',
                          'The «alpha» input works on a «diffuse» output only; here it is not '
                          'used (this map\'s alpha means something else).'),
    'out.alpha_size': ('Альфа на входе другого размера, чем картинка, — не используется.',
                       'The alpha input differs in size from the picture — not used.'),
    'nd.nh_fuse': ('Нормаль и высота: две сети', 'Normal and height: two networks'),
    'nd.nh_fuse.info': ('Готовая цепочка одним узлом: картинка → нормаль DeepBump → высота → '
                        '«Цветовой контраст»; картинка → глубина нейросетью → «Цветовой '
                        'контраст»; обе высоты смешиваются, из результата делается нормаль. На '
                        'выходе высота и нормаль. Любую ветку можно выключить — тогда работает '
                        'одна. Вместо сети можно подать готовую нормаль или высоту на свои '
                        'входы, тогда эта сеть не считается. Радиус 0 выключает свой «Цветовой '
                        'контраст».',
                        'The usual chain as one node: picture → DeepBump normal → height → High '
                        'Pass; picture → network depth → High Pass; the two heights are mixed and '
                        'a normal is made from the result. Outputs: height and normal. Either '
                        'branch can be switched off — then one works. A ready normal or height '
                        'may be fed to its input instead of a network; that network is then not '
                        'run. A radius of 0 switches its High Pass off.'),
    'nh_fuse.none': ('Обе ветки выключены или им не на чем работать: подключите картинку '
                     '(или готовую нормаль / высоту) и включите хотя бы одну ветку.',
                     'Both branches are off or have nothing to work on: connect the picture '
                     '(or a ready normal / height) and switch at least one branch on.'),
    'nh_fuse.both': ('нормаль и глубина', 'normal and depth'),
    'nh_fuse.n': ('только нормаль', 'normal only'),
    'nh_fuse.h': ('только глубина', 'depth only'),
    'np.nh_fuse.rgb': ('картинка', 'picture'),
    'np.nh_fuse.n': ('нормаль', 'normal'),
    'np.nh_fuse.h': ('высота', 'height'),
    'np.nh_fuse.use_n': ('Ветка нормали (DeepBump)', 'Normal branch (DeepBump)'),
    'np.nh_fuse.use_h': ('Ветка глубины (сеть глубины)', 'Depth branch (depth network)'),
    'np.nh_fuse.n_directx': ('Поданная нормаль DirectX', 'Fed normal is DirectX'),
    'np.nh_fuse.hp_n': ('Цветовой контраст по нормали, пикс.', 'High Pass on the normal, px'),
    'np.nh_fuse.hp_h': ('Цветовой контраст по глубине, пикс.', 'High Pass on the depth, px'),
    'np.nh_fuse.wa': ('Вес нормали', 'Weight of the normal'),
    'np.nh_fuse.wb': ('Вес глубины', 'Weight of the depth'),
    'np.nh_fuse.divisor': ('Делитель (без растяжки)', 'Divisor (without stretch)'),
    'np.nh_fuse.directx': ('Выходная нормаль DirectX', 'Output normal is DirectX'),
    'ni.inputs': ('Что подано', 'Fed'),
    'nd.light_back': ('Снятый свет', 'Removed light'),
    'nd.light_back.info': ('Восстанавливает свет, который был снят с картинки: оригинал делится '
                           'на альбедо в линейном свете (например, исходник и альбедо SuperMat). '
                           '«свет» — по яркости, «цвет света» — по каналам. 1 — самое освещённое, '
                           'как у карты AO: годится для «Убрать затенение» и как маска.',
                           'Restores the light that was taken off the picture: the original is '
                           'divided by the albedo in linear light (e.g. the source and SuperMat\'s '
                           'albedo). «light» — by brightness, «light colour» — per channel. 1 is the '
                           'most lit, as in an AO map: fit for «Remove shading» and as a mask.'),
    'np.light_back.rgb': ('оригинал', 'original'),
    'np.light_back.albedo': ('альбедо', 'albedo'),
    'np.light_back.light': ('свет', 'light'),
    'np.light_back.light_rgb': ('цвет света', 'light colour'),
    'np.light_back.floor': ('Порог тёмного альбедо', 'Dark albedo floor'),
    'np.light_back.floor.info': ('Меньше какой яркости альбедо (в линейном свете) не делить: '
                                 'иначе чёрные пятна превращаются во вспышки.',
                                 'Below which albedo brightness (linear) not to divide: '
                                 'otherwise black spots turn into flares.'),
    'np.light_back.smooth': ('Сглаживание, пикс.', 'Smoothing, px'),
    'np.light_back.smooth.info': ('Делятся размытые картинки, а не отдельные пиксели: шум альбедо '
                                  'не превращается в иглы. 0 — попиксельно.',
                                  'Blurred pictures are divided, not single pixels: albedo noise '
                                  'does not turn into spikes. 0 — per pixel.'),
    'np.light_back.norm': ('Привести к 0…1', 'Bring to 0…1'),
    'nv.lit1': ('самое светлое = 1', 'brightest = 1'),
    'nv.avg05': ('среднее = 0.5', 'average = 0.5'),
    'nv.raw': ('как есть', 'as is'),
    'ni.ratio': ('Отношение оригинал/альбедо (медиана)', 'Original/albedo ratio (median)'),
    'ni.ratio.info': ('Насколько сеть сделала альбедо светлее (меньше 1) или темнее (больше 1) '
                      'оригинала в целом.',
                      'How much lighter (below 1) or darker (above 1) than the original the '
                      'network made the albedo overall.'),
    'np.foliage.depth': ('внешняя глубина', 'outside depth'),
    'np.foliage.depth_w': ('Доверие внешней глубине', 'Trust in the outside depth'),
    'np.foliage.depth_w.info': ('Действует, когда ко входу «внешняя глубина» что-то подключено '
                                '(светлое = ближе). Каждый лист берёт глубину из неё — среднее по '
                                'листу — вместо случайной. 0 — порядок прежний, случайный; 1 — '
                                'целиком из внешней глубины. Середина обычно и нужна: соседние '
                                'листья внешняя глубина часто видит одинаковыми, и без разброса '
                                'они снова лягут в одну плоскость. Эта глубина идёт в высоту, AO и '
                                'просвет.',
                                'Works when something is connected to the «outside depth» input '
                                '(light = closer). Each leaf takes its depth from it — the mean '
                                'over the leaf — instead of a random one. 0 — the order stays '
                                'random; 1 — entirely from the outside depth. The middle is usually '
                                'right: neighbouring leaves often look equal to the outside depth, '
                                'and without the scatter they fall into one plane again. This depth '
                                'feeds the height, the AO and the translucency.'),
    'np.foliage.h_mode': ('Высота', 'Height'),
    'np.foliage.h_mode.info': ('Как считать выход «высота». Глубина кроны — каждый лист на своей '
                               'случайной глубине (та же, что для AO и просвета; разброс — '
                               '«Разброс листьев по глубине» на вкладке AO) плюс его выпуклость; '
                               'между листьями ступеньки. Из нормали — то, что в нормали является '
                               'поверхностью: объём пучков и карточки, плавно, без ступенек; '
                               'наклоны отдельных листьев в неё не попадают. По форме куска — как '
                               'было: каждый лист — гребень от края к середине.',
                               'How the «height» output is made. Crown depth — every leaf at its '
                               'own random depth (the same as for the AO and translucency; spread '
                               'is «Leaf depth spread» on the AO tab) plus its bulge; steps between '
                               'leaves. From the normal — the part of the normal that is a surface: '
                               'the volume of clumps and of the card, smooth, no steps; the tilts '
                               'of single leaves do not get into it. By piece shape — as before: '
                               'every leaf a ridge from its edge to the middle.'),
    'nv.depth': ('глубина кроны', 'crown depth'),
    'nv.normal': ('из нормали', 'from the normal'),
    'nv.shape': ('по форме куска (как было)', 'by piece shape (as before)'),
    'ed.spin.z': ('вокруг нормали', 'round the normal'),
    'ed.spin.x': ('через верх и назад', 'over the top and behind'),
    'ed.spin.y': ('через бок и назад', 'over the side and behind'),
    'ed.v3spow': ('Просвет: узость пучка', 'Translucency: beam narrowness'),
    'ed.v3spow.info': ('Просвет листвы (выход «просвет» узла «Листва» или выход «_sss») '
                       'виден, когда свет за карточкой: перетащите точку на кольце света '
                       'во внешнюю полосу «сзади». Чем больше узость, тем ближе к линии '
                       '«камера — свет» он вспыхивает.',
                       'Foliage translucency (the «Foliage» node\'s translucency output or a '
                       '«_sss» output) shows when the light is behind the card: drag the '
                       'point on the light ring into the outer «behind» band. The higher '
                       'the narrowness, the closer to the camera–light line it flares up.'),
    'com.mask.vis': ('показывать маску', 'show mask'),
    'com.batch.done': ('готово: %d, пропущено: %d', 'done: %d, skipped: %d'),
    'com.batch.title': ('Пакетная обработка', 'Batch processing'),

    # --------------------------------------------------- вкладка листвы ----
    'f.preset': ('Тип растительности', 'Vegetation type'),
    'f.preset.info': ('Готовый набор настроек под характер текстуры. Он только '
                      'расставляет ползунки — дальше всё правится вручную.',
                      'A starting point matched to the kind of texture. It only sets '
                      'the sliders; everything stays editable afterwards.'),
    'f.p.leaves': ('Листья', 'Broadleaf'),
    'f.p.needles': ('Хвоя', 'Conifer needles'),
    'f.p.branches': ('Ветки без листьев', 'Bare branches'),
    'f.p.moss': ('Мох и лишайник', 'Moss and lichen'),
    'f.p.grass': ('Трава и почвопокровные', 'Grass and groundcover'),
    'f.p.dense': ('Плотная масса', 'Dense canopy'),
    'f.noalpha': ('Если альфы нет', 'When there is no alpha'),
    'f.noalpha.info': ('Форма листьев читается из силуэта, поэтому без альфа-канала '
                       'эти две вкладки не работают. Если растение нарисовано на '
                       'однородном фоне, альфу можно вырезать из него.',
                       'Leaf shape is read from the silhouette, so without an alpha '
                       'channel these two tabs have nothing to work with. If the plant '
                       'sits on a flat background, an alpha can be cut out of it.'),
    'f.tol': ('Допуск по фону', 'Background tolerance'),
    'f.tol.info': ('Насколько цвет должен отличаться от фона, чтобы считаться листом.',
                   'How far a colour must be from the background to count as foliage.'),
    'f.makealpha': ('Сделать альфу из фона', 'Build alpha from background'),
    'f.alphadone': ('%s   альфа построена из фона', '%s   alpha built from background'),
    'f.alphafail': ('Фон не отделился: %s\n\nПопробуй другой допуск.',
                    'Background did not separate: %s\n\nTry a different tolerance.'),
    'f.detect': ('Разделение на элементы', 'Splitting into elements'),
    'f.detect.info': ('Программа сначала делит массу на отдельные листья, хвоинки или '
                      'травинки, и только потом придаёт каждому объём. От того, как '
                      'она их нашла, зависит всё остальное.',
                      'The mass is first divided into individual leaves, needles or '
                      'blades; only then does each get its shape. Everything else '
                      'depends on how well that division went.'),
    'f.m.silh': ('по силуэту', 'by silhouette'),
    'f.m.silh.info': ('Границы берутся из альфа-канала. Работает, когда листья '
                      'не слиплись в сплошное пятно.',
                      'Boundaries come from the alpha channel. Works when leaves are '
                      'not fused into one solid blob.'),
    'f.m.dense': ('по прожилкам', 'by veins'),
    'f.m.dense.info': ('Границы ищутся по светлым продольным хребтам на самой '
                       'картинке. Для плотной кроны, где силуэт сплошной.',
                       'Boundaries are found as bright elongated ridges in the image '
                       'itself. For dense canopy with no gaps in the silhouette.'),
    'f.m.lum': ('по светлым бугоркам', 'by bright tufts'),
    'f.m.lum.info': ('Каждая кочка — небольшое светлое пятно. Для мха, лишайника '
                     'и мелкой подушки, где нет ни силуэта, ни прожилок.',
                     'Each tuft is a small bright spot. For moss, lichen and fine '
                     'cushion growth, which have neither silhouette nor veins.'),
    'f.split': ('Дробление', 'Split amount'),
    'f.split.info': ('Меньше — режет мельче и элементов получается больше.',
                     'Lower values cut finer and produce more elements.'),
    'f.minarea': ('Минимальный размер элемента, пикс.', 'Minimum element size, px'),
    'f.minarea.info': ('Куски мельче этого не живут своей жизнью, а достаются соседям. '
                       'Главное лекарство от рассыпающихся веток: без него длинная '
                       'ветка рвётся на куски в каждом утолщении и каждый кусок '
                       'получает свой наклон.',
                       'Pieces smaller than this are handed to their neighbours instead '
                       'of living on their own. This is the cure for shattered branches: '
                       'without it a long branch breaks apart at every thickening and '
                       'each fragment gets its own tilt.'),
    'f.shape': ('Форма элемента', 'Element shape'),
    'f.fold': ('Изгиб', 'Cross-section curvature'),
    'f.fold.info': ('1.0 — настоящий цилиндр. Отрицательные значения выгибают лист '
                    'в другую сторону: получается ложбинка, а не горбик. Для лодочек, '
                    'вогнутых листьев и чашечек.',
                    '1.0 is a true cylinder. Negative values bend the leaf the other '
                    'way — a groove instead of a ridge. For cupped and concave leaves.'),
    'f.prof': ('Профиль сечения', 'Cross-section profile'),
    'f.prof.info': ('1 — домиком, 2 — круглый, больше — плоский верх с резким '
                    'завалом у края.',
                    '1 is a tent, 2 is round, higher gives a flat top with a sharp '
                    'fall-off at the edge.'),
    'f.det': ('Мелочь с картинки', 'Detail from the image'),
    'f.det.info': ('Подмешивает мелкий рельеф, снятый с самой текстуры: прожилки, '
                   'зубчики, фактуру коры.',
                   'Adds fine relief taken from the texture itself: veins, serrations, '
                   'bark grain.'),
    'f.seed': ('Зерно случайности', 'Random seed'),
    'f.pad': ('Запас за краем, пикс.', 'Bleed past the edge, px'),
    'f.pad.info': ('Нормаль продолжается наружу за силуэт. Нужно, чтобы при сжатии '
                   'и на мипах по краю листа не вылезал плоский ободок.',
                   'The normal continues outside the silhouette. Prevents a flat rim '
                   'along leaf edges after compression and in mip levels.'),
    'f.orient': ('Разворот элементов', 'Element orientation'),
    'f.tilt': ('Наклон', 'Tilt'),
    'f.tilt.info': ('Насколько каждый элемент повёрнут по-своему. Это то, что '
                    'отличает крону от одного большого мятого листа.',
                    'How much each element turns its own way. This is what separates '
                    'a canopy from one large crumpled leaf.'),
    'f.thin': ('Порог «это ветка»', 'Branch threshold'),
    'f.thin.info': ('Кусок вытянутее этого считается веткой: наклон ему не раздаётся, '
                    'только скругление в трубку. Чтобы увидеть, кого он зацепил, '
                    'включи справа показ «листья и ветки»: красное — ветка, '
                    'зелёное — лист.',
                    'Anything more elongated than this counts as a branch: it gets no '
                    'tilt, only rounding into a tube. Switch the view to «leaves and '
                    'branches» to see the split: red is branch, green is leaf.'),
    'f.blade': ('Вытянутые — это травинки', 'Elongated pieces are blades'),
    'f.blade.info': ('Травинка вытянута так же, как ветка, но наклон ей нужен: это '
                     'отдельный лист, а не сучок, на который всё нанизано. На 1 '
                     'вытянутые куски разворачиваются наравне с остальными.',
                     'A grass blade is as elongated as a branch but does need tilt: it '
                     'is a leaf in its own right, not a stem carrying others. At 1 '
                     'elongated pieces turn just like the rest.'),
    'f.bend': ('Изгиб вдоль элемента', 'Bend along the element'),
    'f.bend.info': ('Дуга по длине куска: один конец отворачивается от света, другой '
                    'к нему. Травинки и ветки прямыми не бывают, и без этого они '
                    'выглядят как палки.',
                    'An arc along the length of a piece: one end turns away from the '
                    'light, the other towards it. Blades and twigs are never straight, '
                    'and without this they read as sticks.'),
    'f.clump': ('Объём пучков', 'Clump volume'),
    'f.clump.info': ('Промежуточный размер: не изгиб отдельного листа и не купол всей '
                     'карточки, а объём пучков. Считается по плотности листвы — где '
                     'масса гуще, там выше. Отрицательные значения делают пучок '
                     'вогнутым.',
                     'The middle scale: neither the individual leaf nor the whole card, '
                     'but the volume of clumps. Derived from foliage density — thicker '
                     'mass sits higher. Negative values make clumps concave.'),
    'f.clm': ('Пучки', 'Clumps'),
    'f.clb': ('Крупные массы', 'Large masses'),
    'f.clb.info': ('Вчетверо крупнее пучков.', 'Four times the clump scale.'),
    'f.cls': ('Размер пучка, пикс.', 'Clump size, px'),
    'f.card': ('Общая форма карточки', 'Overall card shape'),
    'f.card.info': ('Купол на всю карточку по габаритам силуэта. Обе ручки поровну — '
                    'шар, одна — цилиндр. Отрицательные значения дают вогнутую '
                    'карточку — для внутренней стороны кроны и для ниш.',
                    'A dome over the whole card, sized to the silhouette. Both equal '
                    'gives a sphere, one alone gives a cylinder. Negative values make '
                    'the card concave — for the inside of a canopy, or for recesses.'),
    'f.domw': ('Выпуклость по ширине', 'Convexity across width'),
    'f.domh': ('Выпуклость по высоте', 'Convexity across height'),
    'f.hand': ('Ветки вручную', 'Manual branch marking'),
    'f.hand.info': ('Автомат считает веткой всё сильно вытянутое, и на иве это '
                    'ошибается. Здесь можно докрасить руками: помеченные куски '
                    'наклон не получают, только скругление в трубку.',
                    'The automatic rule calls anything very elongated a branch, which '
                    'goes wrong on willow. Here you can mark them by hand: marked '
                    'pieces get no tilt, only rounding into a tube.'),
    'f.brush': ('режим кисти', 'brush mode'),
    'f.brush.info': ('Левая кнопка красит, правая стирает. Пока кисть включена, '
                     'превью не таскается мышью — зум колесом работает.',
                     'Left button paints, right button erases. While the brush is on '
                     'the preview does not pan; the wheel still zooms.'),
    'f.mask.info': ('Оранжевая подсветка помеченного. Снимай галочку, когда нужно '
                    'посмотреть на карту без неё — сама пометка при этом остаётся '
                    'и продолжает действовать.',
                    'Orange highlight over what is marked. Turn it off to look at the '
                    'map without it — the marking itself stays and keeps working.'),
    'f.out': ('Вывод', 'Output'),
    'f.fast': ('быстрое превью для больших текстур', 'fast preview for large textures'),
    'f.fast.info': ('Текстуры крупнее 1024 считаются в превью на половине размера. '
                    'Разделение на элементы там чуть крупнее, чем будет в файле — '
                    'при сохранении всё пересчитывается на полном размере.',
                    'Textures above 1024 are previewed at half size. The element '
                    'division is slightly coarser there than in the saved file — '
                    'saving always recomputes at full resolution.'),
    'f.savenormal': ('Сохранить _n.dds', 'Save _n.dds'),
    'f.status': ('элементов: %d   веткой считается: %.0f %%   помечено вручную: %.1f %%',
                 'elements: %d   counted as branch: %.0f %%   marked by hand: %.1f %%'),
    'f.noalpha.short': ('нет альфа-канала', 'no alpha channel'),
    'f.v.light': ('со светом', 'lit'),
    'f.v.orig': ('оригинал', 'original'),
    'f.v.map': ('саму карту', 'the map itself'),
    'f.v.seg': ('разделение', 'element split'),
    'f.v.branch': ('листья и ветки', 'leaves and branches'),
    'f.err.title': ('Не разобралось как растительность', 'Not readable as foliage'),
    'f.hasalpha': ('альфа есть, непрозрачно %.0f %%', 'alpha present, opaque %.0f %%'),
    'f.noalpha.tag': ('БЕЗ АЛЬФЫ (%s)', 'NO ALPHA (%s)'),
    'f.parse.err': ('ошибка разбора', 'parse error'),
    'f.mask.save': ('Сохранить маску веток', 'Save branch mask'),
    'f.mask.load': ('Загрузить маску веток', 'Load branch mask'),
    'f.mask.saved': ('маска сохранена', 'mask saved'),
    'f.na.head': ('У этой текстуры нет альфа-канала.', 'This texture has no alpha channel.'),
    'f.na.body': ('Форма листьев читается из силуэта, поэтому здесь её взять неоткуда. '
                  'Такую текстуру обрабатывают во вкладке «Материал».\n\n'
                  'Если растение нарисовано на однородном фоне, альфу можно построить '
                  'из него — кнопка слева вверху.',
                  'Leaf shape is read from the silhouette, and there is none here. '
                  'Work on this texture in the «Material» tab instead.\n\n'
                  'If the plant sits on a flat background, an alpha can be built from '
                  'it — the button at the top left.'),

    # ------------------------------------------ вкладка света листвы -------
    'd.delight': ('Снять нарисованный свет', 'Remove painted lighting'),
    'd.delight.info': ('Старые текстуры листвы нарисованы уже со светом и тенью. '
                       'В движке с настоящим освещением это ложится вторым слоем, '
                       'и крона выглядит грязной.',
                       'Old foliage textures come with light and shade already painted '
                       'in. In an engine with real lighting that lands twice and the '
                       'canopy reads as dirty.'),
    'd.un': ('Сила снятия', 'Strength'),
    'd.unr': ('Масштаб света, пикс.', 'Lighting scale, px'),
    'd.ao': ('Затенение в глубине', 'Depth shading'),
    'd.aor': ('Радиус затенения, пикс.', 'Shading radius, px'),
    'd.lift': ('Общая яркость', 'Overall brightness'),
    'd.sat': ('Удержать цветность', 'Keep colourfulness'),
    'd.sat.info': ('Осветлённый участок глаз видит цветнее, даже если отношение '
                   'каналов не изменилось: чем ярче пиксель, тем заметнее его цвет. '
                   'На 1 размах цвета остаётся таким, каким был до снятия света.',
                   'A brightened area looks more colourful even when the channel ratio '
                   'is unchanged: the brighter a pixel, the more its colour shows. At 1 '
                   'the colour spread stays exactly as it was before de-lighting.'),
    'd.aomap': ('Отдельная карта AO', 'Separate AO map'),
    'd.aomap.info': ('Затенение — это доля неба, которую видит точка, и посчитать её '
                     'можно только по глубине: одно затеняет другое, лишь когда лежит '
                     'выше. У карточки глубины нет, поэтому она придумывается — каждый '
                     'лист получает свой случайный сдвиг плюс собственную выпуклость, — '
                     'а дальше по ней идёт обычный обход горизонта, как в SSAO. Дырки '
                     'в альфе при этом считаются небом, а не низкой поверхностью, '
                     'поэтому лист у просвета остаётся светлым.',
                     'Occlusion is the fraction of sky a point can see, and it can only '
                     'be computed from depth: one thing shades another only when it lies '
                     'above it. A card has no depth, so it is invented — each leaf gets '
                     'a random offset plus its own bulge — and then an ordinary horizon '
                     'sweep runs over it, as in SSAO. Holes in the alpha count as sky '
                     'rather than as low ground, so a leaf beside a gap stays bright.'),
    'd.aostr': ('Сила AO', 'AO strength'),
    'd.aow1': ('По плотности листвы', 'From foliage density'),
    'd.aow1.info': ('Где листва лежит в несколько слоёв — темнее. Знает про соседей, '
                    'но ничего не знает про сам лист.',
                    'Where foliage is layered it goes darker. Knows about neighbours, '
                    'nothing about the leaf itself.'),
    'd.aow2': ('По наклону листа', 'From leaf tilt'),
    'd.aow2.info': ('Под равномерным небом точка видит его долю (1 + n.z)/2: '
                    'отвёрнутый лист темнее.',
                    'Under a uniform sky a point sees a fraction (1 + n.z)/2 of it: '
                    'a leaf turned away is darker.'),
    'd.aow3': ('По разделению на элементы', 'From the element split'),
    'd.aow3.info': ('Край листа темнее середины — там он смыкается с тем, что лежит '
                    'под ним. Даёт каждому листу отдельный объём.',
                    'A leaf edge is darker than its middle, where it meets whatever '
                    'lies beneath. Gives every leaf its own volume.'),
    'd.aogam': ('AO: гамма', 'AO: gamma'),
    'd.aocon': ('AO: контраст', 'AO: contrast'),
    # ------------------------------------------------------- просвет ------
    'd.sss': ('Просвет насквозь (translucency)', 'Translucency'),
    'd.sss.info': ('Лист работает как светофильтр: часть света проходит его насквозь '
                   'и выходит с другой стороны окрашенной. Отсюда знакомая картина — '
                   'крона против солнца светится, причём ярче и зеленее, чем она же '
                   'в отражённом свете. Ни нормаль, ни AO этого не дают: они знают '
                   'только про свет, который отразился.',
                   'A leaf works as a colour filter: part of the light passes straight '
                   'through and comes out the other side tinted. Hence the familiar '
                   'sight of a canopy against the sun — glowing, and greener than the '
                   'same canopy in reflected light. Neither the normal map nor the AO '
                   'can give you that: they only know about light that bounced off.'),
    'd.sss.str': ('Сила просвета', 'Translucency strength'),
    'd.sss.thick': ('Толщина листа', 'Leaf thickness'),
    'd.sss.thick.info': ('У края лист тонкий и просвечивает, вдоль жилки толстый и '
                         'не просвечивает. Больше — сильнее гаснет к середине листа.',
                         'A leaf is thin at the edge and lets light through, thick '
                         'along its spine and does not. Higher values fall off faster '
                         'towards the middle of the leaf.'),
    'd.sss.dens': ('Слои листвы', 'Foliage layers'),
    'd.sss.dens.info': ('Один лист просвечивает, пять листьев подряд — уже нет. '
                        'Считается по плотности кроны.',
                        'One leaf lets light through, five in a row do not. Taken from '
                        'canopy density.'),
    'd.sss.branch': ('Ветки не просвечивают', 'Branches are opaque'),
    'd.sss.branch.info': ('Ветки и стволы света не пропускают, и сказать об этом надо '
                          'прямо: по силуэту толстая ветка от толстого листа не '
                          'отличается. Берутся те же ветки, что и на первой вкладке, '
                          'включая помеченные вручную.',
                          'Branches and trunks pass no light, and that has to be stated '
                          'explicitly: by silhouette alone a thick branch is no '
                          'different from a thick leaf. Uses the same branches as the '
                          'first tab, hand-marked ones included.'),
    'd.sss.tint': ('Насыщенность фильтра', 'Filter saturation'),
    'd.sss.tint.info': ('Прошедший свет шёл сквозь пигмент дольше отражённого, поэтому '
                        'он всегда чище по тону: зелёный зеленее, осенний желтее. '
                        'На 0 цвет просвета совпадает с цветом самого листа.',
                        'Transmitted light travelled through the pigment for longer '
                        'than reflected light, so its hue is always purer: green '
                        'greener, autumn yellower. At 0 the transmitted colour matches '
                        'the leaf itself.'),
    'd.sss.gam': ('Просвет: гамма', 'Translucency: gamma'),
    'd.sss.con': ('Просвет: контраст', 'Translucency: contrast'),
    'd.sss.pow': ('Узость пучка', 'Beam narrowness'),
    'd.sss.pow.info': ('Прошедший свет уходит вперёд, по ходу луча, а не во все '
                       'стороны: просвет виден, только когда источник примерно за '
                       'листом. Больше — уже угол, при котором он вспыхивает.',
                       'Transmitted light continues forward along the ray rather than '
                       'spreading everywhere: the glow appears only when the source is '
                       'roughly behind the leaf. Higher values narrow the angle at '
                       'which it lights up.'),
    'd.v.sss': ('карта просвета', 'translucency map'),
    'd.v.backlit': ('против света', 'backlit'),
    'd.v.backlit.info': ('Источник переносится за карточку: то же кольцо, но свет '
                         'сзади. Единственный ракурс, в котором просвет вообще виден.',
                         'The source moves behind the card: the same ring, but lit from '
                         'behind. The only view in which translucency shows at all.'),
    'sv._sss': ('_sss — просвет (цвет + сила в альфе)',
                '_sss — translucency (colour + strength in alpha)'),
    'd.ao.info': ('Затенение — это доля неба, которую точка видит. Для кроны '
                  'вопрос стоит так: сколько соседней листвы её загораживает. '
                  'Считается по плотности листвы на разных расстояниях, с весами '
                  'из телесного угла: близкая листва закрывает больше, дальняя '
                  'меньше, а дырка в кроне не закрывает ничего.',
                  'Occlusion is the fraction of sky a point can see. For a canopy the '
                  'question is how much neighbouring foliage blocks it. It is computed '
                  'from foliage density at a range of distances, weighted by solid '
                  'angle: near foliage blocks more, far foliage less, and a hole in the '
                  'canopy blocks nothing.'),
    'd.aorad': ('Радиус поиска, пикс.', 'Search radius, px'),
    'd.aorad.info': ('Насколько далеко вокруг точки считается листва. Примерно '
                     'размер пучка: дальше этого соседи закрывают уже слишком '
                     'малый кусок неба, чтобы это было видно.',
                     'How far around a point foliage is taken into account. Roughly the '
                     'size of a clump: beyond that, neighbours block too small a part '
                     'of the sky to matter.'),
    'd.aodep': ('Толщина кроны, пикс.', 'Canopy thickness, px'),
    'd.aodep.info': ('Насколько высоко листва стоит над точкой. Чем толще крона, '
                     'тем больше неба закрывает даже далёкая листва и тем мягче '
                     'выходит затенение. Тонкая крона даёт резкое затенение по '
                     'ближайшим соседям.',
                     'How high foliage stands above a point. The thicker the canopy, the '
                     'more sky even distant foliage blocks, and the softer the result. A '
                     'thin canopy gives sharp occlusion from the nearest neighbours '
                     'only.'),
    'd.aoself': ('Собственный объём листа', "Leaf's own volume"),
    'd.aoself.info': ('Небольшая добавка: у края лист сходит на нет и смыкается '
                      'с тем, что под ним, поэтому край чуть темнее середины. Это '
                      'именно добавка — если сделать её основой, каждый лист '
                      'превращается в отдельную черепицу.',
                      'A small addition: a leaf tapers to nothing at its edge and meets '
                      'whatever lies beneath, so the edge is slightly darker than the '
                      'middle. An addition only — make it the main term and every leaf '
                      'turns into a separate shingle.'),
    'd.sss.base': ('Пластина листа', 'Leaf blade'),
    'd.sss.base.info': ('Постоянная толщина самой пластины, одинаковая по всему '
                        'листу. У живого листа она и правда почти постоянна: '
                        'светящегося ободка по краю в природе нет.',
                        'The constant thickness of the blade itself, the same across the '
                        'whole leaf. On a real leaf it genuinely is nearly constant: '
                        'there is no glowing rim around the edge in nature.'),
    'd.sss.veins': ('Жилки', 'Veins'),
    'd.sss.veins.info': ('Жилки и кромки — утолщения, свет через них не идёт. На '
                         'просвет они выходят тёмной сеткой по светящейся пластине, '
                         'и это первое, по чему глаз узнаёт лист против солнца.',
                         'Veins and margins are thickenings and pass no light. Backlit '
                         'they show as a dark network over a glowing blade, and that is '
                         'the first thing by which the eye recognises a leaf against '
                         'the sun.'),
    'd.sss.lay': ('Слои листвы', 'Foliage layers'),
    'd.sss.lay.info': ('Сколько листвы лежит поверх. Берётся то же затенение, что '
                       'и для карты AO, поэтому потемнение выходит в форме соседних '
                       'листьев, а не размытым пятном.',
                       'How much foliage lies on top. It uses the same occlusion as the '
                       'AO map, so the darkening takes the shape of the neighbouring '
                       'leaves rather than a soft blob.'),
    'd.sss.pig': ('Плотность пигмента', 'Pigment density'),
    'd.sss.pig.info': ('Тёмный лист плотнее набит пигментом и пропускает меньше. '
                       'Заодно именно так на текстуре нарисованы листья заднего '
                       'плана — темнее передних.',
                       'A darker leaf holds more pigment and passes less light. It is '
                       'also exactly how background leaves are painted on a texture — '
                       'darker than the ones in front.'),
    'd.sss.edge': ('Край листа тоньше', 'Thinner at the edge'),
    'd.sss.edge.info': ('Небольшая добавка у самой кромки. Держи её маленькой: '
                        'если сделать толщину зависящей в основном от расстояния до '
                        'края, вместо светящегося листа получится светящийся контур.',
                        'A small addition right at the margin. Keep it small: make '
                        'thickness depend mainly on distance from the edge and you get '
                        'a glowing outline instead of a glowing leaf.'),
    'd.sss.shad': ('Тень соседних листьев', 'Shadow from neighbouring leaves'),
    'd.sss.shad.info': ('Только для просмотра, в карту не запекается. Сквозь верхний '
                        'лист бьёт солнце, а на том, что за ним, лежит его силуэт — '
                        'на просвет это видно лучше всего. Тень ездит вместе с '
                        'солнцем, поэтому её место в шейдере, а не в текстуре; здесь '
                        'она нужна, чтобы увидеть, как всё сложится в движке.',
                        'Preview only, never baked into the map. The sun strikes the '
                        'front leaf and its silhouette falls across the one behind — '
                        'backlit, this is the most visible thing of all. The shadow '
                        'travels with the sun, so it belongs in the shader, not in the '
                        'texture; here it shows how it will come together in engine.'),
    'd.sss.shoff': ('Разнос листьев, пикс.', 'Leaf separation, px'),
    'd.sss.shoff.info': ('Насколько далеко разнесены слои листвы по глубине. Больше — '
                         'тень уходит дальше от листа, который её отбрасывает.',
                         'How far apart the foliage layers sit in depth. Higher values '
                         'move the shadow further from the leaf casting it.'),
    'd.aospread': ('Разброс листьев по глубине, пикс.', 'Leaf depth spread, px'),
    'd.aospread.info': ('У карточки нет глубины, а затеняет одно другое только тогда, '
                        'когда одно лежит выше другого. Поэтому глубина придумывается: '
                        'каждый лист из разбиения получает свой случайный сдвиг. Зерно '
                        'то же, что у наклонов на вкладке нормалей, так что глубина и '
                        'нормаль описывают одну и ту же ветку. Больше разброс — '
                        'заметнее контактные тени там, где лист заходит за лист.',
                        'A card has no depth, and one thing shades another only when it '
                        'lies above it. So depth is invented: every leaf from the split '
                        'gets its own random offset. The seed is the same as for the '
                        'tilts on the normals tab, so depth and normal describe one and '
                        'the same twig. More spread means stronger contact shadows '
                        'where one leaf tucks behind another.'),
    'd.aobulge': ('Выпуклость листа, пикс.', 'Leaf bulge, px'),
    'd.aobulge.info': ('Собственный объём листа в той же придуманной глубине: от края '
                       'к середине. Даёт мягкую тень по кромке, которой лист ложится '
                       'на соседа.',
                       "The leaf's own volume within that invented depth, from edge to "
                       'middle. It produces the soft shadow along the margin where a '
                       'leaf rests on its neighbour.'),
    'd.aorad2': ('Радиус обхода, пикс.', 'Sweep radius, px'),
    'd.aorad2.info': ('Насколько далеко ищется горизонт. Примерно размер листа: дальше '
                      'этого соседи закрывают уже слишком малый кусок неба.',
                      'How far the horizon is searched. Roughly one leaf: beyond that, '
                      'neighbours block too small a part of the sky to matter.'),
    'd.aotilt': ('Учитывать наклон листа', "Account for the leaf's tilt"),
    'd.aotilt.info': ('Горизонт отсчитывается не от плоскости карточки, а от плоскости '
                      'самого листа по карте нормалей: у наклонённого листа часть неба '
                      'закрыта им самим. Это честная форма «затенения по наклону» — '
                      'отдельной картой оно было просто синим каналом нормали, а внутри '
                      'обхода горизонта стоит на своём месте. Выше 0.5 лист начинает '
                      'затенять сам себя сильнее, чем его затеняют соседи.',
                      'The horizon is measured not from the plane of the card but from '
                      "the plane of the leaf itself, taken from the normal map: a tilted "
                      'leaf hides part of the sky with its own body. This is the honest '
                      'form of «occlusion from tilt» — as a separate map it was merely '
                      'the blue channel of the normal, but inside a horizon sweep it '
                      'belongs. Above 0.5 a leaf starts shading itself more than its '
                      'neighbours shade it.'),
    'd.sss.behind': ('Листья позади', 'Leaves behind'),
    'd.sss.behind.info': ('Главное, что видно на просвет: сквозь переднюю пластину '
                          'просвечивает тёмный силуэт другого листа — того, что '
                          'заслонил солнце. На карточке он не нарисован, его закрыли, '
                          'но глубина уже есть, и продолжить его под соседа несложно: '
                          'если рядом нашёлся лист заметно ниже — значит он проходит '
                          'под нами. Направление света тут ни при чём, поэтому в карту '
                          'это запекается: что лежит за листом, лежит за ним при любом '
                          'солнце.',
                          'The main thing you see backlit: through the front blade shows '
                          'the dark silhouette of another leaf — the one that blocked '
                          'the sun. It is not painted on the card, being hidden, but the '
                          'depth already exists and continuing it under its neighbour is '
                          'easy: if a leaf sitting distinctly lower is nearby, it passes '
                          'beneath us. Light direction plays no part, so this bakes into '
                          'the map: what lies behind a leaf lies behind it under any '
                          'sun.'),
    'd.sss.reach': ('Насколько заходят друг за друга, пикс.',
                    'How far leaves reach behind, px'),
    'd.sss.reach.info': ('Как далеко лист продолжается под соседним. Примерно половина '
                         'ширины листа: больше — силуэты позади расползаются, меньше — '
                         'остаются узкой каймой у края.',
                         'How far a leaf continues underneath its neighbour. About half '
                         'a leaf width: more and the silhouettes behind spread out, less '
                         'and they stay a narrow band along the edge.'),
    'bent': ('Bent normal (для движков с рассеянным светом по нормали)',
             'Bent normal (for engines with normal-dependent ambient)'),
    'bent.info': ('Обычная нормаль смотрит туда, куда повёрнута поверхность. Bent normal '
                  'смотрит в середину той части неба, которая осталась открытой. В щелях '
                  'она поворачивается к выходу из щели.\n\nЗатенение она даёт только там, '
                  'где движок считает рассеянный свет по нормали — полусферой или по '
                  'карте окружения. В OpenMW рассеянный свет ровный, поэтому затенения не '
                  'будет: нормали в щелях только станут положе, и карта будет выглядеть '
                  'тусклее. Длина вектора, в которой можно было бы хранить затенение, '
                  'движком отбрасывается — он нормирует нормаль сразу после чтения.\n\nДля '
                  'OpenMW затенение кладётся в диффуз: раздел «Рельеф в диффуз», источник '
                  'AO или высота. Проверить, как bent normal выглядит в движке с '
                  'рассеянным светом по нормали, — галочка «рассеянный свет полусферой» '
                  'рядом с кругом света.',
                  'An ordinary normal points where the surface faces. A bent normal '
                  'points at the middle of whatever sky is left open. In crevices it '
                  'turns towards the way out.\n\nIt only produces occlusion where the '
                  'engine computes ambient light from the normal — hemispherically or '
                  "from an environment map. OpenMW's ambient is flat, so there will be no "
                  'occlusion: normals in crevices just get flatter, and the map looks '
                  'duller. The vector length, which could have carried the occlusion, is '
                  'thrown away by the engine — it normalises the normal right after '
                  'reading it.\n\nFor OpenMW, occlusion goes into the diffuse: the "Relief '
                  'into the diffuse" section, source AO or height. To see how a bent '
                  'normal looks in an engine with normal-dependent ambient, tick '
                  '"hemispherical ambient" next to the light circle.'),
    'd.what': ('Что сохранять', 'What to save'),
    'd.cov': ('держать площадь листвы в мипах', 'preserve foliage coverage in mips'),
    'd.cov.info': ('При обычном усреднении альфа расползается, тест прозрачности '
                   'срезает больше пикселей, и крона на расстоянии лысеет.',
                   'Plain averaging spreads the alpha out, the alpha test then cuts '
                   'more pixels, and the canopy goes bald at a distance.'),
    'd.v.pair': ('до и после', 'before and after'),
    'd.v.after': ('только результат', 'result only'),
    'd.v.ao': ('AO отдельно', 'AO alone'),
    'd.v.height': ('высота листа', 'leaf height'),
    'd.hint': ('слева оригинал, справа результат', 'original left, result right'),
    'sv._fixed': ('_fixed — диффуз без света', '_fixed — diffuse without lighting'),
    'sv._ao': ('_ao — затенение', '_ao — occlusion'),
    'sv._n': ('_n — нормаль', '_n — normal map'),
    'sv._nh': ('_nh — нормаль + высота в альфе', '_nh — normal + height in alpha'),
    'sv._h': ('_h — высота', '_h — height'),
    'sv._spec': ('_spec — блик + глянец в альфе', '_spec — specular + gloss in alpha'),
    'sv._rough': ('_rough — шероховатость', '_rough — roughness'),
    'sv._metal': ('_metal — металличность', '_metal — metalness'),

    # ------------------------------------------------ вкладка материала ----
    'm.preset': ('Материал', 'Material'),
    'm.preset.info': ('Пресет только расставляет ползунки — дальше правь руками. '
                      'Камень, дерево, ткань и металл различаются именно материалом, '
                      'а не алгоритмом.',
                      'A preset only sets the sliders; everything stays editable. '
                      'Stone, wood, cloth and metal differ in material, not in '
                      'algorithm.'),
    'p.stone': ('Камень', 'Stone'),
    'p.stone_worn': ('Камень выветренный', 'Weathered stone'),
    'p.brick': ('Кирпич и кладка', 'Brick and masonry'),
    'p.plaster': ('Штукатурка', 'Plaster'),
    'p.metal': ('Металл', 'Metal'),
    'p.metal_rust': ('Металл ржавый', 'Rusted metal'),
    'p.wood': ('Дерево', 'Wood'),
    'p.bark': ('Кора', 'Bark'),
    'p.cloth': ('Ткань', 'Cloth'),
    'p.leather': ('Кожа', 'Leather'),
    'p.skin': ('Кожа и лицо', 'Skin and faces'),
    'p.ground': ('Грунт и песок', 'Ground and sand'),
    'p.moss': ('Мох', 'Moss'),
    'p.glass': ('Стекло', 'Glass'),
    'p.snow': ('Снег', 'Snow'),

    'tab.nodes': ('Узлы', 'Nodes'),
    'g.add': ('Добавить узел', 'Add a node'),
    'g.add.info': ('Граф ничего нового не считает: каждый узел — тот же расчёт, что '
                   'стоит за соответствующим ползунком на вкладках. Новое тут одно: их '
                   'можно соединять как угодно — например снять свет дважды с разными '
                   'радиусами и сложить, или взять нормаль от высоты, собранной из сети '
                   'и освещения сразу.\n\n'
                   'Мышь: тянуть за кружок выхода к кружку входа — соединить; щёлкнуть '
                   'по кружку входа — отсоединить; правая кнопка по узлу — удалить; '
                   'двойной щелчок — посмотреть, что из него выходит.',
                   'The graph computes nothing new: every node is the same calculation '
                   'that sits behind the matching slider on the tabs. The one new thing '
                   'is that they can be wired any way you like — de-light twice at '
                   'different radii and add the results, say, or take the normal from a '
                   'height built from the network and the lighting at once.\n\n'
                   'Mouse: drag from an output circle to an input circle to connect; '
                   'click an input circle to disconnect; right-click a node to delete '
                   'it; double-click to see what comes out of it.'),
    'g.add.btn': ('Добавить', 'Add'),
    'g.del': ('Удалить', 'Delete'),
    'g.reset': ('Сбросить', 'Reset'),
    'g.save': ('Сохранить граф…', 'Save graph…'),
    'g.load': ('Открыть граф…', 'Open graph…'),
    'g.saveout': ('Сохранить карты с выходов…', 'Save maps from the outputs…'),
    'g.batch': ('Обработать папку по графу…',
                'Process a folder with the graph…'),
    'g.noout': ('В графе нет выходных узлов — сохранять нечего.',
                'The graph has no output nodes — nothing to save.'),
    'g.sub': ('нейросеть: окно %d из %d',
              'network: window %d of %d'),
    'g.node': ('считается: %s',
               'computing: %s'),
    'g.noinput': ('%s: не подключен вход «%s»',
                  '%s: input "%s" is not connected'),
    'g.params': ('Параметры узла', 'Node settings'),
    'g.hint': ('Выберите узел, чтобы увидеть его настройки. Двойной щелчок по узлу — '
               'показать его результат внизу.',
               'Select a node to see its settings. Double-click a node to show its '
               'result below.'),
    'g.ok': ('показан узел: %s', 'showing node: %s'),
    'g.badlink': ('так соединить нельзя: разные типы или получилось бы кольцо',
                  'cannot connect: different kinds, or it would make a loop'),
    'g.nosrc': ('сначала откройте текстуру', 'open a texture first'),
    'g.novalue': ('у этого узла нечего показать', 'this node has nothing to show'),

    'nd.source': ('Исходник', 'Source'),
    'nd.repair': ('Починка битых мест', 'Repair damage'),
    'nd.delight': ('Снять свет', 'Remove light'),
    'nd.height_lum': ('Высота из яркости', 'Height from brightness'),
    'nd.height_light': ('Высота по освещению', 'Height from lighting'),
    'nd.depth_net': ('Глубина нейросетью', 'Depth from a network'),
    'nd.from_ao': ('Высота из затенения', 'Height from occlusion'),
    'nd.bands': ('Диапазоны', 'Bands'),
    'nd.tone': ('Тон', 'Tone'),
    'nd.mix': ('Смешать', 'Mix'),
    'nd.normal': ('Нормаль из высоты', 'Normal from height'),
    'nd.normal_light': ('Нормаль по освещению', 'Normal from lighting'),
    'nd.as_normal': ('Это нормаль',
                     'Treat as normal'),
    'nd.as_normal.info': ('Считать картинку на входе готовой картой нормалей.',
                          'Treat the input image as a finished normal map.'),
    'nd.normal_to_height': ('Нормаль → высота',
                            'Normal → height'),
    'nd.normal_to_height.info': ('Высота из карты нормалей по наименьшим квадратам (Франко–Челлаппа). '
                                 '«Не верить крупнее» убирает холм через всю текстуру, который копится '
                                 'из общего перекоса.',
                                 'Height from a normal map by least squares (Frankot–Chellappa). '
                                 '"Distrust above" removes the hill across the whole texture that builds '
                                 'up from an overall tilt.'),
    'nd.esrgan_normal': ('Нормаль нейросетью (ESRGAN)',
                         'Normal by network (ESRGAN)'),
    'nd.esrgan_normal.info': ('Сеть NormalMapGenerator. Модель скачивается в разделе «Нейросети '
                              'ESRGAN» на вкладке материала — там же видна её лицензия. На процессоре '
                              'считается долго.',
                              'The NormalMapGenerator network. The model is downloaded in the "ESRGAN '
                              'networks" section on the material tab, where its licence is shown. '
                              'Slow on a CPU.'),
    'nd.esrgan_franken': ('Франкен-карта (ESRGAN)',
                          'Franken map (ESRGAN)'),
    'nd.esrgan_franken.info': ('Сеть FrankenMapGenerator: высота, шероховатость и текстура в сером.',
                               'The FrankenMapGenerator network: height, roughness and the texture in '
                               'grey.'),
    'nd.parallax_nh': ('Параллакс для OpenMW (_nh)',
                       'Parallax for OpenMW (_nh)'),
    'nd.parallax_nh.info': ('Нормаль в RGB и высота в альфе, уложенная в окно вокруг 128 под нужную '
                            'глубину в сантиметрах (как раздел «Параллакс для OpenMW» на вкладке '
                            'материала).',
                            'Normal in RGB and height in alpha, packed into a window around 128 for '
                            'the wanted depth in centimetres (like the "Parallax for OpenMW" '
                            'section on the material tab).'),
    'np.gray': ('серый',
                'grey'),
    'np.nh': ('_nh',
              '_nh'),
    'np.flip': ('Развернуть зелёный',
                'Flip green'),
    'np.repeat': ('Повтор, юнитов',
                  'Repeat, units'),
    'np.depth': ('Глубина, см',
                 'Depth, cm'),
    'np.ref': ('В 128',
               'At 128'),
    'np.level': ('Уровень вручную',
                 'Manual level'),
    'np.dither': ('Дизеринг',
                  'Dither'),
    'nd.ao': ('Затенение (AO)', 'Occlusion (AO)'),
    'nd.bent': ('Bent normal', 'Bent normal'),
    'nd.spec': ('Отражение', 'Reflection'),
    'nd.seamless': ('Бесшовность', 'Seamless'),
    'nd.out': ('Выход', 'Output'),

    'np.rgb': ('картинка', 'image'), 'np.h': ('высота', 'height'),
    'np.n': ('нормаль', 'normal'), 'np.ao': ('AO', 'AO'),
    'np.a': ('А', 'A'), 'np.b': ('Б', 'B'), 'np.any': ('карта', 'map'),
    'np.light': ('свет', 'light'), 'np.signal': ('сигнал', 'signal'),
    'np.rough': ('шерох.',
                 'rough'),
    'np.spec': ('блик',
                'spec'),
    'np.metal': ('металл', 'metal'),
    'np.hi': ('Порог пересвета', 'Blow-out level'),
    'np.lo': ('Порог провала', 'Crush level'),
    'np.flat': ('Порог отсутствия фактуры', 'Flatness level'),
    'np.grow': ('Отступ маски, пикс.', 'Mask margin, px'),
    'np.repair.seed': ('Зерно пятен', 'Spot seed'),
    'np.repair.seed.info': ('Для способа «пятна основных цветов»: какой набор пятен. Одно и то же '
                            'зерно — одни и те же пятна на превью, при уточнении и в сохранённом '
                            'файле; другое число — другой рисунок.',
                            'For the «spots of main colours» method: which set of spots. The same '
                            'seed gives the same spots on the preview, on the refined pass and in '
                            'the saved file; another number — another pattern.'),
    'np.repair.rim': ('Кайма по склону, пикс.', 'Rim over the slope, px'),
    'np.repair.rim.info': ('Насколько маска заходит от найденного пятна на его склон — соседние '
                           'пиксели чуть светлее порога провала (или чуть темнее порога '
                           'пересвета). Нужна, чтобы починка не брала образец со склона. На тёмных '
                           'текстурах склон — это уже сам тёмный камень, и починка на него '
                           'наезжает: тогда ставьте 0. «Отступ маски» с минусом этого не убирает — '
                           'он только выбрасывает узкие места вроде швов между камнями.',
                           'How far the mask creeps from a found spot over its slope — the '
                           'neighbouring pixels a little lighter than the crush level (or a little '
                           'darker than the blow-out level). It keeps the repair from taking its '
                           'sample from the slope. On dark textures the slope is the dark stone '
                           'itself, and the repair runs onto it: then set 0. A negative «Mask '
                           'margin» does not remove it — it only drops narrow parts such as the '
                           'joints between stones.'),
    'np.mix': ('Насколько заменять', 'How much to replace'),
    'np.method': ('способ', 'method'),
    'np.soft': ('Плавная засветка', 'Smooth gradient'),
    'np.hard': ('Резкие тени', 'Hard shadows'),
    'np.radius': ('Радиус, пикс.', 'Radius, px'),
    'np.sat_keep': ('Удержать цветность', 'Keep colourfulness'),
    'np.detail': ('Мелкость зерна', 'Grain size'),
    'np.mode': ('способ', 'mode'),
    'np.az': ('Откуда светит, °', 'Light direction, °'),
    'np.smooth': ('Сглаживание поперёк света', 'Smoothing across the light'),
    'np.chroma': ('Убрать краску по цвету', 'Remove paint by colour'),
    'np.shadow': ('Убрать резкие тени', 'Remove cast shadows'),
    'np.model': ('модель', 'model'),
    'np.fine': ('Мелкий масштаб', 'Fine scale'),
    'np.tile': ('Размер окна', 'Window size'),
    'np.cutoff': ('Не верить крупнее', 'Distrust above'),
    'np.invert': ('Перевернуть', 'Invert'),
    'np.scale': ('Граница масштабов', 'Scale boundary'),
    'np.low': ('Крупное', 'Large scale'),
    'np.high': ('Мелкое', 'Fine scale'),
    'np.gamma': ('Гамма', 'Gamma'),
    'np.alpha': ('альфа', 'alpha'),
    'np.alpha_cut': ('Порог альфы', 'Alpha threshold'),
    'np.alpha_cut.info': ('Где альфа ниже порога — фон: он не считается впадиной '
                          'и остаётся без затенения. Работает, если подключён вход «альфа».',
                          'Where alpha is below this, it is background: not counted as a pit '
                          'and left unshaded. Works when the «alpha» input is connected.'),
    'np.contrast': ('Контраст', 'Contrast'),
    'np.wa': ('Вес А', 'Weight of A'),
    'np.wb': ('Вес Б', 'Weight of B'),
    'np.strength': ('Сила', 'Strength'),
    'np.amp': ('Размах высоты', 'Height amplitude'),
    'np.mid': ('Средний масштаб', 'Medium scale'),
    'np.big': ('Крупный масштаб', 'Large scale'),
    'np.directx': ('DirectX (зелёный вниз)', 'DirectX (green down)'),
    'np.steep': ('Крутизна', 'Steepness'),
    'np.fix': ('Подогнать под высоту', 'Fit to a height'),
    'np.blue': ('Синий канал',
                'Blue channel'),
    'np.swap': ('Поменять R и G',
                'Swap R and G'),
    'np.amount': ('Сколько подмешать', 'How much to blend in'),
    'np.f0': ('Отражение в лоб (F0)', 'Head-on reflection (F0)'),
    'np.gloss': ('Гладкость', 'Glossiness'),
    'np.gloss_var': ('Разброс гладкости', 'Gloss variation'),
    'np.tint': ('Цвет блика от текстуры', 'Highlight tinted by the texture'),
    'np.band': ('Ширина склейки, пикс.', 'Blend width, px'),
    'np.name': ('имя файла', 'file name'),

    'ui.all': ('все настройки', 'all settings'),
    'ui.all.info': ('Без галочки видно только то, что крутят почти всегда. С галочкой — '
                    'всё, включая тонкие настройки, которые нужны редко и мешают '
                    'смотреть.',
                    'Unticked, only what gets turned nearly every time is shown. Ticked '
                    '— everything, including the fine settings that are rarely needed '
                    'and get in the way.'),
    'ui.more': ('…и ещё %d настройки тут скрыто — нажмите, чтобы показать',
                '…and %d more settings hidden here — click to show'),
    'ui.fold': ('свернуть всё', 'fold all'),
    'ui.stage.src': ('ЭТАП 1 — ИСХОДНИК', 'STAGE 1 — THE SOURCE'),
    'ui.stage.relief': ('ЭТАП 2 — ОТКУДА РЕЛЬЕФ', 'STAGE 2 — WHERE RELIEF COMES FROM'),
    'ui.stage.maps': ('ЭТАП 3 — КАРТЫ', 'STAGE 3 — THE MAPS'),
    'ui.stage.zones': ('ЭТАП 4 — ЗОНЫ И ПРАВКА', 'STAGE 4 — ZONES AND EDITING'),
    'ui.stage.out': ('ЭТАП 5 — ВЫВОД', 'STAGE 5 — OUTPUT'),
    'm.repair': ('Битые участки', 'Damaged areas'),
    'm.repair.info': ('Пересвеченное в белое и провалившееся в чёрное сигнала уже не '
                      'несёт: рельефа там нет, и любая карта, построенная по такому '
                      'месту, будет выдумкой. Здесь эти места находятся по яркости '
                      'вместе с отсутствием фактуры и восстанавливаются.',
                      'Highlights clipped to white and shadows crushed to black carry '
                      'no signal: there is no relief left there, and any map built from '
                      'such a spot is invention. Here those places are found by '
                      'brightness together with missing detail, and restored.'),
    'm.rep.on': ('чинить перед расчётом карт', 'repair before building the maps'),
    'm.rep.hi': ('Порог пересвета', 'Highlight threshold'),
    'm.rep.lo': ('Порог провала', 'Shadow threshold'),
    'm.rep.flat': ('Порог «фактуры не осталось»', 'Flatness threshold'),
    'm.rep.flat.info': ('Сколько фактуры может остаться в месте, чтобы оно считалось битым. '
                        'Светлое или тёмное место с рисунком — просто светлое или тёмное, '
                        'битое — ровное. Чем правее, тем больше берётся; у правого края '
                        'решает одна яркость, как выделение через кривые.',
                        'How much texture a spot may keep and still count as damaged. A '
                        'bright or dark area with detail is just bright or dark; a damaged '
                        'one is featureless. Further right takes more; near the right end '
                        'only brightness decides, like a selection made with Curves.'),
    'm.rep.grow': ('Запас по краю, пикс.', 'Margin, px'),
    'm.rep.grow.info': ('Плюс — расширить маску на столько пикселей. Минус — убрать из '
                        'маски всё, что уже примерно двух таких отступов, а остальное не '
                        'трогать: швы между камнями — тонкие линии, выбитое место — '
                        'широкое, поэтому −4…−8 оставляет швы и чинит выбоины.',
                        'Plus grows the mask by that many pixels. Minus removes from the '
                        'mask everything narrower than about twice that and leaves the rest '
                        'as is: joints between stones are thin lines, a broken-out spot is '
                        'wide, so −4…−8 keeps the joints and repairs the spots.'),
    'm.rep.now': ('Починить', 'Repair'),
    'm.rep.savedif': ('Диффуз без света…', 'Diffuse without light…'),
    'm.rep.mix': ('Насколько заменять', 'How much to replace'),
    'm.rep.mix.info': ('Смесь оригинала и починенного. На единице битое место заменяется '
                       'целиком, на половине только притушивается — иногда нужно именно '
                       'это: пересвет убрать, а рисунок оставить.',
                       'A blend of the original and the repair. At one the damaged place '
                       'is replaced outright, at a half it is only toned down — which is '
                       'sometimes exactly what is wanted: take the blow-out away and keep '
                       'the drawing.'),
    'm.rep.method': ('Способ', 'Method'),
    'm.rep.tone': ('выровнять тон', 'level the tone'),
    'm.rep.tone.info': ('Участок не выдумывается заново, а приводится по уровню '
                        'к соседям. Ничего не сочиняет, но и фактуру не возвращает — '
                        'если её там не осталось, останется пятно.',
                        'The area is not invented, only brought to the level of its '
                        'surroundings. Makes nothing up, but restores no detail either: '
                        'if none is left, a flat patch is what you get.'),
    'm.rep.smooth': ('гладкая заливка', 'smooth fill'),
    'm.rep.smooth.info': ('Бигармоническая интерполяция — самая гладкая поверхность, '
                          'которая сходится с краями дырки. Идеальна для градиентов '
                          'и ровных поверхностей, фактуру не восстанавливает.',
                          'Biharmonic interpolation — the smoothest surface that meets '
                          'the edges of the hole. Ideal for gradients and flat '
                          'surfaces; restores no texture.'),
    'm.rep.patch': ('перенести заплатку', 'transfer a patch'),
    'm.rep.patch.info': ('Во всей текстуре ищется место, чьё окружение лучше всего '
                         'продолжает окружение дырки, и содержимое переносится оттуда. '
                         'Разница по краю снимается подгонкой, поэтому шва не видно. '
                         'Это единственный способ, который возвращает фактуру.',
                         'The whole texture is searched for the place whose surroundings '
                         'best continue the surroundings of the hole, and its content is '
                         'moved in. The edge difference is fitted away, so no seam shows. '
                         'This is the only method that brings detail back.'),
    'm.rep.found': ('найдено: пересвет %.1f %%, провал %.1f %%',
                    'found: %.1f %% blown, %.1f %% crushed'),
    'm.rep.auto': ('искать автоматически', 'find automatically'),
    'm.rep.off': ('— но починка выключена, включите галочку сверху',
                  '— but repairing is off, tick the box above'),
    'm.rep.hand': ('добавлять кистью', 'add with the brush'),

    'm.delight': ('Снять свет со съёмки', 'Remove baked lighting'),
    'm.soft': ('Плавная засветка', 'Smooth gradient'),
    'm.soft.info': ('Общий перепад яркости по кадру — от лампы, от неба, от '
                    'виньетирования объектива.',
                    'The overall brightness gradient across the frame — from a lamp, '
                    'from the sky, from lens vignetting.'),
    'm.hard': ('Резкие тени от солнца', 'Hard sun shadows'),
    'm.hard.info': ('Оценка света строится краесохраняющим фильтром, поэтому граница '
                    'тени не оставляет ореола.',
                    'The lighting estimate is built with an edge-preserving filter, so '
                    'the shadow boundary leaves no halo.'),
    'm.lrad': ('Масштаб света, пикс.', 'Lighting scale, px'),
    'm.rest': ('Вернуть зерно', 'Restore grain'),
    'm.rest.info': ('Мелкая фактура с оригинала обратно в очищенную картинку. '
                    'Подмешивается умножением, а не наложением сверху, поэтому цвет '
                    'не уезжает. Больше 1 — уже подчёркивание.',
                    'Fine detail from the original returned to the cleaned image. Mixed '
                    'in by multiplication rather than overlay, so the colour does not '
                    'drift. Above 1 it becomes emphasis.'),

    'm.hsrc': ('Откуда брать рельеф', 'Where relief comes from'),
    'm.hsrc.info': ('Четыре способа достать высоту из яркости, у каждого своя болезнь. '
                    'Складываются как слои с весами. Пятый слой — внешняя карта '
                    'глубины, он в своём блоке ниже.',
                    'Four ways to get height out of brightness, each with its own '
                    'weakness. They are mixed as weighted layers. The fifth layer is an '
                    'external depth map, in its own block below.'),
    'm.hdet': ('Мелкость зерна', 'Grain size'),
    'm.hdet.info': ('Что считать шумом и краской, а что рельефом.',
                    'Where the line falls between noise and paint on one side and '
                    'relief on the other.'),
    'm.wband': ('Полосовой', 'Band-pass'),
    'm.wband.info': ('Только мелкий рельеф. Крупной формы нет: широкая щель внутри '
                     'остаётся серой, тёмным выходит только её край.',
                     'Fine relief only. No large form: a wide gap stays grey inside and '
                     'only its edge goes dark.'),
    'm.wmulti': ('Многомасштабный', 'Multi-scale'),
    'm.wmulti.info': ('То же на нескольких размерах со спадающим весом. Крупная форма '
                      'возвращается, мелочь не тонет. Обычно основной слой.',
                      'The same at several scales with falling weight. Large form comes '
                      'back and fine detail survives. Usually the main layer.'),
    'm.wlum': ('Яркость целиком', 'Raw brightness'),
    'm.wlum.info': ('Максимум крупной формы, но и краска уходит в рельеф: тёмный '
                    'камень станет ямой.',
                    'Maximum large form, but paint becomes relief too: a dark stone '
                    'turns into a pit.'),
    'm.wguid': ('Краесохраняющий', 'Edge-preserving'),
    'm.wguid.info': ('Яркость без зерна, но с резкими границами областей. Широкая '
                     'щель остаётся глубокой по всей ширине.',
                     'Brightness with the grain removed but region boundaries kept '
                     'sharp. A wide gap stays deep across its whole width.'),
    'm.fall': ('Доля крупного', 'Weight of large scales'),
    'm.grad': ('Радиус краесохраняющего', 'Edge-preserving radius'),

    'm.sfs': ('По освещению на текстуре', 'From the lighting in the texture'),
    'm.sfs.info': ('Остальные слои читают яркость как высоту: где темнее — там ниже. '
                   'Но у освещённой поверхности яркость — это не высота, а УКЛОН: при '
                   'свете с известного направления она с точностью до постоянной равна '
                   'производной высоты вдоль света. Значит высоту надо не брать из '
                   'яркости, а интегрировать её вдоль луча.\n\n'
                   'Разница видна на любой борозде. У слоя «яркость целиком» тёмная '
                   'сторона борозды становится ямой, а светлая — горкой, и борозда '
                   'выходит ступенькой. При интегрировании обе стороны складываются в '
                   'одну яму, как оно и есть.\n\n'
                   'Работает тем лучше, чем честнее снимок: нужен один внятный источник '
                   'света. На рисованной текстуре врёт там же, где врёт яркость.',
                   'The other layers read brightness as height: darker means lower. But '
                   'on a lit surface brightness is not height, it is SLOPE: under light '
                   'from a known direction it equals, up to a constant, the derivative '
                   'of height along the light. So height should not be taken from '
                   'brightness, it should be integrated along the ray.\n\n'
                   'The difference shows on any groove. With the raw brightness layer '
                   'the dark side of a groove becomes a pit and the bright side a ridge, '
                   'and the groove comes out as a step. Integrated, both sides add up to '
                   'one pit, which is what is actually there.\n\n'
                   'The more honest the photograph, the better it works: it needs one '
                   'clear light source. On a painted texture it lies where brightness '
                   'lies.'),
    'm.sfs.w': ('Вес слоя от освещения', 'Weight of the lighting layer'),
    'm.sfs.w.info': ('Смешивается с остальными слоями высоты как обычный слой. Имеет '
                     'смысл ставить основным на фотографических текстурах со внятным '
                     'светом, и оставлять нулём там, где свет нарисован от руки.',
                     'Mixes with the other height layers like any other. Worth making '
                     'the main layer on photographic textures with clear lighting, and '
                     'leaving at zero where the light was painted by hand.'),
    'm.sfs.az': ('Откуда светит, °', 'Light direction, °'),
    'm.sfs.az.info': ('Ноль — свет справа, 90 — сверху, 180 — слева, 270 — снизу. '
                      'Ошибка на 180° переворачивает рельеф: выпуклости становятся '
                      'ямами. Это не изъян расчёта — по одному снимку выпуклость от '
                      'ямы неотличима в принципе, ровно та же иллюзия кратера, что на '
                      'фотографиях Луны.',
                      'Zero is light from the right, 90 from above, 180 from the left, '
                      '270 from below. Being 180° out turns the relief inside out: bumps '
                      'become pits. That is not a flaw in the computation — from a '
                      'single image a bump cannot be told from a pit at all, the same '
                      'crater illusion as in photographs of the Moon.'),
    'm.sfs.smooth': ('Сглаживание поперёк света', 'Smoothing across the light'),
    'm.sfs.smooth.info': ('Поперёк луча рельефа в картинке нет: вдоль борозды, '
                          'повёрнутой ребром к свету, яркость не меняется. Это место '
                          'нечем заполнить, кроме предположения о гладкости. Мало — '
                          'по карте идут полосы вдоль света, много — карта мылится.',
                          'Across the ray there is no relief in the picture: along a '
                          'groove turned edge-on to the light, brightness does not '
                          'change. There is nothing to fill that with but an assumption '
                          'of smoothness. Too little and the map streaks along the '
                          'light, too much and it goes soft.'),
    'm.sfs.chroma': ('Убрать краску по цвету', 'Remove paint by colour'),
    'm.sfs.chroma.info': ('Чистая логика «яркость = уклон» верна только для одноцветной '
                          'поверхности. Спасает то, что тень множит все три канала на '
                          'одно и то же число: в тени камень такой же серый, только '
                          'темнее. Значит затенение меняет яркость и НЕ меняет '
                          'цветность, а краска меняет как раз цветность — мох зеленее '
                          'камня, ржавчина краснее.\n\n'
                          'Ползунок вычитает ту часть перепадов яркости, которую '
                          'объясняют отношения каналов, то есть краску. На одноцветной '
                          'текстуре предсказывать нечем, и он не делает ничего: по '
                          'серой картинке краску от тени не отличить в принципе. '
                          'Поднимать имеет смысл там, где цветных пятен много.',
                          'The plain «brightness equals slope» logic holds only on a '
                          'surface of one colour. What saves it: shadow multiplies all '
                          'three channels by the same number — in shadow a stone is just '
                          'as grey, only darker. So shading changes brightness and does '
                          'NOT change chromaticity, while paint changes exactly that — '
                          'moss is greener than stone, rust is redder.\n\n'
                          'The slider subtracts the part of the brightness variation '
                          'that the channel ratios explain, which is the paint. On a '
                          'single-coloured texture there is nothing to predict with and '
                          'it does nothing: on a grey picture paint cannot be told from '
                          'shadow at all. Worth raising where there are many coloured '
                          'patches.'),
    'm.sfs.shadow': ('Убрать резкие тени', 'Remove cast shadows'),
    'm.sfs.shadow.info': ('Вторая беда чистой логики — падающие тени: перепад на кромке '
                          'тени это не уклон, и проинтегрировать его значит получить '
                          'обрыв на ровном месте. Оценка теней берётся та же, что на '
                          'вкладке «снять свет со съёмки»: краесохраняющий фильтр '
                          'сглаживает фактуру, но не переходит через резкую границу, '
                          'поэтому в оценке остаются именно тени. Её радиус задаётся '
                          'ползунком «масштаб света» там же.\n\n'
                          'Если резкие тени уже сняты на вкладке света (ползунок '
                          '«резкие тени от солнца» не на нуле), поле берётся прямо '
                          'оттуда — как разница оригинала и того, что показано как '
                          '«диффуз без света». Так вычитается ровно то, что вы там '
                          'настроили, а не второе независимое мнение о тех же тенях.\n\n'
                          'На проверке с падающей тенью эта поправка даёт больше всех '
                          'остальных: без неё связь с истинным рельефом 0.50, с ней '
                          '0.70.',
                          'The second trouble with the plain logic is cast shadows: the '
                          'step at a shadow edge is not a slope, and integrating it '
                          'produces a cliff where the surface is flat. The shadow '
                          'estimate is the same one the de-lighting tab uses: an '
                          'edge-preserving filter smooths texture but does not cross a '
                          'sharp border, so what stays in the estimate is the shadows. '
                          'Its radius is the «light scale» slider there.\n\n'
                          'If cast shadows are already being removed on the lighting '
                          'tab (the «hard sun shadows» slider is off zero), the field is '
                          'taken straight from there — as the difference between the '
                          'original and what is shown as «diffuse without light». That '
                          'way what gets subtracted is exactly what you set there, not a '
                          'second independent opinion about the same shadows.\n\n'
                          'In the test with a cast shadow this correction gives more '
                          'than all the others: without it the match to the true relief '
                          'is 0.50, with it 0.70.'),
    'm.sfs.scale': ('Граница масштабов', 'Scale boundary'),
    'm.sfs.scale.info': ('Где для этого слоя проходит граница между крупным и мелким. '
                         'Интегрирование вдоль света раздувает низкие частоты, поэтому '
                         'разделить диапазоны тут полезно так же, как на карте от '
                         'нейросети.',
                         'Where the boundary between large and fine runs for this layer. '
                         'Integration along the light amplifies low frequencies, so '
                         'splitting the bands helps here as much as it does on the '
                         'network map.'),
    'm.sfs.low': ('Крупное', 'Large scale'),
    'm.sfs.low.info': ('Вес крупного в этом слое. Убавить стоит, если после '
                       'интегрирования поплыла общая форма: у пологих неровностей '
                       'делитель мал, и они выходят сильнее, чем есть.',
                       'The weight of the large band in this layer. Worth lowering when '
                       'the overall shape drifts after integration: for gentle '
                       'undulations the divisor is small and they come out stronger than '
                       'they are.'),
    'm.sfs.high': ('Мелкое', 'Fine scale'),
    'm.sfs.high.info': ('Вес мелкого — фактуры, которая читается вблизи и живёт в '
                        'нормали.',
                        'The weight of the fine band — the texture that reads up close '
                        'and lives in the normal map.'),
    'm.sfs.direct': ('Доля нормали по освещению',
                     'Share of the lighting normal'),
    'm.sfs.direct.info': ('0 — обычная нормаль из высоты, 1 — только нормаль по освещению, между '
                          'ними — смесь.',
                          '0 is the ordinary normal from height, 1 is the lighting normal alone, '
                          'anything between is a blend.'),
    'm.sfs.fix': ('Подогнать под высоту', 'Fit to a height'),
    'm.sfs.fix.info': ('Поле наклонов, собранное не из высоты, никакой поверхности не '
                       'отвечает. Починить это можно ровно одним способом: спроецировать '
                       'его на ближайшее поле, у которого высота есть. После этого ротор '
                       'равен нулю точно, а на единице высотой становится та, что этой '
                       'нормали и отвечает, — параллакс и AO наконец описывают ту же '
                       'поверхность.\n\n'
                       'Бесплатно это не даётся. Проекция оставляет только ту часть '
                       'поля, которая вообще может принадлежать поверхности: по одной '
                       'оси остаётся около четверти, по другой три четверти. Карта '
                       'становится мягче и глаже, и та резкость, ради которой всё '
                       'затевалось, сидит как раз в выброшенном. Поэтому ползунок, а не '
                       'галочка: меняете согласованность на вид.',
                       'A slope field not built from a height corresponds to no surface. '
                       'There is exactly one way to fix that: project it onto the '
                       'nearest field that does have a height. The curl is then exactly '
                       'zero, and at one the height becomes the one that matches this '
                       'normal — parallax and AO finally describe the same surface.\n\n'
                       'It is not free. The projection keeps only the part of the field '
                       'that can belong to a surface at all: about a quarter along one '
                       'axis, three quarters along the other. The map gets softer and '
                       'smoother, and the crispness the whole thing was for sits '
                       'precisely in what was thrown away. Hence a slider and not a tick '
                       'box: you trade consistency for looks.'),
    'm.sfs.direct.k': ('Сила',
                       'Strength'),
    'm.sfs.direct.k.info': ('Множитель на R и G. При 1 серое превью слоя 0…1 ложится в канал как '
                            'есть, без пересчёта. Больше единицы — круче, самые яркие и тёмные '
                            'места начинают ложиться почти плашмя.',
                            "A multiplier on R and G. At 1 the layer's grey preview, 0…1, goes into "
                            'the channel as it is. Above one is steeper, and the brightest and '
                            'darkest spots start to lie almost flat.'),
    'm.sfs.direct.swap': ('поменять R и G местами',
                          'swap R and G'),
    'm.sfs.direct.swap.info': ('Какое из двух полей куда класть — вопрос без верного '
                               'ответа: под одним светом настоящей высотой оказывается '
                               'одно, под другим второе. Если рельеф лёг развёрнутым на '
                               'девяносто градусов, переключите.',
                               'Which of the two fields goes where has no right answer: '
                               'under one light the real height is one of them, under '
                               'another it is the other. If the relief came out turned '
                               'by ninety degrees, switch this.'),
    'm.sfs.auto': ('Определить по картинке', 'Work it out from the picture'),
    'm.sfs.found': ('получилось %.0f°, уверенность %.0f %%. Проверьте: если рельеф '
                    'вывернут, прибавьте 180°',
                    'came out at %.0f°, confidence %.0f %%. Check it: if the relief is '
                    'inside out, add 180°'),
    'm.ext.cav': ('это карта затенения, проинтегрировать',
                  'this is an occlusion map, integrate it'),
    'm.ext.cav.info': ('Если во внешнем слое лежит AO, cavity или «карта щелей», то '
                       'тёмное в ней — не низкое место, а вогнутое. Затенение это с '
                       'точностью до множителя вторая производная высоты, поэтому её '
                       'надо проинтегрировать дважды — и тогда тёмная линия шва даёт '
                       'борозду с настоящим профилем, а не стенку той ширины, какой '
                       'нарисована линия. Кирпичи при этом выходят ровными площадками '
                       'на одной высоте.',
                       'If the external layer holds an AO, cavity or «crevice» map, dark '
                       'in it means concave, not low. Occlusion is, up to a factor, the '
                       'second derivative of height, so it has to be integrated twice — '
                       'and then a dark joint line gives a groove with a real profile '
                       'instead of a wall as wide as the line was drawn. The bricks come '
                       'out as flat plateaus at one level.'),
    'm.ext.cut': ('Не верить крупнее', 'Distrust above'),
    'm.ext.cut.info': ('Двойное интегрирование раздувает самые низкие частоты: любая '
                       'пологая неровность, случайно попавшая в карту, станет горой '
                       'через всю текстуру. Ползунок — доля стороны, начиная с которой '
                       'крупному в карте затенения не верят.',
                       'Double integration blows up the lowest frequencies: any gentle '
                       'unevenness that happened to be in the map would become a hill '
                       'across the whole texture. The slider is the fraction of the side '
                       'above which the large scale in the occlusion map is not '
                       'trusted.'),

    'm.ext': ('Внешняя карта глубины', 'External depth map'),
    'm.ext.info': ('Готовая глубина со стороны: MiDaS, скан, рисованная руками, или '
                   'снятая с текущей высоты. Она не выводится из яркости картинки, '
                   'поэтому знает то, чего яркость знать не может. Размер подгоняется '
                   'автоматически.',
                   'Depth from elsewhere: MiDaS, a scan, hand-painted, or lifted from '
                   'the current height. It is not derived from image brightness, so it '
                   'knows what brightness cannot. Resized automatically.'),
    'm.ext.none': ('не загружена', 'not loaded'),
    'm.ext.load': ('Загрузить…', 'Load…'),
    'm.ext.fromh': ('Из высоты', 'From height'),
    'm.ext.fromh.info': ('Снять текущую расчётную высоту во внешний слой, чтобы дальше '
                         'править её кистью, не трогая то, из чего она получена.',
                         'Lift the current computed height into the external layer so it '
                         'can be brushed without disturbing what produced it.'),
    'm.ext.save': ('Сохранить…', 'Save…'),
    'm.ext.drop': ('Убрать', 'Remove'),
    'm.ext.chan': ('канал:', 'channel:'),
    'm.ext.chan.info': ('Из какого канала файла брать глубину. Пригодится, когда карта '
                        'лежит в одном канале чужого файла — например высота в альфе '
                        'нормали (_nh).',
                        'Which channel of the file to read depth from. Useful when the '
                        'map lives in one channel of another file — height in the alpha '
                        'of a normal map (_nh), for instance.'),
    'm.ext.w': ('Вес внешней глубины', 'External depth weight'),
    'm.ext.taken': ('снята с текущей высоты', 'lifted from the current height'),
    'm.ext.hand': ('нарисована вручную', 'painted by hand'),
    'm.ext.brush': ('править кистью', 'edit with the brush'),
    'm.ext.brush.info': ('Левая кнопка поднимает — получается выступ, правая опускает — '
                         'получается углубление. Если внешней карты ещё нет, она '
                         'создаётся ровной серой при первом мазке.',
                         'Left button raises into a bump, right button lowers into a '
                         'recess. If there is no external map yet, a flat grey one is '
                         'created on the first stroke.'),
    'm.ext.force': ('Сила мазка', 'Stroke strength'),
    'm.ext.title': ('Карта глубины (серая)', 'Depth map (greyscale)'),
    'm.ext.saved': ('глубина сохранена: %s', 'depth saved: %s'),

    'd.net': ('Порядок листьев от сети', 'Leaf order from a network'),
    'd.net.info': ('Затенение у карточки держится на придуманной глубине: какой лист '
                   'ближе, решает случай. Сеть предсказания глубины решает это не '
                   'случайно — она видит, какая ветка выходит вперёд. Её ответ '
                   'усредняется по каждому куску разбиения и заменяет собой жребий, '
                   'после чего сам собой становится вернее и AO, и просвет, и силуэты '
                   'листьев позади.\n\n'
                   'Модель и подробность берутся те же, что на вкладке материала.',
                   'Occlusion on a card rests on invented depth: which leaf is in front '
                   'is decided by chance. A depth network decides it otherwise — it sees '
                   'which branch comes forward. Its answer is averaged over each piece '
                   'of the split and takes the place of the draw, after which AO, '
                   'translucency and the silhouettes of leaves behind all get better on '
                   'their own.\n\n'
                   'The model and the detail setting are the ones from the Material tab.'),
    'd.net.w': ('Доверие сети', 'Trust in the network'),
    'd.net.w.info': ('На нуле порядок листьев прежний, случайный. На единице он целиком '
                     'от сети. Середина обычно и нужна: сеть уверенно говорит, какая '
                     'ветка ближе, но внутри одной ветки соседние листья ей одинаковы, '
                     'и без разброса они снова лягут в одну плоскость.',
                     'At zero the leaf order is the old random one. At one it is '
                     'entirely the network\'s. The middle is usually what is wanted: the '
                     'network is confident about which branch is closer, but within one '
                     'branch neighbouring leaves look the same to it, and without some '
                     'scatter they would fall back into one plane.'),

    'm.nn': ('Глубина нейросетью', 'Depth from a network'),
    'm.nn.info': ('Сеть предсказания глубины по одному снимку. Её ответ кладётся во '
                  'внешнюю карту глубины, поэтому вес у него там же — ползунок «вес '
                  'внешней глубины» в разделе выше.',
                  'A network that predicts depth from a single image. Its answer goes '
                  'into the external depth map, so its weight lives there too — the '
                  '«external depth weight» slider in the section above.'),
    'm.nn.model': ('модель:', 'model:'),
    'm.nn.model.info': ('Сеть предсказания глубины по одному снимку. Её учили на парах '
                        '«фотография — измеренная глубина», поэтому она отвечает на '
                        'вопрос «что тут ближе», а не «что тут светлее», — то есть '
                        'знает ровно то, чего яркость знать не может: что кирпич '
                        'выступает, а шов между кирпичами утоплен.\n\n'
                        'Depth Anything V2 точнее держит границы и почти всегда лучше '
                        'на текстурах. MiDaS вдвое меньше и вдвое быстрее, но видит в '
                        'основном общий наклон.',
                        'A network that predicts depth from a single image. It was '
                        'trained on pairs of photograph and measured depth, so it '
                        'answers «what is closer here» rather than «what is brighter '
                        'here» — which is exactly what brightness cannot know: that a '
                        'brick stands out and the mortar between bricks is sunk.\n\n'
                        'Depth Anything V2 keeps edges far better and is almost always '
                        'the better choice on textures. MiDaS is half the size and twice '
                        'as fast, but mostly sees the overall tilt.'),
    'm.nn.fine': ('подробно, окнами', 'detailed, by windows'),
    'm.nn.fine.info': ('Без галочки текстура уходит в сеть целиком одним проходом: '
                       'доли секунды и верная крупная форма. С галочкой она режется на '
                       'перекрывающиеся окна, и каждое сеть рассматривает вблизи — '
                       'мелкие детали появляются, но проходов становятся десятки.',
                       'Unchecked, the texture goes through the network whole in a '
                       'single pass: a fraction of a second and the large shape right. '
                       'Checked, it is cut into overlapping windows and the network '
                       'examines each one close up — small detail appears, but there '
                       'are dozens of passes.'),
    'm.nn.tile': ('Размер окна', 'Window size'),
    'm.nn.tile.info': ('Доля стороны текстуры. Мелкое окно — сеть видит один кирпич '
                       'крупно и разбирает его подробно; крупное — видит кладку и '
                       'отвечает про неё. Считается только при подробном режиме.',
                       'A fraction of the texture side. A small window shows the network '
                       'one brick up close and it works out the detail; a large one '
                       'shows the courses and it answers about those. Used only in the '
                       'detailed mode.'),
    'm.nn.scale': ('Граница масштабов', 'Scale boundary'),
    'm.nn.scale.info': ('Где проходит граница между крупным и мелким. Слева крупным '
                        'считается почти всё, справа — только заметно большее кирпича. '
                        'У кирпичной стены в середине шкалы на крупное приходится '
                        '«этот кирпич выше соседнего», а на мелкое — «внутри кирпича '
                        'выбоина».',
                        'Where the boundary between large and fine runs. On the left '
                        'almost everything counts as large, on the right only what is '
                        'clearly bigger than a brick. On a brick wall, in the middle of '
                        'the range, the large band carries «this brick is higher than '
                        'its neighbour» and the fine one «there is a chip inside the '
                        'brick».'),
    'm.nn.low': ('Крупное', 'Large scale'),
    'm.nn.low.info': ('Вес крупного. Сеть меряет глубину сцены, а не рельеф '
                      'поверхности: на снимке стены дальний край честно дальше '
                      'ближнего, и на карте это выходит перекосом через всю текстуру. '
                      'На нуле от него не остаётся ничего и все кирпичи выходят на одну '
                      'высоту, сохраняя форму каждого; на единице крупное идёт как '
                      'есть.',
                      'The weight of the large band. The network measures scene depth, '
                      'not surface relief: in a photograph of a wall the far edge '
                      'honestly is further than the near one, and on the map that '
                      'becomes a tilt across the whole texture. At zero none of it is '
                      'left and every brick comes to the same level while keeping its '
                      'own shape; at one the large band goes through as it is.'),
    'm.nn.high': ('Мелкое', 'Fine scale'),
    'm.nn.high.info': ('Вес мелкого — того, что меньше границы масштабов. Это оно '
                       'читается вблизи и живёт в нормали, поэтому поднимать его выше '
                       'единицы имеет смысл, когда рельеф «есть, но не чувствуется».',
                       'The weight of the fine band — everything below the scale '
                       'boundary. This is what reads up close and lives in the normal '
                       'map, so raising it above one makes sense when the relief «is '
                       'there but is not felt».'),
    'm.nn.curve': ('Контраст перепадов', 'Transition contrast'),
    'm.nn.curve.info': ('Кривая поверх готовой карты, вокруг середины. На больших '
                        'значениях грань кирпича выходит на ровную площадку, а на стыках '
                        'остаются тёмные борозды. Получается уже не столько глубина, '
                        'сколько карта щелей — и как раз она хорошо ложится в AO: '
                        'высота отсюда идёт в AO дальше по цепочке.',
                        'A curve over the finished map, around the middle. At high '
                        'values a brick face comes out as a flat plateau and dark '
                        'grooves stay at the joints. That is less a depth map than a map '
                        'of crevices — and it is exactly what AO wants: height from here '
                        'feeds AO further down the chain.'),
    'm.nn.run': ('Посчитать', 'Estimate'),
    'm.esr': ('Нейросети ESRGAN: нормаль и франкен-карта',
              'ESRGAN networks: normal and franken map'),
    'm.esr.info': ('Две сети Joey, обученные на текстурах CC0. NormalMapGenerator рисует '
                   'карту нормалей прямо по текстуре. FrankenMapGenerator — '
                   '«франкен-карту»: в одном файле высота (канал B), шероховатость (G) и '
                   'текстура в сером (R).\n\nСначала нажмите «Посчитать» — сети прогоняются '
                   'один раз, ответ запоминается, и ползунки ниже работают сразу. Что сеть '
                   'дала сама по себе, видно на видах «нейросеть: …» справа.\n\nМодели в '
                   'формате PyTorch; программа читает их без PyTorch и один раз переводит '
                   'в ONNX. На процессоре счёт долгий — от минуты на текстуру 1024 и '
                   'больше, с видеокартой в разы быстрее.',
                   'Two networks by Joey, trained on CC0 textures. NormalMapGenerator '
                   'draws a normal map straight from the texture. FrankenMapGenerator '
                   'draws a "franken map": height (channel B), roughness (G) and the '
                   'texture in grey (R) in one file.\n\nPress "Compute" first — the networks '
                   'run once, the answer is kept, and the sliders below work instantly. '
                   'What the network gave on its own is visible in the "network: …" views '
                   'on the right.\n\nThe models are PyTorch files; the program reads them '
                   'without PyTorch and converts them to ONNX once. On a CPU it is slow — '
                   'a minute or more for a 1024 texture; several times faster with a '
                   'graphics card.'),
    'm.esr.lic': ('Лицензия моделей — CC-BY-NC-SA-4.0 (автор Joey): только некоммерческое '
                  'использование. В программу они не входят и скачиваются отдельно.',
                  'The models are licensed CC-BY-NC-SA-4.0 (by Joey): non-commercial use '
                  'only. They are not part of the program and are downloaded separately.'),
    'm.esr.n.on': ('считать нормаль (NormalMapGenerator)',
                   'compute the normal (NormalMapGenerator)'),
    'm.esr.f.on': ('считать франкен-карту (FrankenMapGenerator)',
                   'compute the franken map (FrankenMapGenerator)'),
    'm.esr.run': ('Посчитать',
                  'Compute'),
    'm.esr.file': ('Файл модели…',
                   'Model file…'),
    'm.esr.nmix': ('Доля нормали от сети',
                   'Share of the network normal'),
    'm.esr.nmix.info': ('0 — нормаль программы, 1 — только нормаль от сети. Смешивается до '
                        'нормали по освещению (R и G) и до bent normal.',
                        "0 is the program's normal, 1 is the network normal alone. Blended "
                        'before the lighting normal (R and G) and before the bent normal.'),
    'm.esr.nk': ('Сила нормали от сети',
                 'Strength of the network normal'),
    'm.esr.nk.info': ('Множитель на наклон: больше 1 — круче, меньше — положе.',
                      'A multiplier on the tilt: above 1 is steeper, below is flatter.'),
    'm.esr.nflip': ('развернуть зелёный у нормали от сети',
                    'flip green on the network normal'),
    'm.esr.nflip.info': ('Если выпуклое у сети выходит вогнутым по вертикали — значит, '
                         'соглашение её обучающих карт другое. Сравните на виде «под светом» с '
                         'кругом света сверху.',
                         "If the network's bumps come out as dents vertically, its training maps "
                         'used the other convention. Compare in the "lit" view with the light '
                         'circle at the top.'),
    'm.esr.wh': ('Вес высоты из франкен-карты',
                 'Weight of the franken-map height'),
    'm.esr.wh.info': ('Ещё один слой в смеси высот, наравне со слоями из раздела «Откуда '
                      'брать рельеф».',
                      'One more layer in the height mix, on a par with the layers in "Where '
                      'the relief comes from".'),
    'm.esr.rough': ('Шероховатость из франкен-карты',
                    'Roughness from the franken map'),
    'm.esr.rough.info': ('0 — шероховатость программы, 1 — только от сети.',
                         "0 is the program's roughness, 1 is the network's alone."),
    'm.esr.k.normal': ('нормаль',
                       'normal'),
    'm.esr.k.franken': ('франкен-карта',
                        'franken map'),
    'm.esr.have': ('Модели на месте: %s. Папка моделей: %s',
                   'Models present: %s. Model folder: %s'),
    'm.esr.none': ('нет',
                   'none'),
    'm.esr.done': ('Посчитано для этой текстуры: %s, за %.0f с.',
                   'Computed for this texture: %s, in %.0f s.'),
    'm.esr.title': ('Нейросети ESRGAN',
                    'ESRGAN networks'),
    'm.esr.bad': ('Файл %s не читается как модель ESRGAN:\n\n%s',
                  'File %s cannot be read as an ESRGAN model:\n\n%s'),
    'm.esr.which': ('По имени файла %s не понять, какая это модель. В имени должно быть '
                    '«Normal» или «Franken».',
                    'Cannot tell from the name %s which model it is. The name should '
                    'contain "Normal" or "Franken".'),
    'm.esr.ask': ('Скачать %s?\n\nЛицензия: %s, автор %s — только некоммерческое '
                  'использование. Около 19 МБ на модель.',
                  'Download %s?\n\nLicence: %s, by %s — non-commercial use only. About 19 '
                  'MB per model.'),
    'm.esr.loading': ('%s: скачивается %.1f из %.1f МБ',
                      '%s: downloading %.1f of %.1f MB'),
    'm.esr.nodl': ('Не удалось скачать модель. Скачайте её вручную со страницы\n%s\nи '
                   'укажите кнопкой «Файл модели…» (или положите в %s).\n\n%s',
                   'Could not download the model. Download it by hand from\n%s\nand pick it '
                   'with "Model file…" (or put it in %s).\n\n%s'),
    'm.esr.convert': ('%s: перевод в ONNX (один раз)',
                      '%s: converting to ONNX (once)'),
    'm.esr.pass': ('%s: окно %d из %d',
                   '%s: window %d of %d'),
    'm.v.esrn': ('нейросеть: нормаль',
                 'network: normal'),
    'm.v.esrh': ('нейросеть: высота',
                 'network: height'),
    'm.v.esrr': ('нейросеть: шероховатость',
                 'network: roughness'),
    'm.nn.title': ('Глубина нейросетью', 'Depth by network'),
    'm.nn.ready': ('модель скачана', 'model downloaded'),
    'm.nn.need': ('нужно скачать %.0f МБ', '%.0f MB to download'),
    'm.nn.local': ('считается на вашем компьютере', 'runs on your computer'),
    'm.nn.dev.cpu': ('считается на процессоре', 'runs on the processor'),
    'm.nn.dev.gpu': ('считается на видеокарте (%s)', 'runs on the graphics card (%s)'),
    'm.nn.passes': ('проходов: %d', '%d passes'),
    'm.nn.ask.rt': ('Для счёта нужна библиотека onnxruntime — она запускает модель на '
                    'процессоре. Открытая, лицензия MIT, около 60 МБ. Поставить сейчас?',
                    'Estimating needs the onnxruntime library, which runs the model on '
                    'the processor. Open source, MIT licence, about 60 MB. Install it '
                    'now?'),
    'm.nn.ask.dl': ('Скачать модель %s — %.0f МБ, лицензия %s?\n\n'
                    'Она сохранится в домашней папке и больше скачиваться не будет.',
                    'Download the %s model — %.0f MB, %s licence?\n\n'
                    'It is kept in your home folder and will not be downloaded again.'),
    'm.nn.installing': ('ставлю onnxruntime…', 'installing onnxruntime…'),
    'm.nn.nort': ('Не получилось поставить onnxruntime.',
                  'Could not install onnxruntime.'),
    'm.nn.loading': ('загрузка модели: %.0f из %.0f МБ',
                     'downloading the model: %.0f of %.0f MB'),
    'm.nn.nodl': ('Не удалось скачать модель.\n\nЕё можно взять по ссылке\n%s\n'
                  'и положить файл как есть в папку\n%s\n\nПричина: %s',
                  'Could not download the model.\n\nIt can be taken from\n%s\n'
                  'and the file put as it is into\n%s\n\nReason: %s'),
    'm.nn.working': ('считаю глубину…', 'estimating depth…'),
    'm.nn.pass': ('проход %d из %d', 'pass %d of %d'),

    'nn.da2s': ('Depth Anything V2', 'Depth Anything V2'),
    'nn.midas': ('MiDaS v2.1', 'MiDaS v2.1'),

    'm.guide': ('Подсказка по высоте', 'Height guidance'),
    'm.guide.info': ('Не рисование значений поверх карты, а указание алгоритму. '
                     'Мазок кистью — это не сама высота, а метка «вот эта область»: '
                     'программа сама находит её границы по цвету и по контурам '
                     'картинки и поднимает или опускает ровно её, сохраняя внутри '
                     'весь найденный рельеф.',
                     'Not painting values over the map but pointing the algorithm at '
                     'something. A stroke is not the height itself, it is a marker '
                     'saying «this region»: the program finds its boundaries by colour '
                     'and by the contours of the image and raises or lowers exactly '
                     'that, keeping all the relief it found inside.'),
    'm.guide.brush': ('отмечать области кистью', 'mark regions with the brush'),
    'm.guide.brush.info': ('Левая кнопка отмечает область как поднимаемую, правая — '
                           'как опускаемую. Достаточно провести по ней грубо: границу '
                           'найдёт программа.',
                           'Left button marks a region to raise, right button to lower. '
                           'A rough stroke is enough; the boundary is found for you.'),
    'm.guide.tol': ('Разброс цвета области', 'Region colour tolerance'),
    'm.guide.tol.info': ('Насколько сильно цвет может отличаться от того, по чему '
                         'провели, и всё ещё считаться той же областью. Больше — '
                         'область шире.',
                         'How far a colour may differ from what was stroked and still '
                         'count as the same region. Larger means a wider region.'),
    'm.guide.cont': ('только связанное с мазком', 'only what the stroke touches'),
    'm.guide.cont.info': ('Область не перепрыгивает через контуры: берётся только то, '
                          'что соединено с мазком. Сняв галочку, получишь все похожие '
                          'места сразу по всей текстуре — например все швы кладки.',
                          'The region does not jump across contours: only what is '
                          'connected to the stroke is taken. Unticked, you get every '
                          'similar place across the whole texture at once — all the '
                          'mortar lines, for example.'),
    'm.guide.amt': ('Насколько поднять или опустить', 'How far to raise or lower'),
    'm.guide.flat': ('Выровнять внутри области', 'Flatten inside the region'),
    'm.guide.flat.info': ('Область становится площадкой на одном уровне: рельеф внутри '
                          'сглаживается, а не сдвигается. Для швов кладки, для '
                          'вытертых плит, для всего, что должно быть ровным.',
                          'The region becomes a level plateau: relief inside is '
                          'smoothed rather than shifted. For mortar lines, worn slabs, '
                          'anything that should be flat.'),
    'm.guide.inv': ('поменять поднять и опустить местами', 'swap raise and lower'),
    'm.guide.soft': ('Смягчить край, пикс.', 'Soften the edge, px'),

    'm.hshape': ('Высота: форма и предел', 'Height: shape and limit'),
    'm.tilt': ('Предел крутизны, °', 'Steepness limit, °'),
    'm.tilt.info': ('Если после всех множителей рельеф круче — карта пропорционально '
                    'прижимается. Иначе нормаль вылезает за 90° и появляются пиксели, '
                    'отвёрнутые от камеры.',
                    'If after all the multipliers the relief comes out steeper, the '
                    'whole map is scaled down proportionally. Otherwise the normal goes '
                    'past 90° and pixels end up facing away from the camera.'),
    'm.hinv': ('инвертировать высоту', 'invert height'),
    'm.hdelit': ('высоту считать по картинке со снятым светом',
                 'take height from the de-lit image'),
    'm.hdelit.info': ('По умолчанию высота берётся с ОРИГИНАЛА: на фотографии тень — '
                      'единственный след рельефа, и снимать её перед поиском формы '
                      'значит выбрасывать то, что ищешь. Включать имеет смысл на '
                      'рисованных текстурах, где свет наврал.',
                      'By default height comes from the ORIGINAL: in a photograph the '
                      'shading is the only trace of relief, and removing it before '
                      'looking for shape throws away what you are looking for. Worth '
                      'enabling on painted textures where the lighting lies.'),
    'm.hamp': ('Глубина рельефа, пикс.', 'Relief depth, px'),
    'm.hamp.info': ('Во сколько пикселей превращается полный размах высоты. Для '
                    'parallax это и есть сила эффекта.',
                    'How many pixels the full height range stands for. For parallax '
                    'mapping this is the strength of the effect.'),

    'm.normal': ('Нормаль', 'Normal map'),
    'm.normal.info': ('Нормаль строится из готовой высоты — той, что собрана блоками '
                      'выше, вместе с внешней глубиной и подсказкой. Из диффуза '
                      'напрямую она не берётся никогда.',
                      'The normal is built from the finished height — the one assembled '
                      'in the blocks above, together with external depth and guidance. '
                      'It is never taken from the diffuse directly.'),
    'm.nstr': ('Сила нормали', 'Normal strength'),
    'm.nscales.info': ('Нормаль складывается из трёх масштабов. Мелкий — зерно и '
                       'царапины, средний — форма кирпича, крупный — вся кладка.',
                       'The normal is assembled from three scales. Fine is grain and '
                       'scratches, medium is the shape of one brick, large is the whole '
                       'wall.'),
    'm.nfine': ('Мелкий рельеф', 'Fine relief'),
    'm.nmid': ('Средний', 'Medium'),
    'm.nbig': ('Крупный', 'Large'),
    'm.nscale': ('Размер среднего, пикс.', 'Medium scale, px'),
    'm.nscale.info': ('Крупный берётся вчетверо больше этого.',
                      'The large scale is four times this.'),
    'm.plx': ('Параллакс для OpenMW (_nh)',
              'Parallax for OpenMW (_nh)'),
    'm.plx.info': ('OpenMW читает высоту из альфы карты нормалей, файл с окончанием _nh. '
                   'Смещения нет при альфе 128; весь диапазон альфы — это 4 % длины одного '
                   'повтора текстуры, и отдельной настройки глубины в движке нет (так в '
                   'его шейдере, parallax.glsl). Значит, глубина в сантиметрах зависит от '
                   'того, сколько юнитов занимает повтор текстуры на модели, и управлять '
                   'ею можно только тем, что записано в альфу.\n\nЗдесь задаётся, какой '
                   'глубины рельеф нужен, и высота укладывается в альфу так, чтобы в игре '
                   'вышло именно столько. Результат виден на виде «параллакс (альфа _nh)» '
                   'и на вкладке «3D просмотр» в режиме «как в OpenMW».',
                   'OpenMW reads height from the alpha of the normal map, the file ending '
                   'in _nh. There is no shift at alpha 128; the whole alpha range is 4 % '
                   'of one texture repeat, and the engine has no separate depth setting '
                   '(that is how its shader, parallax.glsl, works). So the depth in '
                   'centimetres depends on how many units one texture repeat covers on the '
                   'model, and the only control is what is written into the alpha.\n\nHere '
                   'you set how deep the relief should be, and the height is packed into '
                   'the alpha so the game shows exactly that. The result is visible in the '
                   '"parallax (_nh alpha)" view and on the "3D preview" tab in "as in '
                   'OpenMW" mode.'),
    'm.plx.rep': ('Повтор текстуры, юнитов',
                  'Texture repeat, units'),
    'm.plx.rep.info': ('Сколько игровых юнитов занимает один повтор текстуры на модели. У стен '
                       'Morrowind обычно 128 или 256. Один юнит — 1.38 см (движок считает 22.1 '
                       'юнита на фут).',
                       'How many game units one repeat of the texture covers on the model. '
                       'Morrowind walls are usually 128 or 256. One unit is 1.38 cm (the '
                       'engine counts 22.1 units to a foot).'),
    'm.plx.dep': ('Глубина рельефа, см',
                  'Relief depth, cm'),
    'm.plx.dep.info': ('Полный размах: от самого глубокого места до самого высокого. Если '
                       'больше, чем движок может показать при таком повторе, — будет урезано '
                       'до предела, и об этом скажет подпись ниже.',
                       'The full range: from the deepest spot to the highest. If it is more '
                       'than the engine can show at this repeat, it is capped at the limit, '
                       'and the note below says so.'),
    'm.plx.ref': ('Что совпадёт с поверхностью модели (альфа 128)',
                  'What lines up with the model surface (alpha 128)'),
    'm.plx.ref.info': ('Какая высота ляжет ровно на геометрию, без смещения. Всё выше будет '
                       'выступать, всё ниже — уходить внутрь.\n\nСамый частый уровень — это '
                       'самая большая ровная площадь. На кладке это либо лицо камней, либо '
                       'заливка швов — смотря что занимает больше места; проверьте на виде '
                       '«параллакс»: то, что легло в 128, там ровного среднего серого цвета. '
                       'Среднее и медиана тянутся к швам, если их много.',
                       'Which height sits exactly on the geometry, unshifted. Everything above '
                       'sticks out, everything below sinks in.\n\nThe most common level is the '
                       'largest flat area. On masonry that is either the stone faces or the '
                       'joint filling — whichever takes up more room; check in the "parallax" '
                       'view: whatever landed on 128 is plain middle grey there. The mean and '
                       'the median drift towards the joints when there are many.'),
    'm.plx.r.mean': ('среднее',
                     'mean'),
    'm.plx.r.median': ('медиана',
                       'median'),
    'm.plx.r.mode': ('самый частый уровень',
                     'most common level'),
    'm.plx.r.manual': ('вручную',
                       'manual'),
    'm.plx.lev': ('Уровень вручную',
                  'Manual level'),
    'm.plx.lev.info': ('Доля от самого низкого места до самого высокого: 0 — всё выступает, 1 '
                       '— всё утоплено.',
                       'A fraction from the lowest spot to the highest: 0 means everything '
                       'sticks out, 1 means everything is sunk.'),
    'm.plx.dith': ('дизеринг перед записью',
                   'dither before writing'),
    'm.plx.dith.info': ('Добавляет упорядоченный шум в одну ступень альфы. Если рельеф '
                        'неглубокий, в альфе занято мало уровней и видны полосы; шум их '
                        'разбивает.',
                        'Adds ordered noise of one alpha step. When the relief is shallow few '
                        'alpha levels are used and banding shows; the noise breaks it up.'),
    'm.plx.norm': ('согласовать нормаль с глубиной',
                   'match the normal to the depth'),
    'm.plx.norm.info': ('Нормаль и параллакс описывают одну поверхность, но обычно задаются '
                        'независимо: сила нормали одним ползунком, глубина другим — и могут '
                        'друг другу противоречить. С этой галочкой нормаль считается из той же '
                        'высоты в тех же сантиметрах: размер пикселя известен из повтора и '
                        'ширины текстуры, и наклон выходит физический. Ползунки силы нормали '
                        'при этом гаснут — сила следует из глубины.',
                        'The normal and the parallax describe one surface, but are usually set '
                        "independently: the normal's strength with one slider, depth with "
                        'another — and they can contradict each other. With this ticked the '
                        'normal is computed from the same height in the same centimetres: the '
                        'pixel size is known from the repeat and the texture width, so the '
                        'slope is physical. The normal strength sliders go grey — strength '
                        'follows from depth.'),
    'm.plx.note': ('Повтор %.0f см. Движок может показать до %.1f см рельефа (±%.1f от '
                   'поверхности), один уровень альфы — %.2f мм. Получилось: выступает до '
                   '%.1f см, уходит вглубь до %.1f см. Занято уровней альфы: %d из 256.',
                   'Repeat %.0f cm. The engine can show up to %.1f cm of relief (±%.1f '
                   'from the surface), one alpha level is %.2f mm. Result: sticks out up '
                   'to %.1f cm, sinks up to %.1f cm. Alpha levels used: %d of 256.'),
    'm.plx.over': ('Запрошено %.1f см, а при таком повторе предел %.1f см — урезано до '
                   'предела.',
                   'Asked for %.1f cm, but at this repeat the limit is %.1f cm — capped at '
                   'the limit.'),
    'm.plx.clip': ('Не влезло и обрезано: %.1f %% пикселей (на виде «параллакс» — '
                   'красным). Сдвиньте уровень поверхности или уменьшите глубину.',
                   'Did not fit and clipped: %.1f %% of pixels (red in the "parallax" '
                   'view). Move the surface level or reduce the depth.'),
    'm.plx.few': ('Уровней мало — в игре будут видны ступеньки. Включите дизеринг или '
                  'увеличьте глубину.',
                  'Few levels — steps will show in the game. Turn on dithering or '
                  'increase the depth.'),
    'm.v.nh': ('параллакс (альфа _nh)',
               'parallax (_nh alpha)'),
    'v3.m.openmw': ('как в OpenMW (_nh)',
                    'as in OpenMW (_nh)'),
    'v3.m.openmw.info': ('Ровно формула движка: один шаг смещения по альфе, которая уйдёт в _nh, '
                         'с нейтралью 128. Глубина здесь не задаётся ползунком — она следует из '
                         'настроек раздела «Параллакс для OpenMW» на вкладке материала. Если в '
                         'игре стоит мод с другим шейдером параллакса (с перекрытием), смотрите '
                         'режим «с перекрытием».',
                         "Exactly the engine's formula: one offset step by the alpha that goes "
                         'into _nh, neutral at 128. Depth is not set by the slider here — it '
                         'follows from the "Parallax for OpenMW" section on the material tab. If '
                         'the game runs a mod with a different parallax shader (with occlusion), '
                         'look at the "with occlusion" mode.'),
    'm.ndir.head': ('Нормаль по освещению (R и G)',
                    'Normal from the lighting (R and G)'),
    'm.ndir.head.info': ('Слой «по освещению» считается дважды — со светом слева и со светом '
                         'снизу, — и два результата кладутся прямо в каналы R и G. Синий '
                         'досчитывается. Высота при этом не участвует вовсе.\n\nСлой берётся с '
                         'теми же настройками, что в разделе «Откуда брать рельеф → По освещению '
                         'на текстуре»: сглаживание, снятие краски и теней, граница масштабов, '
                         'крупное и мелкое. Поэтому то, что видно на превью того слоя, и '
                         'попадает в нормаль. Вес слоя в смеси высот здесь значения не '
                         'имеет.\n\nВыглядит такая нормаль так, будто AO уже запечено: впадины '
                         'темнее при любом свете. Плата — к ней не существует подходящей высоты, '
                         'поэтому параллакс и AO по высоте описывают другую поверхность. Ниже '
                         'показано, насколько сильно, и есть ползунок, который это исправляет.',
                         'The "from the lighting" layer is computed twice — lit from the left '
                         'and lit from below — and the two results go straight into the R and G '
                         'channels. Blue is derived. Height plays no part at all.\n\nThe layer '
                         'uses the same settings as "Where the relief comes from → From the '
                         'lighting on the texture": smoothing, paint and shadow removal, scale '
                         "boundary, coarse and fine. So what you see in that layer's preview is "
                         "what lands in the normal. The layer's weight in the height mix does "
                         'not matter here.\n\nSuch a normal looks as if AO were already baked in: '
                         'hollows stay darker under any light. The price is that no height '
                         'matches it, so parallax and AO built from height describe a different '
                         'surface. How much different is shown below, and a slider fixes it.'),
    'm.ndir.blue': ('Синий канал',
                    'Blue channel'),
    'm.ndir.blue.info': ('Из R и G — синий досчитывается так, чтобы длина вектора была единицей, '
                         'как это делает вкладка «Правка нормалей».\n\nИз AO — синий берётся из '
                         'затенения по высоте, а вектор потом нормируется. В щелях нормаль '
                         'заваливается сильнее и темнеет при любом свете, на ровных площадках '
                         'остаётся как есть.',
                         'From R and G — blue is derived so the vector has unit length, the way '
                         'the "Normal map repair" tab does it.\n\nFrom AO — blue is taken from the '
                         'occlusion of the height, then the vector is normalised. In crevices '
                         'the normal tips further and darkens under any light; on flat areas it '
                         'stays as it is.'),
    'm.ndir.b.unit': ('из R и G',
                      'from R and G'),
    'm.ndir.b.ao': ('из AO',
                    'from AO'),
    'm.ndir.inv': ('вывернуть рельеф',
                   'invert the relief'),
    'm.ndir.inv.info': ('Выпуклое станет вогнутым. По одной картинке эти два варианта не '
                        'различить в принципе, поэтому выбор за вами.',
                        'Bumps become hollows. From a single picture the two cannot be told '
                        'apart at all, so the choice is yours.'),
    'm.ndir.curl': ('Расхождение с высотой (ротор поля наклонов): %.2f. У нормали из высоты '
                    'около 0.1. Параллакс и AO описывают другую поверхность.',
                    'Mismatch with the height (curl of the slope field): %.2f. A normal '
                    'from height is about 0.1. Parallax and AO describe a different '
                    'surface.'),
    'm.ndir.curl.ok': ('Расхождение с высотой: %.2f. Нормаль подогнана, параллакс и AO '
                       'описывают ту же поверхность.',
                       'Mismatch with the height: %.2f. The normal is fitted; parallax and AO '
                       'describe the same surface.'),
    'm.dx': ('зелёный канал в формате DirectX', 'green channel in DirectX convention'),
    'm.dx.info': ('DirectX — зелёный вниз. Так ждут OpenMW, Unreal и большинство '
                  'игровых движков. OpenGL и Blender ждут наоборот.',
                  'DirectX means green points down. That is what OpenMW, Unreal and '
                  'most game engines expect. OpenGL and Blender expect the opposite.'),

    'm.ao': ('Затенение (AO)', 'Ambient occlusion'),
    'm.aor': ('Радиус AO, пикс.', 'AO radius, px'),
    'm.aos': ('Сила AO', 'AO strength'),
    'm.aowc': ('По впадинам рельефа', 'From cavities'),
    'm.aowc.info': ('Точка темнеет настолько, насколько она ниже окружения.',
                    'A point darkens by as much as it sits below its surroundings.'),
    'm.aown': ('По наклону нормали', 'From normal tilt'),
    'm.aown.info': ('Доля видимого неба равна (1 + n.z)/2. Ловит скосы, которые '
                    'впадинами не считаются.',
                    'The visible fraction of sky is (1 + n.z)/2. Catches slopes that '
                    'do not count as cavities.'),
    'm.rd': ('Рельеф в диффуз',
             'Relief into the diffuse'),
    'm.rd.info': ('То, что обычно делают в Фотошопе: накладывают высоту или AO поверх '
                  'текстуры, чтобы щели темнели, а выступы светлели прямо в цвете. Движок '
                  'это сам не сделает, если у модели нет отдельной карты AO, — а в '
                  'Morrowind её нет.\n\nЗапечённое в диффуз затенение потом уже не снять, '
                  'поэтому сохраняйте и исходный диффуз. Сравнить до и после — виды '
                  '«диффуз без света» и «диффуз до подмешивания» или «две панели».',
                  'What is usually done in Photoshop: height or AO laid over the texture '
                  'so that crevices darken and ridges lighten right in the colour. The '
                  'engine will not do it by itself unless the model has a separate AO map '
                  '— and Morrowind has none.\n\nOcclusion baked into the diffuse cannot be '
                  'taken out later, so keep the original diffuse too. To compare before '
                  'and after, use the "diffuse without light" and "diffuse before '
                  'blending" views, or "two panels".'),
    'm.rd.src': ('Что подмешивать',
                 'What to blend in'),
    'm.rd.src.info': ('Высота — выступы светлее, щели темнее, на всех масштабах. AO — только '
                      'затенение в щелях, крупное и гладкое. Внешняя глубина — загруженная '
                      'или посчитанная нейросетью карта глубины (раздел «Внешняя карта '
                      'глубины»).',
                      'Height — ridges lighter, crevices darker, at every scale. AO — only '
                      'occlusion in crevices, large and smooth. External depth — a loaded or '
                      'network-computed depth map ("External depth map" section).'),
    'm.rd.s.height': ('высота',
                      'height'),
    'm.rd.s.ao': ('AO',
                  'AO'),
    'm.rd.s.ext': ('внешняя глубина',
                   'external depth'),
    'm.rd.mode': ('Режим наложения',
                  'Blend mode'),
    'm.rd.mode.info': ('Перекрытие и мягкий свет — как в Фотошопе: и темнят, и осветляют '
                       'относительно середины, то есть добавляют контраст в обе стороны. '
                       'Считаются в sRGB, как в Фотошопе, иначе результат с ним не совпал бы. '
                       'Мягкий свет — то же, но слабее.\n\nУмножение и линейное затемнение '
                       'только темнят. Умножение считается в линейном пространстве — это '
                       'физическое «сколько света дошло»; так работал прежний ползунок «Запечь '
                       'AO в диффуз».',
                       'Overlay and soft light work as in Photoshop: they darken and lighten '
                       'around the middle, adding contrast both ways. They are computed in '
                       'sRGB, as in Photoshop, otherwise the result would not match it. Soft '
                       'light is the same but gentler.\n\nMultiply and linear burn only darken. '
                       'Multiply is computed in linear space — the physical "how much light '
                       'got through"; that is how the old "Bake AO into the diffuse" slider '
                       'worked.'),
    'm.rd.m.overlay': ('перекрытие',
                       'overlay'),
    'm.rd.m.soft': ('мягкий свет',
                    'soft light'),
    'm.rd.m.multiply': ('умножение',
                        'multiply'),
    'm.rd.m.burn': ('линейное затемнение',
                    'linear burn'),
    'm.rd.amt': ('Сила',
                 'Strength'),
    'm.rd.amt.info': ('0 — ничего не меняется. Для перекрытия сила — это контраст '
                      'накладываемого слоя, то же, что «Кривые» поверх него в Фотошопе.',
                      '0 changes nothing. For overlay the strength is the contrast of the '
                      'blended layer, the same as Curves on top of it in Photoshop.'),
    'm.rd.scale.info': ('Граница между крупным и мелким в подмешиваемой карте. Мелкое — это то, '
                        'что в Фотошопе даёт «Цветовой контраст» (High Pass): швы и сколы без '
                        'общего перепада яркости по текстуре.',
                        'The boundary between coarse and fine in the blended map. Fine is what '
                        "Photoshop's High Pass gives: joints and chips without the overall "
                        'brightness drift across the texture.'),
    'm.rd.sat.info': ('Без этого затемнённое уходит в серость, а осветлённое — в кислотный '
                      'цвет.',
                      'Without this, darkened areas go grey and lightened ones go garish.'),
    'm.rd.note': ('Диффуз изменился в среднем на %.1f ступени из 255, у 95 %% пикселей — '
                  'не больше %.1f.',
                  'The diffuse changed by %.1f steps of 255 on average, by at most %.1f '
                  'for 95 %% of pixels.'),
    'm.rd.noext': ('Внешняя глубина не загружена — вместо неё берётся высота.',
                   'No external depth is loaded — the height is used instead.'),
    'm.v.dbase': ('диффуз до подмешивания',
                  'diffuse before blending'),

    'm.tone': ('Тон чёрно-белых карт', 'Tone of the greyscale maps'),
    'm.tone.info': ('Гамма меньше 1 вытягивает тёмное, больше 1 — топит. У высоты тон '
                    'применяется до всего остального, поэтому тянет за собой и '
                    'нормаль, и AO.',
                    'Gamma below 1 lifts the darks, above 1 sinks them. On height the '
                    'tone is applied before everything else, so it drags the normal and '
                    'the AO along with it.'),
    'm.hgam': ('Высота: гамма', 'Height: gamma'),
    'm.hcon': ('Высота: контраст', 'Height: contrast'),
    'm.aogam': ('AO: гамма', 'AO: gamma'),
    'm.aocon': ('AO: контраст', 'AO: contrast'),
    'm.rgam': ('Шероховатость: гамма', 'Roughness: gamma'),
    'm.rcon': ('Шероховатость: контраст', 'Roughness: contrast'),

    'm.spec': ('Отражение', 'Reflectance'),
    'm.spec.info': ('У всех неметаллов отражение в упор около 0.04, и разница между '
                    'камнем, деревом и тканью — в шероховатости, а не в силе блика. '
                    'Разводить их по F0 — привычка из до-PBR времён, и именно она '
                    'даёт блик ярче света.',
                    'Every non-metal reflects about 0.04 head-on, and the difference '
                    'between stone, wood and cloth is roughness, not reflection '
                    'strength. Separating them by F0 is a pre-PBR habit, and it is what '
                    'produces highlights brighter than the light itself.'),
    'm.f0': ('Отражение в упор (F0)', 'Head-on reflectance (F0)'),
    'm.f0.info': ('Диэлектрики 0.02…0.08, металлы 0.5…1.0. Выше — блик станет ярче '
                  'света.',
                  'Dielectrics 0.02…0.08, metals 0.5…1.0. Above that the highlight '
                  'grows brighter than the light.'),
    'm.gloss': ('Глянец', 'Gloss'),
    'm.gvar': ('Разброс глянца', 'Gloss variation'),
    'm.metal': ('Металличность', 'Metalness'),
    'm.tint': ('Подкраска блика альбедо', 'Tint highlight by albedo'),

    'm.zone': ('Второй материал по маске', 'Second material by mask'),
    'm.zone.info': ('На одной текстуре редко бывает один материал: на стене и камень, '
                    'и дерево, и металлическая оковка. Здесь можно задать вторую '
                    'настройку и указать, где она действует. Карты считаются каждая '
                    'со своими параметрами и сшиваются по маске — смешиваются '
                    'результаты, а не ползунки.',
                    'A single texture rarely holds a single material: a wall has stone, '
                    'wood and iron banding at once. Here you can set up a second '
                    'material and say where it applies. Each set of maps is computed '
                    'with its own parameters and stitched by the mask — results are '
                    'blended, not sliders.'),
    'm.zone.on': ('включить второй материал', 'enable the second material'),
    'm.zone.on.info': ('Считается второй полный набор карт, поэтому расчёт занимает '
                       'вдвое больше времени.',
                       'A second full set of maps is computed, so everything takes '
                       'twice as long.'),
    'm.zone.preset': ('Материал зоны', 'Zone material'),
    'm.zone.src': ('Откуда маска', 'Mask source'),
    'm.zone.src.info': ('Маска строится из того, что в текстуре уже есть: из выбранного '
                        'канала по диапазону значений, или из областей, отмеченных '
                        'кистью. Оба источника складываются.',
                        'The mask is built from what the texture already holds: from a '
                        'chosen channel by value range, or from regions marked with the '
                        'brush. Both sources add together.'),
    'm.zone.chan': ('Канал', 'Channel'),
    'm.zone.lo': ('От', 'From'),
    'm.zone.hi': ('До', 'To'),
    'm.zone.soft': ('Мягкость края', 'Edge softness'),
    'm.zone.blur': ('Размыть маску, пикс.', 'Blur the mask, px'),
    'm.zone.inv': ('инвертировать маску', 'invert the mask'),
    'm.zone.brush': ('дописывать маску кистью', 'extend the mask with the brush'),
    'm.zone.brush.info': ('Левая кнопка добавляет область, правая убирает. Границу '
                          'области программа находит сама, как в подсказке по высоте.',
                          'Left button adds a region, right removes one. The boundary '
                          'is found for you, as in height guidance.'),
    'ch.luma': ('яркость', 'brightness'),
    'ch.R': ('R', 'R'),
    'ch.G': ('G', 'G'),
    'ch.B': ('B', 'B'),
    'ch.A': ('альфа', 'alpha'),
    'ch.sat': ('насыщенность', 'saturation'),
    'ch.hue': ('оттенок', 'hue'),
    'ch.height': ('высота', 'height'),
    'ch.ao': ('затенение', 'occlusion'),

    'm.seam': ('Бесшовность и скорость', 'Tiling and speed'),
    'm.fast': ('быстрый предпросмотр (половина размера)', 'fast preview (half size)'),
    'm.fast.info': ('Считает превью на уменьшенной копии. При сохранении всё равно '
                    'считается полный размер.',
                    'Previews on a reduced copy. Saving always computes at full size.'),
    'm.seamon': ('чинить бесшовность', 'fix tiling seams'),
    'm.seamon.info': ('Полоса вдоль краёв подменяется содержимым из середины текстуры '
                      'со сдвигом на половину. Середина непрерывна, поэтому шов '
                      'исчезает.',
                      'A band along the edges is replaced with content from the middle '
                      'of the texture, shifted by half. The middle is continuous, so '
                      'the seam disappears.'),
    'm.band': ('Полоса шва, пикс.', 'Seam band, px'),
    'm.band.info': ('Одинаково применяется ко всем картам, чтобы они не разошлись '
                    'между собой.',
                    'Applied identically to every map so they do not drift apart.'),
    'm.what': ('Что сохранять', 'What to save'),
    'ms._n': ('_n.dds — нормаль', '_n.dds — normal map'),
    'ms._nh': ('_nh.dds — нормаль + высота в альфе (parallax)',
               '_nh.dds — normal + height in alpha (parallax)'),
    'ms._spec': ('_spec.dds — блик + глянец в альфе',
                 '_spec.dds — specular + gloss in alpha'),
    'ms._ao': ('_ao.dds — затенение', '_ao.dds — occlusion'),
    'ms._h': ('_h.dds — высота (displacement)', '_h.dds — height (displacement)'),
    'ms._rough': ('_rough.dds — шероховатость (PBR)', '_rough.dds — roughness (PBR)'),
    'ms._metal': ('_metal.dds — металличность (PBR)', '_metal.dds — metalness (PBR)'),
    'ms._fixed': ('_fixed.dds — диффуз без света', '_fixed.dds — diffuse without light'),
    'm.v.lit': ('под светом', 'lit'),
    'm.v.orig': ('оригинал', 'original'),
    'm.v.diffuse': ('диффуз без света', 'de-lit diffuse'),
    'm.v.height': ('высота', 'height'),
    'm.v.normal': ('нормаль', 'normal'),
    'm.v.ao': ('AO', 'AO'),
    'm.v.spec': ('блик', 'specular'),
    'm.v.rough': ('шероховатость', 'roughness'),
    'm.v.ext': ('внешняя глубина', 'external depth'),
    'm.v.guide': ('подсказка по высоте', 'height guidance'),
    'm.v.zone': ('маска зоны', 'zone mask'),
    'm.v.damage': ('битые участки', 'damaged areas'),
    'm.v.fixed': ('после починки', 'after repair'),

    # ------------------------------------------------- вкладка 3D ---------
    'tab.v3': ('3D просмотр', '3D preview'),
    'v3.cam': ('Камера', 'Camera'),
    'v3.cam.info': ('Плоскость можно наклонить и повернуть. Тяни мышью по картинке — '
                    'вращение, колесо — ближе и дальше. Параллакс в лоб не виден '
                    'вообще: весь смысл этого просмотра в том, чтобы посмотреть на '
                    'поверхность под углом, как на неё смотрят в игре.',
                    'The plane tilts and turns. Drag on the image to rotate, wheel to '
                    'move closer and further. Parallax shows nothing head-on: the whole '
                    'point of this view is to look at the surface at an angle, the way '
                    'you look at it in game.'),
    'v3.pitch': ('Наклон, °', 'Tilt, °'),
    'v3.yaw': ('Поворот, °', 'Turn, °'),
    'v3.dist': ('Удаление', 'Distance'),
    'v3.tiles': ('Повторов текстуры', 'Texture repeats'),
    'v3.tiles.info': ('Сколько раз плитка укладывается по стороне. Это заодно и '
                      'проверка бесшовности: если шов есть, здесь он виден сеткой.',
                      'How many times the tile is laid out along a side. It doubles as a '
                      'tiling check: a seam shows up here as a grid.'),
    'v3.par': ('Параллакс', 'Parallax'),
    'v3.par.info': ('Параллакс сдвигает точку на поверхности вдоль луча зрения по '
                    'карте высоты, и плоская стена начинает вести себя как рельефная: '
                    'при движении камеры выступы перекрывают впадины. Нормаль этого '
                    'не умеет — она только поворачивает свет.',
                    'Parallax shifts the point on the surface along the view ray using '
                    'the height map, so a flat wall starts behaving like a relief one: '
                    'as the camera moves, bumps cover hollows. A normal map cannot do '
                    'that — it only turns the light.'),
    'v3.m.flat': ('выключен', 'off'),
    'v3.m.flat.info': ('Только нормаль и AO, без смещения. Так выглядит текстура '
                       'в движке без parallax.',
                       'Normal and AO only, no offset. This is how the texture looks in '
                       'an engine without parallax.'),
    'v3.m.offset': ('смещение', 'offset'),
    'v3.m.offset.info': ('Один шаг: точка сдвигается на глубину под собой. Дёшево, '
                         'так делали в первых реализациях, но на скользящем взгляде '
                         'изображение плывёт.',
                         'A single step: the point is shifted by the depth beneath it. '
                         'Cheap, and how the first implementations worked, but the '
                         'image swims at grazing angles.'),
    'v3.m.occlusion': ('с перекрытием', 'with occlusion'),
    'v3.m.occlusion.info': ('Луч идёт внутрь поверхности шагами и ищет, где протыкает '
                            'рельеф. Выступы честно закрывают то, что за ними. Так '
                            'считают современные движки, включая OpenMW.',
                            'The ray steps into the surface looking for where it pierces '
                            'the relief, so bumps genuinely hide what is behind them. '
                            'This is what modern engines do, OpenMW included.'),
    'v3.depth': ('Глубина рельефа, % плитки', 'Relief depth, % of a tile'),
    'v3.depth.info': ('В долях ширины одной плитки. Для кладки 4…8 % — примерно '
                      'настоящая глубина шва. Больше 15 % почти всегда выглядит как '
                      'резиновая: кирпич начинает уезжать от собственного шва.',
                      'As a fraction of one tile. For masonry 4–8 % is roughly the real '
                      'depth of a joint. Above 15 % it nearly always looks rubbery: the '
                      'brick starts sliding away from its own joint.'),
    'v3.steps': ('Шагов луча', 'Ray steps'),
    'v3.steps.info': ('Мало шагов — рельеф расслаивается на ступеньки, особенно на '
                      'косом взгляде. Это же ограничение есть и в движке, так что '
                      'ступеньки здесь — честное предупреждение.',
                      'Too few steps and the relief breaks into terraces, especially at '
                      'grazing angles. The engine has the same limit, so terracing here '
                      'is an honest warning.'),
    'v3.shadow': ('собственная тень рельефа', 'relief casts its own shadow'),
    'v3.shadow.info': ('Из найденной точки луч идёт к свету: если по дороге рельеф '
                       'выше, точка в тени. Движок так умеет не всегда — это скорее '
                       'проверка глубины, чем предсказание.',
                       'A ray goes from the found point towards the light: if the relief '
                       'is higher along the way, the point is shaded. Engines do not '
                       'always do this — treat it as a depth check rather than a '
                       'prediction.'),
    'v3.grid': ('показывать границы плиток', 'show tile borders'),
    'v3.grid.info': ('Сетка по стыкам. Удобно ловить шов: на бесшовной текстуре '
                     'рисунок через линию продолжается без разрыва.',
                     'A grid along the joins. Handy for catching a seam: on a seamless '
                     'texture the pattern continues across the line unbroken.'),
    'v3.q': ('Качество и сравнение', 'Quality and comparison'),
    'v3.res': ('Наибольший размер кадра, пикс.',
               'Largest frame size, px'),
    'v3.res.info': ('Кадр считается по размеру окна, но не больше этого числа. Во время '
                    'вращения он вдвое мельче и без тени, а через долю секунды после '
                    'остановки дорисовывается честно — с картами полного размера (они '
                    'считаются один раз после смены настроек).',
                    'The frame is computed at the window size, but no larger than this. '
                    'While rotating it is half size and without shadows, then redrawn '
                    'properly a moment after you stop — with full-size maps (computed once '
                    'after a settings change).'),
    'v3.cmp': ('слева без параллакса, справа с ним',
               'left without parallax, right with it'),
    'v3.cmp.info': ('Один и тот же кадр в двух вариантах. Единственный надёжный '
                    'способ понять, что параллакс вообще даёт на этой текстуре.',
                    'The same frame twice. The only reliable way to see what parallax '
                    'actually contributes on this texture.'),
    'v3.nomaps': ('Сначала открой текстуру — карты берутся с вкладки «Материал».',
                  'Open a texture first — the maps come from the «Material» tab.'),
    'v3.hint': ('тянуть — вращать, колесо — ближе и дальше',
                'drag to rotate, wheel to move closer or further'),
    # ------------------------------------------------ правка нормалей ------
    'x.what': ('Что делать', 'What to do'),
    'x.what.info': ('Открытая текстура считается готовой картой нормалей. Со всеми '
                    'галочками снятыми ничего не меняется — только разбор.',
                    'The open texture is treated as a finished normal map. With every '
                    'box unticked nothing changes — it only reports.'),
    'x.rebuild': ('восстановить Z из X, Y', 'rebuild Z from X and Y'),
    'x.rebuild.info': ('Для карт, где синий канал пустой или забит чем-то посторонним.',
                       'For maps whose blue channel is empty or filled with something '
                       'unrelated.'),
    'x.renorm': ('длину вектора → 1', 'renormalise vectors to 1'),
    'x.renorm.info': ('Неединичная длина читается движком как «поверхность светится '
                      'сильнее просто так», и по карте идут пятна яркости.',
                      'A non-unit length reads to the engine as «this surface is '
                      'brighter for no reason», and the map develops blotches.'),
    'x.clamp': ('поджать Z (убрать отвёрнутые)', 'clamp Z (remove back-facing pixels)'),
    'x.minz': ('Минимальный Z', 'Minimum Z'),
    'x.minz.info': ('Сжимает размах XY так, чтобы Z нигде не падал ниже. При 0.2 '
                    'предельный наклон около 78°.',
                    'Compresses the XY range so Z never drops below this. At 0.2 the '
                    'steepest tilt is about 78°.'),
    'x.flip': ('сменить зелёный (OpenGL ↔ DirectX)', 'flip green (OpenGL ↔ DirectX)'),
    'x.keepalpha': ('сохранить альфу исходника', 'keep the original alpha'),
    'x.over': ('перезаписывать оригинал', 'overwrite the original'),
    'x.over.info': ('Без галочки рядом кладётся копия с «_fix». С галочкой файл '
                    'заменяется на месте — спросит подтверждение один раз. Отката нет, '
                    'так что перед пачкой лучше иметь копию папки.',
                    'Unticked, a copy with «_fix» is written alongside. Ticked, the file '
                    'is replaced in place after one confirmation. There is no undo, so '
                    'copy the folder before running a batch.'),
    'x.savefixed': ('Сохранить исправленную', 'Save the repaired map'),
    'x.report': ('Разбор', 'Report'),
    'x.v.both': ('до и после', 'before and after'),
    'x.v.after': ('только исправленная', 'repaired only'),
    'x.conv': ('Нормаль ↔ высота',
               'Normal ↔ height'),
    'x.conv.info': ('Если открыта карта нормалей — из неё собирается высота: наклоны '
                    'нормали складываются в поверхность по наименьшим квадратам (способ '
                    'Франко–Челлаппы). Сначала применяется починка из раздела выше, потом '
                    'перевод. Если нормаль когда-то была посчитана из высоты, высота '
                    'восстанавливается почти точно; если нет, теряется та часть, которая '
                    'никакой поверхности не принадлежит, — сколько именно, показано '
                    'ниже.\n\nЕсли открыта карта высоты — из неё строится нормаль.\n\nВид '
                    '«результат преобразования» показывает то, что получилось, «обратно» — '
                    'что выходит, если перевести результат назад. По разнице видно, что '
                    'пропало.',
                    "If a normal map is open, a height is assembled from it: the normal's "
                    'slopes are combined into a surface by least squares (the '
                    'Frankot–Chellappa method). The repair options above are applied first, '
                    'then the conversion. If the normal was once computed from a height, '
                    'that height comes back almost exactly; if not, the part that belongs '
                    'to no surface is lost — how much is shown below.\n\nIf a height map is '
                    'open, a normal is built from it.\n\nThe "conversion result" view shows '
                    'what came out, "and back" shows what you get by converting the result '
                    'back. The difference is what was lost.'),
    'x.conv.src': ('Открытая картинка —',
                   'The open image is'),
    'x.src.normal': ('карта нормалей',
                     'a normal map'),
    'x.src.height': ('карта высоты',
                     'a height map'),
    'x.conv.dx': ('DirectX (зелёный вниз), как у OpenMW',
                  'DirectX (green down), as OpenMW uses'),
    'x.conv.dx.info': ('Для нормали на входе — в каком она формате. Для высоты на входе — в '
                       'каком формате строить нормаль. Если рельеф вышел вывернутым по '
                       'вертикали, формат не тот.',
                       'For a normal on input — which format it is in. For a height on input — '
                       'which format to build the normal in. If the relief comes out flipped '
                       'vertically, the format is wrong.'),
    'x.conv.cut': ('Не верить крупнее',
                   'Distrust above'),
    'x.conv.cut.info': ('Только для перевода нормали в высоту. Интеграл копит ошибку на '
                        'крупном: небольшой общий перекос нормали становится холмом через всю '
                        'текстуру. Ползунок вычитает всё крупнее заданной доли стороны. 0 — '
                        'верить всему.',
                        'Only for normal to height. The integral accumulates error at large '
                        'scales: a slight overall tilt in the normal turns into a hill across '
                        'the whole texture. The slider removes everything larger than the given '
                        'fraction of the side. 0 trusts everything.'),
    'x.conv.amp': ('Размах высоты, пикс.',
                   'Height range, px'),
    'x.conv.amp.info': ('Только для перевода высоты в нормаль: на сколько пикселей текстуры '
                        'поднимается белое над чёрным. Больше — круче нормаль.',
                        'Only for height to normal: how many texture pixels white rises above '
                        'black. More means a steeper normal.'),
    'x.conv.save': ('Сохранить результат…',
                    'Save the result…'),
    'x.v.conv': ('результат преобразования',
                 'conversion result'),
    'x.v.back': ('обратно — что уцелело',
                 'and back — what survived'),
    'x.conv.n.note': ('Высота из нормали: размах %.1f пикс. Расхождение нормали с любой '
                      'высотой (ротор): %.2f — у нормали, посчитанной из высоты, около 0. '
                      'Нормаль из этой высоты отличается от исходной в среднем на %.1f°.',
                      "Height from the normal: range %.1f px. The normal's mismatch with any "
                      'height (curl): %.2f — about 0 for a normal computed from a height. The '
                      'normal rebuilt from this height differs from the original by %.1f° on '
                      'average.'),
    'x.conv.h.note': ('Нормаль из высоты: средний наклон %.1f°, у самых крутых мест %.1f°.',
                      'Normal from the height: mean tilt %.1f°, steepest spots %.1f°.'),
    'x.conv.h.rep': ('Открыта карта высоты %d×%d.\nБелое поднято над чёрным на %.0f '
                     'пикс.\n\nРазбор карты нормалей здесь не делается: на входе не нормаль.',
                     'A %d×%d height map is open.\nWhite is raised above black by %.0f '
                     'px.\n\nNo normal-map report here: the input is not a normal.'),
    'x.over.title': ('Перезаписать', 'Overwrite'),
    'x.over.q': ('Заменить файл на месте?\n\n%s\n\nОтката не будет.',
                 'Replace the file in place?\n\n%s\n\nThere will be no undo.'),
    'x.over.batch': ('Все подходящие файлы в папке будут заменены на месте.\n'
                     'Отката не будет. Продолжить?',
                     'Every matching file in the folder will be replaced in place.\n'
                     'There will be no undo. Continue?'),
    'x.r.size': ('размер', 'size'),
    'x.r.length': ('длина вектора', 'vector length'),
    'x.r.outside': ('вне 0.9…1.1', 'outside 0.9…1.1'),
    'x.r.negz': ('Z отрицательный', 'negative Z'),
    'x.r.lowz': ('Z меньше 0.2', 'Z below 0.2'),
    'x.r.tilt': ('наклон', 'tilt'),
    'x.r.uniq': ('уникальных значений R/G/B', 'unique values R/G/B'),
    'x.p.blue': ('синий канал пустой или постоянный — карта двухканальная, '
                 'включи восстановление Z',
                 'the blue channel is empty or constant — the map is effectively '
                 'two-channel, switch on Z rebuild'),
    'x.p.negz': ('есть пиксели, отвёрнутые от зрителя — включи поджатие Z',
                 'there are pixels facing away from the viewer — switch on Z clamping'),
    'x.p.length': ('длина вектора гуляет — включи нормализацию',
                   'vector length drifts — switch on renormalising'),
    'x.p.banding': ('мало уникальных значений (%d в одном из каналов) — карта уже '
                    'пережата блочным сжатием, полосы никуда не денутся',
                    'few unique values (%d in one channel) — the map has already been '
                    'through block compression and the banding will not go away'),
    'x.p.ok': ('явных болезней не вижу', 'no obvious faults'),
    'x.was': ('БЫЛО', 'BEFORE'),
    'x.now': ('СТАЛО', 'AFTER'),

    # ------------------------------------------------------- служебное -----
    'sys.first': ('Первый запуск', 'First run'),
    'sys.need': ('Не хватает библиотек:\n\n    %s\n\nПоставить сейчас? '
                 'Минуту-другую, около 100 МБ.',
                 'Missing libraries:\n\n    %s\n\nInstall them now? '
                 'A minute or two, about 100 MB.'),
    'sys.installing': ('Ставлю библиотеки, подожди…', 'Installing libraries, please wait…'),
    'sys.failed': ('Не установилось', 'Installation failed'),
    'sys.mismatch': ('Файлы разных версий', 'Mismatched file versions'),
    'sys.mismatch.b': ('Не совпадают версии:\n\n    %s\n\nЗамени все файлы из последней '
                       'поставки и положи их в одну папку.',
                       'These versions do not match:\n\n    %s\n\nReplace every file '
                       'with the latest release and keep them in one folder.'),
    'sys.langrestart': ('Язык переключён.', 'Language switched.'),
    'sys.noguide': ('Файл руководства не найден рядом с программой: guide.html',
                    'The manual file was not found next to the program: guide.html'),
    'com.hemi': ('рассеянный свет полусферой (не как в OpenMW)',
                 'hemispherical ambient (not as in OpenMW)'),
    'com.hemi.info': ('Как считать рассеянный свет в превью. Выключено — ровно, как в OpenMW: '
                      'там рассеянная составляющая одна на всю поверхность и от нормали не '
                      'зависит. Включено — полусферой: повёрнутое к небу светлее, как в '
                      'современных движках.\n\nОт этого зависит, виден ли bent normal: при '
                      'ровном рассеянном свете он затенения не даёт вообще.',
                      'How the preview computes ambient light. Off — flat, as in OpenMW: the '
                      'ambient term is the same over the whole surface and does not depend on '
                      'the normal. On — hemispherical: whatever faces the sky is lighter, as '
                      'in modern engines.\n\nThis decides whether a bent normal shows at all: '
                      'under flat ambient it produces no occlusion whatsoever.'),
    'b.w.ext': ('Внешняя глубина из файла или нарисованная кистью относится к открытой '
                'текстуре — в пакете её не будет. Глубина от нейросети в пакете '
                'считается заново, а эта — нет.',
                'External depth from a file or painted by brush belongs to the open '
                'texture — the batch will not use it. Network depth is recomputed in a '
                'batch; this one is not.'),
    'b.w.flat': ('Все остальные слои высоты сейчас в нуле, поэтому высота, нормаль и '
                 'параллакс в пакете выйдут плоскими.',
                 'All other height layers are at zero now, so height, normal and '
                 'parallax will come out flat in the batch.'),
    'b.w.zone': ('Вторая зона по маске в пакете не применяется: её маска относится к '
                 'открытой текстуре.',
                 'The second zone by mask is not applied in a batch: its mask belongs to '
                 'the open texture.'),
    'b.w.hand': ('Подсказки по высоте и битые места, нарисованные кистью, относятся к '
                 'открытой текстуре и в пакете не участвуют. Автоматический поиск битых '
                 'мест повторяется на каждой текстуре.',
                 'Height hints and damage painted by brush belong to the open texture '
                 'and are not used in a batch. Automatic damage detection is repeated on '
                 'every texture.'),
    'b.w.go': ('Продолжить?',
               'Continue?'),
    'm.nn.stale': ('Пока сеть считала, открыли другую текстуру — ответ отброшен.',
                   'Another texture was opened while the network was running — the result '
                   'was discarded.'),
}

# ---------------------------------------------------- сборка 22: узловое окно
S.update({
    # --- панель и меню
    'ed.hello': ('Откройте текстуру (Ctrl+T) или перетащите её в окно. Узел добавляется '
                 'Shift+A, Tab или правой кнопкой.',
                 'Open a texture (Ctrl+T) or drop it on the window. Add nodes with '
                 'Shift+A, Tab or the right mouse button.'),
    'ed.new': ('Новый…', 'New…'),
    'ed.open': ('Открыть…', 'Open…'),
    'ed.save': ('Сохранить', 'Save'),
    'ed.saveas': ('Сохранить как…', 'Save as…'),
    'ed.tex': ('Текстура…', 'Texture…'),
    'ed.samples': ('Образцы текстур', 'Sample textures'),
    'ed.savemaps': ('Сохранить карты…', 'Save maps…'),
    'ed.batch': ('Пакет…', 'Batch…'),
    'ed.classic': ('Классические вкладки', 'Classic tabs'),
    'ed.back': ('← Узлы', '← Nodes'),
    'ed.lit': ('Свет', 'Lit'),
    'ed.draft': ('Черновой проход', 'Draft pass'),
    'ed.pin': ('Закрепить', 'Pin'),
    'ed.pinned': ('(закреплён)', '(pinned)'),
    'ed.hemi': ('Небо и AO', 'Sky and AO'),
    'ed.untitled': ('Без имени', 'Untitled'),
    'ed.notex': ('текстура не открыта', 'no texture open'),
    'ed.empty': ('Пусто. Shift+A — добавить узел.', 'Empty. Shift+A adds a node.'),
    'ed.m.dup': ('Дублировать', 'Duplicate'),
    'ed.m.del': ('Удалить', 'Delete'),
    'ed.m.unlink': ('Отсоединить все провода', 'Disconnect all wires'),
    'ed.prevres': ('превью %d×%d', 'preview %d×%d'),
    'ed.done': ('Готово за %.1f с', 'Done in %.1f s'),
    'ed.errors': ('ошибок: %d (красные узлы)', 'errors: %d (red nodes)'),
    'ed.params': ('Параметры', 'Parameters'),
    'ed.readouts': ('Показания', 'Readouts'),
    'ed.noresult': ('Ещё не посчитано.', 'Not computed yet.'),
    'ed.project': ('Проект', 'Project'),
    'ed.outputs': ('Что будет записано', 'What will be written'),
    'ed.samename': ('диффуз под именем исходника', 'diffuse under the source name'),
    'ed.keys': ('Горячие клавиши', 'Shortcuts'),
    'ed.keys.list': ('Shift+A, Tab, правая кнопка — добавить узел\n'
                     'Тянуть от гнезда — провод; от занятого входа — снять провод\n'
                     'Бросить провод на пустое место — выбрать узел для него\n'
                     'Тянуть по пустому — выделить рамкой; Shift — добавить к выделению\n'
                     'Средняя кнопка или Alt+левая — сдвиг; колесо — масштаб\n'
                     'X, Delete — удалить; Ctrl+D — дублировать; Ctrl+A — всё\n'
                     'F — показать всё; H — миниатюры\n'
                     'Ctrl+Z / Ctrl+Y — отменить / вернуть\n'
                     'Ctrl+N, Ctrl+O, Ctrl+S — новый, открыть, сохранить проект\n'
                     'Ctrl+T — открыть текстуру',
                     'Shift+A, Tab, right button — add a node\n'
                     'Drag from a port — a wire; from an occupied input — pick the wire up\n'
                     'Drop a wire on empty space — choose a node for it\n'
                     'Drag on empty space — box select; Shift — add to selection\n'
                     'Middle button or Alt+left — pan; wheel — zoom\n'
                     'X, Delete — delete; Ctrl+D — duplicate; Ctrl+A — all\n'
                     'F — frame all; H — thumbnails\n'
                     'Ctrl+Z / Ctrl+Y — undo / redo\n'
                     'Ctrl+N, Ctrl+O, Ctrl+S — new, open, save project\n'
                     'Ctrl+T — open texture'),
    'ed.model': ('Нейросеть', 'Network'),
    'ed.m.rt': ('onnxruntime (запуск сетей)', 'onnxruntime (runs the networks)'),
    'ed.m.onnx': ('onnx (перевод весов ESRGAN)', 'onnx (converts ESRGAN weights)'),
    'ed.m.get': ('Скачать', 'Get'),
    'ed.m.working': ('Скачивание или установка…', 'Downloading or installing…'),
    'ed.m.ok': ('Готово.', 'Done.'),
    'ed.m.fail': ('Не получилось.', 'Failed.'),
    'ed.supermat': ('Установка SuperMat', 'SuperMat setup'),
    'ed.supermat.info': ('SuperMat ставится отдельно (install_supermat_portable.bat рядом с Texelith): свой Python с PyTorch, код SuperMat и базовая модель Stable Diffusion 2.1 (около 5 ГБ, скачивается при первом запуске). Texelith только запускает его. Код и веса SuperMat — MIT, базовая модель — CreativeML Open RAIL++-M.',
        'SuperMat is installed separately (install_supermat_portable.bat next to Texelith): its own Python with PyTorch, the SuperMat code and the Stable Diffusion 2.1 base model (about 5 GB, downloaded on first start). Texelith only starts it. SuperMat code and weights are MIT, the base model is CreativeML Open RAIL++-M.'),
    'ed.supermat.repo': ('Папка SuperMat', 'SuperMat folder'),
    'ed.supermat.py': ('python.exe его окружения', "python.exe of its environment"),
    'ed.supermat.w': ('supermat.pth (необязательно; пусто — из checkpoints или с Hugging Face)',
        'supermat.pth (optional; empty — from checkpoints or Hugging Face)'),
    'ed.supermat.run': ('Запущен, устройство: %s', 'Running, device: %s'),
    'ed.supermat.ready': ('Пути заданы. Модель загрузится при первом расчёте.',
                          'Paths are set. The model loads on the first run.'),
    'ed.supermat.starting': ('Запуск и загрузка модели… (до минуты)',
                             'Starting and loading the model… (up to a minute)'),
    'ed.supermat.test': ('Сохранить и проверить запуск', 'Save and test'),
    'supermat.norepo': ('Не найдена папка SuperMat.',
                        'SuperMat folder not found.'),
    'supermat.nopython': ('Не найден python окружения SuperMat.',
                          "SuperMat environment's python not found."),
    'supermat.noweights': ('Файл весов не найден.', 'Weights file not found.'),
    'supermat.notrepo': ('В папке SuperMat нет src/adapters.py — это не папка кода SuperMat.',
                         'No src/adapters.py in the SuperMat folder — not the SuperMat code.'),
    'ed.ask.save': ('Сохранить изменения в проекте?', 'Save changes to the project?'),
    'ed.new.title': ('Новый проект', 'New project'),
    'ed.new.head': ('С чего начать', 'Start from'),
    'ed.new.keeptex': ('Оставить открытую текстуру', 'Keep the open texture'),
    'ed.new.go': ('Создать', 'Create'),
    'ed.texmissing': ('Текстура проекта не найдена: %s', 'Project texture not found: %s'),
    'ed.nodiffuse': ('Диффуз под именем исходника не пишется в папку исходника — выберите '
                     'другую папку.',
                     'A diffuse under the source name is not written into the source '
                     'folder — choose another folder.'),
    'ed.saving': ('Считаю в полном размере и записываю…', 'Computing at full size and writing…'),
    'ed.batch.title': ('Пакетная обработка', 'Batch processing'),
    'ed.b.in': ('Папка с текстурами', 'Texture folder'),
    'ed.b.out': ('Куда писать', 'Output folder'),
    'ed.b.out.info': ('Структура подпапок повторяется. Если папка вывода — отдельная '
                      'папка данных OpenMW, диффуз пишется под своим именем и заменяет '
                      'исходный в игре; в папку исходников он не пишется никогда.',
                      'Subfolders are mirrored. If the output folder is a separate OpenMW '
                      'data folder, the diffuse is written under its own name and replaces '
                      'the original in the game; it is never written into the source '
                      'folder.'),
    'ed.b.rec': ('С подпапками', 'With subfolders'),
    'ed.b.skipdone': ('Пропускать уже сделанные', 'Skip finished ones'),
    'ed.b.compress': ('Сжатие DDS', 'DDS compression'),
    'ed.b.dxt': ('Сжатие делает:', 'Compressed by:'),
    'ed.b.dxt.quicktex': ('quicktex — меньше полос', 'quicktex — fewer bands'),
    'ed.b.dxt.pillow': ('Pillow — быстро, но с полосами', 'Pillow — fast, but banded'),
    'ed.b.dxt.level': ('Качество (0…18):', 'Quality (0…18):'),
    'ed.b.dxt.noqt': ('quicktex не установлен — сожмёт Pillow', 'quicktex is not installed — Pillow will be used'),
    'ed.b.dxt.info': ('Касается и кнопки «Сохранить карты», и классических вкладок. Качество '
                      '14 — как 18 по виду, но в 4–5 раз быстрее; ниже 13 на плавных переходах '
                      'снова появляются полосы.',
                      'Also applies to "Save maps" and the classic tabs. Quality 14 looks the '
                      'same as 18 but is 4–5 times faster; below 13 bands come back on smooth '
                      'gradients.'),
    'ed.b.fmt': ('Формат', 'Format'),
    'ed.b.skip': ('Не трогать (шаблоны через «;»)', 'Leave alone (patterns, «;»-separated)'),
    'ed.b.noin': ('Папка с текстурами не найдена.', 'Texture folder not found.'),
    'ed.b.count': ('Найдено текстур: %d', 'Textures found: %d'),
    'ed.b.count.btn': ('Пересчитать', 'Count'),
    'ed.b.haserr': ('На открытой текстуре в графе %d ошибок. Всё равно начать?',
                    'The graph has %d errors on the open texture. Start anyway?'),
    'ed.b.result': ('Сделано %d, пропущено %d, ошибок %d, %.0f с. Журнал — '
                    'texelith_batch.log в папке вывода.',
                    'Done %d, skipped %d, failed %d, %.0f s. Log: texelith_batch.log in '
                    'the output folder.'),
    'ed.b.start': ('Начать', 'Start'),
    'ed.b.stop': ('Остановить', 'Stop'),
    'ed.quit': ('Выход', 'Quit'),
    'ed.menu.project': ('Проект', 'Project'),
    'ed.menu.edit': ('Правка', 'Edit'),
    'ed.menu.view': ('Вид', 'View'),
    'ed.menu.help': ('Справка', 'Help'),
    'ed.undo': ('Отменить', 'Undo'),
    'ed.redo': ('Вернуть', 'Redo'),
    'ed.addnode': ('Добавить узел', 'Add node'),
    'ed.selall': ('Выделить всё', 'Select all'),
    'ed.frame': ('Показать всё', 'Frame all'),
    'ed.thumbs': ('Миниатюры в узлах', 'Node thumbnails'),
    # --- разделы узлов
    'cat.input': ('Вход', 'Input'),
    'cat.prepare': ('Подготовка', 'Preparation'),
    'cat.height': ('Высота', 'Height'),
    'cat.filter': ('Фильтры', 'Filters'),
    'cat.normal': ('Нормаль', 'Normal'),
    'cat.neural': ('Нейросети', 'Networks'),
    'cat.maps': ('Карты', 'Maps'),
    'cat.openmw': ('OpenMW', 'OpenMW'),
    'cat.output': ('Выход', 'Output'),
    'cat.other': ('Прочее', 'Other'),
    # --- шаблоны
    'tpl.neural': ('Нейросети: DeepBump + глубина → OpenMW',
                   'Networks: DeepBump + depth → OpenMW'),
    'tpl.neural.info': ('Нормаль от DeepBump (обучен на плоских материалах), крупная форма от '
                        'сети глубины, согласованы узлом баланса. Выход: _nh с высотой для '
                        'параллакса, диффуз с AO, _spec. Модели скачиваются кнопкой в узлах.',
                        'Normal from DeepBump (trained on flat materials), large form from a '
                        'depth network, agreed by the balance node. Output: _nh with parallax '
                        'height, diffuse with AO, _spec. Models are downloaded from the '
                        'nodes.'),
    'tpl.classic': ('Без нейросетей → OpenMW', 'No networks → OpenMW'),
    'tpl.classic.info': ('То же, что делала вкладка «Материал»: снятие света, высота '
                         'из яркости и из освещения, нормаль, AO. Работает сразу, без '
                         'скачиваний.',
                         'What the Material tab did: de-lighting, height from '
                         'brightness and from lighting, normal, AO. Works at once, nothing '
                         'to download.'),
    'tpl.supermat': ('SuperMat: PBR → OpenMW', 'SuperMat: PBR → OpenMW'),
    'tpl.supermat.info': ('Базовый цвет, шероховатость и металличность от SuperMat (ставится отдельно), нормаль и высота от DeepBump и сети глубины, всё сведено в то, что читает OpenMW: диффуз, _nh, _spec.',
        'Base colour, roughness and metalness from SuperMat (installed separately), normal and height from DeepBump and a depth network, converted into what OpenMW reads: diffuse, _nh, _spec.'),
    'tpl.normal': ('Быстро: только _n', 'Quick: _n only'),
    'tpl.normal.info': ('Высота из яркости и нормаль из неё. Самое простое.',
                        'Height from brightness and a normal from it. The simplest.'),
    'tpl.fixnormal': ('Готовая нормаль → _nh', 'Finished normal → _nh'),
    'tpl.fixnormal.info': ('Исходник — уже готовая карта нормалей: раскладка зелёного '
                           'определяется сама, сила и наклон правятся, высота интегрируется '
                           'и кладётся в альфу.',
                           'The source is a finished normal map: the green layout is detected, '
                           'strength and tilt adjusted, the height integrated and put into '
                           'alpha.'),
    'tpl.empty': ('Пустой', 'Empty'),
    'tpl.empty.info': ('Только исходник.', 'Only the source.'),
    # --- узлы: новые
    'nd.deepbump': ('Нормаль DeepBump', 'DeepBump normal'),
    'nd.deepbump.info': ('Сеть DeepBump (Hugo Tini, GPL-3.0), обучена на плоских тайлящихся '
                         'материалах. Даёт нормаль и высоту, проинтегрированную из неё. Окна '
                         'раскладываются по кругу через край, поэтому карта тайлится. Высоте '
                         'из нормали нельзя верить в крупной форме — подайте её в «Баланс» '
                         'вместе с глубиной.',
                         'The DeepBump network (Hugo Tini, GPL-3.0), trained on flat tiling '
                         'materials. Gives a normal and a height integrated from it. Windows '
                         'wrap around the edge, so the map tiles. Do not trust the height of '
                         'a normal in the large form — feed it to Balance together with a '
                         'depth map.'),
    'nd.nh_balance': ('Баланс нормаль ↔ высота', 'Balance normal ↔ height'),
    'nd.nh_balance.info': ('Находит одну высоту, которая одновременно ближе всего и к наклонам '
                           'нормали, и к данной высоте, и нормаль ровно от неё. Мелкие детали '
                           'берутся у нормали, крупная форма — у высоты; «Баланс» задаёт, где '
                           'проходит граница. Масштаб высоты подбирается сам по полосе, где '
                           'верят обоим. «Оставить своё» возвращает часть наклонов нормали, '
                           'которые ни одной высотой не объясняются.',
                           'Finds one height that is at once closest to the slopes of the '
                           'normal and to the given height, and a normal exactly from it. Fine '
                           'detail comes from the normal, large form from the height; Balance '
                           'sets where the border lies. The height scale is fitted by itself in '
                           'the band where both are trusted. Keep own brings back part of the '
                           "normal's slopes that no height can explain."),
    'nd.normal_mix': ('Смешать нормали', 'Mix normals'),
    'nd.normal_mix.info': ('«Смешать» — средняя поверхность между А и Б (доля Б — «Сколько '
                           'подмешать»): для двух оценок одного рельефа, например DeepBump и '
                           'нормали из высоты. «Наложить» — рельеф Б поверх А, наклоны '
                           'складываются: для детальной карты поверх основы. «Наложить (RNM)» — '
                           'то же способом игровых движков. Считается на наклонах, а не на '
                           'векторах, поэтому смесь не сплющивается.',
                           'Mix — the surface halfway between A and B (the share of B is "How '
                           'much to blend in"): for two estimates of the same relief, e.g. '
                           'DeepBump and the normal from a height. Add — the relief of B on top '
                           'of A, tilts add up: for a detail map over a base. Add (RNM) — the '
                           'same the way game engines do it. Computed on slopes, not vectors, '
                           'so the mix does not flatten.'),
    'nd.normal_fix': ('Правка нормали', 'Adjust normal'),
    'nd.normal_fix.info': ('Сила наклона, смена формата зелёного, предел наклона.',
                           'Tilt strength, green format switch, tilt limit.'),
    'nd.relief': ('Рельеф в диффуз', 'Relief into diffuse'),
    'nd.relief.info': ('Как в Photoshop: высота или AO поверх текстуры — перекрытие, мягкий '
                       'свет, умножение или линейное затемнение. Сила — в стандартных '
                       'отклонениях карты.',
                       'As in Photoshop: height or AO over the texture — overlay, soft '
                       'light, multiply or linear burn. Strength is in standard deviations of '
                       'the map.'),
    'nd.pbr_openmw': ('PBR → OpenMW (диффуз, _spec)', 'PBR → OpenMW (diffuse, _spec)'),
    'nd.pbr_openmw.info': ('У OpenMW нет PBR: цвет блика берётся из RGB карты _spec, показатель '
                           'блеска — из альфы (альфа × 255). Цвет блика — от серого 0.04 до '
                           'базового цвета по металличности, блеск — из шероховатости, щели '
                           'гасятся по AO. Металл в диффузе затемняется. Выход «маска '
                           'отражения» — карта gloss для NIF: ею OpenMW умножает отражение '
                           'карты окружения (в Morrowind — с Morrowind Code Patch); назначается '
                           'в NifSkope в слот Gloss текстуры модели с картой окружения.',
                           'OpenMW has no PBR: the specular colour comes from the RGB of _spec, '
                           'the shininess exponent from its alpha (alpha × 255). Specular colour '
                           'goes from 0.04 grey to the base colour by metalness, shininess from '
                           'roughness, crevices are dimmed by AO. Metal is darkened in the '
                           'diffuse. The «reflection mask» output is a gloss map for a NIF: '
                           'OpenMW multiplies the environment map reflection by it (Morrowind '
                           'with the Morrowind Code Patch); assign it in NifSkope to the Gloss '
                           'slot of a model that has an environment map.'),
    'nd.supermat': ('SuperMat (PBR)', 'SuperMat (PBR)'),
    'nd.supermat.info': ('Модель SuperMat (MIT): базовый цвет без света, шероховатость, металличность. Нормали и высоты не даёт — их берите из DeepBump или сети глубины. Карты бесшовные: модель считает текстуру четыре раза со сдвигом на полпериода.',
        'The SuperMat model (MIT): base colour without light, roughness, metalness. It gives no normal and no height — take them from DeepBump or a depth network. Seamless: the texture is run four times, rolled by half a period.'),
    # --- узлы: описания старых
    'nd.source.info': ('Открытая текстура и её альфа.', 'The open texture and its alpha.'),
    'nd.repair.info': ('Ищет пересвеченные и провалившиеся места и заделывает их. Что выделено, '
                       'видно в списке над 2D-окном: «маска» (белое — чинится) и «маска поверх» '
                       '(выделенное подкрашено красным). Заливки «средний цвет», «градиент» и '
                       '«пятна» берут цвета только с целых мест светлее порога провала.',
                       'Finds blown-out and crushed spots and fills them. What is selected is in '
                       'the list above the 2D view: "mask" (white is repaired) and "mask over" '
                       '(the selection tinted red). The fills "mean colour", "gradient" and '
                       '"spots" take colours only from intact pixels brighter than the crush '
                       'level.'),
    'np.mask': ('маска', 'mask'),
    'np.overlay': ('маска поверх', 'mask over'),
    'np.preset': ('Материал', 'Material'),
    'np.metal_default': ('Металличность без карты', 'Metalness without a map'),
    'np.fit': ('Вписать карты в материал', 'Fit maps into the material'),
    'np.rough_lo': ('Шероховатость материала от', 'Material roughness from'),
    'np.rough_hi': ('Шероховатость материала до', 'Material roughness to'),
    'np.metal_lo': ('Металличность материала от', 'Material metalness from'),
    'np.metal_hi': ('Металличность материала до', 'Material metalness to'),
    'nv.patch': ('заплатка', 'patch'),
    'nv.tone': ('тон', 'tone'),
    'nv.smooth': ('гладко', 'smooth'),
    'nv.mean': ('средний цвет', 'mean colour'),
    'nv.gradient': ('градиент', 'gradient'),
    'nv.spots': ('пятна основных цветов', 'spots of main colours'),
    'nv.custom': ('свой', 'custom'),
    'nv.stone': ('камень', 'stone'),
    'nv.brick': ('кирпич, штукатурка', 'brick, plaster'),
    'nv.wet': ('мокрый камень', 'wet stone'),
    'nv.ground': ('земля, песок', 'ground, sand'),
    'nv.wood': ('дерево', 'wood'),
    'nv.lacquer': ('лакированное дерево', 'lacquered wood'),
    'nv.cloth': ('ткань', 'cloth'),
    'nv.leather': ('кожа', 'leather'),
    'nv.bone': ('кость, хитин', 'bone, chitin'),
    'nv.glass': ('стекло, кристалл', 'glass, crystal'),
    'nv.ice': ('лёд', 'ice'),
    'nv.iron': ('железо, сталь', 'iron, steel'),
    'nv.rust': ('ржавое железо', 'rusty iron'),
    'nv.gold': ('золото', 'gold'),
    'nv.bronze': ('бронза, двемерский металл', 'bronze, Dwemer metal'),
    'nd.delight.info': ('Снимает плавную засветку и резкие тени, нарисованные в текстуре. '
                        'Выход «свет» — то, что снято.',
                        'Removes the smooth light gradient and hard shadows painted into the '
                        'texture. The light output is what was removed.'),
    'nd.height_lum.info': ('Высота из яркости четырьмя способами: полосовой, многомасштабный, '
                           'вся яркость, краесохраняющий.',
                           'Height from brightness in four ways: band, multi-scale, whole '
                           'brightness, edge-preserving.'),
    'nd.height_light.info': ('Высота по нарисованному освещению (shape from shading), после '
                             'вычитания краски по цвету и падающих теней.',
                             'Height from the painted lighting (shape from shading), after '
                             'removing paint by colour and cast shadows.'),
    'nd.depth_net.info': ('Глубина нейросетью (Depth Anything V2 или MiDaS). Хороша в крупной '
                          'форме, без масштаба.',
                          'Depth from a network (Depth Anything V2 or MiDaS). Good in the '
                          'large form, no scale.'),
    'nd.from_ao.info': ('Высота из карты затенения двойным интегрированием.',
                        'Height from an occlusion map by double integration.'),
    'nd.bands.info': ('Перевешивает крупное и мелкое в высоте.',
                      'Re-weights the large and fine parts of a height.'),
    'nd.tone.info': ('Контраст и гамма серой карты.', 'Contrast and gamma of a grey map.'),
    'nd.mix.info': ('Взвешенная сумма двух высот. С галочкой «Растянуть на 0…1» '
                    'результат всегда растягивается на весь диапазон, поэтому смесь '
                    'тусклых высот выходит контрастной. Без неё сохраняется контраст '
                    'входов, а сумма делится на «Делитель»: при весах 1 и делителе 2 '
                    'это простое среднее, за 0…1 оно не выходит.',
                    'Weighted sum of two heights. With "Stretch to 0…1" the result is '
                    'always stretched to the full range, so a mix of dull heights comes '
                    'out contrasty. Without it the inputs\' contrast is kept and the sum '
                    'is divided by "Divisor": with weights 1 and divisor 2 it is the plain '
                    'average, which cannot leave 0…1.'),
    'np.mix.divisor': ('Делитель (без растяжки)', 'Divisor (without stretch)'),
    'ni.beyond': ('Вышло за 0…1 (смягчено)', 'Beyond 0…1 (softened)'),
    'nd.normal.info': ('Нормаль из высоты по трём масштабам.',
                       'Normal from a height over three scales.'),
    'nd.normal_light.info': ('Нормаль прямо по освещению: R и G — слои «по освещению» для '
                             'двух направлений света.',
                             'Normal straight from the lighting: R and G are the "from '
                             'lighting" layers for two light directions.'),
    'nd.ao.info': ('Затенение из высоты. 1 — открыто.', 'Occlusion from height. 1 is open.'),
    'np.ao.method': ('Способ', 'Method'),
    'np.ao.units': ('Размеры в', 'Sizes in'),
    'nv.cm': ('сантиметрах мира', 'world centimetres'),
    'np.ao.radius_cm': ('Радиус, см', 'Radius, cm'),
    'np.ao.depth_cm': ('Глубина рельефа, см (LSAO, GTAO)', 'Relief depth, cm (LSAO, GTAO)'),
    'np.ao.depth_cm.info': ('Только для LSAO и GTAO: от самого низкого места рельефа до самого '
                            'высокого, в сантиметрах игрового мира — как глубина в «Параллаксе '
                            'для OpenMW». В пиксели переводится через «Повтор»: сколько юнитов '
                            'покрывает текстура по ширине (22,1 юнита в футе).',
                            'LSAO and GTAO only: from the lowest place of the relief to the highest, '
                            'in centimetres of the game world — as the depth in «Parallax for '
                            'OpenMW». Turned into pixels through «Repeat»: how many units the '
                            'texture covers along its width (22.1 units per foot).'),
    'np.repeat_mode': ('Размер повтора', 'Repeat size'),
    'np.repeat_mode.info': ('«Задан в юнитах» — одно число для любой текстуры. «Пропорционально '
                            'разрешению» — столько-то юнитов на пиксель ширины: текстура вдвое '
                            'шире покрывает вдвое больше (так в паке, где разрешение отражает '
                            'размер предмета). Для апскейла, где та же текстура стала крупнее, '
                            'это не подходит — там повтор тот же. При переключении число '
                            'пересчитывается, и на этой текстуре повтор остаётся прежним; режимы '
                            'расходятся только на текстурах другой ширины (в пакете).',
                            '"In units" — one number for any texture. "Proportional to resolution" '
                            '— so many units per pixel of width: a texture twice as wide covers '
                            'twice as much (for a pack where resolution follows the size of the '
                            'object). Not for an upscale, where the same texture just got more '
                            'pixels — its repeat stays the same. Switching converts the number, so on this '
                            'texture the repeat stays as it was; the modes differ only on textures '
                            'of another width (in a batch).'),
    'nv.units': ('задан в юнитах', 'in units'),
    'nv.per_px': ('пропорционально разрешению', 'proportional to resolution'),
    'np.units_px': ('Юнитов на пиксель ширины', 'Units per pixel of width'),
    'np.units_px.info': ('Повтор = это число × ширина полноразмерной текстуры в пикселях. 0,5 — '
                         'текстура 256 пикселей покрывает 128 юнитов (1,77 м). Итог — в показаниях узла.',
                         'Repeat = this number × the width of the full-size texture in pixels. 0.5 '
                         '— a 256 px texture covers 128 units (1.77 m). The result is in the readouts.'),
    'np.relief_sync': ('Общие глубина и повтор', 'Shared depth and repeat'),
    'np.relief_sync.info': ('Узлы «Параллакс для OpenMW», «Рельеф для OpenMW» и «Затенение (AO)» '
                            '(в сантиметрах) с этой галочкой держат одинаковыми глубину в '
                            'сантиметрах и размер повтора, а параллакс и рельеф — ещё и уровень '
                            'поверхности. Изменение в одном сразу переходит в другие. Включённая '
                            'галочка берёт значения у уже отмеченного узла.',
                            '«Parallax for OpenMW», «Relief for OpenMW» and «Occlusion (AO)» (in '
                            'centimetres) nodes with this box ticked keep the depth in centimetres '
                            'and the repeat size the same; parallax and relief also share the '
                            'surface level. A change in one goes to the others at once. Ticking the '
                            'box takes the values of a node already ticked.'),
    'ni.repeat_u': ('Повтор текстуры, юнитов', 'Texture repeat, units'),
    'ni.depth_cm_used': ('Глубина рельефа, см', 'Relief depth, cm'),
    'ni.tilt_mean': ('Средний наклон, °', 'Mean tilt, °'),
    'ni.avail_cm': ('Предел параллакса OpenMW, см', 'OpenMW parallax limit, cm'),
    'nd.relief_omw': ('Рельеф для OpenMW (высота и нормаль)', 'Relief for OpenMW (height and normal)'),
    'nd.relief_omw.info': ('Высота, приведённая к диапазону, который покажет игра, и нормаль ровно '
                           'этого рельефа. Выход «высота» — как альфа _nh: 0,5 — сама поверхность, '
                           'весь диапазон 0…1 — предел параллакса OpenMW (4 % повтора), рельеф '
                           'занимает из него «Глубину рельефа, см». Выход «нормаль» — по настоящим '
                           'уклонам этого рельефа в сантиметрах при заданном повторе. Если подключить '
                           'готовую нормаль (необязательный вход), на выходе будет она, со всеми '
                           'деталями, но с уклонами, приведёнными одним множителем к рельефу в '
                           'сантиметрах; другой формат зелёного распознаётся и переворачивается. '
                           'Галочкой «Общие глубина и повтор» синхронизируется с параллаксом и AO.',
                           'The height brought into the range the game shows, and the normal of '
                           'exactly that relief. The «height» output is as the _nh alpha: 0.5 is the '
                           'surface itself, the whole 0…1 is the OpenMW parallax limit (4 % of the '
                           'repeat), the relief takes «Relief depth, cm» of it. The «normal» output '
                           'comes from the real slopes of that relief in centimetres at the given '
                           'repeat. If a ready normal is connected (optional input), the output is that '
                           'normal with all its detail, its slopes brought to the relief in '
                           'centimetres by one factor; the other green layout is detected and '
                           'flipped. With «Shared depth and repeat» it stays in step with parallax '
                           'and AO.'),
    'np.parallax_nh.ref': ('Что совпадает с поверхностью', 'What lies at the surface'),
    'np.parallax_nh.ref.info': ('Какая высота окажется ровно на поверхности модели (альфа 128): средняя, '
                       'медиана, самый частый уровень (самая большая ровная площадь — на кладке это '
                       'лицо камней или заполнение швов) или «вручную» — доля от низа до верха '
                       'рельефа, «Уровень вручную».',
                       'Which height ends up exactly at the model surface (alpha 128): the mean, '
                       'the median, the most frequent level (the largest flat area — on masonry the '
                       'face of the stones or the joint filling) or «manual» — a fraction from the '
                       'bottom to the top of the relief, «Manual level».'),
    'nv.parallax_nh.mean': ('среднее', 'mean'), 'nv.parallax_nh.median': ('медиана', 'median'),
    'nv.parallax_nh.mode': ('самый частый уровень', 'most frequent level'),
    'nv.parallax_nh.manual': ('вручную', 'manual'),
    'np.relief_omw.ref': ('Что совпадает с поверхностью', 'What lies at the surface'),
    'np.relief_omw.ref.info': ('Какая высота окажется ровно на поверхности модели (альфа 128): средняя, '
                       'медиана, самый частый уровень (самая большая ровная площадь — на кладке это '
                       'лицо камней или заполнение швов) или «вручную» — доля от низа до верха '
                       'рельефа, «Уровень вручную».',
                       'Which height ends up exactly at the model surface (alpha 128): the mean, '
                       'the median, the most frequent level (the largest flat area — on masonry the '
                       'face of the stones or the joint filling) or «manual» — a fraction from the '
                       'bottom to the top of the relief, «Manual level».'),
    'nv.relief_omw.mean': ('среднее', 'mean'), 'nv.relief_omw.median': ('медиана', 'median'),
    'nv.relief_omw.mode': ('самый частый уровень', 'most frequent level'),
    'nv.relief_omw.manual': ('вручную', 'manual'),
    'np.relief_omw.depth_cm': ('Глубина рельефа, см', 'Relief depth, cm'),
    'np.relief_omw.h': ('высота', 'height'),
    'np.relief_omw.n': ('нормаль', 'normal'),
    'ni.n_gain': ('Множитель входной нормали', 'Input normal factor'),
    'ni.n_corr': ('Совпадение входной нормали с высотой, %', 'Input normal agrees with the height, %'),
    'ni.n_flip': ('Зелёный входной нормали перевёрнут', 'Input normal green flipped'),
    'relief_omw.n_mismatch': ('Входная нормаль плохо совпадает с высотой (меньше 50 %): похоже, они '
                              'про разный рельеф, и множитель ненадёжен. Проверьте, что высота и '
                              'нормаль от одной текстуры.',
                              'The input normal agrees poorly with the height (below 50 %): they seem to '
                              'describe different reliefs, and the factor is unreliable. Check that the '
                              'height and the normal come from the same texture.'),
    'ni.yes': ('да', 'yes'), 'ni.no': ('нет', 'no'),
    'ni.step_mm': ('Шаг альфы, мм', 'Alpha step, mm'),
    'ni.clipped_pc': ('Не влезло в диапазон, % пикселей', 'Clipped to the range, % of pixels'),
    'np.ao.units.info': ('«% от размера» — радиус и глубина в процентах от большей стороны '
                         'текстуры: один и тот же проект даёт одинаковый вид на текстурах '
                         'любого размера. «Пикселях» — в пикселях полноразмерной текстуры.',
                         '"% of size" — radius and depth in per cent of the texture\'s larger '
                         'side: the same project gives the same look on textures of any size. '
                         '"Pixels" — in pixels of the full-size texture.'),
    'np.ao.n_depth': ('Глубина из нормали (если подключена)', 'Depth from the normal (if connected)'),
    'np.ao.n_depth.info': ('Если ко входу «нормаль» подключена карта нормалей того же рельефа, '
                           'глубина берётся из неё: подбирается такой масштаб высоты, при котором '
                           'её склоны совпадают с наклонами нормали. Тогда AO соответствует '
                           'рельефу, который игра покажет через эту нормаль, а ползунок глубины '
                           'не действует. Выключите, чтобы задать глубину ползунком. Для GTAO '
                           'наклон точки берётся из нормали в любом случае.',
                           'If a normal map of the same relief is connected to the «normal» input, '
                           'the depth is taken from it: the height scale at which its slopes '
                           'match the normal\'s. The AO then fits the relief the game will show '
                           'through that normal, and the depth slider has no effect. Turn off to '
                           'set the depth with the slider. For GTAO the tilt of each point comes '
                           'from the normal either way.'),
    'ao.nsize': ('Нормаль должна быть того же размера, что высота.',
                 'The normal must be the same size as the height.'),
    'ao.nbad': ('нормаль не от этого рельефа — глубина с ползунка',
                'the normal is not of this relief — depth from the slider'),
    'ni.n_fit': ('Совпадение нормали с высотой, %', 'Normal fits the height, %'),
    'ni.n_bad': ('Нормаль', 'Normal'),
    'ni.depth_used': ('Глубина рельефа, % большей стороны', 'Relief depth, % of the larger side'),
    'ni.depth_px': ('Глубина рельефа, пикс. полного размера', 'Relief depth, px of full size'),
    'np.ao.falloff': ('Спад с расстоянием (LSAO, GTAO)', 'Falloff with distance (LSAO, GTAO)'),
    'np.ao.falloff.info': ('Выключено — как в физике: затенитель считается на любом '
                           'расстоянии, свет неба закрывает всё, что выше горизонта. '
                           'Включено — дальние затенители слабеют, вдвое на расстоянии '
                           '«Радиуса» (ambient obscurance): художественный приём, не физика.',
                           'Off — as in physics: an occluder counts at any distance, everything '
                           'above the horizon blocks the sky. On — far occluders weaken, by half '
                           'at the «Radius» distance (ambient obscurance): an artistic choice, '
                           'not physics.'),
    'nv.pct': ('% от размера текстуры', '% of texture size'),
    'nv.px': ('пикселях', 'pixels'),
    'np.ao.radius_pct': ('Радиус, % от размера', 'Radius, % of size'),
    'np.ao.depth_pct': ('Глубина рельефа, % (LSAO, GTAO)', 'Relief depth, % (LSAO, GTAO)'),
    'np.ao.depth_pct.info': ('Только для LSAO и GTAO: высота 1 соответствует стольким '
                             'процентам большей стороны текстуры. Чем глубже, тем выше '
                             'горизонт и темнее впадины.',
                             'LSAO and GTAO only: height 1 is this many per cent of the '
                             'texture\'s larger side. The deeper, the higher the horizon and the '
                             'darker the hollows.'),
    'np.ao.method.info': ('«По впадинам» — точка темнеет настолько, насколько лежит ниже '
                          'окружения; не зависит от глубины рельефа, устойчиво на шумных '
                          'текстурах. «Line sweep (LSAO)» — честный поиск горизонта по '
                          'линиям (Тимонен, 2013): затенение зависит от настоящей глубины '
                          '(«Глубина рельефа») и от направления; «Радиус» — на каком '
                          'расстоянии затенитель слабеет вдвое, если включён «Спад». «GTAO» (Хименес и др., 2016) — тот же '
                          'поиск горизонта, но считается доля рассеянного света, которая '
                          'реально доходит до поверхности, с учётом её наклона (закон '
                          'косинуса): склон не темнеет только потому, что он склон. '
                          'При выборе способа «Сила» ставится 1,0 для LSAO и GTAO — затенение как '
                          'посчитано, без усиления (GTAO при этом — физическая доля света), '
                          'и 0,9 для «по впадинам»: у него физического масштаба нет.',
                          '"Hollows" — a point darkens by as much as it lies below its '
                          'surroundings; does not depend on the relief depth, robust on '
                          'noisy textures. "Line sweep (LSAO)" — a true horizon search along '
                          'lines (Timonen, 2013): the occlusion depends on the real depth '
                          '("Relief depth") and on direction; "Radius" is the distance at '
                          'which an occluder counts half when «Falloff» is on. "GTAO" (Jimenez et al., 2016) — the same '
                          'horizon search, but it gives the share of diffuse light that '
                          'really reaches the surface, taking its tilt into account (cosine '
                          'law): a slope is not darkened just for being a slope. Picking a '
                          'method sets «Strength»: 1.0 for LSAO and GTAO — the occlusion as '
                          'computed, no boost (GTAO is then the physical share of light), and 0.9 '
                          'for "hollows", which has no physical scale.'),
    'nv.cavity': ('по впадинам', 'hollows'),
    'nv.lsao': ('Line sweep (LSAO)', 'Line sweep (LSAO)'),
    'nv.gtao': ('GTAO', 'GTAO'),
    'np.ao.depth': ('Глубина рельефа, пикс. (LSAO, GTAO)', 'Relief depth, px (LSAO, GTAO)'),
    'np.ao.depth.info': ('Только для LSAO и GTAO: высота 1 соответствует стольким пикселям '
                         'полноразмерной текстуры. Чем глубже, тем выше горизонт и темнее '
                         'впадины.',
                         'LSAO and GTAO only: height 1 is this many pixels of the full-size '
                         'texture. The deeper, the higher the horizon and the darker the '
                         'hollows.'),
    'np.ao.dirs': ('Направлений (LSAO, GTAO)', 'Directions (LSAO, GTAO)'),
    'np.ao.dirs.info': ('Только для LSAO и GTAO: сколько направлений по кругу. Больше — ровнее, '
                        'но дольше (время растёт пропорционально).',
                        'LSAO and GTAO only: how many directions round the circle. More is smoother '
                        'but slower (time grows in proportion).'),
    'nd.bent.info': ('Подмешивает bent normal — направление, откуда приходит рассеянный свет.',
                     'Blends in a bent normal — the direction ambient light comes from.'),
    'nd.spec.info': ('Цвет блика, шероховатость и металличность по диффузу.',
                     'Specular colour, roughness and metalness from the diffuse.'),
    'nd.seamless.info': ('Делает карту тайлящейся, заменяя полосу у краёв.',
                         'Makes a map tile by replacing a band at the edges.'),
    'nd.out.info': ('Что записать и с каким окончанием имени. «Диффуз» — под именем '
                    'исходника (только в другую папку).',
                    'What to write and with which name ending. Diffuse — under the source '
                    'name (only into another folder).'),
    # --- гнёзда и параметры
    'np.base': ('базовый цвет', 'base colour'),
    'np.diffuse': ('диффуз', 'diffuse'),
    'np.s': ('карта', 'map'),
    'np.overlap': ('Перекрытие окон', 'Window overlap'),
    'np.balance': ('Баланс: высота ← → нормаль', 'Balance: height ← → normal'),
    'np.balance.info': ('0 — всё решает высота, 1 — всё решает нормаль. Между ними нормаль '
                        'отвечает за детали мельче границы, высота — за крупнее. Граница в '
                        'пикселях — в показаниях.',
                        '0 — the height decides everything, 1 — the normal. In between the '
                        'normal is in charge of detail finer than the border, the height of '
                        'larger forms. The border in pixels is in the readouts.'),
    'np.keep': ('Оставить своё у нормали', "Keep the normal's own"),
    'np.gain': ('Усилить высоту', 'Boost height'),
    'np.green': ('Зелёный на входе', 'Green of the input'),
    'np.min_z': ('Предел наклона (мин. Z)', 'Tilt limit (min Z)'),
    'np.occl': ('Это AO (1 — открыто)', 'This is AO (1 = open)'),
    'np.metal_dark': ('Затемнять металл', 'Darken metal'),
    'np.ao_diffuse': ('AO в диффуз', 'AO into diffuse'),
    'np.ao_spec': ('AO в блик', 'AO into specular'),
    'np.spec_gain': ('Сила блика', 'Specular strength'),
    'np.pbr_openmw.gloss': ('маска отражения (gloss)', 'reflection mask (gloss)'),
    'np.env_rough': ('Шероховатость гасит отражение', 'Roughness dims the reflection'),
    'np.env_rough.info': ('Для выхода «маска отражения»: карта окружения Morrowind не умеет '
                          'размываться, и шероховатая поверхность с чётким зеркальным '
                          'отражением выглядела бы хромом. 1 — отражение гаснет как '
                          '(1 − шероховатость)², 0 — не гаснет совсем (физически отражённая '
                          'доля от шероховатости не зависит, она только размывается).',
                          'For the «reflection mask» output: a Morrowind environment map cannot '
                          'blur, and a rough surface with a sharp mirror image would look like '
                          'chrome. 1 — the reflection fades as (1 − roughness)², 0 — not at all '
                          '(physically the reflected share does not depend on roughness, it only '
                          'spreads).'),
    'ni.gloss_steps': ('Маска отражения, среднее, шагов из 255', 'Reflection mask, mean, steps of 255'),
    'np.rough_default': ('Шероховатость без карты', 'Roughness without a map'),
    'np.res': ('Размер для модели', 'Model resolution'),
    'nv.overlay': ('перекрытие', 'overlay'),
    'nv.soft': ('мягкий свет', 'soft light'),
    'nv.multiply': ('умножение', 'multiply'),
    'nv.burn': ('линейное затемнение', 'linear burn'),
    'nv.screen': ('экран', 'screen'),
    'nv.darken': ('замена тёмным', 'darken'),
    'nv.lighten': ('замена светлым', 'lighten'),
    'ed.port': ('Выход:', 'Output:'),
    'nd.mix_rgb': ('Смешать цвет', 'Mix colour'),
    'nd.mix_rgb.info': ('Два диффуза как слои в Photoshop: А — нижний, Б — верхний, '
                        '«сколько» — непрозрачность Б, маска (необязательно) — где Б действует.',
                        'Two diffuse maps as Photoshop layers: A at the bottom, B on top, '
                        'amount is B\'s opacity, the optional mask says where B applies.'),
    'nv.detail': ('деталь', 'detail'),
    'nv.linear': ('линейно', 'linear'),
    'nv.mix': ('смешать', 'mix'),
    'nv.add': ('наложить', 'add'),
    'nv.rnm': ('наложить (RNM)', 'add (RNM)'),
    'ni.tilt_a': ('Средний наклон А', 'Mean tilt of A'),
    'ni.tilt_b': ('Средний наклон Б', 'Mean tilt of B'),
    'nv.auto': ('определить', 'detect'),
    'nv.directx': ('DirectX', 'DirectX'),
    'nv.opengl': ('OpenGL', 'OpenGL'),
    'nv.diffuse': ('диффуз (своё имя)', 'diffuse (own name)'),
    # --- показания
    'ni.depth_px': ('Глубина рельефа', 'Relief depth'),
    'ni.curl': ('Ротор нормали', 'Curl of the normal'),
    'ni.curl.info': ('Насколько наклоны не складываются ни в какую высоту. Около 0.1–0.2 — '
                     'честная нормаль, около 0.5 и выше — наклоны без общей высоты (или '
                     'перепутан зелёный).',
                     'How far the slopes fail to add up to any height. About 0.1–0.2 — an '
                     'honest normal, about 0.5 and above — slopes without a common height (or '
                     'green is mixed up).'),
    'ni.crossover': ('Граница нормаль/высота', 'Normal/height border'),
    'ni.scale': ('Подобранный масштаб высоты', 'Fitted height scale'),
    'ni.corr': ('Согласие карт в полосе', 'Agreement in the band'),
    'ni.corr.info': ('Корреляция наклонов нормали и высоты там, где подбирается масштаб. '
                     'Около 0 — высота нормали ничего не подтверждает (масштаб выйдет малым, '
                     'крупная форма сгладится). Минус — высота вывернута относительно '
                     'нормали.',
                     'Correlation of the slopes of the normal and of the height where the '
                     'scale is fitted. Near 0 — the height confirms nothing about the normal '
                     '(the scale comes out small and the large form is flattened). Negative — '
                     'the height is inside out relative to the normal.'),
    'ni.curl_in': ('Ротор на входе', 'Curl in'),
    'ni.curl_out': ('Ротор на выходе', 'Curl out'),
    'ni.angle_n': ('Отличие от входной нормали', 'Difference from the input normal'),
    'ni.angle_h': ('Отличие от нормали высоты', "Difference from the height's normal"),
    'ni.tilt_in': ('Средний наклон до', 'Mean tilt before'),
    'ni.tilt_out': ('Средний наклон после', 'Mean tilt after'),
    'ni.steps': ('Среднее изменение', 'Mean change'),
    'ni.metal': ('Средняя металличность', 'Mean metalness'),
    'ni.rough': ('Средняя шероховатость', 'Mean roughness'),
    'ni.exponent': ('Средний показатель блеска', 'Mean shininess exponent'),
    'ni.rough_in_lo': ('Входная шероховатость, 2 %', 'Incoming roughness, 2 %'),
    'ni.rough_in_hi': ('Входная шероховатость, 98 %', 'Incoming roughness, 98 %'),
    'ni.metal_in_lo': ('Входная металличность, 2 %', 'Incoming metalness, 2 %'),
    'ni.metal_in_hi': ('Входная металличность, 98 %', 'Incoming metalness, 98 %'),
    'ni.green': ('Раскладка зелёного', 'Green layout'),
    'ni.curl_dx': ('Ротор, если DirectX', 'Curl if DirectX'),
    'ni.curl_gl': ('Ротор, если OpenGL', 'Curl if OpenGL'),
    'ni.device': ('Считалось на', 'Ran on'),
    'ed.m.all': ('Установить недостающее', 'Install what is missing'),
    'ed.refining': ('(уточняется…)', '(refining…)'),
    'nd.nomodel': ('Модель «%s» не скачана — кнопка «Установить недостающее» выше.',
                   'Model "%s" is not downloaded — use "Install what is missing" above.'),
    'gpu.head': ('Узлы и 3D-вид', 'Nodes and the 3D view'),
    'gpu.use': ('Считать на видеокарте', 'Compute on the graphics card'),
    'gpu.on': ('Сейчас: видеокарта %s (CuPy).', 'Now: graphics card %s (CuPy).'),
    'gpu.off': ('Сейчас: процессор. Видеокарта не включилась: %s',
                'Now: processor. The card did not start: %s'),
    'gpu.nocupy': ('Сейчас: процессор — CuPy не установлен.',
                   'Now: processor — CuPy is not installed.'),
    'gpu.info': ('CuPy (MIT) считает узлы и 3D-вид на видеокарте NVIDIA. Ставится с '
                 'библиотеками CUDA 13 от NVIDIA (они закрытые), около 1 ГБ. Нужен драйвер '
                 'NVIDIA 580 или новее. Без CuPy всё считается на процессоре, как раньше.',
                 'CuPy (MIT) computes the nodes and the 3D view on an NVIDIA card. It comes '
                 'with the NVIDIA CUDA 13 libraries (closed source), about 1 GB. Needs NVIDIA '
                 'driver 580 or newer. Without CuPy everything runs on the processor as '
                 'before.'),
    'gpu.install': ('Установить CuPy', 'Install CuPy'),
    'gpu.short': ('видеокарта: %s', 'card: %s'),
    'gpu.cpu': ('процессор', 'processor'),
    'ed.v3depth': ('Глубина', 'Depth'),
    'ed.v3game': ('Как в игре, с расстояния', 'As in the game, from'),
    'u.line': ('%.0f юн. = %s = %.1f″', '%.0f units = %s = %.1f″'),
    'u.cm': ('%.1f см', '%.1f cm'),
    'u.m': ('%.2f м', '%.2f m'),
    'np.repeat.info': ('Сколько игровых юнитов занимает один повтор текстуры на модели. '
                       'Юнит Morrowind и OpenMW: 22,1 юнита в футе, то есть 1 юнит = 1,38 см = '
                       '0,54 дюйма; 100 юнитов = 1,38 м = 54,3 дюйма. Перевод текущего '
                       'значения — под ползунком.',
                       'How many game units one texture repeat covers on the model. The '
                       'Morrowind and OpenMW unit: 22.1 units per foot, i.e. 1 unit = 1.38 cm '
                       '= 0.54 inch; 100 units = 1.38 m = 54.3 inches. The current value '
                       'converted is shown under the slider.'),
    'ed.v3dist': ('%.0f юн. = %.1f м (повтор %.0f юн.)', '%.0f units = %.1f m (repeat %.0f u.)'),
    'ed.m.file': ('Файл…', 'File…'),
    'ed.m.badsize': ('Размер файла %d байт, а должен быть %d. Скорее всего, файл докачан не '
                     'до конца или это другой файл.',
                     'The file is %d bytes, it must be %d. Most likely it is incomplete or a '
                     'different file.'),
    'ed.m.nodl': ('Не скачалось: %s\n\nМожно скачать файл браузером:\n%s\nи указать его '
                  'кнопкой «Файл…» рядом с моделью (или положить в папку %s). Уже скачанная '
                  'часть сохранена — повторное нажатие докачивает с того же места.',
                  'Download failed: %s\n\nYou can download the file in a browser:\n%s\nand '
                  'pick it with the File… button next to the model (or put it into %s). The '
                  'part already downloaded is kept — pressing again continues from there.'),
    'ed.m.title': ('Установка', 'Installing'),
    'ed.m.close': ('Закрыть', 'Close'),
    'ed.m.alldone': ('Готово. Можно закрыть.', 'Done. You can close this.'),
    'rt.choose': ('Где считать…', 'Compute on…'),
    'rt.title': ('Где считать нейросети', 'Where networks compute'),
    'rt.head': ('Сборка onnxruntime', 'onnxruntime build'),
    'rt.cur': ('Сейчас: %s', 'Now: %s'),
    'rt.none': ('не установлено', 'not installed'),
    'rt.go': ('Установить', 'Install'),
    'rt.cpu': ('Процессор', 'Processor'),
    'rt.cpu.info': ('onnxruntime, около 15 МБ. Работает везде, но медленнее всего.',
                    'onnxruntime, about 15 MB. Works everywhere, slowest.'),
    'rt.webgpu': ('Видеокарта через WebGPU', 'Graphics card via WebGPU'),
    'rt.webgpu.info': ('onnxruntime-webgpu, около 25 МБ. Считает через Dawn поверх DirectX 12 '
                       'или Vulkan — любая современная видеокарта. Всё открытое, кроме драйвера '
                       'видеокарты. Проверено: результаты DeepBump, Depth Anything и ESRGAN '
                       'совпадают с процессором до 1e-5; скорость на настоящей видеокарте не '
                       'проверялась.',
                       'onnxruntime-webgpu, about 25 MB. Computes through Dawn on DirectX 12 '
                       'or Vulkan — any modern graphics card. All open source except the '
                       'graphics driver. Checked: DeepBump, Depth Anything and ESRGAN results '
                       'match the processor to 1e-5; speed on a real card untested.'),
    'rt.cuda': ('Видеокарта NVIDIA через CUDA', 'NVIDIA card via CUDA'),
    'rt.cuda.info': ('onnxruntime-gpu вместе с библиотеками CUDA 13 и cuDNN от NVIDIA, около '
                     '1,5 ГБ. Библиотеки NVIDIA закрытые. Нужен драйвер NVIDIA версии 580 или '
                     'новее. Обычно самый быстрый вариант для карт NVIDIA.',
                     'onnxruntime-gpu with the NVIDIA CUDA 13 and cuDNN libraries, about 1.5 '
                     'GB. The NVIDIA libraries are closed source. Needs NVIDIA driver 580 or '
                     'newer. Usually the fastest for NVIDIA cards.'),
    'rt.directml': ('Видеокарта через DirectML (Windows)', 'Graphics card via DirectML (Windows)'),
    'rt.directml.info': ('onnxruntime-directml, около 20 МБ. DirectML — закрытая библиотека '
                         'Microsoft, её развитие остановлено. Запасной вариант, если WebGPU не '
                         'заработал.',
                         'onnxruntime-directml, about 20 MB. DirectML is a closed Microsoft '
                         'library, no longer developed. A fallback if WebGPU does not work.'),
    'rt.restart': ('Сейчас загружена прежняя сборка, заменить её на ходу нельзя. Выбор '
                   'сохранён: перезапустите Texelith — смена пройдёт при запуске.',
                   'The previous build is loaded and cannot be replaced while running. The '
                   'choice is saved: restart Texelith and the switch happens at start-up.'),
    'ni.device.info': ('Где работала сеть в этом сеансе. «—» — результат взят из кэша на '
                       'диске, сеть не запускалась.',
                       'Where the network ran in this session. "—" — the result came from '
                       'the disk cache, the network did not run.'),
})

# у узла «Починка битых мест» те же подсказки, что у классической вкладки
S['np.flat.info'] = S['m.rep.flat.info']
S['np.grow.info'] = S['m.rep.grow.info']
S['np.preset.info'] = ('Выбор материала ставит ползунки ниже в типичные для него значения: '
                       'отражение f0, диапазоны шероховатости и металличности, значения без '
                       'карт, затемнение металла, силу блика — и включает «вписать карты». '
                       'Потом их можно подвинуть.',
                       'Picking a material sets the sliders below to values typical for it: '
                       'reflectance f0, roughness and metalness ranges, values without maps, '
                       'metal darkening, highlight gain — and turns on "fit maps". You can '
                       'move them afterwards.')
S['np.fit.info'] = ('Подключённые карты шероховатости и металличности укладываются в диапазон '
                    'материала: их собственный разброс (от 2 до 98 %) растягивается на «от…до», '
                    'рисунок карты сохраняется, меняются только уровни. 0 — карты как есть. '
                    'В показаниях видно, какой разброс пришёл на вход.',
                    'Connected roughness and metalness maps are laid into the material range: '
                    'their own spread (from 2 to 98 %) is stretched onto "from…to", the '
                    'pattern stays, only the levels change. 0 uses the maps as they are. The '
                    'readouts show the incoming spread.')

# --- узлы-фильтры
S.update({
    'nd.highpass': ('Убрать крупное (high-pass)', 'Remove large (high-pass)'),
    'nd.highpass.info': ('Вычитает всё крупнее заданного размера: перепады, пятна, общий наклон. '
                         'Мелкий рельеф остаётся. Выход «Крупное» — то, что убрано.',
                         'Subtracts everything larger than the given size: slopes, blotches, '
                         'overall tilt. The fine relief stays. The "Large scale" output is what '
                         'was removed.'),
    'nd.lowpass': ('Размыть (low-pass)', 'Blur (low-pass)'),
    'nd.lowpass.info': ('Убирает всё мельче заданного размера: шум, зерно, мелкие детали.',
                        'Removes everything smaller than the given size: noise, grain, fine '
                        'detail.'),
    'nd.bandpass': ('Полоса масштабов (band-pass)', 'Band of scales (band-pass)'),
    'nd.bandpass.info': ('Оставляет только масштабы между двумя размерами: мельче — шум, крупнее '
                         '— перепады, обе стороны убираются.',
                         'Keeps only the scales between two sizes: noise below and slopes above '
                         'are both removed.'),
    'nd.median': ('Медиана', 'Median'),
    'nd.median.info': ('Убирает точки и одиночные выбросы, края не размывает.',
                       'Removes specks and single outliers without blurring edges.'),
    'nd.sharpen': ('Резкость', 'Sharpen'),
    'nd.sharpen.info': ('Усиливает детали мельче заданного размера (нерезкая маска).',
                        'Strengthens detail finer than the given size (unsharp mask).'),
    'np.lo_cut': ('Убрать крупнее, доля стороны', 'Remove larger than, share of side'),
    'np.hi_cut': ('Убрать мельче, доля стороны', 'Remove smaller than, share of side'),
    'np.remove': ('Сколько убрать', 'How much to remove'),
    'np.stretch': ('Растянуть на 0…1', 'Stretch to 0…1'),
    'np.win': ('Окно, пикс.', 'Window, px'),
    'np.sharp': ('Сила резкости', 'Sharpening strength'),
    'np.keep_edges': ('Не трогать резкие края (без ореолов)', 'Keep sharp edges (no halos)'),
    'np.edge_level': ('Край — перепад от', 'An edge is a step from'),
    'nd.edge_smooth': ('Сгладить, не трогая края', 'Smooth, keep edges'),
    'nd.edge_smooth.info': ('Сглаживает фактуру и плавные перепады, а резкие края оставляет '
                            'резкими (guided filter).',
                            'Smooths texture and gentle slopes while sharp edges stay sharp '
                            '(guided filter).'),
    'ni.size_px': ('Размер, пикс. текстуры', 'Size, texture px'),
    'ni.small_px': ('Мелкий край, пикс. текстуры', 'Fine edge, texture px'),
    'ni.low_share': ('Доля крупного в разбросе, %', 'Large-scale share of spread, %'),
})
S['np.lo_cut.info'] = ('Размер в долях короткой стороны текстуры: 0.08 у текстуры 1024 — около 80 '
                       'пикселей. Так один граф одинаково работает на любом размере. В показаниях '
                       'тот же размер в пикселях.',
                       'Size as a share of the texture\'s shorter side: 0.08 on a 1024 texture is '
                       'about 80 px, so one graph works the same at any size. The readouts give '
                       'it in pixels.')
S['np.hi_cut.info'] = S['np.lo_cut.info']
S['np.stretch.info'] = ('Включено — результат растягивается на весь диапазон 0…1. Выключено — '
                        'средний уровень остаётся как у входа.',
                        'On — the result is stretched to the full 0…1 range. Off — the mean '
                        'level stays as in the input.')
S['np.keep_edges.info'] = ('Обычное размытие растягивает резкое пятно в широкий бугор, и после '
                          'вычитания вокруг пятна остаётся кольцо — ореол. С этой галкой крупное '
                          'считается с сохранением краёв: резкие перепады остаются в «крупном» '
                          'целиком, кольца нет. Сами такие перепады при этом тоже убираются.',
                          'A plain blur spreads a sharp spot into a wide hump, and subtracting it '
                          'leaves a ring around the spot — a halo. With this on, the large part is '
                          'estimated keeping edges: sharp steps stay in the "large" part whole, no '
                          'ring is left. Those steps are then removed too.')
S['np.edge_level.info'] = ('Какой перепад высоты (в долях 0…1) считать краем. Меньше — больше мест '
                           'считаются краями и меньше ореолов, но и мелкий рельеф у краёв уходит.',
                           'How large a height step (in 0…1) counts as an edge. Lower — more places '
                           'count as edges and fewer halos, but fine relief near edges goes too.')

# --- фильтры и коррекция как в Photoshop / GIMP (сборка 24) ---------------
# Названия — как в русском Photoshop; в скобках — английское, по которому
# фильтр легко найти в справке Photoshop и GIMP.
S.update({
    'cat.adjust': ('Коррекция', 'Adjustments'),
    'cat.filter': ('Фильтры', 'Filters'),
    'nd.highpass': ('Цветовой контраст (High Pass)', 'High Pass'),
    'nd.highpass.info': ('Вычитает всё крупнее радиуса: перепады освещения, пятна, общий наклон. '
                         'Выход «убранное» показывает, что именно ушло.',
                         'Takes away everything larger than the radius: lighting gradients, '
                         'blotches, overall tilt. The «removed» output shows what went.'),
    'nd.lowpass': ('Размытие по Гауссу', 'Gaussian Blur'),
    'nd.lowpass.info': ('Обычное размытие; радиус — сигма гауссианы в пикселях.',
                        'Plain blur; the radius is the Gaussian sigma in pixels.'),
    'nd.box_blur': ('Размытие «Рамка» (Box Blur)', 'Box Blur'),
    'nd.box_blur.info': ('Среднее по квадрату со стороной 2·радиус + 1.',
                         'The mean over a square of side 2·radius + 1.'),
    'nd.motion_blur': ('Размытие в движении (Motion Blur)', 'Motion Blur'),
    'nd.motion_blur.info': ('Среднее вдоль отрезка заданной длины и угла (0° — по горизонтали).',
                            'The mean along a line of the given length and angle (0° — '
                            'horizontal).'),
    'nd.edge_smooth': ('Размытие по поверхности (Surface Blur)', 'Surface Blur'),
    'nd.edge_smooth.info': ('Размывает в пределах радиуса, но перепады больше порога остаются '
                            'резкими.',
                            'Blurs within the radius, but steps above the threshold stay sharp.'),
    'nd.sharpen': ('Контурная резкость (Unsharp Mask)', 'Unsharp Mask'),
    'nd.sharpen.info': ('Усиливает отличие от размытой копии: эффект, радиус и порог — как в '
                        'Photoshop.',
                        'Strengthens the difference from a blurred copy: amount, radius and '
                        'threshold as in Photoshop.'),
    'nd.add_noise': ('Добавить шум (Add Noise)', 'Add Noise'),
    'nd.add_noise.info': ('Равномерный или гауссов шум; одинаковое «зерно» даёт одинаковый шум.',
                          'Uniform or Gaussian noise; the same seed gives the same noise.'),
    'nd.median': ('Медиана (Median)', 'Median'),
    'nd.median.info': ('Убирает точки и одиночные выбросы, края не размывает.',
                       'Removes specks and single outliers without blurring edges.'),
    'nd.dust': ('Пыль и царапины (Dust & Scratches)', 'Dust & Scratches'),
    'nd.dust.info': ('Заменяет медианой только те пиксели, что отличаются от неё больше порога.',
                     'Replaces with the median only the pixels that differ from it by more than '
                     'the threshold.'),
    'nd.denoise': ('Уменьшить шум (Reduce Noise)', 'Reduce Noise'),
    'nd.denoise.info': ('Подавляет шум, сравнивая похожие участки (non-local means). Считается на '
                        'видеокарте, если она включена; без неё — на процессоре, на больших '
                        'текстурах долго.',
                        'Suppresses noise by comparing similar patches (non-local means). Runs on '
                        'the video card when it is on; without it on the processor, slow on large '
                        'textures.'),
    'nd.maximum': ('Максимум (Maximum)', 'Maximum'),
    'nd.maximum.info': ('Наибольшее значение в круге радиуса: светлое расширяется (в GIMP — '
                        '«Расширение»).',
                        'The largest value within the radius: light areas grow (GIMP: Dilate).'),
    'nd.minimum': ('Минимум (Minimum)', 'Minimum'),
    'nd.minimum.info': ('Наименьшее значение в круге радиуса: тёмное расширяется (в GIMP — '
                        '«Сужение»).',
                        'The smallest value within the radius: dark areas grow (GIMP: Erode).'),
    'nd.offset': ('Сдвиг (Offset)', 'Offset'),
    'nd.offset.info': ('Сдвиг с переносом через край. 50 / 50 выводит края плитки в середину — '
                       'шов, если он есть, виден сразу.',
                       'Shift with wrap-around. 50 / 50 brings the tile edges to the middle — a '
                       'seam, if any, is in plain view.'),
    'nd.emboss': ('Тиснение (Emboss)', 'Emboss'),
    'nd.emboss.info': ('Серый 50 % плюс разница значений вдоль направления света.',
                       '50 % grey plus the difference of values along the light direction.'),
    'nd.edges': ('Выделение краёв (Find Edges)', 'Find Edges'),
    'nd.edges.info': ('Сила перепада по Собелю, Превитту, Шарру или Лапласу (в GIMP — «Обнаружение '
                      'краёв»).',
                      'Strength of change by Sobel, Prewitt, Scharr or Laplace (GIMP: Edge '
                      'Detect).'),
    'nd.levels': ('Уровни (Levels)', 'Levels'),
    'nd.levels.info': ('Точки чёрного и белого на входе, гамма средних тонов, точки на выходе — как '
                       'в Photoshop.',
                       'Input black and white points, midtone gamma, output points — as in '
                       'Photoshop.'),
    'nd.curves': ('Кривые (Curves)', 'Curves'),
    'nd.curves.info': ('Кривая уровней, как в диалоге «Кривые» Photoshop и GIMP. Под кривой — '
                       'гистограмма входа. Между точками — плавный сплайн.',
                       'A levels curve, as in the Curves dialogue of Photoshop and GIMP. Under '
                       'the curve — the histogram of the input. A smooth spline between the points.'),
    'nd.paint_mask': ('Нарисовать маску', 'Paint a mask'),
    'nd.paint_mask.info': ('Маска, нарисованная кистью прямо в окне 2D. Выход «поверх картинки» '
                           'показывает текстуру (или то, что подключено ко входу) с красной '
                           'заливкой там, где маска чёрная, — по нему удобно рисовать; выход '
                           '«маска» — сама маска, белое = 1.',
                           'A mask painted with a brush right in the 2D view. The "over the '
                           'picture" output shows the texture (or whatever is connected to the '
                           'input) tinted red where the mask is black, which is handy to paint '
                           'on; the "mask" output is the mask itself, white = 1.'),
    'np.paint_mask.rgb': ('картинка', 'picture'),
    'np.paint_mask.over': ('поверх картинки', 'over the picture'),
    'np.paint_mask.mask': ('маска', 'mask'),
    'np.paint_mask.base': ('Исходно', 'Start with'),
    'nv.black': ('чёрная', 'black'),
    'nv.white': ('белая', 'white'),
    'np.paint_mask.feather': ('Растушёвка, пикс.', 'Feather, px'),
    'np.paint_mask.invert': ('Инвертировать', 'Invert'),
    'ni.strokes': ('Мазков', 'Strokes'),
    'ed.mask': ('Кисть', 'Brush'),
    'ed.mask.how': ('Левая кнопка в окне 2D рисует белым, правая — чёрным. Вид двигается '
                    'средней кнопкой, колесо — масштаб. Каждый мазок — один шаг отмены '
                    '(Ctrl+Z).',
                    'In the 2D view the left button paints white, the right one black. Pan '
                    'with the middle button, zoom with the wheel. Each stroke is one undo '
                    'step (Ctrl+Z).'),
    'ed.mask.size': ('Размер кисти, пикс.', 'Brush size, px'),
    'ed.mask.hard': ('Жёсткость', 'Hardness'),
    'ed.mask.op': ('Непрозрачность', 'Opacity'),
    'ed.mask.n': ('Мазков: %d', 'Strokes: %d'),
    'ed.mask.undo': ('Убрать последний', 'Remove the last'),
    'ed.mask.clear': ('Очистить', 'Clear'),
    'np.foliage.seg': ('разбиение', 'element split'),
    'np.foliage.branch': ('листья и ветки', 'leaves and branches'),
    'ed.curve.in': ('Вход', 'Input'),
    'ed.curve.out': ('Выход', 'Output'),
    'ed.curve.reset': ('Сбросить', 'Reset'),
    'ed.curve.hint': ('Щелчок по графику — новая точка; точку можно тянуть. Убрать — правой '
                      'кнопкой или утащить за край. Вход и выход выбранной точки можно вписать '
                      '(0…255) и нажать Enter.',
                      'Click on the plot — a new point; points can be dragged. Remove — right '
                      'click or drag it out of the plot. The selected point\'s input and output '
                      'can be typed (0…255), then Enter.'),
    'nd.brightness': ('Яркость/Контрастность', 'Brightness/Contrast'),
    'nd.brightness.info': ('Яркость прибавляется, контрастность растягивает разброс вокруг '
                           'среднего.',
                           'Brightness is added, contrast stretches the spread round the mean.'),
    'nd.invert': ('Инверсия (Invert)', 'Invert'),
    'nd.invert.info': ('1 − значение; альфа не меняется.', '1 − value; alpha is kept.'),
    'nd.threshold': ('Изогелия (Threshold)', 'Threshold'),
    'nd.threshold.info': ('Белое там, где яркость не ниже порога, остальное чёрное.',
                          'White where the brightness is at least the threshold, black '
                          'elsewhere.'),
    'nd.posterize': ('Постеризация (Posterize)', 'Posterize'),
    'nd.posterize.info': ('Столько уровней на канал.', 'So many levels per channel.'),
    'nd.equalize': ('Выровнять яркость (Equalize)', 'Equalize'),
    'nd.equalize.info': ('Гистограмма каждого канала становится ровной.',
                         'Each channel\'s histogram is made flat.'),
    'nd.auto_contrast': ('Автоконтраст (Auto Contrast)', 'Auto Contrast'),
    'nd.auto_contrast.info': ('Растягивает на весь диапазон, отбросив заданную долю самых тёмных и '
                              'самых светлых пикселей; каналы вместе, оттенки не сдвигаются.',
                              'Stretches to the full range, clipping the given share of the '
                              'darkest and lightest pixels; channels together, hues stay.'),
    'nd.exposure': ('Экспозиция (Exposure)', 'Exposure'),
    'nd.exposure.info': ('Экспозиция в ступенях, сдвиг и гамма — в линейном свете, как в '
                         'Photoshop.',
                         'Exposure in stops, offset and gamma — in linear light, as in '
                         'Photoshop.'),
    'nd.hue_sat': ('Цветовой тон/Насыщенность', 'Hue/Saturation'),
    'nd.hue_sat.info': ('Сдвиг тона в градусах, насыщенность и яркость −100…100.',
                        'Hue shift in degrees, saturation and lightness −100…100.'),
    'nd.color_balance': ('Цветовой баланс (Color Balance)', 'Color Balance'),
    'nd.color_balance.info': ('Сдвиг цвета в тенях, средних тонах или светах. Для нескольких '
                              'диапазонов — несколько узлов подряд.',
                              'Colour shift in the shadows, midtones or highlights. For several '
                              'ranges chain several nodes.'),
    'nd.desaturate': ('Обесцветить (Desaturate)', 'Desaturate'),
    'nd.desaturate.info': ('Цветная картинка → серая карта: по яркости, среднему, светлоте, '
                           'максимуму или одному каналу.',
                           'Colour picture → grey map: by brightness, average, lightness, maximum '
                           'or one channel.'),
    'nd.bandpass': ('Полоса масштабов (старый узел)', 'Band of scales (old node)'),
    'nd.tone': ('Тон (старый узел)', 'Tone (old node)'),
    # подписи гнёзд и настроек, свои у отдельных узлов
    'np.highpass.h': ('карта', 'map'), 'np.highpass.low': ('убранное', 'removed'),
    'np.highpass.remove': ('Сколько убрать', 'How much to remove'),
    'np.highpass.edge_level': ('Край — перепад от, уровней', 'An edge is a step from, levels'),
    'np.highpass.edge_level.info': ('Какой перепад (из 255 уровней) считать краем. Меньше — '
                                    'меньше ореолов, но и мелкий рельеф у краёв уходит.',
                                    'How large a step (of 255 levels) counts as an edge. Lower — '
                                    'fewer halos, but fine relief near edges goes too.'),
    'np.angle': ('Угол, °', 'Angle, °'),
    'np.height': ('Высота', 'Height'),
    'np.levels': ('Уровней', 'Levels'),
    'np.distance': ('Длина, пикс.', 'Distance, px'),
    'np.threshold': ('Порог, уровней', 'Threshold, levels'),
    'np.sharpen.amount': ('Эффект, %', 'Amount, %'),
    'np.emboss.amount': ('Эффект, %', 'Amount, %'),
    'np.emboss.height': ('Высота, пикс.', 'Height, px'),
    'np.add_noise.amount': ('Эффект, %', 'Amount, %'),
    'np.gaussian': ('По Гауссу (иначе равномерный)', 'Gaussian (else uniform)'),
    'np.mono': ('Монохромный', 'Monochromatic'),
    'np.seed': ('Зерно случайности', 'Random seed'),
    'np.denoise.strength': ('Интенсивность', 'Strength'),
    'np.denoise.detail': ('Сохранить детали', 'Preserve details'),
    'np.dx': ('По горизонтали, % ширины', 'Horizontal, % of width'),
    'np.dy': ('По вертикали, % высоты', 'Vertical, % of height'),
    'np.edges.invert': ('Тёмные линии на белом', 'Dark lines on white'),
    'np.in_black': ('Вход: чёрное', 'Input black'),
    'np.in_white': ('Вход: белое', 'Input white'),
    'np.levels.gamma': ('Гамма средних тонов', 'Midtone gamma'),
    'np.out_black': ('Выход: чёрное', 'Output black'),
    'np.out_white': ('Выход: белое', 'Output white'),
    'np.c0': ('Выход при 0 %', 'Output at 0 %'),
    'np.c25': ('Выход при 25 %', 'Output at 25 %'),
    'np.c50': ('Выход при 50 %', 'Output at 50 %'),
    'np.c75': ('Выход при 75 %', 'Output at 75 %'),
    'np.c100': ('Выход при 100 %', 'Output at 100 %'),
    'np.brightness': ('Яркость', 'Brightness'),
    'np.brightness.contrast': ('Контрастность, %', 'Contrast, %'),
    'np.threshold.level': ('Порог, уровень', 'Threshold level'),
    'np.posterize.levels': ('Уровней', 'Levels'),
    'np.clip_lo': ('Отбросить тёмных, %', 'Clip darkest, %'),
    'np.clip_hi': ('Отбросить светлых, %', 'Clip lightest, %'),
    'np.stops': ('Экспозиция, ступеней', 'Exposure, stops'),
    'np.offset': ('Сдвиг', 'Offset'),
    'np.exposure.gamma': ('Коррекция гаммы', 'Gamma correction'),
    'np.hue': ('Цветовой тон, °', 'Hue, °'),
    'np.saturation': ('Насыщенность', 'Saturation'),
    'np.lightness': ('Яркость', 'Lightness'),
    'np.tone': ('Диапазон', 'Tonal range'),
    'np.cyan_red': ('Голубой — красный', 'Cyan — red'),
    'np.magenta_green': ('Пурпурный — зелёный', 'Magenta — green'),
    'np.yellow_blue': ('Жёлтый — синий', 'Yellow — blue'),
    'np.keep_light': ('Сохранить яркость', 'Preserve luminosity'),
    'np.desaturate.mode': ('По чему', 'Based on'),
    'np.edges.method': ('Оператор', 'Operator'),
    'nv.sobel': ('Собель', 'Sobel'), 'nv.prewitt': ('Превитт', 'Prewitt'),
    'nv.scharr': ('Шарр', 'Scharr'), 'nv.laplace': ('Лаплас', 'Laplace'),
    'nv.shadows': ('тени', 'shadows'), 'nv.midtones': ('средние тона', 'midtones'),
    'nv.highlights': ('света', 'highlights'),
    'nv.luma': ('яркость', 'luminosity'), 'nv.average': ('среднее', 'average'),
    'nv.lightness': ('светлота', 'lightness'), 'nv.max': ('максимум', 'maximum'),
    'nv.r': ('красный канал', 'red channel'), 'nv.g': ('зелёный канал', 'green channel'),
    'nv.b': ('синий канал', 'blue channel'),
})

# --- материалы: от тусклого до блестящего, PBR для любых движков ---------
S.update({
    'np.mat_sync': ('Общий материал с другими PBR-узлами', 'Shared material with other PBR nodes'),
    'np.mat_sync.info': ('Узлы «PBR», «PBR → OpenMW» и «Отражение», у которых стоит эта галка, держат '
                         'настройки материала одинаковыми: правка в одном сразу переносится в '
                         'другие. Когда галку ставят, узел берёт настройки у уже отмеченного.',
                         'The «PBR», «PBR → OpenMW» and «Reflection» nodes with this box ticked '
                         'keep the material settings the same: a change in one goes to the '
                         'others at once. When the box is ticked, the node takes the settings '
                         'of one already ticked.'),
    'np.mat_dull': ('Самый тусклый материал', 'Dullest material'),
    'np.mat_dull.info': ('Из чего сделаны самые матовые места текстуры. Вместе с «самым блестящим» '
                         'задаёт диапазон: карты шероховатости и металличности укладываются в него, '
                         'F0 меняется от одного материала к другому. Выбор сразу ставит ползунки '
                         'ниже; один и тот же материал в обоих списках — текстура из одного '
                         'материала.',
                         'What the most matte parts of the texture are made of. With «shiniest» '
                         'it sets the range: the roughness and metalness maps are laid into it, F0 '
                         'goes from one material to the other. Picking sets the sliders below; the '
                         'same material in both lists means a single-material texture.'),
    'np.mat_shiny': ('Самый блестящий материал', 'Shiniest material'),
    'np.auto_name': ('Материал по имени текстуры', 'Material from the texture name'),
    'np.auto_name.info': ('Если в имени файла есть материал (tx_wood_…, tx_dwrv_…, …_rust…), '
                          'берутся его значения, а при нескольких — диапазон от самого тусклого '
                          'до самого блестящего из них; иначе — ползунки. Найденное — в '
                          'показаниях. Работает и в пакете.',
                          'If the file name names a material (tx_wood_…, tx_dwrv_…, …_rust…), its '
                          'values are used, with several — the range from the dullest to the '
                          'shiniest of them; otherwise the sliders. What was found is in the '
                          'readouts. Works in a batch too.'),
    'np.f0': ('F0 тусклого (отражение в лоб)', 'F0 of the dull end (head-on reflection)'),
    'np.f0_shiny': ('F0 блестящего', 'F0 of the shiny end'),
    'np.f0.info': ('Доля света, отражённая при взгляде в лоб, линейная. У диэлектриков — из '
                   'показателя преломления: вода 0.02, стекло и лак 0.04, камень 0.045, '
                   'самоцветы 0.075. У металлов цвет блика берётся из базового цвета.',
                   'Share of light reflected head-on, linear. For dielectrics it follows from the '
                   'refractive index: water 0.02, glass and varnish 0.04, stone 0.045, gems '
                   '0.075. Metals take the specular colour from the base colour.'),
    'np.fit': ('Уложить карты в материалы', 'Lay maps into the materials'),
    'np.fit.info': ('1 — разброс входных карт растягивается на диапазон материалов (рисунок карты '
                    'сохраняется, меняются уровни); 0 — карты как есть.',
                    '1 — the spread of the incoming maps is stretched onto the materials\' range '
                    '(the pattern stays, the levels change); 0 — the maps as they are.'),
    'np.rough_lo': ('Шероховатость: блестящий край', 'Roughness at the shiny end'),
    'np.rough_hi': ('Шероховатость: тусклый край', 'Roughness at the dull end'),
    'np.metal_lo': ('Металличность от', 'Metalness from'),
    'np.metal_hi': ('Металличность до', 'Metalness to'),
    'ni.materials': ('Материалы по имени', 'Materials from the name'),
    'nd.pbr': ('PBR (любой движок)', 'PBR (any engine)'),
    'nd.pbr.info': ('Стандартные карты PBR с тем же выбором материалов: базовый цвет, '
                    'шероховатость, металличность, AO, упакованная ORM (glTF, Unreal), гладкость и '
                    'металличность+гладкость (Unity), блик и диффуз для схемы specular-glossiness.',
                    'Standard PBR maps with the same material choice: base colour, roughness, '
                    'metalness, AO, packed ORM (glTF, Unreal), smoothness and '
                    'metallic+smoothness (Unity), specular and diffuse for the '
                    'specular-glossiness workflow.'),
    'np.orm': ('ORM', 'ORM'), 'np.smooth': ('гладкость', 'smoothness'),
    'np.ms': ('металл+гладк. (Unity)', 'metal+smooth (Unity)'),
    'np.specular': ('блик (spec-gloss)', 'specular (spec-gloss)'),
    'np.spec_diffuse': ('диффуз (spec-gloss)', 'diffuse (spec-gloss)'),
    'np.pbr.ao': ('AO', 'AO'),
    'nv._gloss': ('_gloss', '_gloss'), 'nv._sss': ('_sss', '_sss'), 'nv._basecolor': ('_basecolor', '_basecolor'), 'nv._orm': ('_orm', '_orm'),
    'nv.stone': ('камень', 'stone'),
    'nv.marble': ('полированный камень, мрамор', 'polished stone, marble'),
    'nv.sandstone': ('песчаник, штукатурка, бетон', 'sandstone, plaster, concrete'),
    'nv.brick': ('кирпич, терракота', 'brick, terracotta'),
    'nv.obsidian': ('обсидиан, эбонит', 'obsidian, ebony'),
    'nv.gem': ('самоцветы, кристалл', 'gems, crystal'),
    'nv.ground': ('земля, песок, пепел', 'ground, sand, ash'),
    'nv.mud': ('грязь, мокрая земля', 'mud, wet ground'),
    'nv.water': ('вода', 'water'),
    'nv.ice': ('лёд', 'ice'),
    'nv.snow': ('снег', 'snow'),
    'nv.wood': ('дерево', 'wood'),
    'nv.bark': ('кора', 'bark'),
    'nv.lacquer': ('лак, лакированное дерево', 'varnish, lacquered wood'),
    'nv.leaves': ('листья, трава', 'leaves, grass'),
    'nv.cloth': ('ткань', 'cloth'),
    'nv.silk': ('шёлк, атлас', 'silk, satin'),
    'nv.leather': ('кожа (выделанная)', 'leather'),
    'nv.fur': ('мех, волосы', 'fur, hair'),
    'nv.skin': ('кожа (тело)', 'skin'),
    'nv.bone': ('кость, рог, зубы', 'bone, horn, teeth'),
    'nv.chitin': ('хитин, панцирь', 'chitin, shell'),
    'nv.ceramic': ('глазурованная керамика', 'glazed ceramic'),
    'nv.wax': ('воск', 'wax'),
    'mat.none': ('не распознаны — по ползункам', 'not recognised — by the sliders'),
    'nv.paper': ('бумага, пергамент', 'paper, parchment'),
    'nv.glass': ('стекло', 'glass'),
    'nv.iron': ('железо (кованое)', 'iron (forged)'),
    'nv.steel': ('сталь (полированная)', 'steel (polished)'),
    'nv.rust': ('ржавчина', 'rust'),
    'nv.gold': ('золото', 'gold'),
    'nv.silver': ('серебро', 'silver'),
    'nv.copper': ('медь', 'copper'),
    'nv.bronze': ('бронза, латунь, двемерский металл', 'bronze, brass, Dwemer metal'),
    'nv.patina': ('патина, потемневший металл', 'patina, tarnished metal'),
})
S.update({
    'np.bent': ('bent normal', 'bent normal'),
    'nd.bent.info': ('Направление, откуда реально приходит рассеянный свет. Выход «bent normal» — '
                     'отдельная карта для движков, которые её читают (Unreal, Unity HDRP); выход '
                     '«нормаль» — обычная нормаль с подмешанной bent (для остальных). Вход нормали '
                     'можно не подключать.',
                     'The direction diffuse light actually comes from. The «bent normal» output is '
                     'a separate map for engines that read one (Unreal, Unity HDRP); the «normal» '
                     'output is the ordinary normal with the bent one blended in (for the rest). '
                     'The normal input may stay unconnected.'),
})
S.update({
    'nd.albedo_match': ('Подогнать альбедо к исходнику', 'Match albedo to source'),
    'nd.albedo_match.info': ('Возвращает сгенерированное альбедо (SuperMat, снятие света) к '
                             'исходному диффузу в целом: общую яркость, оттенок и насыщенность, '
                             'чтобы цвет не уезжал. По месту не подгоняет — иначе вернулось бы '
                             'нарисованное освещение. «Чёткость из исходника» добавляет мелкую '
                             'деталь диффуза, если альбедо мылит. С подключённой нормалью '
                             'яркость и оттенок считаются по рельефу: подбирается запечённый свет, '
                             'и плоские места сохраняют цвет исходника.',
                             'Brings a generated albedo (SuperMat, de-light) back towards the '
                             'source diffuse overall: lightness, cast and saturation, so the colour '
                             'does not drift. Nothing local — that would bring the painted lighting '
                             'back. «Detail from source» adds the diffuse\'s fine detail when the '
                             'albedo is soft. With a normal connected, lightness and cast come '
                             'from the relief: the baked light is fitted and flat places keep the '
                             'source colour.'),
    'np.albedo': ('альбедо', 'albedo'), 'np.ref': ('исходник', 'source'),
    'np.albedo_match.light': ('Яркость как у исходника', 'Lightness as in source'),
    'np.cast': ('Оттенок как у исходника', 'Cast as in source'),
    'np.albedo_match.saturation': ('Насыщенность как у исходника', 'Saturation as in source'),
    'np.albedo_match.detail': ('Чёткость из исходника', 'Detail from source'),
    'np.albedo_match.spread': ('Крупные пятна как у исходника', 'Large patches as in source'),
    'np.albedo_match.spread.info': ('Разброс светлоты на крупном масштабе (то, что видно издалека). '
                                    'Снятие затенения выравнивает крупные светлые и тёмные пятна, '
                                    'и текстура издалека выглядит молочной. Здесь собственные пятна '
                                    'альбедо растягиваются до разброса исходника (не больше чем в 3 '
                                    'раза); из исходника берётся только это одно число, не его '
                                    'рисунок, так что его тени не возвращаются. 1 — до конца, 0 — '
                                    'выключено.',
                                    'Spread of lightness on a large scale (what is seen from afar). '
                                    'De-lighting evens out large light and dark patches and the '
                                    'texture looks milky from a distance. Here the albedo\'s own '
                                    'patches are stretched to the source\'s spread (at most 3×); '
                                    'only that one number comes from the source, not its pattern, '
                                    'so its shading does not return. 1 — all the way, 0 — off.'),
    'np.albedo_match.dark_cut': ('Чёрные дефекты исходника темнее', 'Source black defects darker than'),
    'np.albedo_match.dark_cut.info': ('Места исходника темнее этого (с запасом в несколько пикселей) '
                                      'считаются дефектами — дырами апскейла, — не идут в подсчёт '
                                      'крупных пятен и не дают чёткости. 0 — выключено.',
                                      'Places of the source darker than this (with a margin of a few '
                                      'pixels) count as defects — holes of an upscale — and are left '
                                      'out of the large-patch measurement and of the detail. 0 — off.'),
    'ni.spread_k': ('Крупные пятна усилены в', 'Large patches stretched by'),
    'np.detail_r': ('Мельче чего деталь, пикс.', 'Detail finer than, px'),
    'ni.light_diff': ('Разница яркости до подгонки', 'Lightness difference before'),
    'ni.sat_ratio': ('Насыщенность исходник / альбедо', 'Saturation source / albedo'),
    'np.albedo_match.n': ('нормаль', 'normal'),
    'np.albedo_match.h': ('высота', 'height'),
    'np.albedo_match.shadow': ('падающие тени', 'cast shadows'),
    'np.albedo_match.shadows': ('Не учитывать падающие тени', 'Leave out cast shadows'),
    'np.albedo_match.shadows.info': ('Нужна подключённая нормаль. Найденный свет пускается по '
                                     'рельефу, как тени в экранном пространстве, только по карте '
                                     'высоты: места, куда свет загораживает выступ, в подгонку не '
                                     'идут. Высота — со входа «высота», а без него считается из '
                                     'нормали. Маска — на выходе «падающие тени».',
                                     'Needs a connected normal. The found light is marched over '
                                     'the relief, like screen-space shadows but over the height '
                                     'map: places a bump hides from the light are left out of the '
                                     'fit. The height comes from the «height» input, or without it '
                                     'from the normal. The mask is on the «cast shadows» output.'),
    'albedo_match.size': ('Нормаль и высота должны быть того же размера, что альбедо.',
                          'The normal and the height must be the size of the albedo.'),
    'ni.gain': ('Свет на плоском месте (множитель альбедо)', 'Light on a flat place (albedo gain)'),
    'ni.light_az': ('Свет: направление, ° (0 — справа, 90 — сверху)',
                    'Light: direction, ° (0 — right, 90 — top)'),
    'ni.light_el': ('Свет: высота над поверхностью, °', 'Light: elevation, °'),
    'ni.light_share': ('Доля направленного света, %', 'Directional share of the light, %'),
    'ni.shadow_pc': ('Падающие тени, % пикселей', 'Cast shadows, % of pixels'),
    'ni.fit_px': ('Пикселей в подгонке, %', 'Pixels in the fit, %'),
})

# --- узел «Листва»: подписи те же, что на вкладках листвы ------------------
S.update({
    'cat.foliage': ('Листва', 'Foliage'),
    'nd.foliage': ('Листва (карта с альфой)', 'Foliage (alpha card)'),
    'nd.foliage.info': ('Движок вкладок листвы одним узлом: листья вырезаются из силуэта один '
                        'раз, и из этого все карты — нормали по листьям, высота, AO кроны, bent '
                        'normal, просвет, проверочные «разбиение» и «листья и ветки». Альфу '
                        'подайте с выхода «А» исходника, ветки вручную — со входа «ветки '
                        'вручную» (белое = ветка). Настройки и умолчания — как на вкладке. '
                        'Диффуз без света — отдельный узел «Снять свет».',
                        'The foliage tabs\' engine as one node: the leaves are cut out of the '
                        'silhouette once, and all maps come from that — per-leaf normals, height, '
                        'crown AO, bent normal, translucency, and the check views «element '
                        'split» and «leaves and branches». Feed the alpha from the source\'s «A» '
                        'output, hand-marked branches into «branches by hand» (white = branch). '
                        'Settings and defaults are those of the tab. The de-lit diffuse is the '
                        'separate «Remove light» node.'),
    'np.sss': ('просвет', 'translucency'), 'np.sss_a': ('сила просвета', 'translucency amount'),
    'np.foliage.a': ('альфа', 'alpha'), 'np.foliage.n': ('нормаль', 'normal'),
})
for _k, _src in (('split', 'f.split'), ('min_area', 'f.minarea'), ('tilt', 'f.tilt'),
                 ('fold', 'f.fold'), ('blade', 'f.blade'), ('bend', 'f.bend'), ('seed', 'f.seed'),
                 ('profile', 'f.prof'), ('detail', 'f.det'), ('pad', 'f.pad'),
                 ('thin', 'f.thin'), ('clump_mid', 'f.clm'), ('clump_big', 'f.clb'),
                 ('clump_scale', 'f.cls'), ('dome_w', 'f.domw'), ('dome_h', 'f.domh'),
                 ('ao_str', 'd.aostr'), ('ao_spread', 'd.aospread'), ('ao_bulge', 'd.aobulge'),
                 ('ao_radius', 'd.aorad2'), ('ao_tilt', 'd.aotilt'), ('ao_gamma', 'd.aogam'),
                 ('ao_contrast', 'd.aocon'), ('sss_base', 'd.sss.base'),
                 ('sss_reach', 'd.sss.reach'), ('sss_layers', 'd.sss.lay'),
                 ('sss_pigment', 'd.sss.pig'), ('sss_edge', 'd.sss.edge'),
                 ('sss_branch', 'd.sss.branch'), ('sss_gamma', 'd.sss.gam'),
                 ('sss_contrast', 'd.sss.con'),
                 ('sss_str', 'd.sss.str'), ('sss_tint', 'd.sss.tint'),
                 ('sss_behind', 'd.sss.behind'), ('sss_veins', 'd.sss.veins')):
    S['np.foliage.' + _k] = S[_src]
    S.setdefault('np.' + _k, S[_src])
    if _src + '.info' in S:
        S['np.foliage.' + _k + '.info'] = S[_src + '.info']
# у ручек без своей подсказки — подсказка их раздела на вкладке
for _k, _src in (('clump_mid', 'f.clump'), ('dome_w', 'f.card'), ('dome_h', 'f.card')):
    S.setdefault('np.foliage.' + _k + '.info', S[_src + '.info'])
for _v in ('silh', 'dense', 'lum'):
    S['nv.leaf_' + _v] = S['f.m.' + _v]
S.update({
    'tpl.foliage': ('Листва: карта с альфой → OpenMW', 'Foliage: alpha card → OpenMW'),
    'tpl.foliage.info': ('Узел «Листва»: нормали по листьям с высотой (_nh), диффуз без '
                         'нарисованного света, AO кроны. Нужна текстура с альфой.',
                         'The «Foliage» node: per-leaf normals with height (_nh), the diffuse '
                         'without painted light, crown AO. Needs a texture with alpha.'),
})
S.update({
    'ni.spec_steps': ('Средний блик, шагов из 255', 'Mean specular, steps of 255'),
})
S.update({
    'ed.dxall': ('Нормали: DX / GL ▾', 'Normals: DX / GL ▾'),
    'ed.dxall.dx': ('Все узлы — DirectX (зелёный вниз)', 'All nodes — DirectX (green down)'),
    'ed.dxall.gl': ('Все узлы — OpenGL (зелёный вверх)', 'All nodes — OpenGL (green up)'),
    'ed.dxall.done': ('Раскладка %s у %d узлов', 'Layout %s on %d nodes'),
})
S.update({
    'nd.patch': ('Заплатка (Patch)', 'Patch'),
    'nd.patch.info': ('Ручная заплатка, как инструмент «Заплатка» в Photoshop: обвести в 2D место, '
                      'которое надо исправить, и перетащить обводку туда, откуда взять замену. '
                      'Заплаток может быть сколько угодно; текстура — плитка, сдвиг идёт через край.',
                      'Manual patch, like the Patch tool in Photoshop: outline in the 2D view the place '
                      'to fix and drag the outline to where the replacement should come from. Any '
                      'number of patches; the texture is a tile, the shift wraps round the edge.'),
    'np.seamless': ('Бесшовно (подогнать края)', 'Seamless (blend the edges)'),
    'np.seamless.info': ('Включено — переносится фактура, а яркость и цвет на краю плавно '
                         'подгоняются к окружению (смешивание по градиентам), шва нет. Выключено — '
                         'простое копирование с растушёвкой края.',
                         'On — the texture is carried over and the brightness and colour at the edge '
                         'are smoothly brought to the surroundings (gradient-domain blend), no seam. '
                         'Off — a plain copy with a feathered edge.'),
    'np.feather': ('Растушёвка края, пикс.', 'Edge feather, px'),
    'np.patches': ('Заплатки', 'Patches'),
    'ni.patches': ('Заплаток', 'Patches'),
    'ed.patch': ('Заплатки', 'Patches'),
    'ed.patch.how': ('В 2D: левой кнопкой обведите место, которое надо исправить. С автоподбором '
                     'этого достаточно; без него перетащите обводку на место, откуда взять '
                     'замену. Правая кнопка — снять '
                     'обводку, средняя — двигать вид, колесо — масштаб. Жёлтым — исправленные '
                     'места, голубым пунктиром — откуда взято.',
                     'In 2D: outline the place to fix with the left button. With automatic '
                     'picking that is all; without it, drag the outline onto the place to take the '
                     'replacement from. Right button — drop the '
                     'outline, middle — pan, wheel — zoom. Yellow — fixed places, blue dashes — '
                     'where the replacement came from.'),
    'ed.patch.n': ('Заплаток: %d', 'Patches: %d'),
    'ed.patch.undo': ('Убрать последнюю', 'Remove the last'),
    'ed.patch.clear': ('Убрать все', 'Remove all'),
})
S.update({'np.patch.h': ('карта', 'map')})
S.update({
    'np.auto': ('Источник подбирать автоматически', 'Pick the source automatically'),
    'np.auto.info': ('Включено — достаточно обвести место: откуда взять замену, программа найдёт '
                     'сама — там, где окрестность совпадает лучше всего (как «С учётом содержимого» '
                     'в Photoshop). Выключено — обводку перетаскивают на источник вручную.',
                     'On — just outline the place: the program finds where to take the '
                     'replacement from itself, where the surroundings match best (like '
                     'Content-Aware in Photoshop). Off — drag the outline onto the source by '
                     'hand.'),
    'ed.patch.wait': ('Заплатка: дождитесь, пока узел посчитается', 'Patch: wait until the node is computed'),
    'ed.patch.nofit': ('Заплатка: подходящего места не нашлось — обводка слишком большая',
                       'Patch: no place fits — the outline is too large'),
})
S.update({
    'np.spec_gain.info': ('Ручной множитель сверх физики. Силу блика материала уже задают F0 и '
                          'шероховатость, поэтому при выборе материала он ставится в 1; меньше 1 — '
                          'приглушить блик по вкусу.',
                          'A manual multiplier on top of the physics. The material\'s specular '
                          'strength is already set by F0 and roughness, so picking a material resets '
                          'it to 1; below 1 dims the highlight to taste.'),
})
S.update({
    'nd.normal_flatfix': ('Починить плоские пятна нормали', 'Repair flat normal spots'),
    'nd.normal_flatfix.info': ('Пересвеченное или провалившееся в чёрное место исходника сеть не '
                               'видит и отвечает одной постоянной нормалью — плоским пятном. Узел '
                               'находит такие пятна и заполняет их рельефом, взятым оттуда, где '
                               'окрестность совпадает лучше всего (как автозаплатка), или, без '
                               'галки «С рельефом», плавным продолжением наклона вокруг.',
                               'A blown-out or crushed spot in the source gives the network nothing '
                               'to see, and it answers with one constant normal — a flat blob. The '
                               'node finds such blobs and fills them with relief taken from where the '
                               'surroundings match best (like the auto patch) or, with «With relief» '
                               'off, with a smooth continuation of the tilt around.'),
    'np.spread': ('Плоско, если разброс меньше, °', 'Flat if the spread is below, °'),
    'np.normal_flatfix.min_area': ('Пятно не меньше, пикс.', 'Spot at least, px'),
    'np.normal_flatfix.grow': ('Захватить край, пикс.', 'Take in the edge, px'),
    'ni.filled': ('Заполнено, %', 'Filled, %'),
    'np.normal_flatfix.detail': ('С рельефом (заплатка из похожего места)',
                                 'With relief (patch from a similar place)'),
})
S.update({
    'nd.normal_flatfix': ('Починить срезанное (плато)', 'Repair clipped spots (plateaus)'),
    'np.normal_flatfix.n': ('карта', 'map'),
    'np.normal_flatfix.spread': ('Нормаль: плоско, если разброс меньше, °',
                                 'Normal: flat if the spread is below, °'),
    'np.normal_flatfix.level': ('Высота: плоско, если разброс меньше, уровней',
                                'Height: flat if the spread is below, levels'),
})
S['nd.normal_flatfix.info'] = (
    'Срезанные места — плоские пятна на нормали или плато на высоте (вершины и дно, упёршиеся '
    'в край диапазона, или места, где исходник пересвечен). Узел подключается к нормали или к '
    'высоте, находит такие места и закрывает их рельефом из места, где окрестность совпадает '
    'лучше всего (как автозаплатка). У нормали без галки «С рельефом» — плавным продолжением '
    'наклона.',
    'Clipped spots are flat blobs on a normal or plateaus on a height (tops and bottoms that hit '
    'the end of the range, or places where the source is blown out). Connect the node to a normal '
    'or a height: it finds such spots and covers them with relief from where the surroundings '
    'match best (like the auto patch). For a normal with «With relief» off — a smooth '
    'continuation of the tilt.')
S.update({
    'ed.b.layout': ('Раскладка карт:', 'Map layout:'),
    'ed.b.layout.flat': ('всё вместе (рядом с исходником)', 'all together (next to the source)'),
    'ed.b.layout.by_map': ('по папкам вида карты (n, nh, spec…)', 'folder per kind of map (n, nh, spec…)'),
    'ed.b.layout.by_texture': ('по папке на текстуру', 'folder per texture'),
    'ed.b.layout.info': ('Подпапки исходной папки сохраняются внутри. Для игры карты должны лежать '
                         'рядом с текстурой — раскладка по папкам удобна для просмотра и отбора. '
                         'То же действует и для «Сохранить карты…».',
                         'Subfolders of the input folder are kept inside. For the game the maps must '
                         'lie next to the texture — the folder layouts are for viewing and sorting. '
                         'The same applies to «Save maps…».'),
})

S.update({
    'np.azimuth': ('Откуда светит, °', 'Light direction, °'),
    'np.azimuth.info': ('0° — справа, 90° — сверху, 180° — слева, 270° — снизу.',
                        '0° — from the right, 90° — from the top, 180° — from the left, '
                        '270° — from below.'),
    'np.dir_amount': ('Свет с одной стороны, доля', 'Light from one side, share'),
    'np.dir_amount.info': ('0 — небо светит равномерно со всех сторон (обычное AO). Больше — свет '
                           'приходит в основном с направления «Откуда светит», и затеняется то, что '
                           'закрыто с той стороны.',
                           '0 — the sky lights evenly from all sides (plain AO). Higher — light '
                           'comes mostly from «Light direction», and what is hidden on that side '
                           'gets darker.'),
    'np.spread': ('Ширина источника, °', 'Source width, °'),
    'np.elevation': ('Высота света над поверхностью, °', 'Light elevation above the surface, °'),
    'np.shadows.depth': ('Высота рельефа, пикс.', 'Relief height, px'),
    'np.shadows.depth.info': ('Сколько пикселей по высоте занимает весь диапазон карты высоты: '
                              'больше — длиннее тени.',
                              'How many pixels tall the whole height range is: higher — longer '
                              'shadows.'),
    'np.softness': ('Мягкость края, °', 'Edge softness, °'),
    'np.shadows.radius': ('Искать на расстоянии до, пикс.', 'Search up to, px'),
    'nd.shadows': ('Тени (в экранном пространстве)', 'Shadows (screen space)'),
    'nd.shadows.info': ('Тени, которые рельеф отбрасывает при свете с заданной стороны. Выход «свет» '
                        '— для узла «Убрать затенение», «маска» — 1 там, где тень.',
                        'Shadows the relief casts under light from the given side. The «light» '
                        'output is for «Remove shading», «mask» is 1 in shadow.'),
    'ni.in_shadow': ('В тени, %', 'In shadow, %'),
    'nd.unshade': ('Убрать затенение', 'Remove shading'),
    'nd.unshade.info': ('Убирает из текстуры нарисованное затенение: подключите AO или выход «свет» '
                        'узла «Тени». Цвет делится на затенение в линейном свете; «нижний предел» '
                        'не даёт чёрным местам превратиться в шум.',
                        'Takes the painted shading out of the texture: connect an AO map or the '
                        '«light» output of «Shadows». The colour is divided by the shading in '
                        'linear light; «floor» keeps black spots from turning into noise.'),
    'np.shade': ('затенение', 'shading'),
    'np.unshade.strength': ('Сколько убрать', 'How much to remove'),
    'np.floor': ('Нижний предел делителя', 'Divisor floor'),
    'ed.tab.materials': ('Материалы', 'Materials'),
    'ed.tab.ranges': ('Диапазоны', 'Ranges'),
    'mg.stone': ('Камень и минералы', 'Stone and minerals'),
    'mg.ground': ('Земля, вода, лёд', 'Ground, water, ice'),
    'mg.plant': ('Дерево и растения', 'Wood and plants'),
    'mg.body': ('Ткань, кожа, тело', 'Cloth, leather, body'),
    'mg.made': ('Рукотворное', 'Made things'),
    'mg.metal': ('Металлы', 'Metals'),
    'ed.mat.hint': ('Нажмите шар — материал добавится или уберётся. Один материал — вся текстура '
                    'из него; несколько — делят шкалу блеска ниже, от тусклого слева к блестящему '
                    'справа.',
                    'Click a sphere to add or remove a material. One material — the whole texture '
                    'is of it; several share the shininess scale below, from dull on the left to '
                    'shiny on the right.'),
    'ed.mat.info': ('%s: шероховатость %.2f…%.2f, металличность %.1f…%.1f, f0 %.3f',
                    '%s: roughness %.2f…%.2f, metalness %.1f…%.1f, f0 %.3f'),
    'ed.mat.equal': ('Поровну', 'Equal shares'),
    'ed.mat.pick': ('Пипетка: щёлкнуть по текстуре в 2D (вспомогательно)',
                    'Pipette: click the texture in 2D (a helper)'),
    'ed.mat.pick_val': ('Блеск в этой точке %.2f — %s', 'Shininess here %.2f — %s'),
    'ed.mat.pick_cutd': ('за срезом тусклого (получит самое тусклое)', 'beyond the dull cut'),
    'ed.mat.pick_cuts': ('за срезом блестящего (получит самое блестящее)', 'beyond the shiny cut'),
    'ed.mat.pick_cd': ('Срез тусклого сюда', 'Dull cut here'),
    'ed.mat.pick_b': ('Ближайшую границу сюда', 'Nearest border here'),
    'ed.mat.pick_cs': ('Срез блестящего сюда', 'Shiny cut here'),
    'ed.mat.pick_norough': ('Нет входной карты шероховатости — пипетке нечего показать.',
                            'No roughness map connected — nothing to show.'),
    'ed.mat.clear': ('Убрать все — свои ползунки', 'Remove all — own sliders'),
    'ed.mat.nohist': ('Нет карты шероховатости — гистограммы нет', 'No roughness map — no histogram'),
    'ed.mat.sliders': ('свои ползунки', 'own sliders'),
    'ed.mat.dull': ('тусклое', 'dull'),
    'ed.mat.shiny': ('блестящее', 'shiny'),
    'ed.mat.cuts': ('срез %.1f %% | %.1f %%', 'cut %.1f %% | %.1f %%'),
    'ed.mat.note': ('Пока выбран хоть один материал, ползунки вкладки «Диапазоны» не действуют. '
                    'Границы полос и метки срезов на шкале можно тянуть мышью. Полосу, '
                    'взятую за середину, можно перетащить на место другой — материалы '
                    'поменяются местами (если карта шероховатости ошиблась и, скажем, мех '
                    'вышел самым блестящим).',
                    'While any material is chosen, the sliders of the «Ranges» tab do nothing. '
                    'The band borders and the cut marks on the scale can be dragged. A band '
                    'grabbed by its middle can be dragged onto another — the two materials swap '
                    '(when the roughness map got it wrong and, say, fur came out the shiniest).'),
    'ed.mat.manual': ('Порядок полос свой, не по блеску материалов. «По блеску» вернёт обычный.',
                      'The band order is your own, not by the materials\' shininess. «By '
                      'shininess» brings the usual one back.'),
    'ed.mat.by_gloss': ('По блеску', 'By shininess'),
    # --- Albedo Fix (Albedo-Fix/albedo_fix.py): доводка готовых текстур
    'af.title': ('Доводка готовых текстур', 'Finishing ready textures'),
    'af.intro': ('Возвращает в готовые текстуры то, что сгладила цепочка альбедо, не запуская сети заново: '
                 'крупные пятна (вид издалека), мелкую фактуру мельче разрешения SuperMat и заново '
                 'запекает AO в диффуз для обычного OpenMW. Нужны исходники, на которых шёл пакет, и '
                 'папка с результатом пакета (diffuse, _fixed, _orm). Входные папки не трогаются.',
                 'Puts back into ready textures what the albedo chain smoothed out, without running '
                 'the networks again: large patches (the look from afar), fine texture finer than '
                 'SuperMat\'s resolution, and bakes AO into the diffuse for plain OpenMW again. Needs '
                 'the sources the batch was run on and the batch result folder (diffuse, _fixed, '
                 '_orm). The input folders are not touched.'),
    'af.src': ('Исходники', 'Sources'),
    'af.done_dir': ('Результат пакета', 'Batch result'),
    'af.dst': ('Куда писать', 'Output folder'),
    'af.spread': ('Вернуть крупные пятна', 'Bring back large patches'),
    'af.weight': ('Насколько', 'How far'),
    'af.spread_hint': ('Свои пятна альбедо растягиваются до разброса светлоты исходника на крупном масштабе '
                       '(вне дефектов); рисунок исходника не переносится.',
                       'The albedo\'s own patches are stretched to the source\'s large-scale '
                       'lightness spread (outside defects); the source\'s pattern is not copied.'),
    'af.detail_on': ('Добрать мелкую фактуру из исходника', 'Add the fine texture from the source'),
    'af.detail': ('Сила', 'Strength'),
    'af.r_old': ('Радиус, уже взятый «Подгонкой альбедо», пикс.', 'Radius already taken by «Fit albedo», px'),
    'af.sm_res': ('Разрешение SuperMat', 'SuperMat resolution'),
    'af.detail_hint': ('Добирается полоса от уже взятого радиуса до ширина ÷ разрешение SuperMat: всё, '
                       'что SuperMat не видел. Для 2048 при 512 — до 4 пикс.',
                       'The band from the radius already taken up to width ÷ SuperMat resolution is '
                       'added: everything SuperMat did not see. For 2048 at 512 — up to 4 px.'),
    'af.thr': ('Чёрные дефекты исходника темнее', 'Source black defects darker than'),
    'af.thr_hint': ('Такие места (с запасом) не идут в подсчёт и не дают фактуры.',
                    'Such places (with a margin) are not counted and give no texture.'),
    'af.ao': ('Сила AO в новом диффузе', 'AO strength in the new diffuse'),
    'af.ao_old': ('AO, уже запечённый в диффуз', 'AO already baked into the diffuse'),
    'af.ao_old_hint': ('Нужно только для текстур без _fixed: альбедо тогда получается делением диффуза на AO '
                       'с этой силой.', 'Only for textures without _fixed: the albedo is then the '
                       'diffuse divided by AO at this strength.'),
    'af.v_src': ('Исходник', 'Source'),
    'af.v_now': ('Сейчас', 'Now'),
    'af.v_new': ('Станет', 'After'),
    'af.readout': ('Крупные пятна × %.2f · дефекты %.1f %%', 'Large patches × %.2f · defects %.1f %%'),
    'af.nodone': ('Выберите папку с результатом пакета.', 'Choose the batch result folder.'),
    'af.same': ('Папка вывода не может совпадать с входной или лежать внутри неё.',
                'The output folder must not be an input folder or lie inside one.'),
    'af.none': ('В папке результата не найдено диффузов.', 'No diffuse textures found in the result folder.'),
    'af.found': ('Найдено текстур: %d', 'Textures found: %d'),
    'af.done': ('Найдено %d, обработано %d, ошибок %d.', 'Found %d, processed %d, errors %d.'),
    'af.cpu': ('процессор', 'processor'),
    'af.r_src': ('Исходники: %s', 'Sources: %s'),
    'af.r_done': ('Результат пакета: %s', 'Batch result: %s'),
    'af.r_dst': ('Папка вывода: %s', 'Output: %s'),
    'af.r_opt': ('Настройки: %s', 'Settings: %s'),
    'af.r_dev': ('Считалось на: %s', 'Computed on: %s'),
    'af.r_stopped': ('Остановлено пользователем.', 'Stopped by the user.'),
    'af.r_error': ('ОШИБКА', 'ERROR'),
    'af.r_k': ('крупные пятна × %.2f', 'large patches × %.2f'),
    'af.r_band': ('фактура %.1f…%.1f пикс.', 'texture %.1f…%.1f px'),
    'af.r_def': ('дефекты %.1f %%', 'defects %.1f %%'),
    'af.r_nosrc': ('исходник не найден — только AO', 'source not found — AO only'),
    'af.r_nofixed': ('нет _fixed — альбедо из диффуза', 'no _fixed — albedo from the diffuse'),
    'af.r_noorm': ('нет _orm — без AO', 'no _orm — no AO'),
    # --- AO Strength (AO-Strength/ao_strength.py): сила AO в готовых диффузах
    'ao_s.title': ('Сила AO в готовых текстурах', 'AO strength in ready textures'),
    'ao_s.intro': ('Меняет, насколько затенение (AO) темнит готовый диффуз. AO берётся из канала R '
                   'карты ORM (или ARM) с тем же именем: tx_name.dds ↔ tx_name_orm.dds. Расчёт — '
                   'в линейном свете, как запекает «PBR → OpenMW»: цвет × AO^(новая − запечённая). '
                   'Исходные папки не трогаются: результат пишется в отдельную.',
                   'Changes how much ambient occlusion (AO) darkens a ready diffuse. AO is taken '
                   'from the R channel of the ORM (or ARM) map with the same name: tx_name.dds ↔ '
                   'tx_name_orm.dds. Computed in linear light, the way «PBR → OpenMW» bakes it: '
                   'colour × AO^(new − baked). The source folders are not touched: results go to '
                   'a separate one.'),
    'ao_s.dif': ('Папка с диффузами', 'Folder with the diffuse textures'),
    'ao_s.orm': ('Папка с ORM (пусто — та же)', 'Folder with the ORM maps (empty — the same)'),
    'ao_s.dst': ('Куда писать', 'Output folder'),
    'ao_s.recursive': ('Со вложенными папками', 'Include sub-folders'),
    'ao_s.old': ('Сила AO, уже запечённая', 'AO strength already baked in'),
    'ao_s.new': ('Новая сила AO', 'New AO strength'),
    'ao_s.old_hint': ('Запечённая: 0 — в диффузе AO нет (чужой пак), 1 — Texelith по умолчанию '
                      '(«AO в диффузе» узла «PBR → OpenMW»). Если в папке текстуры обоих видов, '
                      'обработайте их двумя прогонами.',
                      'Baked: 0 — the diffuse has no AO (a pack from elsewhere), 1 — Texelith\'s '
                      'default («AO in the diffuse» of «PBR → OpenMW»). If the folder has both '
                      'kinds, process them in two runs.'),
    'ao_s.before': ('Было', 'Before'),
    'ao_s.after': ('Станет', 'After'),
    'ao_s.rescan': ('Найти заново', 'Scan again'),
    'ao_s.nosrc': ('Выберите папку с диффузами.', 'Choose the folder with the diffuse textures.'),
    'ao_s.same': ('Папка вывода не может совпадать с исходной или лежать внутри неё.',
                  'The output folder must not be a source folder or lie inside one.'),
    'ao_s.none': ('Не найдено ни одной пары «диффуз + _orm/_arm».',
                  'No «diffuse + _orm/_arm» pairs found.'),
    'ao_s.found': ('Найдено пар: %d', 'Pairs found: %d'),
    'ao_s.done': ('Найдено %d, обработано %d, ошибок %d.', 'Found %d, processed %d, errors %d.'),
    'ao_s.r_src': ('Диффузы: %s', 'Diffuse: %s'),
    'ao_s.r_orm': ('ORM: %s', 'ORM: %s'),
    'ao_s.r_dst': ('Папка вывода: %s', 'Output: %s'),
    'ao_s.r_k': ('Сила AO: запечённая %.2f → новая %.2f', 'AO strength: baked %.2f → new %.2f'),
    'ao_s.r_stopped': ('Остановлено пользователем.', 'Stopped by the user.'),
    'ao_s.r_error': ('ОШИБКА', 'ERROR'),
    'ao_s.r_note': ('средний AO %.2f, яркость %+.0f %%', 'mean AO %.2f, brightness %+.0f %%'),
    # --- Normal Flip (normal_flip.py): нормали OpenGL → DirectX пачкой
    'nf.title': ('Нормали OpenGL → DirectX', 'Normals OpenGL → DirectX'),
    'nf.intro': ('OpenMW читает карты нормалей в формате DirectX (зелёный вниз). Здесь готовые '
                 'карты _n и _nh, сделанные для OpenGL, переворачиваются по зелёному каналу. '
                 'DDS правится прямо в сжатых блоках — без пересжатия, мипы и альфа (высота _nh) '
                 'не меняются. Исходная папка не трогается: результат пишется в отдельную.',
                 'OpenMW reads normal maps in the DirectX layout (green down). Here ready _n and '
                 '_nh maps made for OpenGL get their green channel flipped. DDS files are edited '
                 'right in the compressed blocks — no recompression, the mips and the alpha (the '
                 '_nh height) stay. The source folder is not touched: results go to a separate '
                 'one.'),
    'nf.src': ('Папка с картами', 'Folder with the maps'),
    'nf.dst': ('Куда писать', 'Output folder'),
    'nf.recursive': ('Со вложенными папками', 'Include sub-folders'),
    'nf.auto': ('Проверить каждую и перевернуть только OpenGL',
                'Check each and flip only the OpenGL ones'),
    'nf.all': ('Перевернуть все без проверки', 'Flip all, no checking'),
    'nf.copy_dx': ('Копировать и те, что уже в DirectX (полный набор в папке вывода)',
                   'Also copy the ones already in DirectX (a full set in the output folder)'),
    'nf.start': ('Начать', 'Start'),
    'nf.stop': ('Остановить', 'Stop'),
    'nf.nosrc': ('Выберите папку с картами.', 'Choose the folder with the maps.'),
    'nf.same': ('Папка вывода не может совпадать с исходной или лежать внутри неё.',
                'The output folder must not be the source folder or lie inside it.'),
    'nf.none': ('В папке нет файлов, оканчивающихся на _n или _nh (.dds, .png, .tga, .bmp).',
                'No files ending in _n or _nh (.dds, .png, .tga, .bmp) in the folder.'),
    'nf.overwrite': ('В папке %s уже есть файлы. Совпадающие по имени будут перезаписаны. '
                     'Продолжить?', 'The folder %s already has files. Those with the same names '
                     'will be overwritten. Go on?'),
    'nf.found': ('Найдено карт: %d', 'Maps found: %d'),
    'nf.done': ('Найдено %d, перевёрнуто %d, уже DirectX %d, неясно %d, ошибок %d.',
                'Found %d, flipped %d, already DirectX %d, unclear %d, errors %d.'),
    'nf.report': ('Отчёт: %s', 'Report: %s'),
    'nf.r_src': ('Исходная папка: %s', 'Source: %s'),
    'nf.r_dst': ('Папка вывода: %s', 'Output: %s'),
    'nf.r_mode': ('Режим: %s; вложенные папки: %s', 'Mode: %s; sub-folders: %s'),
    'nf.r_stopped': ('Остановлено пользователем.', 'Stopped by the user.'),
    'nf.r_flipped': ('ПЕРЕВЁРНУТ', 'FLIPPED'),
    'nf.r_dx': ('уже DirectX', 'already DirectX'),
    'nf.r_copied': ('скопирован', 'copied'),
    'nf.r_unclear': ('НЕЯСНО, не тронут', 'UNCLEAR, left alone'),
    'nf.r_curl': ('проверка: как DX %.2f / как GL %.2f (меньше — верный)',
                  'check: as DX %.2f / as GL %.2f (smaller is right)'),
    'nf.r_error': ('ОШИБКА', 'ERROR'),
    'nf.r_inplace': ('в блоках, %s', 'in place, %s'),
    'nf.r_decoded': ('раскрыт и записан как DXT5: %s', 'decoded and written as DXT5: %s'),
    'nf.r_picture': ('картинка, зелёный обращён', 'picture, green inverted'),
    'ed.repeat_cut': ('Повтор %.0f юнитов не помещается в ползунок другого режима: стало %.0f.',
                      'A repeat of %.0f units does not fit the other mode\'s slider: it is now %.0f.'),
    'np.mats_manual': ('Свой порядок материалов на шкале', 'Own order of the materials on the scale'),
    'np.mats_manual.info': ('Ставится на шкале материалов перетаскиванием полосы; кнопка «По блеску» '
                            'снимает.', 'Set on the material scale by dragging a band; the «By '
                            'shininess» button clears it.'),
    'ed.mat.none': ('Материалы не выбраны — действуют ползунки вкладки «Диапазоны».',
                    'No materials chosen — the sliders of the «Ranges» tab apply.'),
    'np.clip_dull': ('Срез тусклого, % пикселей', 'Cut dull, % of pixels'),
    'np.clip_dull.info': ('Столько процентов самых шершавых пикселей не учитывается, когда '
                          'ищется разброс карты шероховатости: они просто получают самое тусклое '
                          'значение. Нужно, когда тусклее всего фон или мусор, который на модель '
                          'не ложится, — иначе он сжимает остальную текстуру к блестящему краю.',
                          'This share of the roughest pixels is left out when the spread of the '
                          'roughness map is found: they just get the dullest value. Useful when '
                          'the dullest thing is a background or junk that does not go on the '
                          'model — else it squeezes the rest of the texture to the shiny end.'),
    'np.clip_shiny': ('Срез блестящего, % пикселей', 'Cut shiny, % of pixels'),
    'np.clip_shiny.info': ('Столько процентов самых гладких пикселей не учитывается, когда ищется '
                           'разброс карты шероховатости: они получают самое блестящее значение. '
                           'Нужно, когда самым блестящим оказался фон, который на модель не '
                           'ложится, — иначе основная текстура бликует меньше, чем надо.',
                           'This share of the smoothest pixels is left out when the spread of the '
                           'roughness map is found: they get the shiniest value. Useful when the '
                           'shiniest thing is a background that does not go on the model — else '
                           'the main texture shines less than it should.'),
    'nd.antialias': ('Сглаживание лесенки (антиалиасинг)', 'Anti-aliasing'),
    'nd.antialias.info': ('Убирает ступеньки с краёв карты. Алгоритм на выбор: FXAA, сглаживание '
                          'вдоль контура или «размыть и перерезать» для масок и двухцветных карт.',
                          'Takes the staircase off edges. Algorithm to choose: FXAA, smoothing '
                          'along contours, or «blur and cut again» for masks and two-colour maps.'),
    'np.antialias.algo': ('Алгоритм', 'Algorithm'),
    'np.antialias.algo.info': ('FXAA (NVIDIA, Т. Лоттес) — ищет края по яркости и пересчитывает '
                               'пиксель со сдвигом поперёк края; быстрый общий вариант. «Вдоль '
                               'контура» — направление края по тензору структуры, размытие только '
                               'вдоль него: ступеньки уходят, резкость поперёк остаётся. «Маска» — '
                               'размыть и снова перерезать так, чтобы край был в один пиксель: '
                               'лучше всего для масок, альфы, порога, двух сплошных цветов; детали '
                               'тоньше радиуса пропадают. На детальной фактуре FXAA и контур '
                               'трогают и саму фактуру — смягчают её.',
                               'FXAA (NVIDIA, T. Lottes) — finds edges by brightness and resamples '
                               'the pixel shifted across the edge; a fast general choice. «Along '
                               'contours» — the edge direction from the structure tensor, blur only '
                               'along it: the stairs go, the sharpness across stays. «Mask» — blur '
                               'and cut again to a one-pixel edge: best for masks, alpha, '
                               'threshold, two flat colours; details thinner than the radius '
                               'fade. On detailed texture FXAA and contours touch the texture '
                               'itself too — soften it.'),
    'np.antialias.amount': ('Сила', 'Amount'),
    'np.antialias.amount.info': ('0…1. Для FXAA — насколько смягчать одиночные пиксельные '
                                 'ступеньки (в оригинале 0,75); для остальных — доля результата.',
                                 '0…1. For FXAA — how much single-pixel steps are softened (0.75 '
                                 'in the original); for the others — the share of the result.'),
    'np.antialias.radius': ('Радиус, пикс.', 'Radius, px'),
    'np.antialias.radius.info': ('Для «вдоль контура» и «маски»: масштаб ступенек в пикселях '
                                 'текстуры полного размера. FXAA его не использует.',
                                 'For «along contours» and «mask»: the scale of the stairs, in '
                                 'pixels of the full-size texture. FXAA does not use it.'),
    'nv.fxaa': ('FXAA', 'FXAA'),
    'nv.contour': ('вдоль контура', 'along contours'),
    'nv.mask': ('маска (размыть и перерезать)', 'mask (blur and cut)'),
    'ni.touched': ('Изменено пикселей (больше 1/255), %', 'Pixels changed (over 1/255), %'),
    'np.unshade.auto': ('Сила автоматически', 'Strength automatically'),
    'np.unshade.auto.info': ('Сила находится по самой картинке: насколько она на деле темнеет там, '
                             'где карта затенения говорит «тень» (сравниваются мелкие перепады '
                             'яркости, места у смены цвета не учитываются). «Сколько убрать» тогда — '
                             'множитель к найденному: 1 — как найдено.',
                             'The strength is found from the picture itself: how much it really '
                             'darkens where the shade map says «shade» (fine brightness changes '
                             'are compared, places at a change of colour are left out). «How much '
                             'to remove» then multiplies it: 1 — as found.'),
    'np.unshade.no_halo': ('Без ореолов', 'No halos'),
    'np.unshade.no_halo.info': ('Пиксель не высветляется ярче освещённых мест того же цвета рядом '
                                '(окно — 3 % размера). Убирает светлые каймы там, где карта '
                                'затенения темнее, чем картинка, — обычно у смены цвета. Большая '
                                'ровная тень шире окна может недосветлиться.',
                                'A pixel is not made brighter than the lit places of its own colour '
                                'nearby (window — 3 % of the size). Removes light rims where the '
                                'shade map is darker than the picture — usually at a change of '
                                'colour. A large even shadow wider than the window may stay a bit '
                                'dark.'),
    'unshade.size': ('Карта затенения должна быть того же размера, что картинка.',
                     'The shade map must be the size of the picture.'),
    'unshade.noauto': ('не найдена — по ползунку', 'not found — by the slider'),
    'ni.auto_s': ('Найденная сила', 'Strength found'),
    'ni.auto_used': ('Пикселей для оценки, %', 'Pixels for the estimate, %'),
    'ni.halo_px': ('Удержано от ореолов, % пикселей', 'Held back from halos, % of pixels'),
    'np.auto_az': ('Направление света — по картинке', 'Light direction from the picture'),
    'ni.angle_est': ('Свет по картинке, °', 'Light from the picture, °'),
    'ni.light_dir': ('Направленность света', 'Light directionality'),
    'ni.light_dir.info': ('0…1: насколько в текстуре виден свет с одной стороны. Ниже ~0.3 '
                          'направленного света нет, и рельеф по освещению выдаст пятна. Высокое '
                          'значение на кирпиче или дереве может давать сама кладка или волокна.',
                          '0…1: how much light from one side the texture shows. Below ~0.3 there '
                          'is no directional light, and height from lighting gives blobs. A high '
                          'value on brick or wood may come from the courses or the grain.'),
})
for _c in 'gba':
    S['np.out_pack.%s_ch.info' % _c] = S['np.out_pack.r_ch.info']
for _c in 'gb':
    S['np.out_pack.%s_val.info' % _c] = S['np.out_pack.r_val.info']
