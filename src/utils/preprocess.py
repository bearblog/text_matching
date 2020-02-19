import re


def clear_character(sentence):
    pattern1 = '[a-z0-9]'
    pattern2 = '\[.*?\]'
    pattern3 = re.compile(u'[^\s1234567890:：' + '\u4e00-\u9fa5]+')
    pattern4 = '[’!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]+'
    line1 = re.sub(pattern1, '', sentence)  # 去除英文字母和数字
    line2 = re.sub(pattern2, '', line1)  # 去除表情
    line3 = re.sub(pattern3, '', line2)  # 去除其它字符
    new_sentence = re.sub(pattern4, '', line3)  # 去掉残留的冒号及其它符号
    # new_sentence = ''.join(line4.split())  # 去除空白
    return new_sentence
