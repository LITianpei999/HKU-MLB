import sys
# import os

import re

from zhon.hanzi import punctuation
from zhon.hanzi import characters

import chardet
def CheckCode(filename):
    adchar=chardet.detect(filename.encode())
    if adchar['encoding']=='utf-8':
        filename=filename.decode('utf-8')
    else:
        filename=filename.encode('utf-8').decode('gbk') 
    return filename

def lm_find_unchinese(file):
    pattern = re.compile(r'[\u4e00-\u9fa5]')
    unchinese = re.sub(pattern,"",file) #排除汉字
    unchinese = re.sub('[{}]'.format(punctuation),"",unchinese) #排除中文符号
    #print("unchinese:",unchinese)
    return unchinese


def lm_find_chinese(m_str):
    pattern = re.compile(r'[^\u4e00-\u9fa5]')
    chinese = re.sub(pattern, '', m_str)
    print("chinese:",chinese)

def lm_find_chinese_symbol(m_str):
    t_symbol = re.findall("[{}]".format(punctuation),m_str)
    print("chinese symbols:",t_symbol)

def lm_find_chinese_and_symbol(m_str):
    lm_find_unchinese(m_str)
    lm_find_chinese(m_str)
    lm_find_chinese_symbol(m_str)

def lm_delete_chinese_and_symbol(m_str):
    print("delete chinese and symbol")
    
# 测试用例
with open(r"txt_path.txt","r",encoding='utf-8') as txt_path:
    while 1:
        text_path = txt_path.readline()[:-1]
        if not text_path:
            break
        filename=text_path
        fname=CheckCode(filename)
        with open(fname,'r+',encoding='gb18030',errors='ignore') as fp:
            content = fp.read()
            print("查找到的中文符号：",re.findall("[{}]".format(punctuation),content))
            print("中文汉字：",re.findall("[{}]".format(characters),content))
            lm_find_chinese(content)
            unchinese =  lm_find_unchinese(content)
            fp.seek(0,0)
            fp.truncate()
            fp.write(unchinese)

def main(argv=None):
#    lm_find_chinese_and_symbol(line)
    print("main")

#if __name__ == "__main__":
#    sys.exit(main())