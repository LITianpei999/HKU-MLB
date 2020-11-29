import os
import sys
import fnmatch
import win32com.client
 
PATH = os.path.abspath(os.path.dirname(sys.argv[0]))
    # sys.argv[0] 为 F:/PyCharm/untitled5/doc_totxt.py
    # os.path.dirname(sys.argv[0]) 为 F:/PyCharm/untitled5
PATH_DATA = os.path.abspath(os.path.dirname(sys.argv[0]))   #word文档路径
 
 
def to_txt():
    wordapp = win32com.client.gencache.EnsureDispatch("Word.Application")
    try:
        for root, dirs, files in os.walk(PATH_DATA):
            for _dir in dirs:
                pass
            for _file in files:
                if not (fnmatch.fnmatch(_file, '*.doc') or fnmatch.fnmatch(_file, '*.docx')):
                    continue
                print('_file:',_file)
                file = os.path.join(root, _file)
                wordapp.Documents.Open(file)
                if fnmatch.fnmatch(_file, '*.doc'): #匹配doc文档
                    file = file[:-3] + 'txt'
                else:               #匹配docx文档
                    file = file[:-4] + 'txt'
                wordapp.ActiveDocument.SaveAs(file, FileFormat=win32com.client.constants.wdFormatText)
                wordapp.ActiveDocument.Close()
                #os.remove(os.path.join(root, _file)) #将word文档删除，使文件夹只剩下txt文档
    finally:
        wordapp.Quit()
    print("well done!")
 
if __name__ == '__main__':
    to_txt()