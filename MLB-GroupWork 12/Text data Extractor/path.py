import os
import traceback



class PathError(Exception):
    def __init__(self, message, code):
        self.message = "PathError: " + message
        self.code = code
 
 
def check_path(path):
    """
    Check path.
    :param path: <str> Input path.
    :return: <str> path
    """
    if not os.path.exists(path):
        raise PathError("directory path url %s is not exist." %  path, 500)
    if not os.path.isdir(path):
        raise PathError("path url %s is not a directory." % path, 500)
    if path[-1] == "/":
        path = path.strip("/")
        path = "/" + path
 
    return path
 
 
def cd(path):
    """
    Traverse the directory and add the file path to the list.
    :param path: <str> Valid path.
    :return: <list> file_list
    """
    cd_list = os.listdir(path)
    file_list = []
    for ele in cd_list:
        temp_path = path + "/" + ele
        if os.path.isfile(temp_path):
            file_list.append(temp_path)
        else:
            pre_list = cd(temp_path)
            file_list.extend(pre_list)
    return file_list
 
 
def print_files(files):
    """
    Write path to txt file.
    :param files: <list> file list.
    :return: <None>
    """
    open("files.txt", "w").write("")
    if len(files) == 0:
        open("files.txt", "w",encoding='utf-8').write("None")
        print("write success.")
        return
 
    with open("files.txt", "w",encoding='utf-8') as txt_files:
        for file in files:
            txt_files.write(file + "\n")
    txt_files.close()
    print("write success.")
    return
 
 
# main method
path = r"C:\Users\24508\Desktop\Resume&Job_Description\Original_Resumes"
 
try:
    path = check_path(path)
    files = cd(path)
    print_files(files)
except PathError as e:
    print(e.message + " errcode " + str(e.code))
    print("errmag: \n%s" % traceback.format_exc())