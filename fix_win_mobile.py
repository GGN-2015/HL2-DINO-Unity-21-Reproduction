import os

DIRNOW = os.path.dirname(os.path.abspath(__file__))

# 递归获取目录中的所有文件
# 忽略所有 .git 文件
def iter_all(folder_or_file_path:str) -> list[str]:

    # 获取绝对路径
    if not os.path.isabs(folder_or_file_path):
        folder_or_file_path = os.path.abspath(folder_or_file_path)

    # 文件
    if os.path.isfile(folder_or_file_path):
        return [folder_or_file_path]
    
    # 目录
    if os.path.isdir(folder_or_file_path):
        
        # 忽略 .git 目录
        if os.path.basename(folder_or_file_path).startswith(".git"):
            return []
        
        ans = []
        for filename in os.listdir(folder_or_file_path):
            newpath = os.path.join(folder_or_file_path, filename)
            ans += iter_all(newpath)
        return ans
    
    # 不知道什么类型
    return []

def select_suffix(list_of_str:list[str], suffix:str) -> list[str]:
    return [
        term
        for term in list_of_str
        if term.endswith(suffix)
    ]

def find_aim_vcxproj() -> str:
    BAN_SEGMENT = "Il2CppOutputProject"
    arr = []
    for x in select_suffix(iter_all(DIRNOW), ".vcxproj"):
        if x.find(BAN_SEGMENT) == -1:
            arr.append(x)
    if len(arr) != 1:
        raise FileNotFoundError("More than 1 vcxproj file found!")
    return arr[0]

def erase_win_mobile(vcxproj:str, encoding="utf-8") -> bool:
    if not os.path.isfile(vcxproj):
        raise FileNotFoundError(f"file {vcxproj} not found.")
    new_lines = []
    line_id = []
    with open(vcxproj, "r", encoding=encoding) as fp:
        content = fp.read()
        for idx, line in enumerate(content.split("\n")):
            new_lines.append(line)
            if line.find("WindowsMobile") != -1:
                line_id.append(idx)
    
    if len(line_id) == 0:
        return False
    
    else:

        # 删除附近的三行
        new_lines[line_id[0] - 1] = ""
        new_lines[line_id[0] + 0] = ""
        new_lines[line_id[0] + 1] = ""
        with open(vcxproj, "w", encoding=encoding) as fpout:
            fpout.write("\n".join(new_lines) + "\n")
        
        return True

if __name__ == "__main__":
    # 找到这个项目文件
    vcxproj = find_aim_vcxproj()

    # 删除 WindowsMobile 相关代码
    flag = erase_win_mobile(vcxproj)
    if flag:
        print("WindowsMobile ItemGroup Erased!")
    else:
        print("WindowsMobile ItemGroup Not Found!")
