import csv

with open('final.csv','w',encoding='utf-8',newline='') as f:

    # 基于文件对象构建 csv写入对象
    csv_writer = csv.writer(f)

    # 构建列表头
    csv_writer.writerow(["ID","Category","Resume"])

    # 写入csv文件内容
    id=0
    with open(r"txt_path.txt","r",encoding='utf-8') as txt_path:
        while 1:
            text_path = txt_path.readline()[:-1]
            if not text_path:
                break
            str=text_path
            str2=str[63:]
            end=str2.find("\\")
            category=str2[:end]
            with open(text_path,"r",encoding='gb18030') as f2:
                resume=''
                while 1:
                    ss=f2.readline()
                    if not ss:
                        break
                    resume=resume+' '.join(ss.lower().strip().split())
                exp_location=resume.find('experience')
                edu_location=resume.find('education',exp_location)
                if exp_location!=-1:
                    id=id+1
                    if edu_location!=-1:
                        ans=resume[exp_location:edu_location-1].encode('utf-8')
                    else:
                        ans=resume[exp_location:].encode('utf-8')
                    if len(ans)<200:
                        ans=resume[exp_location:].encode('utf-8')                       
                    csv_writer.writerow([id,category,ans])


