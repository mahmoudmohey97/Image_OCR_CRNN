import os

class ReadData:
    def __init__(self, path):
        self.path = path
    
    def read_file(self, file_path):
        data = ""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = f.read()
        return data.strip()
    
    #Read txt files and save content in dictionary
    def parse_data_from_folder(self):
        text_data_dict = dict()
        files = os.listdir(self.path)
        
        for file in files:
            if file.endswith(".txt"):
                file_path = self.path + "\\" + file
                file_name = file.split(".")[0]
                file_content = self.read_file(file_path)
                text_data_dict[file_name] = file_content
        return text_data_dict