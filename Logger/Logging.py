from datetime import datetime


class LogClass:


    def __init__(self, folder_path, file_name):
        self.folder_path = folder_path
        self.file_name = file_name

    def log(self,log_type,log_message):

        try:
            self.now = datetime.now()
            self.date = self.now.date()
            self.current_time = self.now.strftime("%H:%M:%S")

            with open(self.folder_path + self.file_name, 'a+') as file:
                file.write(str(self.date) + "\t" + str(self.current_time) + "\t"+log_type+'\t' + log_message + "\n")
        except Exception as e:
            raise e
