from datetime import datetime

class Observer:
    def __init__(self, fs):
        self.subscribers = []
        self.fs = fs

    def subscribe(self, callback):
        """添加订阅者函数"""
        self.subscribers.append(callback)

    def unsubscribe(self, callback):
        """移除订阅者函数"""
        self.subscribers.remove(callback)

    def notify(self, data):
        """通知所有订阅者函数"""
        # sample_point_ranges = self.convert_to_sample_points(data)
        # for callback in self.subscribers:
        #     callback(sample_point_ranges)
        for callback in self.subscribers:
            callback(data)
            
    # def convert_to_sample_points(self, data):
    #     """将时间范围格式的数据转换为采样点格式"""
    #     sample_point_ranges = {}
    #     # print("convert_to_sample_points(start,end, range)", data)

    #     for index, (start, end) in data.items():
    #         print("convert_to_sample_points",data)
    #         # 假设时间格式是 "%H:%M:%S"
    #         start_time = datetime.strptime(start, '%H:%M:%S')
    #         end_time = datetime.strptime(end, '%H:%M:%S')
            
    #         # 转换为秒数
    #         start_seconds = start_time.hour * 3600 + start_time.minute * 60 + start_time.second
    #         end_seconds = end_time.hour * 3600 + end_time.minute * 60 + end_time.second

    #         start_sample = int(start_seconds * self.fs)
    #         end_sample = int(end_seconds * self.fs)

    #         sample_point_ranges[index] = (start_sample, end_sample)

    #     return sample_point_ranges

    