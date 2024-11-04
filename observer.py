class Observer:
    def __init__(self):
        self.subscribers = []

    def subscribe(self, callback):
        """添加订阅者函数"""
        self.subscribers.append(callback)

    def unsubscribe(self, callback):
        """移除订阅者函数"""
        self.subscribers.remove(callback)

    def notify(self, data):
        """通知所有订阅者函数"""
        for callback in self.subscribers:
            callback(data)